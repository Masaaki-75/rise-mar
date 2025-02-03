import os
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from warnings import warn
from networks.proct import Encoder, init_weights_with_scale
from utilities.clip_models import load as load_clip_model
from utilities.misc import ensure_tuple_rep
from configs import CLIP_RESNET_PATH, VGG_PATH



class CQAEncoder(Encoder):
    def forward(self, x, cond=None, use_cp=False, **attn_kwargs):
        y = self.patch_embed(x)  # [B, 24, H, W]
        
        def _inner_forward(y):
            y = self.enc_layer1(y, cond=cond, **attn_kwargs)  # [B, 24, H, W]
            y = self.enc_down1(y)  # [B, 48, H/2, W/2]
            
            y = self.enc_layer2(y, cond=cond, **attn_kwargs)  # [B, 48, H/2, W/2]
            y = self.enc_down2(y)  # [B, 96, H/4, W/4]
        
            y = self.neck(y, cond=cond, **attn_kwargs)  # [B, 96, H/4, W/4]
            return y
        
        if use_cp:
            return cp.checkpoint(_inner_forward, y, use_reentrant=False)
        else:
            return _inner_forward(y)
        
    def forward_multiscale(self, x, cond=None, use_cp=False, **attn_kwargs):
        y = self.patch_embed(x)  # [B, 24, H, W]
        
        def _inner_forward(y):
            y = self.enc_layer1(y, cond=cond, **attn_kwargs)  # [B, 24, H, W]
            s1 = y
            
            y = self.enc_down1(y)  # [B, 48, H/2, W/2]
            y = self.enc_layer2(y, cond=cond, **attn_kwargs)  # [B, 48, H/2, W/2]
            s2 = y
            
            y = self.enc_down2(y)  # [B, 96, H/4, W/4]
            y = self.neck(y, cond=cond, **attn_kwargs)  # [B, 96, H/4, W/4]
            
            size = y.shape[2:]
            y = torch.cat((
                F.interpolate(s1, size, mode='nearest-exact'),
                F.interpolate(s2, size, mode='nearest-exact'),
                y
            ), dim=1)
            return y
        
        if use_cp:
            return cp.checkpoint(_inner_forward, y, use_reentrant=False)
        else:
            return _inner_forward(y)


class CQA(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1, 
        window_size=8, 
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        drop_proj_rates=[0., 0., 0., 0., 0.],
        drop_path_rates=[0., 0., 0., 0., 0.],
        prompt_dim=-1,
        prompt_len=360,
        use_rope=True,
        use_spectrals=[True, True, True, False, False],
        use_bounded_scale=True,
        use_scale_for_mlp=True,
        use_global_scale=False,
        block_kwargs={'norm_type':'INSTANCE'},
        do_multiscale=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        self.prompt_dim = prompt_dim
        self.prompt_len = prompt_len
        
        assert embed_dims[1] == embed_dims[3]
        assert embed_dims[0] == embed_dims[4]
        use_spectrals = ensure_tuple_rep(use_spectrals, 5)
        drop_path_rates = ensure_tuple_rep(drop_path_rates, 5)
        drop_proj_rates = ensure_tuple_rep(drop_proj_rates, 5)
        former_dict = dict(
            window_size=window_size, embed_dims=embed_dims, mlp_ratios=mlp_ratios, depths=depths, num_heads=num_heads, 
            attn_ratio=attn_ratio, drop_proj_rates=drop_proj_rates, drop_path_rates=drop_path_rates, use_rope=use_rope,
            use_bounded_scale=use_bounded_scale, use_scale_for_mlp=use_scale_for_mlp,
            prompt_dim=prompt_dim)
        encoder_dict = dict(use_spectrals=use_spectrals)
        self.encoder = CQAEncoder(in_channels=in_channels, **encoder_dict, **former_dict, **block_kwargs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        proj_in_dim = sum(embed_dims[:3]) if do_multiscale else embed_dims[2]
        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out_channels))
        
        self.do_multiscale = do_multiscale

        if prompt_dim > 0:
            self.squeeze = nn.Linear(prompt_len, prompt_dim, bias=True)
            self.global_scale = nn.Linear(prompt_dim, out_channels, bias=True) if use_global_scale else None
            init_weights_with_scale([self.squeeze, self.global_scale], 0.1)
        else:
            self.squeeze = self.global_scale = None
            
        self.memory_length = 300
        self.memory_device = 'cuda'
        self.memory_bank = {
            'proj': [None for _ in range(self.memory_length)],
            'label': [None for _ in range(self.memory_length)]
        }
        self.memory_static = 0

    @staticmethod
    def pad_image(x, patch_size):
        h, w = x.shape[-2:]
        mod_pad_h = (patch_size - h % patch_size) % patch_size
        mod_pad_w = (patch_size - w % patch_size) % patch_size
        if mod_pad_h > 0 or mod_pad_w > 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def _forward_features(self, x, cond=None, use_cp=False, **attn_kwargs):
        y = self.encoder.patch_embed(x)  # [B, 24, H, W]
        
        def _inner_forward(y):
            y = self.encoder.enc_layer1(y, cond=cond, **attn_kwargs)  # [B, 24, H, W]
            s1 = y
            
            y = self.encoder.enc_down1(y)  # [B, 48, H/2, W/2]
            y = self.encoder.enc_layer2(y, cond=cond, **attn_kwargs)  # [B, 48, H/2, W/2]
            s2 = y
            
            y = self.encoder.enc_down2(y)  # [B, 96, H/4, W/4]
            y = self.encoder.neck(y, cond=cond, **attn_kwargs)  # [B, 96, H/4, W/4]
            
            size = y.shape[2:]            
            if self.do_multiscale:
                y = torch.cat((F.interpolate(s1, size), F.interpolate(s2, size), y), dim=1)
            return y
        
        if use_cp:
            return cp.checkpoint(_inner_forward, y, use_reentrant=False)
        else:
            return _inner_forward(y)
    
    def forward_feature(self, x, cond=None, use_cp=False, do_multiscale=None, **attn_kwargs):
        # - x: the image to be reconstructed. [B, 1, H, W]
        # - context: the contextual pair(s), can be a simple phantom. [B, S, 2, H, W]
        # - cond: the view-aware prompts. [B, 1, Nv]
        do_multiscale = self.do_multiscale if do_multiscale is None else do_multiscale
        
        if cond is not None:
            assert cond.shape[-1] == self.prompt_len, f'Expected prompt length {self.prompt_len}, got {cond.shape}.'
        if self.squeeze is not None:
            cond = self.squeeze(cond).reshape(cond.shape[0], -1).contiguous()  # [B, 1, Nv] -> [B,1,C]
            
        x = self.pad_image(x, self.patch_size)
        y = self._forward_features(x, cond=cond, use_cp=use_cp, **attn_kwargs)
        return y
    
    def vectorize(self, features):
        features = self.pool(features)
        projections = torch.flatten(features, start_dim=1)
        projections = F.normalize(projections, dim=-1)
        return projections
    
    def forward(self, x, cond=None, use_cp=False, **attn_kwargs):
        y = self.forward_feature(x, cond, use_cp=use_cp, **attn_kwargs)
        y = self.vectorize(y)
        y = self.proj(y)
        return y
    
    def forward_all(self, x, cond=None, use_cp=False, **attn_kwargs):
        z = self.forward_feature(x, cond, use_cp=use_cp, **attn_kwargs)
        z = self.vectorize(z)
        y = self.proj(z)
        return y, z
    
    def update_memory_bank(self, proj, label):
        # proj shape: (batch, dim)
        # label shape: (batch,)
        proj = proj.detach().to(self.memory_device)
        label = label.detach().to(self.memory_device)
        batch_size = proj.shape[0]
        for i in range(batch_size):
            new_proj = proj[i, ...].unsqueeze(0)
            new_label = label[i, ...].unsqueeze(0)
            self.memory_bank['proj'].pop()
            self.memory_bank['proj'].insert(0, new_proj)
            self.memory_bank['label'].pop()
            self.memory_bank['label'].insert(0, new_label)
            self.memory_static = min(self.memory_static + 1, self.memory_length)
    
    def get_multiclass_contrastive_loss(self, proj_list, label_list, temp=0.1, eps=1e-6):
        if self.memory_static < self.memory_length:
            self.update_memory_bank(proj_list[0], label_list[0])
            return torch.tensor(0)
        
        z = torch.cat((*proj_list, *self.memory_bank['proj']), dim=0)
        label = torch.cat((*label_list, *self.memory_bank['label']), dim=0)
        label = label.squeeze(1).long()
        num_projs = z.shape[0]
        self.update_memory_bank(proj_list[0], label_list[0])
        
        sim = torch.mm(z, z.T) / temp
        exp_sim = torch.exp(sim - torch.max(sim, dim=1, keepdim=True)[0]) + eps
        mask_pos: torch.Tensor = (label.unsqueeze(1) == label.unsqueeze(0)).float()
        mask_pos = mask_pos.fill_diagonal_(0).to(sim.device)
        mask_anchor_out = torch.ones((num_projs, num_projs)).fill_diagonal_(0).to(sim.device)
        pos_cnt = torch.sum(mask_pos, dim=-1)
        idx = torch.where(pos_cnt > 0)[0]
        
        if len(idx) == 0:
            return torch.tensor(0)
        
        log_prob = - torch.log(exp_sim / (torch.sum(exp_sim * mask_anchor_out, dim=1, keepdim=True)))
        loss_per_sample = torch.sum(log_prob * mask_pos, dim=1)[idx] / pos_cnt[idx]
        loss = torch.mean(loss_per_sample)
        return loss

    def get_contrastive_loss(self, lq_projs, hq_projs, temp=0.1, eps=1e-6):
        B = lq_projs.shape[0]
        z = torch.cat((lq_projs, hq_projs), dim=0)
        sim = torch.mm(z, z.T) / temp
        exp_sim = torch.exp(sim - torch.max(sim, dim=1, keepdim=True)[0]) + eps
        
        label = torch.cat([torch.zeros(B), torch.ones(B)]).to(z.device)
        mask_pos = (label.unsqueeze(1) == label.unsqueeze(0)).float()
        mask_pos = mask_pos.fill_diagonal_(0).to(sim.device)
        mask_anchor_out = torch.ones((2*B, 2*B)).fill_diagonal_(0).to(sim.device)
        pos_cnt = torch.sum(mask_pos, dim=-1)
        idx = torch.where(pos_cnt > 0)
        
        log_prob = - torch.log(exp_sim / (torch.sum(exp_sim * mask_anchor_out, dim=1, keepdim=True)))
        loss_per_sample = torch.sum(log_prob * mask_pos, dim=1)[idx] / pos_cnt[idx]
        loss = torch.mean(loss_per_sample)
        
        return loss


class CQACLIP(nn.Module):
    def __init__(
        self, 
        in_channels=1,
        out_channels=10, 
        weight_path=CLIP_RESNET_PATH,
        memory_length=300,
        **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = load_clip_model(weight_path, device='cpu').visual.to('cpu')
        embed_dims = [64, 256, 512, 1024, 2048]
        #(64, H/4, W/4), (256, H/4, W/4), (512, H/8, W/8), (1024, H/16, W/16), (2048, H/32, W/32)
        min_img_size = 8
        self.pool = nn.AdaptiveAvgPool2d(min_img_size)
        proj_in_dim = sum(embed_dims) * min_img_size * min_img_size

        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out_channels))
        
        self.memory_length = memory_length
        self.memory_device = 'cuda'
        self.memory_bank = {
            'proj': [None for _ in range(self.memory_length)],
            'label': [None for _ in range(self.memory_length)]
        }
        self.memory_static = 0
        
    def forward_feature(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feature_list = self.encoder.forward_features(x)
        feature = torch.cat([self.pool(f) for f in feature_list], dim=1)
        return feature
    
    def vectorize(self, features):
        projections = torch.flatten(features, start_dim=1)
        projections = F.normalize(projections, dim=-1)
        return projections
    
    def forward(self, x):
        z = self.forward_feature(x)
        z = self.vectorize(z)
        y = self.proj(z)
        return y
    
    def forward_all(self, x):
        z = self.forward_feature(x)
        z = self.vectorize(z)
        y = self.proj(z)
        return y, z
    
    def update_memory_bank(self, proj, label):
        # proj shape: (batch, dim)
        # label shape: (batch,)
        proj = proj.detach().to(self.memory_device)
        label = label.detach().to(self.memory_device)
        batch_size = proj.shape[0]
        for i in range(batch_size):
            new_proj = proj[i, ...].unsqueeze(0)
            new_label = label[i, ...].unsqueeze(0)
            self.memory_bank['proj'].pop()
            self.memory_bank['proj'].insert(0, new_proj)
            self.memory_bank['label'].pop()
            self.memory_bank['label'].insert(0, new_label)
            self.memory_static = min(self.memory_static + 1, self.memory_length)
    
    def get_multiclass_contrastive_loss(self, proj_list, label_list, temp=0.1, eps=1e-6):
        if self.memory_static < self.memory_length:
            self.update_memory_bank(proj_list[0], label_list[0])
            return torch.tensor(0)
        
        z = torch.cat((*proj_list, *self.memory_bank['proj']), dim=0)
        label = torch.cat((*label_list, *self.memory_bank['label']), dim=0)
        label = label.squeeze(1).long()
        num_projs = z.shape[0]
        self.update_memory_bank(proj_list[0], label_list[0])

        sim = torch.mm(z, z.T) / temp
        exp_sim = torch.exp(sim - torch.max(sim, dim=1, keepdim=True)[0]) + eps
        mask_pos = (label.unsqueeze(1) == label.unsqueeze(0)).float()
        mask_pos = mask_pos.fill_diagonal_(0).to(sim.device)
        mask_anchor_out = torch.ones((num_projs, num_projs)).fill_diagonal_(0).to(sim.device)
        pos_cnt = torch.sum(mask_pos, dim=-1)
        idx = torch.where(pos_cnt > 0)[0]
        
        if len(idx) == 0:
            return torch.tensor(0)
        
        log_prob = - torch.log(exp_sim / (torch.sum(exp_sim * mask_anchor_out, dim=1, keepdim=True)))
        loss_per_sample = torch.sum(log_prob * mask_pos, dim=1)[idx] / pos_cnt[idx]
        loss = torch.mean(loss_per_sample)
        return loss
    


class CQAVGG(CQACLIP):
    def __init__(
        self, 
        in_channels=1,
        out_channels=10,
        weight_path=VGG_PATH,
        memory_length=300
    ):
        super().__init__()
        from torchvision.models import vgg16
        vgg_model = vgg16(pretrained=False)
        if weight_path is not None and os.path.exists(weight_path):
            vgg_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        else:
            warn(f'Checkpoint for VGG does not exist: {weight_path}')
            
        for param in vgg_model.parameters():
            param.requires_grad = True
        self.encoder = vgg_model.features[:16]
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        
        self.in_channels = in_channels
        embed_dims = [64, 128, 256]
        #(64, H, W), (128, H/2, W/2), (256, H/4, W/4),
        min_img_size = 8
        self.pool = nn.AdaptiveAvgPool2d(min_img_size)
        proj_in_dim = sum(embed_dims) * min_img_size * min_img_size

        self.proj = nn.Sequential(
            nn.Linear(proj_in_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out_channels))
        
        self.memory_length = memory_length
        self.memory_device = 'cuda'
        self.memory_bank = {
            'proj': [None for _ in range(self.memory_length)],
            'label': [None for _ in range(self.memory_length)]
        }
        self.memory_static = 0
        
    def forward_feature(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        output = {}
        for name, module in self.encoder._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
                
        feature_list = list(output.values())
        feature = torch.cat([self.pool(f) for f in feature_list], dim=1)
        return feature
    