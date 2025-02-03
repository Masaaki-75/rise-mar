import math
import einops as E
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from modules.basic_layers import get_conv_block, get_norm_layer, get_act_layer, DropPath
from modules.rope import RotaryEmbedding, apply_rot_embed
AVAIL_FAST_ATTN = hasattr(F, 'scaled_dot_product_attention')


"""
Modified from ProCT
- Paper: Prompted Contextual Transformer for Incomplete-View CT Reconstruction
- Code: https://github.com/Masaaki-75/proct
"""


def init_weights_with_scale(modules, scale=1):
    if not isinstance(modules, list):
        modules = [modules]
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias.data, 0.0)
            

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.GELU(),
            nn.Conv2d(hidden_features, out_features, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            nn.init.trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
    
    
class SpatialFreqConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, padding_mode='reflect', norm_type=None, norm_kwargs=None, 
        act_type='RELU', act_kwargs=None, fft_type='ortho', reduction=2,):
        super().__init__()
        mid_channels = out_channels // reduction
        ffc_in_channels = ffc_out_channels = 2 * mid_channels
        conv_kwargs = dict(stride=1, padding_mode=padding_mode)
        block_kwargs = dict(norm_type=norm_type, norm_kwargs=norm_kwargs, act_type=act_type, act_kwargs=act_kwargs)
        self.fft_type = fft_type
        
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_in = get_conv_block(
            in_channels, mid_channels, kernel_size=1, bias=False, adn_order='CNA', **conv_kwargs, **block_kwargs)
        self.conv = nn.Conv2d(ffc_in_channels, ffc_out_channels, kernel_size=1, **conv_kwargs)
        self.norm = get_norm_layer(norm_type=norm_type, args=[ffc_out_channels])
        self.act = get_act_layer(act_type=act_type, kwargs=act_kwargs)
        self.conv_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False, **conv_kwargs)
        
        init_weights_with_scale([self.spatial, self.conv_in, self.conv, self.norm, self.conv_out], 0.1)
    
    def forward(self, x: torch.Tensor):
        out = self.spatial(x)
        x_fft = self.conv_in(x)
        ffted = torch.fft.rfftn(x_fft, dim=(-2, -1), norm=self.fft_type)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)  # (batch, c, h, w/2+1, 2)
        ffted = E.rearrange(ffted, "B C H W L -> B (C L) H W")#.contiguous()
        
        ffted = self.conv(ffted)
        ffted = self.act(self.norm(ffted))
        ffted = E.rearrange(ffted, "B (C L) H W -> B C H W L", L=2)#.contiguous()
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        
        ifft_shape_slice = x_fft.shape[2:]
        out_fft = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=(-2, -1), norm=self.fft_type)
        out_fft = self.conv_out(x_fft + out_fft)
        return out + out_fft


class SpatialConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, padding_mode='reflect', norm_type=None, norm_kwargs=None, 
        act_type='RELU', act_kwargs=None, fft_type='ortho', reduction=2,):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        out = self.spatial(x)
        return out
        
        
class RLN(nn.Module):
    """
    Revised LayerNorm, modified from DehazeFormer
    """
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        nn.init.trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)
        nn.init.trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, x: torch.Tensor):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True, unbiased=True)
        # possible solution to NaN Loss: unbiased=False:
        # https://stackoverflow.com/questions/66542007/transformer-model-output-nan-values-in-pytorch
        # std = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        normalized_x = (x - mean) / (std + self.eps)

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_x * self.weight + self.bias
            
        return out, rescale, rebias
    
    
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, use_rope=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope
        
        if not use_rope:
            relative_positions = self.get_relative_positions(self.window_size)
            self.register_buffer("relative_positions", relative_positions)
            self.meta = nn.Sequential(
                nn.Linear(2, 256, bias=True),
                nn.GELU(),
                nn.Linear(256, num_heads, bias=True))
        else:
            self.relative_positions = RotaryEmbedding(head_dim, max_res=512)

    def forward(self, qkv:torch.Tensor, use_fast_attn=False):
        B_, N, _ = qkv.shape
        H = W = int(math.sqrt(N))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: [3, B_, Nh, N, C//N_h]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        if not self.use_rope:
            relative_position_bias = self.meta(self.relative_positions)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn_bias = relative_position_bias.unsqueeze(0)
        else:
            sin_emb, cos_emb = self.relative_positions.get_embed((H, W))
            q = apply_rot_embed(q, sin_emb, cos_emb)
            k = apply_rot_embed(k, sin_emb, cos_emb)
            attn_bias = None
        
        # q, k, v: [B_, Nh, N, C//N_h]
        if AVAIL_FAST_ATTN and use_fast_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale, attn_mask=attn_bias)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = (attn + attn_bias) if attn_bias is not None else attn
            attn = attn.softmax(dim=-1)
            x = attn @ v
            
        x = x.transpose(1, 2).reshape(B_, N, self.dim)
        
        return x

    def get_relative_positions(self, window_size):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())
        return relative_positions_log


class TokenMixer(nn.Module):
    def __init__(
        self, network_depth, dim, num_heads, window_size, shift_size, 
        use_attn=False, use_rope=False, use_spectral=False, 
        drop_proj=0., **block_kwargs):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size
        self.network_depth = network_depth  # for initialization of MLP
        self.use_attn = use_attn
        block_kwargs_ = dict(norm_type=None, norm_kwargs=None, act_type='RELU', act_kwargs=None)
        block_kwargs_.update(block_kwargs)
        conv_op = SpatialFreqConv if use_spectral else SpatialConv
        self.conv = conv_op(dim, dim, padding_mode='reflect', **block_kwargs_)
        self.V = nn.Conv2d(dim, dim, 1)
        self.QK = nn.Conv2d(dim, dim * 2, 1) if use_attn else None
        self.attn = WindowAttention(dim, window_size, num_heads, use_rope=use_rope) if use_attn else None
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop_proj = nn.Dropout(drop_proj) if drop_proj > 0. else nn.Identity()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape
            
            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std)        
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        
        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def window_partition(self, x, window_size):
        # "B H W C -> B H/win win W/win win C -> B H/win W/win win win C -> (B*H*W/win/win) win*win C"
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def forward(self, x, **attn_kwargs):
        # x: [B, Ciq, H, W],
        H, W = x.shape[-2:]
        V = self.V(x)  # [B, Ciq, H, W]
        attn_out = None
        
        if self.use_attn:
            QK = self.QK(x)
            QKV = torch.cat([QK, V], dim=1)
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = self.window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
            attn_windows = self.attn(qkv, **attn_kwargs)
            shifted_out = self.window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
            attn_out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            attn_out = attn_out.permute(0, 3, 1, 2)
        
        conv_out = self.conv(V)
        x_out = self.proj(conv_out) if attn_out is None else self.proj(conv_out + attn_out)
        x_out = self.drop_proj(x_out)
        return x_out
    
    

class TransformerBlock(nn.Module):
    """
    norm -> attn -> res -> norm -> mlp -> res
    """
    def __init__(
        self, network_depth, dim, num_heads, mlp_ratio=4., norm_layer=RLN, 
        prompt_dim=24, window_size=8, shift_size=0, drop_proj=0., drop_path=0., 
        use_attn=True, use_rope=False, use_spectral=False,
        use_bounded_scale=False, use_scale_for_mlp=True, **block_kwargs):
        super().__init__()
        self.use_attn = use_attn
        self.use_scale_for_mlp = use_scale_for_mlp
        self.norm = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = TokenMixer(
            network_depth, dim, num_heads=num_heads, window_size=window_size, shift_size=shift_size, drop_proj=drop_proj, 
            use_attn=use_attn, use_rope=use_rope, use_spectral=use_spectral,  **block_kwargs)
        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.local_scale = None
        if prompt_dim > 0:
            self.local_scale = nn.Sequential(nn.Linear(prompt_dim, dim, bias=True), nn.Tanh()) if use_bounded_scale else nn.Linear(prompt_dim, dim, bias=True)

        init_weights_with_scale([self.norm, self.attn, self.mlp], 1)
        init_weights_with_scale([self.local_scale], 0.1)
    
    
    def forward(self, x, cond=None, **attn_kwargs):
        if self.use_attn: 
            y, rescale, rebias = self.norm(x)
            y = self.attn(y, **attn_kwargs)
            y = y * rescale + rebias
        else:
            y = self.attn(x, **attn_kwargs)
        
        if cond is not None and self.local_scale is not None:
            local_scale = self.local_scale(cond)  # TODO: check size, check where to merge
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
        x = x + self.drop_path(y)
        
        y = self.mlp(x)
        if cond is not None and self.local_scale is not None and self.use_scale_for_mlp:
            y = y * local_scale.view(-1, y.shape[1], 1, 1)
        x = x + self.drop_path(y)
        
        return x
    
    
    
class BasicLayer(nn.Module):
    def __init__(
        self, network_depth, dim, depth, num_heads, mlp_ratio=4., window_size=8, 
        attn_ratio=0., attn_loc='last', drop_proj=0., drop_path=0.,
        use_rope=False, use_spectral=True, prompt_dim=24, 
        use_bounded_scale=False, use_scale_for_mlp=False, **block_kwargs):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth  # [8, 8, 8, 4, 4]
        norm_layer = RLN
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        else:
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(
                network_depth=network_depth,
                dim=dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_proj=drop_proj,
                drop_path=dpr[i],
                use_attn=use_attns[i], 
                use_rope=use_rope,
                use_spectral=use_spectral,
                prompt_dim=prompt_dim, 
                use_bounded_scale=use_bounded_scale,
                use_scale_for_mlp=use_scale_for_mlp,
                **block_kwargs)
            for i in range(depth)])

    def forward(self, x, cond=None, **attn_kwargs):        
        for blk in self.blocks:
            x = blk(x, cond=cond, **attn_kwargs)
        return x



class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = patch_size if kernel_size is None else kernel_size
        padding = (kernel_size - patch_size + 1) // 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=kernel_size, stride=patch_size,
            padding=padding, padding_mode='reflect')

        init_weights_with_scale(self.proj, 0.1)
        
    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_channels=3, embed_dim=96, kernel_size=None, output_padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        kernel_size = patch_size if kernel_size is None else kernel_size
        padding = (kernel_size - patch_size + output_padding) // 2
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=kernel_size, stride=patch_size,
            padding=padding, output_padding=output_padding)  
        # only zero-pad is supported for convt, so better not padding

        init_weights_with_scale(self.proj, 0.1)
        
    def forward(self, x):
        return self.proj(x)


class FeatureFuser(nn.Module):
    def __init__(self, in_dims, out_dim=None, fusion_type='cat', use_se_fusion=True, reduction=8):
        super().__init__()
        if fusion_type == 'add':
            assert all(d == in_dims[0] for d in in_dims)
            in_dim = in_dims[0]
        else:
            in_dim = sum(in_dims)
        out_dim = in_dim if out_dim is None else out_dim
        self.fusion_type = fusion_type
        if use_se_fusion:
            from modules.basic_blocks import SEBlock
            self.se = SEBlock(in_dim, reduction=reduction)
        else: 
            self.se = nn.Identity()
        self.conv = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        init_weights_with_scale([self.se, self.conv])
    
    @staticmethod
    def assert_same_shape(tensor_tuple, dim='all'):
        shape = tensor_tuple[0].shape
        if dim == 'all':
            assert all(_.shape == shape for _ in tensor_tuple)
        else:
            assert all(_.shape[dim] == shape[dim] for _ in tensor_tuple)
    
    def forward(self, *in_feats):
        if self.fusion_type == 'cat':
            self.assert_same_shape(in_feats, dim=-3)
            x = torch.cat(in_feats, dim=-3)
        elif self.fusion_type == 'add':
            self.assert_same_shape(in_feats)
            for y in in_feats:
                x = torch.add(x, y)
        x = self.se(x)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, 
        in_channels=1,  
        window_size=8,
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        drop_proj_rates=[0., 0., 0., 0., 0.],
        drop_path_rates=[0., 0., 0., 0., 0.],
        use_rope=False,
        use_spectrals=[True, True, True, True, True],
        prompt_dim=24, 
        use_bounded_scale=True, 
        use_scale_for_mlp=True,
        **block_kwargs):
        super().__init__()
        
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        network_depth = sum(depths)
        common_params = dict(
            network_depth=network_depth, attn_loc='last', window_size=window_size, prompt_dim=prompt_dim, 
            use_bounded_scale=use_bounded_scale, use_scale_for_mlp=use_scale_for_mlp, use_rope=use_rope)
        layer_params = [
            dict(
                dim=embed_dims[i], depth=depths[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratio[i],  drop_proj=drop_proj_rates[i], drop_path=drop_path_rates[i], 
                use_spectral=use_spectrals[i]) for i in range(len(embed_dims))
        ]
            
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(patch_size=1, in_channels=in_channels, embed_dim=embed_dims[0], kernel_size=3)
        
        self.enc_layer1 = BasicLayer(**common_params, **layer_params[0], **block_kwargs)
        self.enc_down1 = PatchEmbed(patch_size=2, in_channels=embed_dims[0], embed_dim=embed_dims[1])  # downsampling
        
        self.enc_layer2 = BasicLayer(**common_params, **layer_params[1], **block_kwargs)
        self.enc_down2 = PatchEmbed(patch_size=2, in_channels=embed_dims[1], embed_dim=embed_dims[2])  # downsampling
        
        # bottleneck
        self.neck = BasicLayer(**common_params, **layer_params[2], **block_kwargs)

    def forward(self, x,  cond=None, **attn_kwargs):
        y = self.patch_embed(x)  # [B, 24, H, W]
        
        y = self.enc_layer1(y, cond=cond, **attn_kwargs)  # [B, 24, H, W]
        s1 = y
        y = self.enc_down1(y)  # [B, 48, H/2, W/2]
        
        y = self.enc_layer2(y, cond=cond, **attn_kwargs)  # [B, 48, H/2, W/2]
        s2 = y
        y = self.enc_down2(y)  # [B, 96, H/4, W/4]
       
        y = self.neck(y, cond=cond, **attn_kwargs)  # [B, 96, H/4, W/4]
        return y, s1, s2


if __name__ == '__main__':
    
    layer = PatchUnEmbed(patch_size=1, out_channels=1, embed_dim=16)
    x = torch.randn((2, 16, 15, 15))
    y = layer(x)
    print(y.shape)
    