import os
import sys
sys.path.append("..")
import tqdm
import random
from warnings import warn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from utilities.logger import LongziliLogger
from trainers.basic_mar_trainer import BasicMARTrainer, set_requires_grad
from datasets.risemar_dataset import RiseMARDataset
from networks.mar import RiseMARNet
from configs import (
    PAIRED_META1, PAIRED_META2, PAIRED_META3, 
    UNPAIRED_META1, UNPAIRED_META2, UNPAIRED_META3,
    UNDERTRAINED_WEIGHTS, SEED
)


class CQATrainer(BasicMARTrainer):
    def __init__(self, opt, net, **kwargs):
        super().__init__(opt=opt)
        self.opt = opt
        self.net = net
        self.prepare_dataset()
        self.prepare_methods()
    
    def prepare_dataset(self):
        opt = self.opt
        common_kwargs = dict(
            seed=SEED, min_hu=opt.min_hu, max_hu=opt.max_hu, 
            num_train=opt.num_train, num_val=opt.num_val)
        train_kwargs = dict(flip_prob=opt.flip_prob, rot_prob=opt.rot_prob, mode='train')
        
        self.train_dataset = RiseMARDataset(
            deepl_json=PAIRED_META1, unpaired_json1=UNPAIRED_META1,
            dental_json=PAIRED_META2, unpaired_json2=UNPAIRED_META2, 
            pelvic_json=PAIRED_META3, unpaired_json3=UNPAIRED_META3,
            **train_kwargs, **common_kwargs)
        self.val_dataset = RiseMARDataset(
            deepl_json=PAIRED_META1, unpaired_json1=UNPAIRED_META1,
            dental_json=PAIRED_META2, unpaired_json2=UNPAIRED_META2, 
            pelvic_json=PAIRED_META3, unpaired_json3=UNPAIRED_META3,
            mode='val', **common_kwargs)
        
    def prepare_methods(self):
        self.mu2hu = self.train_dataset.cttool.mu2hu
        self.hu2mu = self.train_dataset.cttool.hu2mu
        self.normalize_hu = self.train_dataset.cttool.normalize_hu
        self.denormalize_hu = self.train_dataset.cttool.denormalize_hu
        
    def prepare_logger(self):
        opt = self.opt
        self.logger = LongziliLogger(
            log_name=str(opt.tensorboard_dir),
            project_name=opt.wandb_project,
            config_opt=opt,
            checkpoint_root_path=opt.checkpoint_root,
            tensorboard_root_path=opt.tensorboard_root,
            wandb_root_path=opt.wandb_root,
            use_wandb=opt.use_wandb,
            log_interval=opt.log_interval,)
    
    def prepare_dataloader(self):
        opt = self.opt
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
            sampler=train_sampler, drop_last=True, pin_memory=True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=opt.num_workers, sampler=val_sampler,)
    
    def prepare_optimizer(self):
        self.optimizer = self.get_optimizer(self.net.parameters(), 'adam') 
        self.scheduler = self.get_scheduler(self.optimizer, 'mstep')
        if self.opt.load_opt:
            self.resume(mode='opt')
    
    def prepare_auxiliary_nets(self):
        self.aux_weight_paths = UNDERTRAINED_WEIGHTS
        self.aux_nets = dict()
        aux_net_dict = dict(base_channels=64, norm_type='INSTANCE', act_type='RELU')

        # Check if the undertrained network weights are correctly set.
        anatomies = list(self.aux_weight_paths.keys())
        for anatomy in anatomies:
            quality_levels = list(self.aux_weight_paths[anatomy].keys())
            self.aux_nets[anatomy] = {level:None for level in quality_levels}
            for level in quality_levels:
                paths = self.aux_weight_paths[anatomy][level]
                for path in paths:
                    if not os.path.exists(path):
                        print(f"Warnings: pretrained weight for undertrained network ({anatomy}, {level}) does not exist: {path}")
                        self.aux_weight_paths[anatomy][level].remove(path)
                
                if len(self.aux_weight_paths[anatomy][level]) > 0:
                    net = self.sync_network_batchnorm(RiseMARNet(1, 1, **aux_net_dict))
                    set_requires_grad(net, False)
                    self.aux_nets[anatomy][level] = net.to(self.device).eval()
        
    def load_random_auxiliary_nets(self,):
        anatomies = list(self.aux_weight_paths.keys())
        for anatomy in anatomies:
            quality_levels = list(self.aux_weight_paths[anatomy].keys())
            for level in quality_levels:
                paths = self.aux_weight_paths[anatomy][level]
                if len(paths) > 0:
                    aux_ckpt_path = random.choice(paths)
                    weights = torch.load(aux_ckpt_path, map_location='cuda')['net']
                    self.aux_nets[anatomy][level].load_state_dict(weights)

    @torch.no_grad()
    def infer_auxillary_net(self, x, flag, use_lq=False):
        batch_size = x.shape[0]
        y = x.clone().detach()
        key = 'low-quality' if use_lq else 'mid-quality'
        
        for b in range(batch_size):
            if (flag[b] == 0 or flag[b] == 2) and (self.aux_nets['torso'][key] is not None):
                y[b:b+1] = self.aux_nets['torso'][key](x[b:b+1])
            elif (flag[b] == 1) and (self.aux_nets['dental'][key] is not None):
                y[b:b+1] = self.aux_nets['dental'][key](x[b:b+1])
                
        return y.clip(0, 1)
        
    def move_network_to_cuda(self, net, use_ddp=True):
        net = net.to(self.device)
        if use_ddp:
            net = torch.nn.parallel.DistributedDataParallel(
                net, device_ids=[self.opt.local_rank], output_device=self.opt.local_rank,
                broadcast_buffers=False) #  find_unused_parameters=True
            #[NOTE] broadcast_buffers=False can somehow prevent the INPLACE ERROR
        return net
    
    def sync_network_batchnorm(self, net):
        try:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        except Exception as err:
            warn(f'Failed to sync batchnorm due to the following error: {err}')
        return net
    
    def fit(self):
        opt = self.opt
        self.net = self.sync_network_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device('cuda', opt.local_rank)
        
        if self.opt.load_net:
            self.resume(mode='net')
        self.net = self.move_network_to_cuda(self.net)
        self.net._set_static_graph()
        
        self.prepare_auxiliary_nets()
        self.prepare_logger()
        self.prepare_dataloader()
        self.prepare_optimizer()

        start_epoch = self.epoch
        self.nan_epoch, self.nan_iter = None, None
        
        for self.epoch in range(start_epoch, opt.epochs):
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.scheduler.step()

            info_dict = {
                'epoch': self.epoch,
                'batch_size': self.opt.batch_size,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}
            self.logger.log_info_dict(info_dict)
            
            if (self.epoch % self.opt.save_epochs == 0):
                self.save_net(self.net)
                self.save_opt(self.optimizer)
            
            self.val()
    
    def get_batched_data(self, data):
        ma_img, gt_img, li_img, mask = self.move_data_to_device([data[k] for k in ('ma_img', 'gt_img', 'li_img', 'mask')])
        unp_ma_img = self.move_data_to_device(data['unp_ma_img'])
        unp_mf_img = self.move_data_to_device(data['unp_mf_img'])
        unp_ma_qua = self.move_data_to_device(data['unp_ma_qua'])
        unp_mf_qua = self.move_data_to_device(data['unp_mf_qua'])
        dataset_flag = data['dataset_flag']
        return ma_img, gt_img, li_img, mask, unp_ma_img, unp_mf_img, unp_ma_qua, unp_mf_qua, dataset_flag
    
    def get_quality_labels(self, ma_img: torch.Tensor, gt_img: torch.Tensor, mask: torch.Tensor, dataset_flag: torch.Tensor):
        res = (ma_img - gt_img).abs().clip(0,1)
        fov = (gt_img > self.normalize_hu(-900)).float()
        batch_size = ma_img.shape[0]
        quality = torch.empty((batch_size,), device=res.device)  # [batch_size,]
        for b in range(batch_size):
            flag = dataset_flag[b:b+1]        
            if flag == 0:
                quality[b:b+1] = self.get_deepl_quality(res[b:b+1], mask[b:b+1], fov[b:b+1])
            elif flag == 1:
                quality[b:b+1] = self.get_dental_quality(res[b:b+1], mask[b:b+1])
            elif flag == 2:
                quality[b:b+1] = self.get_pelvic_quality(res[b:b+1], mask[b:b+1])
            else:
                raise NotImplementedError(f"Dataset flag {flag} not implemented.")
            
        return quality
    
    @staticmethod
    def get_deepl_quality(res, metal_mask, fov=1):
        """
        Quality classification rule for simulated DeepLesion dataset. 
        The thresholds listed below are only for the metal artifact simulation process 
        in the original RISE-MAR paper, and may require modification when adapting 
        to customized simulation processes or new datasets.
        """
        if torch.is_tensor(fov):
            fov = fov.to(metal_mask.device)
        intersect = (metal_mask * fov).sum()
        mask_area = metal_mask.sum()
        r = max(intersect / mask_area, 0.6)
        severity = r * (res * (1 - metal_mask) * fov).sum(dim=[1,2,3])  # [batch_size,]
        device = severity.device
        thresholds = torch.tensor([200, 450, 600, 900, 1000, 1200, 1500, 2000, 3000], device=device)
        values = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], device=device)
        output = values[-1] * torch.ones_like(severity, device=device)  # [batch_size,]
        for i in range(len(thresholds) - 1, -1, -1):
            output = torch.where(severity < thresholds[i], values[i], output)
        return output.float()
    
    @staticmethod
    def get_dental_quality(res, metal_mask, **kwargs):
        """
        Quality classification rule for simulated (private) Dental dataset. 
        The thresholds listed below are only for the metal artifact simulation process 
        in the original RISE-MAR paper, and may require modification when adapting 
        to customized simulation processes or new datasets.
        """
        severity = (res * (1 - metal_mask)).sum(dim=[1,2,3])
        device = severity.device
        thresholds = torch.tensor([200, 400, 600, 900, 1200, 1800, 2400, 3300, 4500], device=device)
        values = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], device=device)
        output = values[-1] * torch.ones_like(severity, device=device)  # [batch_size,]
        for i in range(len(thresholds) - 1, -1, -1):
            output = torch.where(severity < thresholds[i], values[i], output)
        return output.float()

    @staticmethod
    def get_pelvic_quality(res, metal_mask, **kwargs):
        """
        Quality classification rule for simulated CTPelvic1K dataset. 
        The thresholds listed below are only for the metal artifact simulation process 
        in the original RISE-MAR paper, and may require modification when adapting 
        to customized simulation processes or new datasets.
        """
        severity = (res * (1 - metal_mask)).sum(dim=[1,2,3])
        device = severity.device
        thresholds = torch.tensor([200, 400, 600, 900, 1200, 1800, 2400, 3300, 4500], device=device)
        values = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], device=device)
        output = values[-1] * torch.ones_like(severity, device=device)  # [batch_size,]
        for i in range(len(thresholds) - 1, -1, -1):
            output = torch.where(severity < thresholds[i], values[i], output)
        return output.float()
    
    @staticmethod
    def paste_metal(x, y, mask):
        return x * mask + y * (1 - mask)
    
    @staticmethod
    def mixup(x, y, alpha=0.5):
        if alpha == 1:
            return x
        elif alpha == 0:
            return y
        else:
            return x * alpha + y * (1 - alpha)
        
    @staticmethod
    def logit2quality(logits):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        classes = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
        refs = (torch.arange(1, num_classes+1)).repeat(batch_size, 1).float().to(classes.device)
        qualities = (classes * refs).sum(dim=-1)  # [batch_size,]
        return qualities
    
    @staticmethod    
    def get_random_numbers_with_gap(lower_bound=0.0, upper_bound=1.0, min_gap=0.2):
        assert upper_bound - lower_bound > min_gap, "The range should be larger than the minimum gap."
        max_first = upper_bound - min_gap
        # Generate the first number in the range [lower_bound + min_gap, upper_bound]
        first = torch.rand(1).item() * (max_first - lower_bound) + lower_bound + min_gap
        # Generate the second number in the range [lower_bound, first - min_gap]
        second = torch.rand(1).item() * (first - min_gap - lower_bound) + lower_bound
        return first, second   # (first > second)

    def dqaug(self, lq_img, hq_img, flag=0, mq_img=None, use_mixup=True, use_moderate=True):
        """Diverse Quality Augmentation. 
        Generate two images that have a quality between `lq_img` (low-quality image, e.g., the original
        metal artifact-affected image) and `hq_img` (high-quality images, e.g., ground-truth artifact-
        free image). 
        
        Args:
            lq_img (torch.Tensor): Low quality image with metal artifacts
            hq_img (torch.Tensor): High quality image with fewer artifacts
            flag (int, optional): Dataset flag. 0=torso, 1=dental, 2=AAPM, 3=pelvic. Defaults to 0.
            mq_img (torch.Tensor, optional): Medium quality image if available. Defaults to None.
            use_mixup (bool, optional): Whether to use mixup augmentation. Defaults to True.
            use_moderate (bool, optional): Whether to use moderate quality images from prior network. Defaults to True.
            
        Returns:
            tuple: A pair of images (mlq_img, mhq_img) where mhq_img has higher quality than mlq_img.
            The quality difference is controlled by randomly sampling mixing coefficients with a minimum gap.
            
        The method generates image pairs with different quality levels by:
        1. Randomly choosing first image from low quality input
        2. Generating second higher quality image through:
           - Mixup between lq and hq images with random coefficients
           - Using prior network inference
           - Using provided medium quality image
           - Using high quality ground truth
        The specific generation strategy depends on use_mixup and use_moderate flags.
        """
        
        # The quality level within this function specifies the relative quality
        # of the two images to be returned. 0 (bad) -> 4 (good). The relative quality
        # level of the first output image is among [0,1,2,3], while that of the second
        # output image is among [1,2,3,4]
        first_quality_level = random.choice([0, 1, 2, 3])
        mq_alpha, gq_alpha = self.get_random_numbers_with_gap(min_gap=0.3)
        is_lucky = np.random.rand() < 0.5
        is_mq_available = mq_img is not None
                
        def _get_higher_quality(curr_quality_level):
            if use_mixup and (not use_moderate):
                if curr_quality_level == 1:
                    return self.mixup(lq_img, hq_img, alpha=mq_alpha)
                elif curr_quality_level in [2, 3]:
                    if is_mq_available and is_lucky:
                        return mq_img
                    else:
                        self.mixup(lq_img, hq_img, alpha=gq_alpha)
                else:
                    return hq_img
                
            elif (not use_mixup) and use_moderate:
                if curr_quality_level == 1:
                    return self.infer_auxillary_net(lq_img, flag=flag, use_lq=True)
                elif curr_quality_level in [2, 3]:
                    if is_mq_available and is_lucky:
                        return mq_img
                    else:
                        return self.infer_auxillary_net(lq_img, flag=flag, use_lq=False)
                else:
                    return hq_img
                
            elif use_moderate and use_mixup:
                if curr_quality_level == 1:
                    return self.mixup(lq_img, hq_img, alpha=mq_alpha)
                elif curr_quality_level == 2:
                    if is_lucky:
                        return self.infer_auxillary_net(lq_img, flag=flag, use_lq=True)
                    else:
                        return self.mixup(lq_img, hq_img, alpha=gq_alpha)
                elif curr_quality_level == 3:
                    if is_lucky:
                        return self.infer_auxillary_net(lq_img, flag=flag, use_lq=False)
                    else:
                        return mq_img if is_mq_available else hq_img
                else:
                    return hq_img
            else:
                return mq_img if is_mq_available else hq_img
        
        if first_quality_level == 0:
            second_quality_level = random.choice([1, 2, 3, 4])
            mlq_img = lq_img
            mhq_img = _get_higher_quality(second_quality_level)
        elif first_quality_level == 1:
            second_quality_level = random.choice([2, 3, 4])
            mlq_img = _get_higher_quality(first_quality_level)
            mhq_img = _get_higher_quality(second_quality_level)
        elif first_quality_level == 2:
            second_quality_level = random.choice([3, 4])
            mlq_img = _get_higher_quality(first_quality_level)
            mhq_img = _get_higher_quality(second_quality_level)
        elif first_quality_level == 3:
            mlq_img = _get_higher_quality(first_quality_level)
            mhq_img = hq_img
            
        return mlq_img, mhq_img
    
    def update_memory_bank(self, x, label):
        is_ddp = hasattr(self.net, 'module')
        with torch.no_grad():
            if is_ddp:
                proj = self.net.module.forward_feature(x)
                proj = self.net.module.vectorize(proj)
            else:
                proj = self.net.vectorize(self.net.forward_feature(x))
        self.net.train()
        
        if is_ddp:
            return self.net.module.update_memory_bank(proj, label)
        else:
            return self.net.update_memory_bank(proj, label)
    
    @staticmethod
    def get_classification_loss(pred, gt):
        # pred shape: [batch_size, num_classes], float type
        # gt shape: [batch_size, 1], long type
        if len(gt.shape) == 2:
            gt = gt.squeeze(1)
        return F.cross_entropy(pred, gt.long()-1)
    
    def get_multiclass_contrastive_loss(self, proj_list, label_list, temp=1, eps=1e-6):
        if hasattr(self.net, 'module'):
            return self.net.module.get_multiclass_contrastive_loss(
                proj_list, label_list, temp=temp, eps=eps)
        else:
            return self.net.get_multiclass_contrastive_loss(
                proj_list, label_list, temp=temp, eps=eps)
        
    def forward_net(self, ma_img, output_logits=True):
        logits = self.net(ma_img)
        return logits if output_logits else (10 * torch.sigmoid(logits))
    
    def forward_net_all(self, ma_img, output_logits=True, use_cp=False):
        if hasattr(self.net, 'module'):
            logits, projs = self.net.module.forward_all(ma_img, use_cp=use_cp)
        else:
            logits, projs = self.net.forward_all(ma_img, use_cp=use_cp)
        
        if not output_logits:
            logits = (10 * torch.sigmoid(logits))
        return logits, projs
    
    def train(self):
        loader = self.train_loader
        self.net.train()
        
        pbar = tqdm.tqdm(loader, ncols=100, disable=self.opt.local_rank!=0)
        for i, data in enumerate(pbar): 
            ma_img, gt_img, li_img, mask, unp_ma_img, unp_mf_img, unp_ma_qua, unp_mf_qua, dataset_flag = self.get_batched_data(data)
            
            if i == 0 or i == len(pbar)//3:
                self.load_random_auxiliary_nets()
            
            if np.random.rand() < 0.5:
                gt_img = self.paste_metal(ma_img, gt_img, mask)
            
            ma_img, mf_img = self.dqaug(ma_img, gt_img, flag=dataset_flag, mq_img=li_img, use_mixup=True, use_moderate=True)

            quality1 = self.get_quality_labels(ma_img, gt_img, mask, dataset_flag)  # [batch_size,]
            quality2 = self.get_quality_labels(mf_img, gt_img, mask, dataset_flag)
            pred_logit1, pred_proj1 = self.forward_net_all(ma_img, output_logits=True)
            pred_logit2, pred_proj2 = self.forward_net_all(mf_img, output_logits=True)
            
            loss = self.get_classification_loss(pred_logit1, quality1) + self.get_classification_loss(pred_logit2, quality2)
            is_loss_nan1 = torch.isnan(loss)
            if is_loss_nan1:
                self.nan_epoch = self.epoch if self.nan_epoch is None else self.nan_epoch
                self.nan_iter = self.logger.step if self.nan_iter is None else self.nan_iter
                print(f'NaN found for CE in Epoch {self.nan_epoch} Iter {self.nan_iter}')
            
            is_loss_nan2 = False
            loss_factor2 = float(self.opt.loss_factor2)
            if self.epoch >= 0 and (loss_factor2 > 0):
                if unp_mf_img is not None:
                    self.update_memory_bank(unp_mf_img, unp_mf_qua)
                if unp_ma_img is not None:
                    self.update_memory_bank(unp_ma_img, unp_ma_qua)
                loss_c = self.get_multiclass_contrastive_loss((pred_proj1, pred_proj2), (quality1, quality2))
                is_loss_nan2 = torch.isnan(loss_c)
                
                if is_loss_nan1:
                    loss = loss_c
                if is_loss_nan2:
                    self.nan_epoch = self.epoch if self.nan_epoch is None else self.nan_epoch
                    self.nan_iter = self.logger.step if self.nan_iter is None else self.nan_iter
                    print(f'NaN found for contrastive in Epoch {self.nan_epoch} Iter {self.nan_iter}')
                
                if (not is_loss_nan1) and (not is_loss_nan2):
                    loss = loss + loss_c * loss_factor2
            
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            pbar.set_postfix({'loss': '%.2f' % loss.item()})
            pbar.update(1)
            
            if (not is_loss_nan1) and (not is_loss_nan2) and (i % 20000 == 0):
                # Timely save non-nan checkpoints so we don't waste time resuming
                self.save_net(self.net, name=f'notnan_{int(self.logger.step)}')
                self.save_opt(self.optimizer, name=f'notnan_{int(self.logger.step)}')
            
            self.logger.tick()
            self.logger.log_scalar_dict({'loss': loss.item()}, training_stage='train')

        self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'train')

    @torch.no_grad()
    def val(self,):
        self.net.eval()
        losses_ma, losses_mf, losses = [], [], []
        pbar = tqdm.tqdm(self.val_loader, ncols=100, disable=self.opt.local_rank != 0)
        for i, data in enumerate(pbar):
            ma_img, gt_img, li_img, mask, unp_ma_img, unp_mf_img, unp_ma_qua, unp_mf_qua, dataset_flag = self.get_batched_data(data)
            ma_img = self.mixup(ma_img, gt_img)
            mf_img = self.infer_auxillary_net(ma_img, flag=dataset_flag, use_lq=False)
            
            quality = self.get_quality_labels(ma_img, gt_img, mask, dataset_flag)
            pred_quality = self.logit2quality(self.forward_net(ma_img, output_logits=True))
            quality2 = self.get_quality_labels(mf_img, gt_img, mask, dataset_flag)
            pred_quality2 = self.logit2quality(self.forward_net(mf_img, output_logits=True))
            
            loss_ma = F.l1_loss(pred_quality, quality).item()
            loss_mf = F.l1_loss(pred_quality2, quality2).item()
            loss = (loss_ma + loss_mf) / 2
            losses_ma.append(loss_ma)
            losses_mf.append(loss_mf)
            losses.append(loss)
            
            pbar.set_postfix({'loss': '%.2f' % loss})
            pbar.update(1)
                
            self.logger.log_scalar('loss', loss, training_stage='val')
            self.logger.log_scalar('loss_ma', loss_ma, training_stage='val')
            self.logger.log_scalar('loss_mf', loss_mf, training_stage='val')
        
        print(f'Epoch: {self.epoch}: MA: {np.mean(losses_ma)}')
        print(f'Epoch: {self.epoch}: MF: {np.mean(losses_mf)}')
        print(f'Epoch: {self.epoch}: MA+MF: {np.mean(losses)}')
        self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')
        

