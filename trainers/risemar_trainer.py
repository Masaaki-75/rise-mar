
import os
import sys
sys.path.append("..")
import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.distributed
import torch.distributed as dist
from warnings import warn
from torch.utils.data import DataLoader
from modules.ema import EfficientEMA
from networks.cqa import CQA
from utilities.metrics import compute_rec_metrics
from utilities.logger import LongziliLogger
from trainers.basic_mar_trainer import BasicMARTrainer
from datasets.risemar_dataset import RiseMARDataset
from configs import (
    PAIRED_META1, PAIRED_META2, PAIRED_META3,
    UNPAIRED_META1, UNPAIRED_META2, UNPAIRED_META3,
    PRETRAINED_TORSO_MAR_PATH,
    PRETRAINED_DENTAL_MAR_PATH,
    PRETRAINED_CQA_PATH, SEED
)


class RiseMARTrainer(BasicMARTrainer):
    def __init__(self, opt, net, **kwargs):
        super().__init__(opt=opt)
        self.opt = opt
        self.net = net
        self.prepare_dataset()
        self.prepare_methods()
        
    def prepare_dataset(self):
        opt = self.opt
        dataset_name = opt.dataset_name.lower()
        common_kwargs = dict(
            seed=SEED, min_hu=opt.min_hu, max_hu=opt.max_hu, 
            num_train=opt.num_train, num_val=opt.num_val)
        train_kwargs = dict(flip_prob=opt.flip_prob, rot_prob=opt.rot_prob, mode='train')
        
        if dataset_name in ['deepl', 'deepl-deepl']:
            self.train_dataset = RiseMARDataset(
                deepl_json=PAIRED_META1, unpaired_json1=UNPAIRED_META1, **train_kwargs, **common_kwargs)
            # Validation set 1: on the paired DeepLesion dataset (as source domain)
            self.val_dataset = RiseMARDataset(deepl_json=PAIRED_META1, mode='val', **common_kwargs)
            # Validation set 2: on the paired CTPelvic1K dataset (as target domain)
            self.val_dataset2 = RiseMARDataset(deepl_json=PAIRED_META3, mode='val', **common_kwargs)
            
        elif dataset_name in ['dental', 'dental-dental']:
            self.train_dataset = RiseMARDataset(
                dental_json=PAIRED_META2, unpaired_json2=UNPAIRED_META2, **train_kwargs, **common_kwargs)
            self.val_dataset = RiseMARDataset(
                dental_json=PAIRED_META2, unpaired_json2=None, mode='val', **common_kwargs)
            self.val_dataset2 = None
            
        elif dataset_name in ['deepl', 'deepl-pelvic']:
            self.train_dataset = RiseMARDataset(
                deepl_json=PAIRED_META1, unpaired_json1=UNPAIRED_META3, **train_kwargs, **common_kwargs)
            # Validation set 1: on the paired DeepLesion dataset (as source domain)
            self.val_dataset = RiseMARDataset(deepl_json=PAIRED_META1, mode='val', **common_kwargs)
            # Validation set 2: on the paired CTPelvic1K dataset (as target domain)
            self.val_dataset2 = RiseMARDataset(deepl_json=PAIRED_META3, mode='val', **common_kwargs)
            
        else:
            raise NotImplementedError(f"Unsupported dataset: {opt.dataset_name}")
    
    def prepare_methods(self):
        self.mu2hu = self.train_dataset.cttool.mu2hu
        self.hu2mu = self.train_dataset.cttool.hu2mu
        self.normalize_hu = self.train_dataset.cttool.normalize_hu
        self.denormalize_hu = self.train_dataset.cttool.denormalize_hu
        
    def prepare_logger(self):
        opt = self.opt
        self.logger = LongziliLogger(
            log_name = str(opt.tensorboard_dir),
            project_name = opt.wandb_project,
            config_opt = opt,
            checkpoint_root_path = opt.checkpoint_root,
            tensorboard_root_path = opt.tensorboard_root,
            wandb_root_path = opt.wandb_root,
            use_wandb = opt.use_wandb,
            log_interval = opt.log_interval,)
    
    def prepare_dataloader(self):
        opt = self.opt
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
            sampler=train_sampler, drop_last=True, pin_memory=True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, num_workers=opt.num_workers, sampler=val_sampler,)
        if self.val_dataset2 is not None:
            val_sampler2 = torch.utils.data.distributed.DistributedSampler(self.val_dataset2, shuffle=False)
            self.val_loader2 = DataLoader(self.val_dataset2, batch_size=1, num_workers=opt.num_workers, sampler=val_sampler2,)
        
    def prepare_optimizer(self):
        self.optimizer = self.get_optimizer(self.net.parameters(), 'adam')
        self.scheduler = self.get_scheduler(self.optimizer, 'mstep')
        if self.opt.load_opt:
            self.resume(mode='opt')
    
    def prepare_ema(self):
        opt = self.opt
        net_ema = copy.deepcopy(self.get_module(self.net)).to(self.device)
        self.ema = EfficientEMA(net_ema, decay=opt.ema_momentum, device=self.device, sync_interval=opt.log_interval)
        self.ema.register()
        
    def prepare_quality_assessor(self):
        net_dict = dict(do_multiscale=True,in_channels=1,out_channels=10,attn_ratio=[0,1/2,1,0,0],drop_path_rates=0.1,use_spectrals=[True,True,True,False,False])
        self.quality_assessor = CQA(**net_dict)
        quality_checkpath = PRETRAINED_CQA_PATH
        state_dict = torch.load(quality_checkpath, map_location='cpu')
        if 'net' in state_dict.keys():  # Adapt to BasicMARTrainer's setting of saving checkpoint.
            state_dict = state_dict['net']
        self.quality_assessor.load_state_dict(state_dict)
        self.quality_assessor = self.quality_assessor.eval().to(self.device)
    
    def move_network_to_cuda(self, net):
        net = net.to(self.device)
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[self.opt.local_rank], 
            output_device=self.opt.local_rank, 
            broadcast_buffers=False) #  find_unused_parameters=True
        return net
    
    def sync_network_batchnorm(self, net):
        try:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        except Exception as err:
            warn(f'Failed to sync batchnorm due to the following error: {err}')
        return net
    
    def initialize_with_pretrained_prior(self):
        # [NOTE] As reported in the paper, we found that initializing MAR networks 
        # with (supervised) pretrained weights significantly improves the stability of
        # unsupervised or semi-supervised training. 
        dataset_name = self.opt.dataset_name.lower()
        if dataset_name in ['deepl']:
            net_checkpath = PRETRAINED_TORSO_MAR_PATH
            try:
                self.net.load_state_dict(torch.load(net_checkpath, map_location='cpu')['net'])
                print(f"Initialized MAR network with pretrained weights for dataset {dataset_name}")
            except Exception as err:
                warn(f"Failed to initialize MAR network with pretrained weights for dataset {dataset_name} due to the following exception: {err}")
        elif dataset_name in ['dental']:
            net_checkpath = PRETRAINED_DENTAL_MAR_PATH
            try:
                self.net.load_state_dict(torch.load(net_checkpath, map_location='cpu')['net'])
                print(f"Initialized MAR network with pretrained weights for dataset {dataset_name}")
            except Exception as err:
                warn(f"Failed to initialize MAR network with pretrained weights for dataset {dataset_name} due to the following exception: {err}")
    
    def fit(self):
        opt = self.opt
        self.net = self.sync_network_batchnorm(self.net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device('cuda', opt.local_rank)
        
        self.initialize_with_pretrained_prior()
        if self.opt.load_net:
            self.resume(mode='net', strict=False)
        
        self.net = self.move_network_to_cuda(self.net)
        self.prepare_quality_assessor()
        self.prepare_ema()
        self.prepare_logger()
        self.prepare_dataloader()
        self.prepare_optimizer()

        start_epoch = self.epoch
        for self.epoch in range(start_epoch, opt.epochs):
            self.train_loader.sampler.set_epoch(self.epoch)
            self.train()
            self.scheduler.step()
            
            info_dict = {
                'epoch': self.epoch,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr']}
            self.logger.log_info_dict(info_dict)
            
            if (self.epoch % self.opt.save_epochs == 0):
                self.save_net(self.net)
                self.save_opt(self.optimizer)
                self.save_net(self.ema.model, 'ema')
            
            self.val()
    
    def get_batched_data(self, data):
        ma_img, gt_img, li_img, mask = self.move_data_to_device([data[k] for k in ('ma_img', 'gt_img', 'li_img', 'mask')])
        unp_ma_img = self.move_data_to_device(data['unp_ma_img'])
        unp_mf_img = self.move_data_to_device(data['unp_mf_img'])
        unp_ma_qua = self.move_data_to_device(data['unp_ma_qua'])
        unp_mf_qua = self.move_data_to_device(data['unp_mf_qua'])
        return ma_img, gt_img, li_img, mask, unp_ma_img, unp_mf_img, unp_ma_qua, unp_mf_qua
    
    @staticmethod
    def logit2quality(logits):
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        classes = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
        refs = (torch.arange(1, num_classes+1)).repeat(batch_size, 1).float().to(classes.device)
        qualities = (classes * refs).sum(dim=-1)  # [batch_size,]
        return qualities
    
    @torch.no_grad()
    def assess_quality(self, x):
        quality = self.logit2quality(self.quality_assessor(x))  # [batch_size,]
        return quality
    
    @torch.no_grad()
    def get_pseudo_label(self, ma_img):
        return self.ema.model(ma_img).detach().clip(0, 1)

    def get_source_domain_loss(self, ma_img, gt_img, valid_mask=None):
        if valid_mask is None:
            return F.l1_loss(ma_img, gt_img)
        else:
            return F.l1_loss(ma_img * valid_mask, gt_img * valid_mask)

    def get_target_domain_loss(self, ma_img, mf_img):
        mf_pseudo = self.get_pseudo_label(ma_img)
        quality = self.assess_quality(mf_pseudo)
        
        q0 = float(self.opt.qua_thres)  # minimum allowed quality
        q1 = float(self.opt.qua_thres2)  # maximum allowed quality
        q = quality.min()
        if q >= q0 and q <= q1:
            if np.random.rand() < 0.5:  # augmentation by manipulating pseudo artifacts
                artifact = ma_img - mf_pseudo
                ma_img = (mf_img + artifact).clip(0, 1)
                mf_pseudo = mf_img
            mf_img_pred = self.net(ma_img).clip(0, 1)
            return F.l1_loss(mf_img_pred, mf_pseudo)
        else:
            return torch.tensor(0.)
            
    def train(self,):
        self.net.train()
        self.ema.model.eval()
        loss_factor2 = self.opt.loss_factor2
        
        pbar = tqdm.tqdm(self.train_loader, ncols=120, disable=self.opt.local_rank != 0)
        for i, data in enumerate(pbar):
            ma_img_src, gt_img_src, _, mask, ma_img_tgt, mf_img_tgt, _, _ = self.get_batched_data(data)
            valid_mask = 1 - mask
            
            self.optimizer.zero_grad()
            pr_img_src = self.net(ma_img_src).clip(0, 1)
            loss = self.get_source_domain_loss(pr_img_src, gt_img_src, valid_mask)

            if loss_factor2 > 0:
                mf_img = mf_img_tgt if np.random.rand() <= 0.5 else gt_img_src
                loss2 = self.get_target_domain_loss(ma_img_tgt, mf_img) 
                loss = loss + loss2 * loss_factor2
                
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.net)
            self.ema.apply_shadow()
            
            psnr, ssim, quality = self.compute_metrics(pr_img_src, gt_img_src, valid_mask)
            pbar.set_postfix({'loss': '%.2f' % (loss.item()), 'psnr': '%.2f' % psnr})
            pbar.update(1)
            
            self.logger.tick()
            iter_log = {'loss': loss, 'psnr': psnr, 'ssim': ssim, 'quality': quality}
            self.logger.log_scalar_dict(iter_log, training_stage='train')

        self.logger.log_scalar(force=True, log_type='epoch', training_stage='train')

    @torch.no_grad()
    def val(self,):
        self.ema.apply_shadow()
        self.net.eval()
        self.ema.model.eval()
        
        # Source domain validation
        pbar = tqdm.tqdm(self.val_loader, ncols=120, disable=self.opt.local_rank != 0)
        for i, data in enumerate(pbar):
            ma_img_src, gt_img_src, _, mask, _, _, _, _ = self.get_batched_data(data)            
            valid_mask = 1 - mask
            
            pr_img_src = self.net(ma_img_src).clip(0,1)
            pr_img_src_ps = self.get_pseudo_label(ma_img_src)
            
            loss = F.l1_loss(pr_img_src, gt_img_src, valid_mask)
            psnr, ssim, quality = self.compute_metrics(pr_img_src, gt_img_src, valid_mask)
            psnr_ps, ssim_ps, quality_ps = self.compute_metrics(pr_img_src_ps, gt_img_src, valid_mask)
            pbar.set_postfix({'loss': '%.2f' % (loss.item()), 'psnr': '%.2f' % psnr})
            pbar.update(1)
            
            val_log = {
                'loss': loss, 'psnr': psnr, 'ssim': ssim, 'quality': quality,
                'psnr_ps': psnr_ps, 'ssim_ps': ssim_ps, 'quality_ps': quality_ps}
            self.logger.log_scalar_dict(val_log, training_stage='val')
        
        # Target domain validation (another paired dataset as target domain)
        if self.val_loader2 is not None:
            pbar2 = tqdm.tqdm(self.val_loader2, ncols=120, disable=self.opt.local_rank != 0)
            for i,data in enumerate(pbar2):
                ma_img_tgt, gt_img_tgt, _, mask, _, _, _, _ = self.get_batched_data(data)
                valid_mask = 1 - mask
                
                pr_img_tgt = self.net(ma_img_tgt).clip(0,1)
                pr_img_tgt_ps = self.get_pseudo_label(ma_img_tgt)
                
                loss = F.l1_loss(pr_img_tgt, gt_img_tgt, valid_mask)
                psnr, ssim, quality = self.compute_metrics(pr_img_tgt, gt_img_tgt, valid_mask)
                psnr_ps, ssim_ps, quality_ps = self.compute_metrics(pr_img_tgt_ps, gt_img_tgt, valid_mask)
                pbar2.set_postfix({'loss2': '%.2f' % (loss.item()), 'psnr2': '%.2f' % psnr})
                pbar2.update(1)

                val_log = {
                    'loss2': loss, 'psnr2': psnr, 'ssim2': ssim, 'quality2': quality,
                    'psnr2_ps': psnr_ps, 'ssim2_ps': ssim_ps, 'quality2_ps': quality_ps}
                self.logger.log_scalar_dict(val_log, training_stage='val')

        self.logger.log_scalar(force=True, log_type='epoch', training_stage = 'val')


    def compute_metrics(self, pr_img, gt_img, valid_mask=None, data_range=1):
        valid_mask = 1.0 if valid_mask is None else valid_mask
        rmse, psnr, ssim = compute_rec_metrics(pr_img * valid_mask, gt_img * valid_mask, data_range=data_range)
        quality = self.assess_quality(pr_img).mean().item()
        return psnr, ssim, quality
    
