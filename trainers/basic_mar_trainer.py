import os
import sys
sys.path.append("..")
import warnings
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import deque
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad_(requires_grad)


class LossScaler:
    def __init__(self, max_len=50, scale_factor=2):
        self.history = deque(maxlen=max_len)
        self.scale_factor = scale_factor

    def __call__(self, loss):
        self.history.append(loss.detach().item())
        min_loss = min(self.history)

        if loss.item() > min_loss * self.scale_factor:
            scaled_loss = loss * (min_loss * (self.scale_factor/2) / loss.item())
            return scaled_loss
        else:
            return loss

class BasicMARTrainer:
    def __init__(self, opt=None):
        assert opt is not None
        self.opt = opt
        self.checkpoint_path = os.path.join(opt.checkpoint_root, opt.checkpoint_dir)
        self.iter = 0
        self.itlog_intv = opt.log_interval
        self.num_epochs = opt.epochs
        self.epoch = 0
        
        self.multigpu = False
        self.rgb_dict = {'r':255,'g':0,'b':0}
        self.loss_scaler = LossScaler(max_len=50, scale_factor=3)

    @staticmethod
    def move_data_to_device(data, device=None):
        def _to_device(d):
            if (device is None) and hasattr(d, 'cuda'):
                return d.cuda()
            elif (device is not None) and hasattr(d, 'to'):
                return d.to(device)
            else:
                return d
            
        if isinstance(data, (list, tuple)):
            return [_to_device(d) for d in data]
        elif isinstance(data, dict):
            return {k:_to_device(v) for (k,v) in data.items()}
        else:
            return _to_device(data)
    
    def get_optimizer(self, net_params, optimizer_name, lr=None, **kwargs):
        opt = self.opt
        lr = opt.lr if lr is None else lr
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            if len(kwargs) == 0:
                kwargs = dict(betas=(opt.beta1, opt.beta2))
            return torch.optim.Adam(net_params, lr=lr, **kwargs)
        elif optimizer_name == 'sgd':
            if len(kwargs) == 0:
                kwargs = dict(momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
            return torch.optim.SGD(net_params, lr=lr, **kwargs)
        elif optimizer_name == 'adamw':
            if len(kwargs) == 0:
                kwargs = dict(betas=(self.opt.beta1, self.opt.beta2))
            return torch.optim.AdamW(net_params, lr=lr, **kwargs)
        else:
            raise NotImplementedError(f'Currently only support optimizers among: Adam, AdamW and SGD, got {optimizer_name}.')
        
    def get_scheduler(self, optimizer=None, scheduler_name='step', last_epoch=-1, **kwargs):
        opt = self.opt
        optimizer = self.optimizer if optimizer is None else optimizer
        scheduler_name = scheduler_name.lower()
        if scheduler_name == 'step':
            if len(kwargs) == 0:
                kwargs = dict(step_size=opt.step_size, gamma=opt.step_gamma)
            return torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **kwargs)
        elif scheduler_name == 'mstep':
            if len(kwargs) == 0:
                kwargs = dict(milestones=opt.milestones, gamma=opt.step_gamma)
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, last_epoch=last_epoch, **kwargs)
        elif scheduler_name == 'exp':
            if len(kwargs) == 0:
                kwargs = dict(gamma=opt.step_gamma)
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler_name == 'cosine':
            if len(kwargs) == 0:
                kwargs = dict(T_max=opt.epochs, eta_min=1e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        else:
            warnings.warn(f'Currently only support schedulers among Step, MultiStep, Exp, Cosine, got {scheduler_name}. So using none (constant).')
            return None

    @staticmethod
    def save_checkpoint(data, path, name:str, epoch:int):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, name + f'_e{epoch}.pkl')
        torch.save(data, checkpoint_path)
    
    @staticmethod
    def get_module(net):
        if hasattr(net, 'module'):
            return net.module
        else:
            return net
    
    def sync_network_batchnorm(self, net):
        try:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        except Exception as err:
            warnings.warn(f'Failed to sync batchnorm due to the following error: {err}')
        return net

    def save_net(self, net=None, name='', **kwargs):
        net_name = self.opt.net_name
        if net is None:
            net_ckpt = {'net': self.get_module(self.net).state_dict(),}
        else:
            net_ckpt = {'net': self.get_module(net).state_dict(),}
        
        name = f"{net_name}-{name}-net"
        self.save_checkpoint(net_ckpt, self.checkpoint_path, name, self.epoch)

    def save_opt(self, optim=None, name='', **kwargs):
        net_name = self.opt.net_name    
        if optim is None:
            opt_check = {'optimizer': self.optimizer.state_dict(), 'epoch': self.epoch,}
        else:
            opt_check = {'optimizer': optim.state_dict(), 'epoch': self.epoch,}
        
        name = f"{net_name}-{name}-opt"
        self.save_checkpoint(opt_check, self.checkpoint_path, name, self.epoch)

    def load_opt(self, opt=None):
        opt_checkpath = self.opt.opt_checkpath
        opt_checkpoint = torch.load(opt_checkpath, map_location='cpu')
        if opt is None:
            self.optimizer.load_state_dict
        else:
            opt.load_state_dict(opt_checkpoint['optimizer'])
        self.epoch = opt_checkpoint['epoch']
        print(f'Finish loading optimizer from {opt_checkpath}')

    def resume(self, mode='net', strict=True):
        if mode == 'net':
            self.load_net(strict=strict)
        elif mode == 'opt':
            self.load_opt()
        else:
            raise NotImplementedError(f'Unsupported mode for resuming: {mode}')
    
    def load_net(self, net=None, strict=True):
        print(f"Loading network with `strict={strict}`")
        net_checkpath = self.opt.net_checkpath
        if net is None:
            self.net.load_state_dict(torch.load(net_checkpath, map_location='cpu')['net'], strict=strict)
        else:
            net.load_state_dict(torch.load(net_checkpath, map_location='cpu')['net'], strict=strict)
        print(f'Finish loading network from {net_checkpath}')

    # reduce function
    def reduce_value(self, value, average=True):
        world_size = dist.get_world_size()
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.opt.local_rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value /= world_size
        return value.cpu()

    def reduce_loss(self, loss, average=True):
        return self.reduce_value(loss, average=average)
        
    def fit():
        raise NotImplementedError

    def train():
        pass

    def val():
        pass



