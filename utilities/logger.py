import os
import torch
import logging
import torchvision
import numpy as np
import torch.distributed as dist

def only_on_rank0(func):
    '''wrapper for only log on the first rank'''
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)
    return wrapper

def create_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


class LongziliLogger(object):
    """
    A custom logger class designed to simplify log management in the training loop.
    Note: This class should be initialized after torch.distributions.

    Arguments:
        log_name: The name of the log.
        project_name: The name of the project.
        resume: Whether to resume training.
        config_opt: The argument object containing all the hyperparameters.
        log_root_path: The root path for the logs.
        checkpoint_root_path: The root path for the checkpoints.
        tensorboard_root_path: The root path for TensorBoard.
        use_wandb: Whether to use wandb for log management.
        wandb_root_path: The root path for wandb.
        log_interval: The interval for log printing.

    Usage:
        # 1. Initialize the logger.
        self.logger = Longzili_Logger(
            log_name=str(wandb_dir),
            project_name=opt.wandb_project,
            config_opt=opt,
            checkpoint_root_path=opt.checkpoint_root,
            tensorboard_root_path=opt.tensorboard_root,
            wandb_root_path=opt.wandb_root,
            use_wandb=True,
            log_interval=opt.log_interval,)

        # 2. Use the methods inside.
            2.1 Use `tick` to record steps in the training loop.
            2.2 Use `log_info`, `log_scalar`, `log_image` to record logs.
            2.3 Use the following methods to log epoch:
                self.logger.log_scalar(force=True, log_type='epoch', training_stage='train')
                self.logger.log_scalar(force=True, log_type='epoch', training_stage='val')
    """
    
    def __init__(self, 
                log_name: str,
                project_name: str,
                resume = False,
                config_opt = None,
                log_root_path = None,
                checkpoint_root_path = None,
                tensorboard_root_path = None,
                use_wandb=False, 
                use_tb=False,
                wandb_root_path = None,
                log_interval=100,
                log_interval_image = None):

        # check device and so on 
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.cards = dist.get_world_size()
        else:   
            self.rank = 0
            self.cards = 1  
        
        print(f'Training on {self.cards} cards')

        if use_wandb and wandb_root_path is None:
            wandb_root_path = './logger'
        if log_root_path is not None:
            tensorboard_root_path = os.path.join(log_root_path, 'tensorboard')
            checkpoint_root_path = os.path.join(log_root_path, 'checkpoint')
        else:
            if tensorboard_root_path is None:
                tensorboard_root_path = './logger'
            if checkpoint_root_path is None:
                checkpoint_root_path = './logger'

        self.use_wandb = use_wandb
        self.use_tb = use_tb
        if use_wandb:
            self.wandb_path = os.path.join(wandb_root_path, log_name)
            create_directories(self.wandb_path)
        if use_tb:
            self.tensorboard_path = os.path.join(tensorboard_root_path, log_name)
            create_directories(self.tensorboard_path)
        self.checkpoint_path = os.path.join(checkpoint_root_path, log_name)
        create_directories(checkpoint_root_path)

        # initialize state
        self.log_interval = log_interval
        self.log_interval_image = log_interval * 30 if log_interval_image is None else log_interval_image
        self.step = 0
        self.values = {'train': {}, 'val': {}, 'test': {}}   

        # initialize mg_logger, tensorboard, wandb and so on
        if self.rank == 0:
            self.mg_logger = logging.getLogger(__name__)
            handler = logging.FileHandler(os.path.join(checkpoint_root_path, f"mg_logger.log"))
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.mg_logger.addHandler(handler)
            self.mg_logger.setLevel(logging.INFO)
            # initialize tensorboard
            if use_tb:
                self.tb_logger = self._tensorboard_init(self.tensorboard_path, resume=resume)
            # initialize wandb
            if use_wandb:
                self.wandb_logger = self._wandb_init(log_name, project_name, self.wandb_path, config_opt, resume=resume)


    @only_on_rank0
    def log_image(self, tag, value, log_type="iter", force=False, step = 0, training_stage = 'train'):
        if log_type == "iter" and self.step % self.log_interval_image != 0 and not force:
            return
        elif log_type == "epoch" and not force:
            return
        value = value.detach().cpu()
        grid = torchvision.utils.make_grid(value, normalize=True)  # BCWH -> grid image
        grid = grid.permute(1, 2, 0)  # HWC
        grid = grid.numpy()
        tag = f'{training_stage}/{tag}'
        if self.use_tb:
            self.tb_logger.add_image(tag=tag, img_tensor=grid.transpose((2, 0, 1)), global_step = step, dataformats='CHW')
        # img = Image.fromarray((grid * 255).astype(np.uint8)) # convert to PIL Image
        if self.use_wandb:
            import wandb
            self.wandb_logger.log({tag: wandb.Image(grid)}, step=self.step)

    # @only_on_rank0
    def log_scalar(self, tag=None, value=None, force=False, log_type="iter", log_list=None, training_stage='train', step = None):
        assert log_type in ["iter", "epoch"]
        assert training_stage in ["train", 'val', 'test', 'info']
        if step is not None:
            step = step
        else:
            step = self.step

        if tag is not None and value is not None:
            if tag not in self.values[training_stage]:
                self.values[training_stage][tag] = []
            # if value is tensor and need grad
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            self.values[training_stage][tag].append(self._reduce_value(value))

        if not force:
            if log_type == "iter" and step % self.log_interval == 0:
                self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
            elif log_type == "epoch":
                self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
                self._clear_all_values(log_list, training_stage=training_stage) 
        else:
            self._write_log_for_tags(log_list, log_type=log_type, training_stage=training_stage, step = step)
            if log_type == "epoch":
                self._clear_all_values(log_list, training_stage=training_stage) 

    @only_on_rank0
    def log_info(self, tag, value, step=None, training_stage='info'):
        """
        Log information once. This information is independent of steps or iterations.
        """
        assert training_stage in ["train", 'val', 'test', 'info']
        if step is None:
            step = self.step
        self._write_log(tag, value, training_stage=training_stage, step=step)

    @only_on_rank0
    def log_info_dict(self, dict_values, step=None, **kwargs):
        """
        Log multiple information terms simultaneously. This information is independent of steps or iterations.
        """
        assert isinstance(dict_values, dict)
        if step is None:
            step = self.step
        for tag, value in dict_values.items():
            self._write_log(tag, value, step=step)
            
    # @only_on_rank0
    def log_scalar_dict(self, dict_values = None, log_type="iter", force=False, training_stage='train'):
        """
        Log multiple scalars (stored in a dictionary) simultaneously. These scalars can change over steps.
        """
        assert isinstance(dict_values, dict)
        for tag, value in dict_values.items():
            self.log_scalar(tag, value, log_type=log_type, force=force, training_stage=training_stage)

    @only_on_rank0
    def log_image_dict(self, dict_images, log_type="iter", force=False, training_stage='train'):
        """
        Log multiple images (stored in a dictionary) simultaneously.
        """
        assert isinstance(dict_images, dict)
        for tag, value in dict_images.items():
            self.log_image(tag, value, log_type=log_type, force=force, training_stage = training_stage)

    @only_on_rank0
    def _write_log_for_tags(self, log_list, log_type='iter', training_stage='train', step = None):
        step = self.step if step is None else step
        for tag in self.values[training_stage]:
            
            if log_list is None or tag in log_list:  
                if not self.values[training_stage][tag]:
                    continue
                if log_type == 'iter':
                    # log according to interval
                    mean_value = np.mean(self.values[training_stage][tag][-self.log_interval:])  
                else:  
                    # log according to epoch
                    mean_value = np.mean(self.values[training_stage][tag])
                
                if training_stage in ['val', 'test']:
                    pass
                
                self._write_log(training_stage + '/' + log_type + '/' + tag, mean_value, step = step)

    @only_on_rank0
    def _write_log(self, tag, value, step):
        if self.use_tb:
            self.tb_logger.add_scalar(tag, value, global_step = step)
        if self.use_wandb:
            self.wandb_logger.log({tag: value}, step=step)

    def _clear_all_values(self, log_list, training_stage='train'):
        for tag in self.values[training_stage]:
            if log_list is None or tag in log_list:
                self.values[training_stage][tag] = []

    @only_on_rank0
    def _update_step(self):
        self.step += self.cards

    def tick(self):
        self._update_step()

    @staticmethod
    def _wandb_init(log_name, project_name, wandb_path, config_opt, resume=False):
        import wandb
        if config_opt.wandb_key is not None and len(config_opt.wandb_key) > 0:
            wandb.login(key=config_opt.wandb_key)
            
        os.makedirs(wandb_path, exist_ok=True)
        wandb_logger = wandb.init(
            project=project_name,
            name=str(log_name),
            dir=wandb_path,
            resume=resume, 
            config=config_opt, 
            reinit=True,)
        return wandb_logger

    @staticmethod
    def _tensorboard_init(tensorboard_path, resume, flush_secs = 3):
        from torch.utils.tensorboard import SummaryWriter
        if resume:
            tb_logger = SummaryWriter(tensorboard_path, flush_secs=flush_secs, resume=True)
        else:
            tb_logger = SummaryWriter(tensorboard_path, flush_secs=flush_secs)
        return tb_logger

    def _reduce_value(self, value, average=True):
        # Reduce value for multi-GPU training and testing
        # **Warning:** Make sure that every process can reach this function, otherwise it will wait forever.
        try:
            world_size = dist.get_world_size()
        except Exception as err:
            world_size = 1
        if world_size < 2:  # single GPU
            return value
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if not value.is_cuda:
            value = value.cuda(self.rank)
        with torch.no_grad():
            dist.all_reduce(value)   # get reduce value
            if average:
                value = value.float()
                value /= world_size
        return value.cpu().item()

