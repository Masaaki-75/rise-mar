import argparse
import torch

from utilities.misc import fix_seed
from networks.mar import RiseMARNet
from networks.cqa import CQA, CQACLIP, CQAVGG

from trainers.cqa_trainer import CQATrainer
from trainers.risemar_trainer import RiseMARTrainer
from trainers.supervised_trainer import SupervisedMARTrainer
from configs import WANDB_KEY, SEED
#os.environ['WANDB_MODE'] = 'online' 


def get_parser():
    parser = argparse.ArgumentParser(description='MAIN FUNCTION PARSER')
    # logging interval by iter
    parser.add_argument('--log_interval', type=int, default=500, help='logging interval by iteration')
    # tensorboard
    parser.add_argument('--use_tensorboard', action='store_true', default=False,)
    parser.add_argument('--tensorboard_root', type=str, default='', help='root path of tensorboard, project path')
    parser.add_argument('--tensorboard_dir', type=str, default='', help='detail folder of tensorboard')
    # wandb config
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_key', default=WANDB_KEY, type=str)
    parser.add_argument('--wandb_project', type=str, default='RISEMAR')
    parser.add_argument('--wandb_root', type=str, default='')
    parser.add_argument('--wandb_dir', type=str, default='')
    # tqdm config
    parser.add_argument('--use_tqdm', action='store_true', default=False)

    # DDP
    parser.add_argument('--local-rank', type=int, default=-1, help='node rank for torch distributed training')
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for torch distributed training')
    
    # data_path
    parser.add_argument('--img_size', default=512, nargs='+', type=int, help='combined image size')
    parser.add_argument('--crop_size', default=128, nargs='+', type=int, help='crop size for medical volume cropping.')
    parser.add_argument('--patch_size', default=16, nargs='+', type=int, help='patch size for patchfied training.')

    # dataset
    parser.add_argument('--dataset_name', default='deepl', type=str, help='name of the dataset')
    parser.add_argument('--num_train', default=10000, type=float, help='number/ratio of training examples')
    parser.add_argument('--num_val', default=1000, type=float, help='number/ratio of validation examples')

    # dataloader
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers, 4 is a good choice')
    parser.add_argument('--drop_last', default=False, action='store_true', help='dataloader droplast')
    
    # optimizer
    parser.add_argument('--accum_steps', default=1, type=int)
    parser.add_argument('--optimizer', default='adam', type=str, help='name of the optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')    
    parser.add_argument('--beta1', default=0.5, type=float, help='Adam beta1')    
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta2')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay for optimizer')
    parser.add_argument('--epochs', default=60, type=int, help='number of training epochs')
    parser.add_argument('--warmup_steps', default=10, type=int, help='number of epochs for warmup')
    parser.add_argument('--clip_grad', default=-1, type=float, help='clipped value of gradient.')
    
    # scheduler
    parser.add_argument('--scheduler', default='', type=str, help='name of the scheduler')
    parser.add_argument('--step_size', default=10, type=int, help='step size for StepLR')
    parser.add_argument('--milestones', nargs='+', type=int, help='milestones for MultiStepLR')
    parser.add_argument('--step_gamma', default=0.5, type=float, help='learning rate reduction factor')
    parser.add_argument('--min_lr', default=0., type=float)
    
    # checkpath && resume training
    parser.add_argument('--checkpoint_root', type=str, default='', help='where to save the checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='test', help='detail folder of checkpoint')
    parser.add_argument('--save_epochs', default=1, type=int, help='interval of epochs for saving checkpoints')
    parser.add_argument('--save_net_only', default=False, type=bool, help='only save the network param, discard the optimizer and scheduler')
    parser.add_argument('--load_net', default=False, action='store_true', help='load network param or not')
    parser.add_argument('--load_opt', default=False, action='store_true', help='load optimizer and scheduler or not')
    parser.add_argument('--net_checkpath', default='', type=str, help='checkpoint path for network param')
    parser.add_argument('--opt_checkpath', default='', type=str, help='checkpoint path for optimizer and scheduler param')
    parser.add_argument('--log_images', default=False, action='store_true')
    
    # network hyper args
    parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data (2D/3D)')
    parser.add_argument('--loss_type_list', nargs='+', default='', type=str, help='loss functions list')
    parser.add_argument('--net_name', default='', type=str, help='name of the network, e.g. unet.') 
    parser.add_argument('--net_dict', default="{}", type=str, help='dictionary specifying network architecture.')
    
    # EMA
    parser.add_argument('--ema_momentum', default=0.999, type=float)
    
    # CT setting
    parser.add_argument('--min_hu', default=-1024, type=int)
    parser.add_argument('--max_hu', default=3072, type=int)
    parser.add_argument('--flip_prob', default=0, type=float, help='probability of random flip for data augmentation')
    parser.add_argument('--rot_prob', default=0, type=float, help='probability of random rotation for data augmentation')

    # SemiMAR
    parser.add_argument('--loss_factor2', default=0, type=float, help='')
    parser.add_argument('--qua_thres', default=7, type=float, help='')
    parser.add_argument('--qua_thres2', default=10, type=float, help='')
    
    return parser



def main(opt):
    net_name = opt.net_name
    print('torch.__version__: ', torch.__version__)
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    print('torch.backends.cudnn.version(): ', torch.backends.cudnn.version())
    print('torch.cuda.device_count(): ', torch.cuda.device_count())
    print('Network name: ', net_name)
    print('Random seed: ', SEED)
    
    # Fix random seed 
    fix_seed(SEED)
    
    # Select networks
    if net_name in ['supervised', 'risemar']:
        net_dict = dict(in_channels=1, out_channels=1, base_channels=64, norm_type='INSTANCE', act_type='RELU')
        net_dict.update(eval(opt.net_dict))
        net = RiseMARNet(**net_dict)
    elif net_name == 'cqa':
        net_dict = dict(
            in_channels=1, out_channels=10, drop_path_rates=0.1, prompt_dim=-1, 
            use_rope=True, block_kwargs={'norm_type':'INSTANCE'})
        net_dict.update(eval(opt.net_dict))
        net = CQA(**net_dict)
    elif net_name == 'cqaclip':
        net_dict = dict(out_channels=10)
        net_dict.update(eval(opt.net_dict))
        net = CQACLIP(**net_dict)
    elif net_name == 'cqavgg':
        net_dict = dict(out_channels=10)
        net_dict.update(eval(opt.net_dict))
        net = CQAVGG(**net_dict)
    else:
        print(f'Unsupported network: {net_name}')    
        net = None

    # Select trainers
    if net_name in ['cqa', 'cqaclip', 'cqavgg']:
        trainer = CQATrainer(opt, net)
    elif net_name in ['risemar',]:
        trainer = RiseMARTrainer(opt, net)
    else:  # common supervised trainer
        trainer = SupervisedMARTrainer(opt, net)

    trainer.fit()


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    main(opt)

