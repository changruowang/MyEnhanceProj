import argparse
import os

from util import util
from util import mydist
import copy
import torch


# from MyEnhanceProj.util import mydist
# import MyEnhanceProj.util.util as utils

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--output_mode', type=str, default='direct_img', help='chooses ...')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--config', default='', type=str, help='Path to the config file.')
        self.parser.add_argument('--dataroot', default='', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')

        self.parser.add_argument('--box_filter', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--use_distribute_train', action='store_true', help='use distribute train')

        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--patchSize', type=int, default=64, help='then crop to this size')
        
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_patchD', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 / cuda:1 / cpu')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

        self.parser.add_argument('--which_direction', type=str, default='A', help='A or ')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--skip', type=float, default=0.8, help='B = net.forward(A) + skip*A')
        

        # self.parser.add_argument('--train_base_decom', type=bool, default=True, help='output exposure num')
        self.parser.add_argument('--which_epoch', type=str, default=' ', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--K_GA', type=float, default=1.0, help='生成器GA损失权重')
        # self.parser.add_argument('--use_pre_decom', type=bool, default=True, help='生成器GA损失权重')
        self.parser.add_argument('--K_GL', type=float, default=0.2, help='光照平滑损失')
        self.parser.add_argument('--K_GP', type=float, default=1.0, help='光照平滑损失')
        self.parser.add_argument('--milestones', default='100,125,150', help='')

        self.parser.add_argument("--spatio_size", type=int, default=64)
        self.parser.add_argument("--feat_dim", type=int, default=-1)
        # self.parser.add_argument("--start_r", type=int, default=1, help="start layer to use resblock")
        self.parser.add_argument("--mc", type=int, default=512, help="# max channel")

        self.parser.add_argument('--use_mse', action='store_true', help='MSELoss')
        self.parser.add_argument('--l1', type=float, default=10.0, help='L1 loss weight is 10.0')
        self.parser.add_argument('--use_norm', type=float, default=1, help='L1 loss weight is 10.0')
        self.parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
        self.parser.add_argument('--use_ragan', action='store_true', help='use ragan')
        self.parser.add_argument('--vgg', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
        self.parser.add_argument('--vgg_choose', type=str, default='relu5_1', help='choose layer for vgg')
        self.parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
        self.parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
        self.parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
        self.parser.add_argument('--fcn', type=float, default=0, help='use semantic loss')
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--instance_norm', type=float, default=0, help='use instance normalization')
        self.parser.add_argument('--syn_norm', action='store_true', help='use synchronize batch normalization')
        self.parser.add_argument('--tanh', action='store_true', help='tanh')
        self.parser.add_argument('--linear', action='store_true', help='tanh')
        self.parser.add_argument('--new_lr', action='store_true', help='tanh')
        self.parser.add_argument('--multiply', action='store_true', help='tanh')
        self.parser.add_argument('--noise', type=float, default=1, help='variance of noise')
        self.parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_norm', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')

        self.parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        self.parser.add_argument('--patchD_3', type=int, default=0, help='choose the number of crop for patch discriminator')
        # self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
        self.parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--high_times', type=int, default=400, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--norm_attention', action='store_true', help='normalize attention map')
        # self.parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
        # self.parser.add_argument('--lighten', action='store_true', help='normalize attention map')
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.initialized = True

    def merg_config(self):
        
        if(not os.path.exists(self.opt.config)):
            return
        print("load config from file.")
        import yaml
        with open(self.opt.config, 'r') as stream:
            cfg = yaml.safe_load(stream)
            for k, v in cfg.items():
                setattr(self.opt, k, v)
            # print(type(self.opt))
            return cfg

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        ###测试的的时候不会被配置文件覆盖的 选项  
        if not self.isTrain:
            opt_cp = copy.deepcopy(self.opt)

        cfg = self.merg_config()

        # save to the disk
        args = vars(self.opt)
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if mydist.is_main_rank() and self.isTrain:
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.yaml')
            with open(file_name, 'wt') as opt_file:
                print('------------ Options -------------')
                opt_file.write('####------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                    print('%s: %s' % (str(k), str(v)))
                opt_file.write('####-------------- End ----------------\n')
                print('-------------- End ----------------')

        self.opt.milestones = [int(num) for num in self.opt.milestones.split(",")]

        self.opt.isTrain = self.isTrain   # train or test


        self.opt.dataroot = cfg['train_dataroot']

        if not self.isTrain:   ## 训练的时候才开启
            self.opt.use_distribute_train = False
            self.opt.dataroot = opt_cp.dataroot
            self.opt.dataset_mode = opt_cp.dataset_mode
            self.opt.resize_or_crop = opt_cp.resize_or_crop
            self.opt.device = opt_cp.device
            self.opt.which_epoch = opt_cp.which_epoch
            self.opt.phase = opt_cp.phase
            
        if self.opt.use_distribute_train:
            # torch.backends.cudnn.benchmark = True
            # torch.backends.cudnn.deterministic = False
            # torch.backends.cudnn.enabled = True

            # os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
            torch.distributed.init_process_group(backend="nccl",init_method='env://')
            local_rank = torch.distributed.get_rank()
            self.opt.local_rank = local_rank
            torch.cuda.set_device(local_rank)
            self.opt.device = torch.device("cuda", local_rank)
            
        else:
            self.opt.device = torch.device(self.opt.device)
        
        return self.opt
