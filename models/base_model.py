import os
import torch
import sys


from util import mydist
from util import util as utils
# from ..util import mydist
# from ..util import util

# from torch.utils.data.distributed import DistributedSampler



class BaseModel():
    '''基础模型类  其他的  restore_model.py  restore_gan_model.py  restore_fusion_model.py中的模型 继承此类
    '''
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.optimizers = {}   ### 将所有模型 优化器 学习率调整器 保存在 字典中 方便统一管理
        self.opt = opt
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if opt.device!='cpu' else torch.Tensor
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = opt.device

        if opt.output_mode == 'line_trans':
            assert(opt.output_nc == 12)
            assert(opt.tanh == False and opt.skip == 0 and not opt.self_attention)    ## 输出不能限制在-1，1   
        elif opt.output_mode == 'cur_trans':
            assert(opt.output_nc == 24)
            assert(opt.tanh == True and opt.skip == 0 and not opt.self_attention)
        elif opt.output_mode == 'fusion':
            assert(opt.output_nc == 2)
            assert(opt.tanh == True and opt.skip == 0 and not opt.self_attention)
        elif opt.output_mode == 'direct_img':
            assert(opt.output_nc == 3)


    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def line_transform(self, img, coeff):
        ''' HDR-net中的 线性变换 根据网络预测的 3x4 的变换参数矩阵对 图象进行变换
        Args:
            img: 待变换的图象 tensor
            coeff: 参数矩阵 长宽和img大小一样 通道数为 12
        '''
        R = torch.sum(img * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(img * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(img * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

    def cur_transform(self, x, coeff):
        ''' ZeroDEC 中的 曲线变换 
        Args:
            img: 待变换的图象 tensor
            coeff: 参数矩阵 长宽和img大小一样
        '''
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(coeff, 3, dim=1)

        x = x + r1*(torch.pow(x,2)-x)
        x = x + r2*(torch.pow(x,2)-x)
        x = x + r3*(torch.pow(x,2)-x)
        enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + r6*(torch.pow(x,2)-x)	
        x = x + r7*(torch.pow(x,2)-x)
        enhance_image = x + r8*(torch.pow(x,2)-x)
		
        return enhance_image

    def torch_normal_ch(self, input_):
        ''' 通道归一化 
        '''
        sum_ = torch.sum(input_, dim=1,keepdim=True)
        return input_ / (sum_ + 0.00001)

    def fusion_transform(self, x1, x2, w):
        ''' 格局权重 融合 x1  x2 图象
        '''
        w_ = self.torch_normal_ch((w + 1) / 2)
        # print(w_.shape)
        # print(w_[:, 0:1, :, :].shape, x1.shape,  x2.shape)
        return w_[:, 0:1, :, :] * x1 + w_[:, 1:2, :, :] * x2

    def set_requires_grad(self, nets, requires_grad=False):
        ''' 设置是否求梯度
        '''
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def print_network(net, name):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('%s total number of parameters: %d' % (name, num_params))


    def print_networks(self):
        '''打印所有网络结构
        '''
        for model_name, model_opt in self.optimizers.items():
            self.print_network(model_opt['model'], model_name)
            print('\n')

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        if not mydist.is_main_rank():
            return 
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        latest_filename = '%s_net_%s.pth' % ('latest', network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {}

        save_dict['epoch'] = epoch_label
        if isinstance(network, dict):
            save_dict['model'] = mydist.get_model_param(network['model'])
            save_dict['optim'] = network['optim'].state_dict()
            save_dict['lr_sch'] = network['lr_sch'].state_dict()
        else: 
            save_dict['model'] = mydist.get_model_param(network) 

        torch.save(save_dict, save_path)
        utils.symlink(save_filename, os.path.join(self.save_dir, latest_filename))

    ### 一次性保存所有模型 包括优化器  
    def save_networks(self, epoch_label):
        '''一次性保存所有模型 包括优化器  
        '''
        if not mydist.is_main_rank():
            return
        for model_name, model_opt in self.optimizers.items():
            self.save_network(model_opt, model_name, epoch_label)

    ### 加载一个网络
    def load_network(self, network, network_label, epoch_label): 
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path), strict=False)

    def load_networks(self, epoch_label):
        ''' 一次性加载所有模型 权重 包括优化器  
        '''
        for model_name, model_opt in self.optimizers.items():
            save_filename = '%s_net_%s.pth' % (epoch_label, model_name)
            save_path = os.path.join(self.save_dir, save_filename)
            if not os.path.exists(save_path):
                return 
            try:
                model_opt['model'].load_state_dict(torch.load(save_path, map_location=self.device)['model'],strict=False) 
            except:
                # new_params = {k.replace('module.', '', 1):v for k, v in torch.load(save_path).items()}
                model_opt['model'].load_state_dict(torch.load(save_path, map_location=self.device),strict=False)
            print('Success load model wights from %s !'%epoch_label)
                
    #### resum 训练
    def resum_networks(self, epoch_label):
        for model_name, model_opt in self.optimizers.items():
            save_filename = '%s_net_%s.pth' % (epoch_label, model_name)
            save_path = os.path.join(self.save_dir, save_filename)
            resum_dict = torch.load(save_path, map_location=self.device)
            model_opt['model'].load_state_dict(resum_dict['model'])
            model_opt['optim'].load_state_dict(resum_dict['optim'])
            model_opt['lr_sch'].load_state_dict(resum_dict['lr_sch'])

    ### 初始化优化器 可以自定义
    def add_optimeizer(self, model, opt, name, lr=None):
        lr = opt.lr if lr is None else lr
        opti = {'model':None, 'optim':None, 'lr_sch':None}
        opti['model'] = model
        if self.isTrain:
            opti['optim'] = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(opt.beta1, 0.999))
            opti['lr_sch'] = torch.optim.lr_scheduler.MultiStepLR(opti['optim'], milestones=self.opt.milestones, gamma=0.2)    
            model.train() 
        else:
            model.eval() 
        self.optimizers[name] = opti
        return opti['optim'] 

    ### 统一更新所有优化器的学习率
    def update_learning_rate(self):
        for _, opt in self.optimizers.items():
            opt['lr_sch'].step()
        
