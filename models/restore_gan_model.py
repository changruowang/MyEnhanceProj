#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 基于GAN的图像提亮结构，集成了训练和测试逻辑，由train.py test.py文件调用。
                 主要实现了 以下功能：LSG/WGAN  RaGAN  UnetGAN  局部GAN   VGG感知损失
@Date          : 2021/03/29 14:19:50
@Author        : changruowang
@version       : 1.0
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import OrderedDict
# from torch.autograd import Variable
from collections import OrderedDict
# from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
import random
from . import networks
from . import loss
import sys


class RestoreGANModel(BaseModel):
    def name(self):
        return 'RestoreGANModel'

    def initialize(self, opt):
        self.opt = opt

        BaseModel.initialize(self, self.opt)
        ### 设置生成器
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, skip=True, opt=opt)
        ### 设置生成器的优化器
        self.optimizer_G = self.add_optimeizer(self.netG_A, opt, 'G_A')

        if self.isTrain:
            self.netG_A.train()

            ### 设置全局判别器
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(6 if opt.use_rotate_aug else 3, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, False, opt=opt)
            self.optimizer_D_A = self.add_optimeizer(self.netD_A, opt, 'D_A')

            ### 设置局部判别器
            if self.opt.patchD_3 > 0:
                self.netD_P = networks.define_D(3, opt.ndf,
                                            'no_norm_4',
                                            opt.n_layers_patchD, opt.norm, use_sigmoid, True, opt=opt)
                self.optimizer_D_P = self.add_optimeizer(self.netD_P, opt, 'D_P')

            ## 其他损失
            if opt.vgg > 0:
                self.vgg_loss = loss.PerceptualLoss(opt)

            if opt.use_wgan:
                assert(not use_sigmoid)  ##　wgan 的对抗器输出不能加激活函数
                self.criterionGAN = loss.DiscLossWGANGP()
            else:
                self.criterionGAN = loss.GANLoss(use_lsgan=not opt.no_lsgan, opt=self.opt)

            self.criterionLSGAN = loss.GANLoss(use_lsgan=True, opt=self.opt)
            self.criterionMSE = loss.L_MSE()
            self.criterionSpatial = loss.SpatialConsistencyLoss(grad_normal=False, pre_avg=False)

            self.load_networks(opt.which_epoch)

            if opt.continue_train:
                self.resum_networks(opt.which_epoch)


        else: 
            self.netG_A.eval()
            self.load_networks(opt.which_epoch)
            
        print('---------- Networks initialized -------------')
        self.print_networks()
        print('-----------------------------------------------')


    def set_input(self, input):
        """设置输入数据
        args: 
            input: 字典格式，例如 {'A': A_img, 'B': B_img} 其中 A_img/B_img为tensor。
            'real_A': 原始低光照图
            'real_input_A_ref': 低光照图A 同场景 不对齐的参考图（根据参数设置是否模糊）
            'real_A_ref': 低光照图A 同场景 不对齐的参考图
            'real_B': 正常亮度的图
            'A_motion_mask': 预先计算的运动掩码
            'real_A_enhance': 预先匹配增强的图象
        """
        input_A =  input['A']  ### 输入
        input_B = input['B'] 
        input_A_ref = input['A_input_ref']
        A_ref = input['A_ref']
        A_enhance = input['A_enhance']
        input_A_gray = input['A_gray']

        self.real_A_enhance = A_enhance.to(device=self.device)

        try:
            self.image_paths = input['A_paths']
        except:
            self.image_paths = None

        self.real_A = input_A.to(device=self.device) 
        self.real_B = input_B.to(device=self.device)

        self.real_A_gray = input_A_gray.to(device=self.device)
        self.real_input_A_ref = input_A_ref.to(device=self.device)
        self.real_A_ref = A_ref.to(device=self.device)

        if self.isTrain:
            A_motion_mask = input['motion_mask']
            self.motion_mask = A_motion_mask.to(device=self.device)

    @torch.no_grad()
    def predict(self):
        """该模型测试代码 
        """
        self.get_R_out(self.netG_A.forward(self.real_A, self.real_input_A_ref, self.real_A_gray))

        real_A = util.tensor2im(self.real_A.data)
        real_A_ref = util.tensor2im(self.real_A_ref.data)
        fake_B = util.tensor2im(self.fake_B.data)
        re_od =OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('ref_A', real_A_ref)])

        return re_od

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_A(self, epoch):
        """对全局判别器进行 优化 更新梯度
        """
        ### unet-gan 文章中的设置，mix batch的概率是慢慢增加的，增加到0.5，起始的epoch不开启mix  full_batch_mixup 默认true
        if self.opt.slow_mixup and self.opt.full_batch_mixup:
            # r_mixup is the chance that we select a mixed batch instead of
            # a normal batch. This only happens in the setting full_batch_mixup.
            # Otherwise the mixed loss is calculated on top of the normal batch.
            r_mixup = 0.5 * min(1.0, epoch/(self.opt.niter*0.3)) # r is at most 50%, after reaching warmup_epochs
        elif not self.opt.slow_mixup and self.opt.full_batch_mixup:
            r_mixup = 0.5
        else:
            r_mixup = 0.0
        
        ### mix-操作和 unet_dis要一起使用
        use_unet_dis = self.opt.which_model_netD == 'unet_dis'
        use_mixup_in_this_round = (self.opt.mix_argue and use_unet_dis and (torch.rand(1).detach().item() < r_mixup))
        self.use_mixup_in_this_round = use_mixup_in_this_round 

        if self.opt.slow_mixup and not self.opt.full_batch_mixup:
            mixup_coeff = min(1.0, epoch/(self.opt.niter*0.3) )#use without full batch mixup
        else:
            mixup_coeff = 1.0

        loss_fake = 0
        loss_real = 0
        self.loss_D_A = 0
        self.loss_D_normal = 0

        if self.opt.mix_argue:
            self.mixed_img = self.mix_target*self.real_B+(1-self.mix_target)*self.fake_B_detach
            dis_mix_input = torch.cat((self.mixed_img, self.real_B_condition), dim=1) if self.opt.use_rotate_aug else \
                                                                                         self.mixed_img
        ### 判别器使用 条件GAN的形式 即输入为两张图 cat
        if self.opt.use_rotate_aug:
            dis_real_input = torch.cat((self.real_B, self.real_B_condition), dim=1)
            dis_fake_input = torch.cat((self.fake_B_detach, self.real_A_condition), dim=1)
        else:
            dis_real_input = self.real_B 
            dis_fake_input = self.fake_B.detach()

        ###### 计算 对抗 损失 
        def cal_dis_loss(pred_fake_, pred_real_, real_input_=None, fake_input_=None, out_sel=None):
            if self.opt.use_wgan:  
                loss_D = pred_fake_.mean() - pred_real_.mean() + \
                            self.criterionGAN.calc_gradient_penalty(self.netD_A,  real_input_.data, fake_input_.data, out_sel)
                # print(pred_fake_.mean().item(), pred_real_.mean().item())
            elif not self.opt.hybrid_loss:
                loss_D = (self.criterionGAN(pred_real_ - torch.mean(pred_fake_), True) +
                                    self.criterionGAN(pred_fake_ - torch.mean(pred_real_), False)) / 2
            else:
                loss_D = (self.criterionGAN(pred_fake_, False) + self.criterionGAN(pred_real_, True)) / 2
            return loss_D
        #######   
        ### 计算判别器常规的 GAN 损失
        if not (use_mixup_in_this_round and self.opt.full_batch_mixup):
            if use_unet_dis:
                pred_real, pred_real_middle = self.netD_A.forward(dis_real_input)
                pred_fake, pred_fake_middle = self.netD_A.forward(dis_fake_input)
                # self.loss_D_normal += cal_dis_loss(pred_fake_middle, pred_real_middle, dis_real_input, dis_fake_input, 1)
  
            else:
                pred_real = self.netD_A.forward(dis_real_input)
                pred_fake = self.netD_A.forward(dis_fake_input)

            self.loss_D_normal += cal_dis_loss(pred_fake, pred_real,  dis_real_input, dis_fake_input, 0 if use_unet_dis else None)
            self.loss_D_A += self.loss_D_normal

        #### 计算 判别器 mix 相关的损失 
        if use_mixup_in_this_round and use_unet_dis:
            fake_target = torch.tensor([0.0]).to(self.opt.device)

            pred_mix, pred_mix_middle = self.netD_A.forward(dis_mix_input) 

            self.loss_mix_argue = self.criterionGAN(pred_mix, self.mix_target) + \
                                self.criterionGAN(pred_mix_middle, fake_target.expand_as(pred_mix_middle)) * mixup_coeff
            self.pred_mix = pred_mix.detach()
            mixed_pred =  pred_real*self.mix_target + pred_fake*(1-self.mix_target)
            self.loss_mix_consist = mixup_coeff*1.0*F.mse_loss(pred_mix, mixed_pred )

            self.loss_D_A += self.loss_mix_consist + self.loss_mix_argue

        ####如果不使用 WGAN，则判别器学的慢点 
        if not self.opt.use_wgan:  
            self.loss_D_A *= 0.3   

        self.loss_D_A.backward()

    def backward_D_P(self):
        """对局部判别器进行 优化 更新梯度
        """
        #### 局部判别器只写了 LSG损失
        self.loss_D_P = 0
        for fake_patchs_layer, real_patchs_layer in zip(self.fake_patchs_sample, self.real_patchs_sample):
         
            pred_real = self.netD_P.forward(real_patchs_layer)
            pred_fake = self.netD_P.forward(fake_patchs_layer.detach())

            if self.opt.hybrid_loss:
                loss_D_real = self.criterionLSGAN(pred_real, True)
                loss_D_fake = self.criterionLSGAN(pred_fake, False)

                self.loss_D_P += (loss_D_real + loss_D_fake) * 0.5
            else:
                AssertionError('1')
        self.loss_D_P = self.loss_D_P * self.opt.K_GP * 0.5
        self.loss_D_P.backward()    
  
    def get_R_out(self, Ls, epoch=None):
        """在生成器输出结果后，集中在这个函数中计算一些 计算损失的时候 用的上的中间过程
        1. 从生成的fake图中 做局部patch的采样
        """

        self.fake_B = Ls[0]
        self.latent_real_A = Ls[1]

        if not self.isTrain:
            return

        ### patch采样
        if self.opt.patchD_3 > 0:
            self.fake_patchs_sample = []
            self.input_patchs_sample = []
            self.real_patchs_sample = []

            fake_patchs, real_patchs, input_patchs = self.samlpe_one_layer(Ls[0])
            
            self.fake_patchs_sample.append(fake_patchs)
            self.input_patchs_sample.append(input_patchs)
            self.real_patchs_sample.append(real_patchs)

        if self.opt.mix_argue:
            sz = self.opt.fineSize
            
            n_mixed = self.fake_B.size(0)
            self.mix_target = torch.cat(
                [util.CutMix(sz).to(self.opt.device).view(1,1,sz,sz) for _ in range(n_mixed)], dim=0)

        if self.opt.use_rotate_aug:
            self.fake_B_detach = util.center_rotate_crop(self.fake_B.detach(), 0, p=256)
            self.real_A_condition = util.center_rotate_crop(self.real_A_ref, p=256)
            self.real_B_condition = util.center_rotate_crop(self.real_B, p=256)
            self.real_B = util.center_rotate_crop(self.real_B, 0, p=256)  ### 只crop
            
    def samlpe_one_layer(self, images, masks=None):
        w = images.size(3)
        h = images.size(2)

        fake_patchs_ = []
        real_patchs_ = []
        input_patchs_ = []

        if masks is None:   ### balance sample
            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                fake_patchs_.append(images[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                real_patchs_.append(self.real_B[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize])
                input_patchs_.append(self.input[:,:, h_offset_1:h_offset_1 + self.opt.patchSize,
                    w_offset_1:w_offset_1 + self.opt.patchSize]) 
            return torch.cat(fake_patchs_, 0), torch.cat(real_patchs_, 0), torch.cat(input_patchs_, 0)
            
    def backward_G(self, epoch):
        use_unet_dis = self.opt.which_model_netD == 'unet_dis'

        self.loss_G_A = 0
        #### 设置判别器的输入
        if self.opt.use_rotate_aug:
            dis_input = torch.cat((self.fake_B, self.real_A_ref), dim=1)
            dis_real_input = torch.cat((self.real_B, self.real_B_condition), dim=1) 
        else:
            dis_input = self.fake_B
            dis_real_input = self.real_B

        ### 如果是 Unet GAN 
        if use_unet_dis:
            pred_fake, pred_fake_middle = self.netD_A.forward(dis_input)
        else:
            pred_fake = self.netD_A.forward(dis_input)

        #### 损失是 WGAN 的形式还是 raGAN的形式  还是 LSG GAN
        if self.opt.use_wgan:
            self.loss_G_A = -pred_fake.mean()
        elif self.opt.use_ragan:
            if use_unet_dis:
                pred_real, pred_real_middle = self.netD_A.forward(dis_real_input)
                self.loss_G_A += ((self.criterionGAN(pred_real_middle - torch.mean(pred_fake_middle), False, mask=None) + \
                            self.criterionGAN(pred_fake_middle - torch.mean(pred_real_middle), True, mask=None)) / 2)
            else:
                pred_real = self.netD_A.forward(dis_real_input)

            self.loss_G_A += ((self.criterionGAN(pred_real - torch.mean(pred_fake), False, mask=None) + \
                            self.criterionGAN(pred_fake - torch.mean(pred_real), True, mask=None)) / 2)
                   
        else:
            self.loss_G_A += self.criterionGAN(pred_fake, True)
            if use_unet_dis:
                self.loss_G_A += self.criterionGAN(pred_fake_middle, True)

        self.loss_G_A = self.loss_G_A * self.opt.K_GA 

        ### 计算局部判别损失
        self.loss_G_P = 0
        if self.opt.patchD_3 > 0:
            for fake_patchs_layer, real_patchs_layer in zip(self.fake_patchs_sample, self.real_patchs_sample):

                pred_fake_patch = self.netD_P.forward(fake_patchs_layer)
                if self.opt.hybrid_loss:
                    self.loss_G_P += self.criterionLSGAN(pred_fake_patch, True)
                else:   ### raGAN
                    pred_real_patch = self.netD_P.forward(real_patchs_layer)
                    self.loss_G_P += (self.criterionLSGAN(pred_real_patch - torch.mean(pred_fake_patch), False) +
                                        self.criterionLSGAN(pred_fake_patch - torch.mean(pred_real_patch), True)) / 2
               
            self.loss_G_P  = self.loss_G_P  * self.opt.K_GP
        ### 计算 VGG 损失  和局部 VGG 损失
        self.loss_vgg = 0
        if self.opt.vgg > 0: 
            self.loss_vgg += self.vgg_loss.compute_vgg_loss(self.fake_B, self.input) 
            if self.opt.patch_vgg and self.opt.patchD_3 > 0:
                loss_vgg_patch = 0  
                loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.fake_patchs_sample[0], self.input_patchs_sample[0]) 
                self.loss_vgg  += loss_vgg_patch 
            self.loss_vgg = self.loss_vgg  * self.opt.vgg

        ### 其他 MSE 损失 梯度损失 等
        self.loss_mse = 0
        self.loss_spa = 0
 
        self.loss_mse = self.criterionMSE(self.fake_B, self.real_A_enhance, mask=self.motion_mask) * self.opt.K_MSE
        self.loss_mse += self.criterionMSE(self.fake_B, self.real_A_ref, mask=1-self.motion_mask) * self.opt.K_MSE 

        self.loss_spa = self.criterionSpatial(self.fake_B, self.real_A_enhance, mask=self.motion_mask) * self.opt.K_GSPA
        # self.loss_spa += self.criterionSpatial(self.fake_B, self.real_A_ref, mask=1-self.motion_mask) * self.opt.K_GSPA * 0.0
        self.loss_G = self.loss_G_A + self.loss_mse + self.loss_G_P + self.loss_spa
  
        self.loss_G.backward()


    def optimize_parameters(self, epoch):
        ### 前向  生成 fake img
        self.input, self.input_gray, self.input_ref = self.real_A, self.real_A_gray, self.real_input_A_ref
      
        self.get_R_out(self.netG_A.forward(self.input, self.input_ref, self.input_gray), epoch=epoch)

        ### 优化全局判别器
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D_A.zero_grad()  
        self.backward_D_A(epoch)           
        self.optimizer_D_A.step()       
        self.set_requires_grad(self.netD_A, False)
        ### 优化局部判别器
        if self.opt.patchD_3 > 0:
            self.set_requires_grad(self.netD_P, True)
            self.optimizer_D_P.zero_grad()  
            self.backward_D_P()
            self.optimizer_D_P.step()
            self.set_requires_grad(self.netD_P, False)
        ### 优化生成网络
        # if epoch_iter % 3:
        self.optimizer_G.zero_grad()  ### 优化生成器
        self.backward_G(epoch)        ### 求生成器损失   fake false 
        self.optimizer_G.step()  


    def get_current_errors(self, epoch):
        """获取训练过程的损失信息, train.py调用
        args:
            epoch: 没用到
        return:
            将当前各项损失存储为 字典结构返回。例如 {}
        """
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        l_mse = self.loss_mse.item()
        l_spa = self.loss_spa.item()
        loss_od = OrderedDict([('D_A', D_A), ('G_A', G_A), ('l_mse', l_mse), ('l_spa', l_spa)])

        if self.opt.patchD_3 > 0:
            loss_od['D_P'] = self.loss_D_P.item()
            loss_od['G_P'] = self.loss_G_P.item()

        if self.opt.vgg > 0:
            loss_od['vgg'] = self.loss_vgg.item()

        if self.opt.mix_argue and self.use_mixup_in_this_round:
            loss_od['mix_consist'] = self.loss_mix_consist.item()
            loss_od['mix_argue'] = self.loss_mix_argue.item()

        return loss_od

    def get_current_visuals(self):
        """获取训练过程中的中间结果图象
        return:
            将图片存储为 字典结构返回。例如 {'A':A_img}  其中A_img图片 为 np 格式
        """
        real_A_enhance = util.tensor2im(self.real_A_enhance.data)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_A_ref = util.tensor2im(self.real_A_ref.data)
        # input_ref = util.tensor2im(self.input_ref.data)
        motion_mask = util.tensor2im(self.motion_mask.data*2-1)

        image_od = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),('mask', motion_mask), \
            ('real_A_ref', real_A_ref), ('A_enhance', real_A_enhance)])

        # image_od['fake_patch'] = util.tensor2im(self.fake_patchs_sample[0].data)
        # image_od['input_patch'] = util.tensor2im(self.input_patchs_sample[0].data)
        # image_od['real_patch'] = util.tensor2im(self.real_patchs_sample[0].data)
        
        if self.opt.use_rotate_aug:
            assert(self.real_B_condition.shape[-1] == 256)
            image_od['condition_img'] = util.tensor2im(self.real_B_condition.data)

        if self.use_mixup_in_this_round:
            image_od['mixed'] = util.tensor2im(self.mixed_img.data)
            image_od['mixed_pred'] = util.tensor2im(self.pred_mix.data)
         
        return image_od      
        
