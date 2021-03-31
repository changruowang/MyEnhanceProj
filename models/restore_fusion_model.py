#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description : 多阶段融合模型，集成了 编解码模块训练过程，mask分割模块训练过程 和 特征融合
                训练过程backbone只支持 endec_atten_net
@Date        :2021/03/29 14:15:21
@Author      : changruowang
@version     : 1.0
'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from collections import OrderedDict
# from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
# from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
from . import loss
import sys
import torchvision


class FusionModel(BaseModel):
    def name(self):
        return 'FusionModel'

    def initialize(self, opt):
        self.opt = opt

        # assert(opt.opti_mode == 0 and opt.aligen_AB == True or opt.opti_mode > 0)

        BaseModel.initialize(self, self.opt)

        skip = True if opt.skip > 0.1 else False
        ## 初始化生成网路  opt.which_model_netG == endec_atten_net
        ## 这里的 skip 表示 跳过解码器的权重加载
        skip = self.opt.opti_mode==3
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, skip=skip, opt=opt)
     

        self.optimizer_G = self.add_optimeizer(self.netG_A, opt, 'G_A')

        if self.isTrain:
            ### opt.K_GA > 0 则开启GAN
            if opt.K_GA > 0:
                self.netD_A = networks.define_D(6, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, False, False, opt=opt)
                self.optimizer_D_A = self.add_optimeizer(self.netD_A, opt, 'D_A', lr=opt.lr*4)
                self.criterionGAN = loss.DiscLossWGANGP()
                
                self.vgg_loss = loss.PerceptualLoss(opt)

                ### opt.patchD_3 > 0 则开启局部GAN
                if self.opt.patchD_3 > 0:
                    self.netD_P = networks.define_D(6, opt.ndf,
                                        opt.which_model_netP,
                                        opt.n_layers_D, opt.norm, False, False, opt=opt)
                    self.optimizer_D_P = self.add_optimeizer(self.netD_P, opt, 'D_P', lr=opt.lr*4)


            self.load_networks(opt.which_epoch)

            if opt.continue_train:
                self.resum_networks(opt.which_epoch)

            self.criterionTV = loss.L_RTV(opt) 
            self.criterionMSE = loss.L_MSE(clamp=True)
            self.criterionExposur = loss.L_exp(31)
            self.criterionSpatial = loss.SpatialConsistencyLoss(grad_normal=False, pre_avg=False)
            self.criterionCosine = loss.L_cosine(clamp=True)

        else: 
            self.netG_A.eval()
            self.load_networks(opt.which_epoch)
            
        print('---------- Networks initialized -------------')
        self.print_networks()
        print('-----------------------------------------------')


    def set_input(self, input):
        self.batch_data = input
        
        input_A =  input[self.opt.which_direction]  ### 输入
        input_B = input['B'] 
        input_A_ref = input['A_input_ref']
        A_ref = input['A_ref']
        A_enhance = input['A_enhance']
        # input_A_gray = input['A_gray']

        self.real_A_enhance = A_enhance.to(device=self.device)

        try:
            self.image_paths = input['A_paths']
        except:
            self.image_paths = None

        # self.real_A = self.real_A_enhance
        self.real_A = input_A.to(device=self.device) 
        self.real_B = input_B.to(device=self.device)

        # self.real_A_gray = input_A_gray.to(device=self.device)
        self.real_input_A_ref = input_A_ref.to(device=self.device)
        self.real_A_ref = A_ref.to(device=self.device)
        
        # if self.isTrain:
        A_motion_mask = input['motion_mask']
        self.motion_mask = A_motion_mask.to(device=self.device)

    @torch.no_grad()
    def predict(self):
        self.input, self.input_gray, self.input_ref = self.real_A, self.real_A_enhance, self.real_input_A_ref
        fake_B, pred_mask= self.netG_A.forward(self.input, self.input_ref)  # self.motion_mask

        real_A = util.tensor2im(self.real_A.data)
        real_A_ref = util.tensor2im(self.real_A_ref.data)
        fake_B = util.tensor2im(fake_B.data)
        re_od =OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('ref_A', real_A_ref)])

        re_od['pre_mask'] = util.tensor2im(pred_mask.detach()*2-1)

        return re_od

    @torch.no_grad()
    def test(self):
        criterionMSE = loss.L_MSE()
        criterionSpatial = loss.SpatialConsistencyLoss(grad_normal=False, pre_avg=False).to(self.device)
        criterionCosine = loss.L_cosine(clamp=True)
        self.input, self.input_gray, self.input_ref = self.real_A, self.real_A_enhance, self.real_input_A_ref
        self.fake_B , pred_mask = self.netG_A.forward(self.input, self.input_ref, self.input_gray)

        loss_mse1 = criterionMSE(self.fake_B, self.real_A_ref, mask=1-self.motion_mask, out_map=True) * 60
        loss_mse2 = criterionMSE(self.fake_B, self.real_A_enhance, mask=self.motion_mask, out_map=True) * 100
        loss_cos = criterionCosine(self.fake_B, self.real_A_enhance, mask=self.motion_mask, out_map=True)
        
        # print(torch.max(loss_cos))

        loss_spa1 = criterionSpatial(self.fake_B, self.real_A_enhance, mask=self.motion_mask, out_map=True) * 20  ##  mask=self.motion_mask
        loss_spa2 = criterionSpatial(self.fake_B, self.real_A, mask=None, out_map=True) * 20

        # loss_spa += criterionSpatial(self.fake_B, self.real_A_ref, mask=1-self.motion_mask) * 100

        # loss_cos = loss.torch_normal_img(loss_cos)
        # loss_mse2 = loss.torch_normal_img(loss_mse2)
        # loss_spa = loss.torch_normal_img(loss_spa)  ### 归一化后无法对比训练前后损失的变化 但是归一化可以对比一幅图中不同位置的损失分布

        real_A = util.tensor2im(self.real_A.data)
        real_A_enhance = util.tensor2im(self.real_A_enhance.data)
        real_A_ref = util.tensor2im(self.real_A_ref.data)
        fake_B = util.tensor2im(self.fake_B.data)
        motion_mask = util.tensor2im(pred_mask.data)

        
        loss_mse1 = torch.mean(loss_mse1, dim=1, keepdim=True)
        loss_mse2 = torch.mean(loss_mse2, dim=1, keepdim=True)
        loss_spa1 = torch.mean(loss_spa1, dim=1, keepdim=True)
        loss_spa2 = torch.mean(loss_spa2, dim=1, keepdim=True)

        # loss_mse1 = util.tensor2im(2*loss_mse1.data-1)
        loss_cos = util.tensor2im(2*loss_cos.data-1)
        loss_mse1 = util.tensor2im(2*loss_mse1.data-1)
        loss_mse2 = util.tensor2im(2*loss_mse2.data-1)
        loss_spa1 = util.tensor2im(2*loss_spa1.data-1)
        loss_spa2 = util.tensor2im(2*loss_spa2.data-1)
        re_od =OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('loss_cos', loss_cos), ('mask', motion_mask),\
            ('mse1', loss_mse1), ('fake_B', fake_B), ('mse2', loss_mse2), ('spa1', loss_spa1),('spa2', loss_spa2)
        ])

        return re_od

    def get_image_paths(self):
        return self.image_paths

    def cal_base_loss_items(self, fake_image, ref_img=None,use_mask=True, out_vector=False, pre_avg=False):
        if use_mask:
            fg_mask = self.motion_mask
            bg_mask = 1- self.motion_mask
        else:
            fg_mask = None
            bg_mask = None

        ref_img = self.real_A_ref if ref_img is None else ref_img

        mse1 = self.criterionMSE(fake_image, ref_img, mask=bg_mask, out_vector=out_vector) 
        mse2 = self.criterionMSE(fake_image, self.real_A_enhance, mask=fg_mask, out_vector=out_vector) 

        cos1 = self.criterionCosine(fake_image, ref_img, mask=bg_mask, out_vector=out_vector) 
        cos2 = self.criterionCosine(fake_image, self.real_A_enhance, mask=fg_mask, out_vector=out_vector)
        
        spa1 = self.criterionSpatial(fake_image, ref_img, mask=bg_mask, out_vector=out_vector) * 100 
        spa2 = self.criterionSpatial(fake_image, self.real_A_enhance, mask=fg_mask, out_vector=out_vector) * 100  

        self.loss_cos1 = cos1.item()
        self.loss_cos2 = cos2.item()
        self.loss_spa1 = spa1.item()
        self.loss_spa2 = spa2.item()
        self.loss_mse1 = mse1.item()
        self.loss_mse2 = mse2.item()

        return mse1,mse2,cos1,cos2,spa1,spa2

    def prepare_gan_data(self, fake_image, mask):
        """
        @description  : 在 GAN 融合的模式下 计算需要的数据  1. 条件图象生成， 2. 局部块的生成
        ---------
        @param  : 
            fake_image: 生成器生成的假样本
            mask: 运动前景的掩码 用于提取局部块
        -------
        @Returns  :
        -------
        """
        half_patch = 32

        fake_image_ = util.center_rotate_crop(fake_image, 0, 0, p=256)

        self.fake_B_256 = fake_image_
        self.real_A_condition = util.center_rotate_crop(self.real_A_ref, p=256)
        self.real_B_condition = util.center_rotate_crop(self.real_B, p=256)
        self.real_B = util.center_rotate_crop(self.real_B, 0, 0, p=256)  ### 只crop


        if self.opt.patchD_3 == 0:
            return 
        w = fake_image_.size(3)
        h = fake_image_.size(2)
        
        fake_patches = []
        real_A_condition_patches = []
        real_B_pathes = []
        real_B_condition_pathes = []

        for fb, ca, cb, rb, m in zip(fake_image_, self.real_A_condition, self.real_B_condition, self.real_B, mask):
            idx_pt = torch.nonzero(m)[:, 1:]
            ys, xs = idx_pt[:, 0], idx_pt[:, 1]
            # print(ys)
            idx = ((ys >= half_patch) & (xs >= half_patch) & (ys <= h - half_patch) & (xs <= w - half_patch))
            idx_pt = idx_pt[idx, :]    

            if len(idx_pt) > self.opt.patchD_3:
                poses = idx_pt[random.sample(range(0, len(idx_pt)), self.opt.patchD_3), :].cpu().numpy().tolist()        
            else: 
                x = random.sample(range(half_patch,  w - half_patch-1), self.opt.patchD_3)
                y = random.sample(range(half_patch,  h - half_patch-1), self.opt.patchD_3)
                poses = zip(y, x)
                
            for pos in poses:
                y, x = pos[-2], pos[-1]
                
                x1, x2, y1, y2 = x-half_patch, x+half_patch, y-half_patch, y+half_patch

                # if x1<0 or y1<0 or y2>=h or x2>=w:
                #     continue
                
                fake_patches.append(fb[:, y1:y2, x1:x2])        
                real_A_condition_patches.append(ca[:, y1:y2, x1:x2])
                real_B_pathes.append(rb[:, y1:y2, x1:x2])
                real_B_condition_pathes.append(cb[:, y1:y2, x1:x2])

        self.fake_patches = torch.stack(fake_patches, 0)
        self.real_A_condition_patches = torch.stack(real_A_condition_patches, 0)
        self.real_B_pathes = torch.stack(real_B_pathes, 0)
        self.real_B_condition_pathes = torch.stack(real_B_condition_pathes, 0)

        # print(self.fake_patches.shape)

    def backward_fusion_adv(self, fake_image, pre_mask):  
        """
        @description  : 基于GAN的融合训练模式 过程 就是通过GAN的方法 来从头优化公用的解码器
        ---------
        @param  : 
        -------
        @Returns  :
        -------
        """
        # mask_crop = util.center_rotate_crop(self.motion_mask.detach(), 0,0, p=256)
        fake_B_detach = util.center_rotate_crop(fake_image.detach(), 0, 0, p=256)
        real_A_condition = util.center_rotate_crop(self.real_A_ref, p=256)
        real_B_condition = util.center_rotate_crop(self.real_B, p=256)
        

        real_B = util.center_rotate_crop(self.real_B, 0, p=256)  ### 只crop

        self.optimizer_D_A.zero_grad()  
        dis_fake_input = torch.cat((fake_B_detach, real_A_condition), dim=1)
        dis_real_input = torch.cat((real_B, real_B_condition), dim=1) 

        pred_real = self.netD_A.forward(dis_real_input)
        pred_fake = self.netD_A.forward(dis_fake_input)
        
        loss_D_A = pred_fake.mean() - pred_real.mean()
        #  + self.criterionGAN.calc_gradient_penalty(self.netD_A,  dis_real_input.data, dis_fake_input.data)
        loss_D_A.backward()
        self.optimizer_D_A.step()

        self.optimizer_G.zero_grad()  
        gen_fake_input = torch.cat((fake_image, self.real_A_ref), dim=1)
        fake_scalar = self.netD_A.forward(gen_fake_input)
        loss_G_A = (- torch.mean(fake_scalar)) * self.opt.K_GA
        loss_vgg = self.vgg_loss.compute_vgg_loss(fake_image, self.real_A) * self.opt.vgg
        l1_loss = self.criterionMSE(fake_image, self.real_A_ref, mask=1-self.motion_mask, out_vector=False) 
        (loss_G_A + loss_vgg).backward()
        self.optimizer_G.step()

        self.loss_D_A = loss_D_A.item()
        self.loss_G_A = loss_G_A.item()
        self.loss_vgg = loss_vgg.item()
        self.loss_L1 = l1_loss.item()
        self.loss_tv = 0
        self.loss_mask = 0

    def basic_dis_back(self, fake_image, fake_condition, real_image, real_condition, D_model):
        dis_fake_input = torch.cat((fake_image, fake_condition), dim=1)
        dis_real_input = torch.cat((real_image, real_condition), dim=1) 

        pred_real = D_model.forward(dis_real_input)
        pred_fake = D_model.forward(dis_fake_input)

        loss = pred_fake.mean() - pred_real.mean()
        loss.backward()
        return loss.item()

    def basic_adv_back(self, fake_img, fake_condition, D_model):
        gen_fake_input = torch.cat((fake_img, fake_condition), dim=1)
        fake_scalar = self.netD_A.forward(gen_fake_input)
        return (- torch.mean(fake_scalar)) 

    # def backward_fusion_adv(self, fake_image, pre_mask):  
    #     self.prepare_gan_data(fake_image, self.motion_mask)

    #     self.optimizer_D_A.zero_grad()    # fake_B_256  real_A_condition
    #     self.loss_D_A = self.basic_dis_back(self.fake_B_256.detach(), self.real_A_condition, \
    #                                         self.real_B, self.real_B_condition, self.netD_A)
    #     self.optimizer_D_A.step()

    #     if self.opt.patchD_3 > 0:
    #         self.optimizer_D_P.zero_grad()  
    #         self.loss_D_P = self.basic_dis_back(self.fake_patches.detach(), self.real_A_condition_patches, 
    #                                             self.real_B_pathes, self.real_B_condition_pathes, self.netD_P)
    #         self.optimizer_D_P.step()


    #     self.optimizer_G.zero_grad()  
    #     loss_G_A = self.basic_adv_back(fake_image, self.real_A_ref, self.netD_A) * self.opt.K_GA
    #     loss_G_P = 0
    #     if self.opt.patchD_3 > 0:
    #         loss_G_P = self.basic_adv_back(self.fake_patches, self.real_A_condition_patches, self.netD_P) * self.opt.K_GP
    #         self.loss_G_P = loss_G_P.item()

    #     loss_vgg = self.vgg_loss.compute_vgg_loss(fake_image, self.real_A) * self.opt.vgg
    #     l1_loss = self.criterionMSE(fake_image, self.real_A_ref, mask=1-self.motion_mask, out_vector=False) 

    #     # loss_tv = self.criterionTV(pre_mask, self.motion_mask) * self.opt.K_TV
    #     # loss_mask = torch.mean((pre_mask * (1 - self.motion_mask))**2) * self.opt.K_MASK

    #     (loss_G_A + loss_vgg + loss_G_P).backward()
    #     self.optimizer_G.step()

    #     self.loss_G_A = loss_G_A.item()
    #     self.loss_vgg = loss_vgg.item()
    #     self.loss_L1 = l1_loss.item()
    #     self.loss_tv = 0
    #     self.loss_mask = 0

    def backward_fusion(self, fake_image, pre_mask):
        """
        @description  : 基于 mask损失的 融合模式。就是通过mask损失的方法 来从头优化公用的解码器
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        self.optimizer_G.zero_grad() 

        ms1, ms2, cos1, cos2, spa1, spa2 = self.cal_base_loss_items(fake_image, use_mask=True)
        ms2 = ms2*self.opt.K_MSE
        cos2 = cos2*self.opt.K_COSINE
        spa1 = spa1*self.opt.K_GSPA
        loss_fus = ms1 + ms2 + spa2 + spa1 + cos2 + cos1   ### 提高背景的权重 训练 mask 
        
        loss_tv = self.criterionTV(pre_mask, self.motion_mask) * self.opt.K_TV
        loss_mask = torch.mean((pre_mask * (1 - self.motion_mask))**2) * self.opt.K_MASK
        loss_fus += (loss_mask+loss_tv)                          ### mask 权重   

        loss_fus.backward()
        self.optimizer_G.step()  

        self.loss_cos2 = cos2.item()
        self.loss_spa1 = spa1.item()
        self.loss_mse2 = ms2.item()
        self.loss_tv = loss_tv.item()
        self.loss_mask = loss_mask.item()
        self.loss_fus = loss_fus.item()


    def backward_endec(self, A_res, B_res):
        self.optimizer_G.zero_grad() 

        # ms1_low, ms2_low, _, cos2_low, _, spa2_low = self.cal_base_loss_items(A_res, use_mask=True)
        # loss_A_res = ms1_low + ms2_low + cos2_low + spa2_low   ## 低光照路径主要目的是  在不产生阴影的情况下提亮
        
        ms1_low, _, cos1_low, _ , _, spa2_low = self.cal_base_loss_items(A_res, use_mask=False)
        loss_A_res = ms1_low + spa2_low   

        ms1_ref, _, cos1_ref, _, spa1_ref, _ = self.cal_base_loss_items(B_res, ref_img=self.real_B, use_mask=False)
        loss_B_res = ms1_ref + cos1_ref + spa1_ref

        (loss_A_res + loss_B_res).backward()

        self.optimizer_G.step()  

        self.loss_B_res = loss_B_res.item()
        self.loss_A_res = loss_A_res.item()

        self.loss_mse1 = ms1_ref.item()
        self.loss_spa1 = spa1_ref.item()

        self.loss_mse2 = ms1_low.item()
        self.loss_spa2 = spa2_low.item()
    
    def backward_mask(self, fake_image, pre_mask):
        self.optimizer_G.zero_grad() 

        ms1, ms2, cos1, cos2, spa1, spa2 = self.cal_base_loss_items(fake_image, use_mask=True)
        ms2 = ms2*self.opt.K_MSE
        cos2 = cos2*self.opt.K_COSINE
        spa1 = spa1*self.opt.K_GSPA
        loss_fus = ms1 + ms2 + spa2 + spa1 + cos2 + cos1   ### 提高背景的权重 训练 mask 
        
        loss_tv = self.criterionTV(pre_mask, self.motion_mask) * self.opt.K_TV
        loss_mask = torch.mean((pre_mask * (1 - self.motion_mask))**2) * self.opt.K_MASK
        loss_fus += (loss_mask+loss_tv)                          ### mask 权重   

        loss_fus.backward()
        self.optimizer_G.step()  

        self.loss_cos2 = cos2.item()
        self.loss_spa1 = spa1.item()
        self.loss_mse2 = ms2.item()
        self.loss_tv = loss_tv.item()
        self.loss_mask = loss_mask.item()
        self.loss_fus = loss_fus.item()

    def backward_ohem(self, fake_image, pre_mask, image_pool=None):
        self.optimizer_G.zero_grad() 

        out_vector = (image_pool is not None)

        ms1, ms2, cos1, cos2, spa1, spa2 = self.cal_base_loss_items(fake_image, use_mask=True, out_vector=out_vector)
        
        loss_mask  = torch.mean((pre_mask * (1 - self.motion_mask))**2) * 0.2  
        loss_ohem = ms2 + spa2 + cos2  

        if image_pool is not None:
            image_pool.add_example(self.batch_data, loss_ohem)
            return

        (loss_ohem + loss_mask).backward()
        self.optimizer_G.step()  
        self.loss_ohem = loss_ohem.item()
        self.loss_mask = loss_mask.item()
        self.loss_spa = spa2.item()


    def optimize_parameters(self, epoch, is_backward=None, image_pool=None):

        self.loss_mask, self.loss_fus = 0, 0
        # self.opti_mode = 0 if epoch <= -1 else 1 if self.opt.K_GA == 0 else 3

        if is_backward is not None:
            self.opt.opti_mode = 2
            self.netG_A.set_grad_model((3,), True)
            self.netG_A.set_grad_model((1,2,4,5,0), False)
            self.fake_B, self.pred_mask = self.netG_A.forward(self.real_A, self.real_B)
            self.backward_ohem(self.fake_B, self.pred_mask, image_pool=image_pool)
            return

        if self.opt.opti_mode == 0:
            self.A_restore, self.B_restore = \
                    self.netG_A.forward_endec_path(self.real_A, self.real_A_ref, self.real_B)
            self.backward_endec(self.A_restore, self.B_restore)
     
        elif self.opt.opti_mode == 1:
            self.fake_B, self.pred_mask = self.netG_A.forward_mask_path(self.real_A, self.real_A_ref)
            self.backward_mask(self.fake_B, self.pred_mask)
           
        elif self.opt.opti_mode == 3:
            self.fake_B, self.pred_mask = self.netG_A.forward_fus_path(self.real_A, self.real_A_ref) #  mask=self.motion_mask
            if self.opt.K_GA > 0:
                self.backward_fusion_adv(self.fake_B, self.pred_mask)
            else:
                self.backward_fusion(self.fake_B, self.pred_mask)

    def get_current_errors(self, epoch):
        ans = OrderedDict()

        if self.opt.opti_mode == 3:
            ans['loss_mask'] = self.loss_mask
            ans['loss_tv'] = self.loss_tv 

            if self.opt.K_GA > 0:
                ans['loss_G_A'] = self.loss_G_A
                ans['loss_D_A'] = self.loss_D_A
                ans['loss_vgg'] = self.loss_vgg 
                if self.opt.patchD_3 > 0:
                    ans['loss_D_P'] = self.loss_D_P
                    ans['loss_G_P'] = self.loss_G_P
                return ans
         
        ans['loss_mse1'] = self.loss_mse1  
        ans['loss_mse2'] = self.loss_mse2
        ans['loss_spa1'] = self.loss_spa1
        ans['loss_spa2'] = self.loss_spa2

        if self.opt.opti_mode == 2:
            ans['loss_ohem'] = self.loss_ohem
            ans['loss_mask'] = self.loss_mask
            ans['loss_spa'] = self.loss_spa
            return ans

        if self.opt.opti_mode == 0: 
            ans['loss_A_res'] = self.loss_A_res
            ans['loss_B_res'] = self.loss_B_res
            
        if self.opt.opti_mode == 1:
            ans['loss_cos2'] = self.loss_cos2
            ans['loss_cos1'] = self.loss_cos1
            ans['loss_fus'] = self.loss_fus
            ans['loss_mask'] = self.loss_mask
            ans['loss_tv'] = self.loss_tv

        return ans
        

    def get_current_visuals(self):
        
        real_A_enhance = util.tensor2im(self.real_A_enhance.data)
        real_A = util.tensor2im(self.real_A.data) # A_restore

        fake_B = util.tensor2im(self.A_restore.data) if self.opt.opti_mode == 0 else \
                util.tensor2im(self.fake_B.data)
        
        real_A_ref = util.tensor2im(self.real_A_ref.data)  
        motion_mask = util.tensor2im(self.motion_mask.data*2-1)

        image_od = OrderedDict([('fake_B', fake_B), ('real_A_ref', real_A_ref), ('mask', motion_mask),
                ('A_enhance', real_A_enhance),('real_A', real_A) ])
        
        if self.opt.opti_mode > 0:
            image_od['atten'] = util.tensor2im(self.pred_mask.detach()*2-1)
                            
        if self.opt.opti_mode == 0:
            image_od['real_B'] = util.tensor2im(self.real_B.data)
            image_od['B_restore'] = util.tensor2im(self.B_restore.data)

        if self.opt.patchD_3 > 0 and self.opt.K_GA > 0:
            image_od['fake_patch'] = util.tensor2im(self.fake_patches.data)
            image_od['fake_condition'] = util.tensor2im(self.real_A_condition_patches.data)
        return image_od      
        


   # def backward_fusion_adv(self, fake_image):  

    #     # mask_crop = util.center_rotate_crop(self.motion_mask.detach(), 0,0, p=256)
    #     fake_B_detach = util.center_rotate_crop(fake_image.detach(), 0, 0, p=256)
    #     real_A_condition = util.center_rotate_crop(self.real_A_ref, p=256)
    #     real_B_condition = util.center_rotate_crop(self.real_B, p=256)
        

    #     real_B = util.center_rotate_crop(self.real_B, 0, p=256)  ### 只crop

    #     self.optimizer_D_A.zero_grad()  
    #     dis_fake_input = torch.cat((fake_B_detach, real_A_condition), dim=1)
    #     dis_real_input = torch.cat((real_B, real_B_condition), dim=1) 

    #     pred_real = self.netD_A.forward(dis_real_input)
    #     pred_fake = self.netD_A.forward(dis_fake_input)
        
    #     loss_D_A = pred_fake.mean() - pred_real.mean()
    #     #  + self.criterionGAN.calc_gradient_penalty(self.netD_A,  dis_real_input.data, dis_fake_input.data)
    #     loss_D_A.backward()
    #     self.optimizer_D_A.step()

    #     self.optimizer_G.zero_grad()  
    #     gen_fake_input = torch.cat((fake_image, self.real_A_ref), dim=1)
    #     fake_scalar = self.netD_A.forward(gen_fake_input)
    #     loss_G_A = (- torch.mean(fake_scalar)) * self.opt.K_GA
    #     loss_vgg = self.vgg_loss.compute_vgg_loss(fake_image, self.real_A) * self.opt.vgg
    #     l1_loss = self.criterionMSE(fake_image, self.real_A_ref, mask=1-self.motion_mask, out_vector=False) 
    #     (loss_G_A + loss_vgg).backward()
    #     self.optimizer_G.step()

    #     self.loss_D_A = loss_D_A.item()
    #     self.loss_G_A = loss_G_A.item()
    #     self.loss_vgg = loss_vgg.item()
    #     self.loss_L1 = l1_loss.item()