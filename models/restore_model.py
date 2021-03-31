#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 基于传统图象损失的图像提亮结构，集成了训练和测试逻辑，由train.py train_ohem.py 
                 test.py文件调用。主要实现了基于mask的 MSE 损失，梯度一致性损失，亮度损失，余弦相
                 似度损失，以及难样本挖掘训练逻辑。
@Date          : 2021/03/29 14:21:12
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
import util.util as util
from collections import OrderedDict
# from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks, hdrnet
from . import loss
import sys
import torchvision


class RestoreModel(BaseModel):
    def name(self):
        return 'RestoreModel'

    def initialize(self, opt):
        self.opt = opt

        BaseModel.initialize(self, self.opt)

        skip = True if opt.skip > 0.1 else False
       
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, skip=skip, opt=opt)
       
        self.optimizer_G = self.add_optimeizer(self.netG_A, opt, 'G_A')

        if self.isTrain:

            self.load_networks(opt.which_epoch)

            if opt.continue_train:
                self.resum_networks(opt.which_epoch)


            self.criterionSSIM = loss.SSIM()
            self.criterionMSE = loss.L_MSE(clamp=True)
            self.criterionExposur = loss.L_exp(31)
            self.criterionSpatial = loss.SpatialConsistencyLoss(grad_normal=False, pre_avg=False)
            self.criterionCosine = loss.L_cosine(clamp=True)

            if opt.output_mode == 'line_trans' or opt.output_mode == 'cur_trans':
                self.criterionTV = loss.L_TV() 
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
        """该模型测试代码 
        """
        self.input, self.input_gray, self.input_ref = self.real_A, self.real_A_enhance, self.real_input_A_ref
        self.get_R_out(self.netG_A.forward(self.input, self.input_ref, self.input_gray))

        real_A = util.tensor2im(self.real_A.data)
        real_A_ref = util.tensor2im(self.real_A_ref.data)
        fake_B = util.tensor2im(self.fake_B.data)
        re_od =OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('ref_A', real_A_ref)])

        if self.opt.which_model_netG == 'atten_net':
            re_od['pre_mask'] = util.tensor2im(self.latent_real_A.detach()*2-1)

        return re_od

    # get image paths
    def get_image_paths(self):
        return self.image_paths
  
    def get_R_out(self, Ls):
        """将模型输出的结果做转换，例如如果网络输出的为变换矩阵，则在此处通过变换矩阵计算最终增强结果, 结构内部调用
        args:
            Ls: 网络输出的结果以List格式输入 
        """
        if self.opt.output_mode == 'line_trans':
            self.coeffs = Ls
            self.fake_B = self.line_transform(self.input, Ls)
        elif self.opt.output_mode == 'cur_trans':
            self.coeffs = Ls
            self.fake_B = self.cur_transform(self.input, Ls)
        elif self.opt.output_mode == 'direct_img':
            self.fake_B = Ls[0]
            self.latent_real_A = Ls[1]

        if not self.isTrain:
            return
            
    def backward_G(self, epoch, is_backward=True, image_pool=None):
        """计算损失 并反向传播
        args:
            epoch: 没用到
            is_backward： 是否反向传播，在难样本挖掘中使用
            image_pool： 用于存储难样本的数据池
        """
        self.loss_vgg = torch.tensor(0)
        ### 
        self.loss_exp = 0
        self.loss_spa = 0
 
        out_vector = (image_pool is not None)
        bg_mask = 1 - self.motion_mask
        fg_mask = None

        ### 前景背景 L2 损失 
        self.loss_mse1 = self.criterionMSE(self.fake_B, self.real_A_ref, mask=bg_mask, out_vector=out_vector) * self.opt.K_MSE 
        self.loss_mse2 = self.criterionMSE(self.fake_B, self.real_A_enhance, mask=fg_mask, out_vector=out_vector) * self.opt.K_MSE * 0
        self.loss_mse = self.loss_mse1 + self.loss_mse2

        ### 前景背景 余弦相似度损失
        self.loss_cos1 = self.criterionCosine(self.fake_B, self.real_A_ref, mask=bg_mask, out_vector=out_vector) * self.opt.K_COSINE 
        self.loss_cos2 = self.criterionCosine(self.fake_B, self.real_A_enhance, mask=fg_mask, out_vector=out_vector) * self.opt.K_COSINE * 0
        self.loss_cos = self.loss_cos2 + self.loss_cos1

        ### 前景背景 梯度相似损失
        self.loss_spa += self.criterionSpatial(self.fake_B, self.real_A_ref, mask=bg_mask, out_vector=out_vector) * self.opt.K_GSPA 
        self.loss_spa += self.criterionSpatial(self.fake_B, self.real_A_enhance, mask=fg_mask, out_vector=out_vector) * self.opt.K_GSPA

        self.loss_G = self.loss_spa + self.loss_mse + self.loss_cos

        self.loss_G += self.criterionSSIM(self.fake_B, self.real_A_ref)

        ### 如果输出的是 线性变换、曲线变换的参数，需要加TV损失
        if self.opt.output_mode == 'line_trans' or self.opt.output_mode == 'cur_trans':
            self.loss_tv = self.criterionTV(self.coeffs) * self.opt.K_TV
            self.loss_G += self.loss_tv

        ### 存储难样本
        if image_pool is not None:
            image_pool.add_example(self.batch_data, self.loss_cos)

        ### 如果是只是 ohem 中的前向步骤 就退出 不进行反传
        if not is_backward:
            return

        self.loss_G.backward()

        ### 曲线变换的参数，ZeroDEC原文代码中也对 梯度进行了裁剪 否则会梯度爆炸
        if self.opt.output_mode == 'cur_trans':
            torch.nn.utils.clip_grad_norm(self.netG_A.parameters(), 0.1)

    def optimize_parameters(self, epoch, is_backward=True, image_pool=None):
        """优化模型，由外部文件 train.py/train_ohem调用
        args:
            epoch: 没用到
            is_backward： 是否反向传播，在难样本挖掘中使用
            image_pool： 用于存储难样本的数据池
        """
        # G_A and G_B
        self.optimizer_G.zero_grad()  ### 优化生成器

        self.input, self.input_gray, self.input_ref = self.real_A, self.real_A_enhance, self.real_input_A_ref

        if is_backward:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        self.get_R_out(self.netG_A.forward(self.input, self.input_ref, self.input_gray))

        self.backward_G(epoch, is_backward, image_pool)        ### 求生成器损失   fake false 
        self.optimizer_G.step()  
        # if random.randint(0, 10) >= 6:

    def get_current_errors(self, epoch):
        """获取训练过程的损失信息, train.py/train_ohem调用
        args:
            epoch: 没用到
        return:
            将当前各项损失存储为 字典结构返回。例如 {}
        """
        loss_all = self.loss_G.item() 
        # vgg = self.loss_vgg.item()
        # l_exp = self.loss_exp.item()
        l_spa = self.loss_spa.item()
        l_mse = self.loss_mse.item()
        l_mse1 = self.loss_mse1.item()
        l_cos = self.loss_cos.item()
        # l_cos1 = self.loss_cos1.item()

        ans = OrderedDict([('loss_all', loss_all), ('spa', l_spa), ("mse", l_mse),\
                    ("mse1", l_mse1), ("l_cos", l_cos)])

        if self.opt.which_model_netG == 'atten_net':
            ans['mask'] = self.loss_mask.item()

        if self.opt.output_mode == 'line_trans' or self.opt.output_mode == 'cur_trans':
            ans['tv'] = self.loss_tv.item()

        return ans
        

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
        input_A_ref = util.tensor2im(self.input_ref.data)
        motion_mask = util.tensor2im(self.motion_mask.data*2-1)

        image_od = OrderedDict([('fake_B', fake_B), ('real_B', real_B), ('mask', motion_mask),
                ('real_A_ref', real_A_ref), ('A_enhance', real_A_enhance),('input_A_ref', input_A_ref)])
        
        if self.opt.which_model_netG == 'atten_net':
            image_od['atten'] = util.tensor2im(self.latent_real_A.detach()*2-1)

        return image_od      
        
