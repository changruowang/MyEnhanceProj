#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 从 .h5 格式的文件中读取训练数据
@Date          : 2021/03/29 14:22:15
@Author        : changruowang
@version       : 1.0
'''
import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, hist_noraml, motionDetection, store_dataset
import random
import numpy as np
from PIL import Image
import PIL
from pdb import set_trace as st
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)




class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        ''' opt中数据集加载相关的重要参数，详细见 HD5F 结构中的注释，和HD5F结构的区别是，这个类是直接读取图片，
        在测试的时候用，将测试图片放在 testA 文件夹中 参考图放在 testB 文件夹中，A/B中对应图象命名要一样
        ''' 
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        pre_enhance_img = hist_noraml(A_img, B_img)
        A_motion_mask = motionDetection(B_img, pre_enhance_img)
        
        if self.opt.box_filter:
            # A_input_ref = cv2.boxFilter(np.array(B_img), ddepth=-1, ksize=(31, 31), normalize=True)
            V = cv2.cvtColor(np.array(B_img), cv2.COLOR_RGB2HSV)[:,:,2]
            V = cv2.boxFilter(V, ddepth=-1, ksize=(127, 127), normalize=True)
            A_input_ref = np.stack((V,V,V), axis=2)
        else:
            A_input_ref = B_img.copy()

        # A_motion_mask = self.transform(A_motion_mask)
        A_input_ref = self.transform(A_input_ref)
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)
        pre_enhance_img = self.transform(pre_enhance_img)

        # if self.opt.random_mask:
        #     A_motion_mask = self.random_ff_mask(A_img.shape[1:3]).squeeze(0) 
        # else:
        A_motion_mask = A_motion_mask.astype(np.float32) / 255.0
        A_motion_mask = torch.from_numpy(A_motion_mask.copy()).unsqueeze(0)
        
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            assert True, '该类只在测试的时候使用，未实现数据扩增，请设置resize_or_crop参数为no'  
      
        return {'A': A_img, 'B': B_img,'A_ref': B_img, 'motion_mask':A_motion_mask, 'A_gray': A_gray,
                'A_paths': A_path, 'B_paths': B_path, 'A_enhance':pre_enhance_img,  'A_input_ref':A_input_ref}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


