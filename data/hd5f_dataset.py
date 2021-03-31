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
from data.base_dataset import BaseDataset, data_augmentation, get_img_rot_broa
import random
import numpy as np
from pdb import set_trace as st
import h5py
import cv2
import json

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)



import random
class HD5FDataset(BaseDataset):
    ''' 加载预先处理的 .h5 格式的训练数据集
    '''
    def initialize(self, opt):
        ''' opt中数据集加载相关的重要参数：
                opt.dataroot： 包含.h5的数据集文件路径
                opt.mask_type：
                    poly: 使用数据集中预先存在的 mask
                    rect： 使用同路径下，标注的前景运动框 .json 来生成矩形的mask
                opt.random_mask：最终输出的 mask 是使用random随机生成的mask 还是输出 由 opt.mask_type 参数决定的mask
                opt.resize_or_crop： 测试的时候为 'no' 训练的时候不为 'no' 就行
                opt.aligen_AB： AB 图像是否是同一场景，如果为False ，那么真实样本 B 就是从所有正常图象中随便选，否则就是A的参考图
                opt.box_filter： True，则将输出的 'A_input_ref' 模糊，否则就等于A_ref，A_input_ref 一般用于和 低光图 cat输入网络 
                opt.fineSize： 最终的 patch_size 大小（ 320 ）
                opt.motion_sample_percent： crop 运动区域的概率 0-1，如果取0，则训练样本完全是从对齐的区域中crop
        ''' 
        self.opt = opt
        self.f5_file = opt.dataroot

        f = h5py.File(self.f5_file ,'r', libver='latest', swmr=True)
        self.A_imgs = f['low_data'][:].astype(np.uint8)
        self.A_enhance_imgs = f['pre_enhance_data'][:].astype(np.uint8)
        self.B_imgs = f['normal_data'][:].astype(np.uint8)


        try: 
            self.mask_type = opt.mask_type
        except:
            self.mask_type = 'poly'

        (base_path, _) = os.path.split(self.f5_file)
        if self.mask_type == 'poly':
            try:
                self.motion_masks = f['motion_mask'][:].astype(np.uint8)
            except:
                f = h5py.File(os.path.join(base_path, 'motion_masks.h5') ,'r', libver='latest', swmr=True)
                self.motion_masks = f['motion_mask'][:].astype(np.uint8)

        elif self.mask_type == 'rect':
            with open(os.path.join(base_path, 'box_annos.json'), 'r') as f:
                self.boxes = json.load(f)

        self.A_size = len(self.A_imgs)
        self.B_size = len(self.B_imgs)
        
        self.rand_idx = list(range(len(self)))
        if not self.opt.aligen_AB:   
            random.shuffle(self.rand_idx)   ### 打乱 B image

    def crop_except_mask(self, mask):
        ''' 从图中随机crop 出 完全不包含 mask标注的 patch, 就是避开运动区域 采集样本 
        '''
        p = self.opt.fineSize
        hp = p // 2
        h, w = mask.shape 
        
        contours, _ =cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y, x = np.where(mask == 0)

        idx = ((y < h-hp) & (x < w-hp) & (y >= hp) & (x >= hp))
        idx_back = idx
        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            rx, ry = rx + rw // 2, ry + rh // 2
            idx = idx & ((np.abs(y - ry) > hp + rh//2) | (np.abs(x - rx) > hp + rw//2))

        y, x = y[idx], x[idx]

        if(len(x) == 0):    
            return random.randint(0, w - p), random.randint(0, h - p)

        poses_list = list(zip(y, x))

        y, x = poses_list[random.randint(0,len(poses_list)-1)]
        y, x = y - hp, x - hp
        assert(y >= 0 and x >= 0 and y < h-p and x < w-p)
        return x, y

    def crop_from_mask(self, mask):
        ''' 从图中随机 crop 出 包含 mask标注的 patch, 就是crop 包含运动区域的样本，crop到运动区域的概率
        可调 
        '''
        if self.opt.motion_sample_percent < 0.1:
            return self.crop_except_mask(mask)

        sel = 255 if random.random() < self.opt.motion_sample_percent else 0

        p = self.opt.fineSize
        hp = p // 2
        h, w = mask.shape
        y, x = np.where(mask == sel)
        
        idx = ((y < h-hp) & (x < w-hp) & (y >= hp) & (x >= hp))
        y, x = y[idx], x[idx]

        if(len(x) == 0):
            return random.randint(0, w - p), random.randint(0, h - p)

        poses_list = list(zip(y, x))
        y, x = poses_list[random.randint(0,len(poses_list)-1)]

        y, x = y - hp, x - hp
        assert(y >= 0 and x >= 0 and y < h-p and x < w-p)
        return x, y

    def get_motion_mask(self, idx, img_h, img_w):
        ''' 获取事先标注的 mask 图象，主要有两种标注形式
            poly: 通过 motionDetection 检测出的掩码图象，直接和原图一起打包在h5文件中
            rect: 同文件夹下的 json 文件标注的数据框
        Args：
            idx: 样本序号
            img_h， img_w: 原图的高，宽
        '''
        if self.mask_type == 'poly':
            A_mask = self.motion_masks[idx % self.A_size,:,:]
            B_mask = self.motion_masks[self.rand_idx[idx] % self.B_size,:,:]
            return A_mask, B_mask         
        elif self.mask_type == 'rect':

            A_boxes = self.boxes[str(idx % self.A_size)]
            B_boxes = self.boxes[str(self.rand_idx[idx] % self.B_size)]

            def set_rect_area_mask(boxes, w, h):
                mask = np.zeros((h, w), dtype=np.uint8)
                for box in boxes:
                    x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], w), min(box[3], h)
                    mask[y1:y2, x1:x2] = 255
                return mask

            A_mask_ = set_rect_area_mask(A_boxes, img_w, img_h)
            B_mask_ = set_rect_area_mask(B_boxes, img_w, img_h)
            return A_mask_, B_mask_

    def __getitem__(self, index):
        ''' 读取一帧数据，返回字典格式 键的具体含义：
            'A': 原始低光照图
            'input_A_ref': 低光照图A 同场景 不对齐的参考图（根据参数设置是否模糊）
            'A_ref': 低光照图A 同场景 不对齐的参考图
            'B': 正常亮度的图
            'motion_mask': 预先计算的运动掩码
            'A_enhance': 预先匹配增强的图象‘’
            'input_img': 等同于 A 
            'A_gray': A 的灰度图象
        '''
        A_img = self.A_imgs[index % self.A_size,:,:,:].astype(np.float32)/255.0
        
        A_enhance_img = self.A_enhance_imgs[index % self.A_size,:,:,:].astype(np.float32)/255.0
        A_ref_img = self.B_imgs[index % self.B_size,:,:,:].astype(np.float32)/255.0
        B_img = self.B_imgs[self.rand_idx[index] % self.B_size,:,:,:].astype(np.float32)/255.0
        ### 是否将输入参考图模糊
        if self.opt.box_filter:
            A_input_ref = cv2.boxFilter(A_ref_img, ddepth=-1, ksize=(31, 31), normalize=True)
            # V = cv2.cvtColor(A_ref_img, cv2.COLOR_RGB2HSV)[:,:,2]
            # V = cv2.boxFilter(V, ddepth=-1, ksize=(127, 127), normalize=True)
            # A_input_ref = np.stack((V,V,V), axis=2)
        else:
            A_input_ref = A_ref_img.copy()

        ### 读取 运动区域的 标注
        A_motion_mask, B_motion_mask = self.get_motion_mask(index, A_img.shape[0], A_img.shape[1])
       
        ### 数据变换，包括随机crop 和 镜像旋转等
        if self.opt.resize_or_crop == 'no':
            A_motion_mask = A_motion_mask.astype(np.float32) / 255.0
     
        else:
            ### A 和 A_ref 要保证 是做的一致的变换
            rand_mode = random.randint(0, 7)
            p = self.opt.fineSize
       
            x, y = self.crop_from_mask(A_motion_mask)

            A_img = data_augmentation(A_img[y:y+p, x:x+p, :], rand_mode)
            A_ref_img = data_augmentation(A_ref_img[y:y+p, x:x+p, :], rand_mode)
            A_enhance_img = data_augmentation(A_enhance_img[y:y+p, x:x+p, :], rand_mode)
            A_input_ref = data_augmentation(A_input_ref[y:y+p, x:x+p, :], rand_mode)

            if self.opt.random_mask:
                A_motion_mask = self.random_ff_mask((p, p)).squeeze(0)            
            else:
                A_motion_mask = A_motion_mask.astype(np.float32) / 255.0
                A_motion_mask = data_augmentation(A_motion_mask[y:y+p, x:x+p], rand_mode)

        if not self.opt.aligen_AB and self.opt.resize_or_crop != 'no': 
            rand_mode = random.randint(0, 7)
            x, y = self.crop_from_mask(B_motion_mask)
        
        B_img = data_augmentation(B_img[y:y+p, x:x+p, :], rand_mode)
        A_motion_mask = torch.from_numpy(A_motion_mask.copy()).unsqueeze(0)

        ### 转torch格式，数值范围转换到 -1 - 1 
        A_img = 2*torch.from_numpy(A_img.copy()).permute(2,0,1)-1
        A_enhance_img = 2*torch.from_numpy(A_enhance_img.copy()).permute(2,0,1)-1
        A_input_ref = 2*torch.from_numpy(A_input_ref.copy()).permute(2,0,1)-1
        A_ref_img = 2*torch.from_numpy(A_ref_img.copy()).permute(2,0,1)-1
        B_img = 2*torch.from_numpy(B_img.copy()).permute(2,0,1)-1

        input_img = A_img
        ### 计算 A 的灰度值
        r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
        A_gray = torch.unsqueeze(A_gray, 0)

        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img, 'motion_mask':A_motion_mask,
                'A_ref':A_ref_img, 'A_enhance':A_enhance_img, 'A_input_ref':A_input_ref}


    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'HD5FDataset'


