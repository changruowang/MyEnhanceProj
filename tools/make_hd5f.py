#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 制作 hd5f 格式的训练数据
@Date          : 2021/03/29 14:35:38
@Author        : changruowang
@version       : 1.0
'''
from glob import glob
import json
import os
from tqdm import tqdm
import h5py
import numpy as np
from python_cv_apply import hist_noraml, motionDetection
from itertools import chain
import cv2

def get_paired_image_names(base_path, level1='0ev', level2='-3ev'):
    """
    @description  : 获取成对的图像路径  给定数据是-2ev -3ev放在同一个文件夹下的，所以根据文件名提取对应
                    图片。xxxx_-2evxxx.png   xxxx_-3evxxxx.png
    ---------
    @param  : 
        base_path: 包含图象的路径
        level1: 需要提取的 对象 
        leves2: 同
    -------
    @Returns  : 返回 level1 level2 的图片路径，以及图片的原始名字，三个列表一一对应
    -------
    """
    img_lists = glob(base_path+'*.jpg')

    temp_names = set()   ## 
    for img_name in img_lists:
        _, name = os.path.split(img_name)
        name = name.replace('0ev', ',')
        name = name.replace('-2ev', ',')
        name = name.replace('-3ev', ',')
        temp_names.add(name)

    temp_names = list(temp_names)              
    temp_names.sort()

    level1_paths = []
    level2_paths = []
    names = []
    for name in temp_names:
        level2_path = os.path.join(base_path, name.replace(',', level2))
        level1_path = os.path.join(base_path, name.replace(',', level1))
      
        if os.path.exists(level1_path) and os.path.exists(level2_path):
            level1_paths.append(level1_path)
            level2_paths.append(level2_path)  ### 在写入 hd5f 前一定要确保每个数据都有效  不能有跳
            names.append(name[:-4])
        else:
            # if os.path.exists(low_path):
            #     os.remove(low_path)   
            print(level1_path, level2_path)

    assert(len(level1_paths) == len(level2_paths))
    return level1_paths, level2_paths, names



def make_low_light_dataset(base_path, level):
    """
    @description  : 用于将文件夹下的图片制作成 .h5 格式训练数据集
    ---------
    @param  : 
        base_path: 包含图象的路径
        level： -2ev 或者 -3ev，表示需要制作的数据集类型
    -------
    @Returns  : 制作的.h5数据集中包含以下 字典内容。他们对应位置是成对的。
        low_data： 低光照图象数据
        normal_data： 参考图数据
        pre_enhance_data： 预增强的结果
        motion_mask： 预先估计的运动区域，也可以是手工标注的mask图片
    -------
    """
    high_paths, low_paths, names = get_paired_image_names(base_path, '0ev', level)

    low_data = np.zeros((len(low_paths), 750, 1000, 3), dtype=np.uint8) 
    normal_data = np.zeros((len(low_paths), 750, 1000, 3), dtype=np.uint8)   
    pre_enhance_data = np.zeros((len(low_paths), 750, 1000, 3), dtype=np.uint8)   
    masks = np.zeros((len(low_paths), 750, 1000), dtype=np.uint8)   

    par = tqdm(range(len(low_paths)), total=(len(low_paths)))

    for idx in par:
        low_path = low_paths[idx]
        normal_path = high_paths[idx]

        img = cv2.imread(low_path)[0:750, 0:1000,:]
        ref = cv2.imread(normal_path)[0:750, 0:1000,:]

        out = hist_noraml(img, ref)
        mask = motionDetection(ref, out)

        low_data[idx,:,:,:] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normal_data[idx,:,:,:] = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
        pre_enhance_data[idx,:,:,:] = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        masks[idx,:,:] = mask

    f = h5py.File(os.path.join(base_path, 'out.h5'),'w')
    f['low_data'] = low_data 
    f['normal_data'] = normal_data 
    f['pre_enhance_data'] = pre_enhance_data 
    f['motion_mask'] = masks 
    f.close()  

        
if __name__ == "__main__":
    level = '-3ev'   ###  or  -2ev
    base_path = 'D:\\changruowang\\data\\low_light_resize_qc\\'
    make_low_light_dataset(base_path， level )
    # packJsonFiles()

