#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from glob import glob
import json
import os
from tqdm import tqdm
from itertools import chain
import cv2
from make_hd5f import get_paired_image_names


###　x1 y1  x2  y2
def packJsonFiles(imgs_dir, anno_dir, level):
    """
    @description  : 将使用 labelme 标注软件生成的所有图片的json注释文件打包成整个json文件
    ---------
    @param  : 
        imgs_dir: 包含所有训练图片的文件夹，这个主要是保证 打包的每个注释文件对应的图片都存在,且和hd5f
                  制作的数据集中每个位置图片是对应的。所以这两个都使用 get_paired_image_names 函数读取
                  训练数据
        anno_dir: 输出json文件的保存路径
        level：需要制作的数据集类型  -2ev/-3ev
    -------
    @Returns  : 输出的json格式做了简化，键值和.h制作的数据中对应位置的图片对应
        {0：[[], [], []], 1: [[], [] ...]} ，坐标格式为 x1 y1  x2  y2 左上右下
    -------
    """
    level_low, level_high = level, '0ev'
    _, _, names = get_paired_image_names(imgs_dir, level_high, level_low)

    pbar = tqdm(enumerate(names), total=(len(names)))

    out_dict = {}

    for img_idx, name in pbar:

        low_anno = os.path.join(anno_dir, name.replace(',', level_low)+'.json')
        high_anno = os.path.join(anno_dir, name.replace(',', level_high)+'.json')

        boxes_list = []

        def read_one_json_boxes(path_):
            if not os.path.exists(path_):
                return []
            with open(path_, 'r') as f:
                shapes = json.load(f)['shapes']
                return [list(map(int, list(chain.from_iterable(sp['points'])))) for sp in shapes]
                      
        boxes_list += read_one_json_boxes(low_anno)
        boxes_list += read_one_json_boxes(high_anno)

        out_dict[img_idx] = boxes_list

    with open(os.path.join(imgs_dir, 'box_annos_%s.json'%level_low), 'w') as f:
        json.dump(out_dict, f)

if __name__ == "__main__":
    imgs_dir = 'D:\\changruowang\\data\\low_light_resize_qc\\'
    anno_dir = 'D:\\changruowang\\data\\low_light_resize_qc\\anno\\'
    level = '-2ev'
    packJsonFiles(imgs_dir, anno_dir, level)