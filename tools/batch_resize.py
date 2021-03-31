from glob import glob
import os
from tqdm import tqdm
import numpy as np
import cv2
from make_hd5f import get_paired_image_names

def batch_resize_images(base_path, save_path, level='-3ev'):
    """
    @description  : 批量将文件夹下的图片 resize 四倍
    ---------
    @param  : 
        base_path: 包含图片的文件夹路径
        save_path: 输出图片存储路径
        level: 要 resize 的数据对象 -2ev -3ev or 0ev
    -------
    @Returns  : 
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

        ### 缩放
        img = cv2.imread(low_path)
        ref = cv2.imread(normal_path)
        height, width, _ = img.shape
        img = cv2.resize(img, (int(width * 0.25), int(height * 0.25)))
        ref = cv2.resize(ref, (int(width * 0.25), int(height * 0.25)))
        cv2.imwrite(os.path.join(save_path, names[idx].replace(',', level)), img)

if __name__ == "__main__":
    level = '-3ev' 
    base_path = 'D:\\changruowang\\data\\low_light_resize_qc\\'
    save_path = 'D:\\changruowang\\data\\low_light_resize_qc2\\'

    batch_resize_images(base_path, save_path, level)