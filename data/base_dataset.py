import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random
import numpy as np
import math
import cv2
import os
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class BaseDataset(data.Dataset):
    ''' 数据加载的 基础 类 
    '''
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.  用于生成随机mask
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape[0]
        width = shape[1]
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)

    def initialize(self, opt):
        pass


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def store_dataset(dir):
    """
    @description  : 从文件夹中读取所有 图片 和 图片名
    ---------
    @param  : dir文件路径
    -------
    @Returns  : 图片列表PIL格式 和 图片名列表 一一对应
    -------
    """
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        fnames = sorted(fnames)
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')
                images.append(img)
                all_path.append(path)

    return images, all_path

def motionDetection(ref_, out_):
    ''' 用于检测 运动前景和背景的差异像素
    Args：
        ref_: 参考图像
        out_: 余弦匹配增强的结果
    Returns：
        分割出的掩码图像  运动区域为 255 背景为 0
    '''
    out_ = np.array(out_)
    ref_ = np.array(ref_)
    ref = cv2.cvtColor(ref_, cv2.COLOR_RGB2BGR)     #转为灰度图
    out = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)     #转为灰度图

    ref_gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)     #转为灰度图
    out_gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)     #转为灰度图
    flow = cv2.calcOpticalFlowFarneback(out_gray,ref_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # 计算光流以获取点的新位置
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(ref)     # 为绘制创建掩码图片
    hsv[...,0] = ang*180/np.pi/2   #色调范围：0°~360°
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    motion_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    motion_mask = cv2.cvtColor(motion_mask, cv2.COLOR_BGR2GRAY)
    ret1, motion_mask = cv2.threshold(motion_mask, 10, 255, cv2.THRESH_BINARY)  # cv2.THRESH_OTSU
    
    
    diff_mask = (np.mean(np.abs(out.astype(np.float32)/255.0 - ref.astype(np.float32)/255.0), axis=2) * 255).astype(np.uint8)
    ret1, diff_mask = cv2.threshold(diff_mask, 10, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)  
    diff_mask = cv2.erode(diff_mask, kernel, iterations = 1)
    motion_mask = cv2.erode(motion_mask, kernel, iterations = 1)
    diff_mask = cv2.dilate(diff_mask, kernel, iterations = 1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations = 1)
    
    mask = cv2.multiply(diff_mask, motion_mask)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 2)

    # mask = cv2.GaussianBlur(mask,(5,5),0) 
    return mask


def hist_noraml(img, ref):
    ''' 直方图匹配
    Args：
        ref_: 参考图像
        img: 低光照图象
    Returns：
        增强后的结果
    '''
    img = np.array(img)
    ref = np.array(ref)
    
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        # print(i)
        hist_img, _ = np.histogram(img[:, :, i], 256)   # get the histogram
        hist_ref, _ = np.histogram(ref[:, :, i], 256)
        cdf_img = np.cumsum(hist_img)   # get the accumulative histogram
        cdf_ref = np.cumsum(hist_ref)
    
        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))   # find the smallest number in tmp, get the index of this number
            out[:, :, i][img[:, :, i] == j] = idx

    out = Image.fromarray(out)
    return out



def get_transform(opt):
    ''' 获取torchvision接口的数据扩增方法
    Args：
       opt：配置文件，配置文件具体参数有单独注释
    Returns：
        torchvision接口的数据扩增方法
    '''
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        zoom = 1 + 0.1*radom.randint(0,4)
        osize = [int(400*zoom), int(600*zoom)]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))  
    # elif opt.resize_or_crop == 'no':
    #     osize = [384, 512]
    #     transform_list.append(transforms.Scale(osize, Image.BICUBIC))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def data_augmentation(image, mode):
    ''' 根据参数 对numpy 格式的 图像进行变换
    Args：
        mode: 变换模式
            0： 不变换直接返回
            1-7：为不同的变换方式 主要是 旋转和镜像操作的组合
        img: numpy格式图象
    Returns：
        变换后的图象
    '''
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)
 
def get_img_rot_broa(img, degree):
    height, width = img.shape[:2]
 
    # 旋转后的尺寸
    height_new = int(width * math.fabs(math.sin(math.radians(degree))) +
                     height * math.fabs(math.cos(math.radians(degree))))
    width_new = int(height * math.fabs(math.sin(math.radians(degree))) +
                    width * math.fabs(math.cos(math.radians(degree))))
 
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
 
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
 
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new))
    return img_rotated



def __scale_width(img, target_width):
    ''' 将 图像 的  w 边长度缩放到指定的大小
    Args：
        img: PIL格式图像
        target_width： 期望的图象的宽   
    Returns：
        缩放后的图
    '''
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
