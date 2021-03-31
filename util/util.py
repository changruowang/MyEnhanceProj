# from __future__ import print_function
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import torch
import os
import collections
from torch.optim import lr_scheduler
import torch.nn.init as init
import random
import math
from torch.nn import functional as F


def max_per_image(input_):
    ''' 每个 batch 中的每张图分别求最大 输入 [b,c,h,w], 输出[b,]
    '''
    assert(len(input_.shape) == 4)
    return input_.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]

def min_per_image(input_):
    ''' 每个 batch 中的每张图分别求最小 输入 [b,c,h,w], 输出[b,]
    '''
    assert(len(input_.shape) == 4)
    return input_.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]

def torch_normal_img(input_):
    ''' 每个 batch 中的每张图分别归一化，可以理解为batch数据中 每张图分别归一化
    因此最大最小值要在每张图中分别求，而不能是全局最大最小
    '''
    max_ = torch.max(torch.max(input_, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] 
    min_ = torch.min(torch.min(input_, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] 

    return (input_ - min_) / (max_ - min_ + 0.001)

def sum_per_image(input_):
    ''' 每个 batch 中的每张图分别求和 输入 [b,c,h,w], 输出[b,]
    '''
    assert(len(input_.shape) == 4)
    return input_.sum(dim=-1).sum(dim=-1).sum(dim=-1)

def mean_per_image(input_):
    ''' 每个 batch 中的每张图分别求平均 输入 [b,c,h,w], 输出[b,]
    '''
    assert(len(input_.shape) == 4)
    return input_.mean(dim=-1).mean(dim=-1).mean(dim=-1)

def pad_tensor(input, divide=16):
    ''' 用于给输入tensor pad 使得尺可以 整除 divide，因为unet中有下采样，防止不能整除导致出错
    Args：
        input：待padding的tensor
        divide: 需要满足的整除因子
    Returns:
        padding完成之后的tensor，和左右上下分别padding的宽度，用于 pad_tensor_back 函数的复原
    '''
    height_org, width_org = input.shape[2], input.shape[3]
    # divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    ''' 和上面的函数对应，在网络最终输出的时候 将图象 crop 回原始尺寸
    Args：
       ad_left, pad_right, pad_top, pad_bottom 参数为pad_tensor 函数的返回
    '''
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


def random_boundingbox(size, lam):
    ''' 生成随机的  矩形框
    Args:
        lam: 生成边界框的最大面积
        size： 原始图象尺寸
    Returns:
        左上角 右下角的  x1 y1 x2 y2 坐标
    '''
    width , height = size, size

    r = np.sqrt(1. - lam)
    w = np.int(width * r)
    h = np.int(height * r)
    x = np.random.randint(width)
    y = np.random.randint(height)

    x1 = np.clip(x - w // 2, 0, width)
    y1 = np.clip(y - h // 2, 0, height)
    x2 = np.clip(x + w // 2, 0, width)
    y2 = np.clip(y + h // 2, 0, height)

    return x1, y1, x2, y2

def CutMix(imsize):
    """
    @description  : 根据图象尺寸生成随机的 矩形 mask 用于 gan 的 Mix 操作
    ---------
    @param  : 
        imsize: 图象尺寸
    -------
    @Returns  :
        返回和图象尺寸一样大的mask
    -------
    """
    lam = np.random.beta(1,1)
    x1, y1, x2, y2 = random_boundingbox(imsize, lam)
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (imsize * imsize))
    map = torch.ones((imsize,imsize))
    map[x1:x2,y1:y2]=0
    if torch.rand(1)>0.5:
        map = 1 - map
        lam = 1 - lam
    # lam is equivalent to map.mean()
    return map#, lam


class AffineGridGen():
    """
    @description  : pytorch自带的 图象仿射变换中的旋转功能会对图象进行缩放，该类为重新实现的函数，
    旋转图象不会出现缩放的问题, 不需要单独调用该函数
    ---------
    @param  : 
    -------
    @Returns  :
    -------
    """
    def __init__(self, height, width,lr=1):
        super(AffineGridGen, self).__init__()
        self.lr = lr
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)
        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.height), 0), repeats = self.width, axis = 0).T, 0)*height/width
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(-1, 1, 2.0/self.width), 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = torch.from_numpy(self.grid.astype(np.float32))
        #print(self.grid)

    def __call__(self, input1):
        self.input1 = input1
        output = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        self.batchgrid = torch.zeros(torch.Size([input1.size(0)]) + self.grid.size())
        for i in range(input1.size(0)):
            self.batchgrid[i] = self.grid

        if input1.is_cuda:
            self.batchgrid = self.batchgrid.cuda()
            output = output.cuda()

        for i in range(input1.size(0)):
            output = torch.bmm(self.batchgrid.view(-1, self.height*self.width, 3), torch.transpose(input1, 1, 2)).view(-1, self.height, self.width, 2)

        return output

### bias 随机偏移的像素范围  angle 随机旋转的角度范围  
def center_rotate_crop(img_torch, angle=30, bias=15, p=256):
    """
    @description  :  以图象中心为原点，对图象进行随机旋转和平移，由于调用的是 pytorch自带的放射变换的函数，所以
        支持反向传播
    ---------
    @param  : 
        img_torch: 输出待变换的tensor
        angle: 随机旋转的角度范围  [-angle/2, angle/2]
        bias: 随机偏移的像素范围
        p： 最终输出的图象尺寸, 由于旋转后会有黑边，所以要crop出略小于原图的图象用于训练。本项目中输入图象
            320x320，旋转后 crop 256x256
    -------
    @Returns  :  返回变换后的图象
    -------
    """
    b, _, h, w = img_torch.shape
    if angle == 0:
        y, x = (h-p)//2, (w-p)//2
        return img_torch[:, :, y:y+p, x:x+p]

    bw, bh = float(bias) / w,  float(bias) / h
    angle = angle*3.1415926/180
    thetas = []
    for i in range(b):
        alpha = random.random()*angle - 0.5*angle
        deltax, deltay = random.random()*bw - 0.5*bw, random.random()*bh - 0.5*bh

        theta = torch.tensor([
            [math.sin(-alpha), math.cos(alpha), deltax],
            [math.cos(alpha), math.sin(alpha), deltay]
        ], dtype=torch.float)
        thetas.append(theta)
### 计算旋转后的图象大小
    # nW = math.ceil(h*math.fabs(math.sin(angle))+w*math.cos(angle))
    # nH = math.ceil(h*math.cos(angle)+w*math.fabs(math.sin(angle)))
### 旋转后的图象维持不变 会有图象内容损失
    nW = w
    nH = h

    g = AffineGridGen(nH, nW)
    grid_out = g(torch.stack(thetas, dim=0))
    grid_out[:,:,:,0] = grid_out[:,:,:,0]*nW / w
    grid_out[:, :, :, 1] = grid_out[:, :, :, 1] * nW/h
 
    output = F.grid_sample(img_torch, grid_out.to(img_torch.device))
    _,_,h,w = output.shape
    y, x = (h-p)//2, (w-p)//2
    return output[:, :, y:y+p, x:x+p]

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy arrayS
# 输入神经网络中 将 [0, 1]的数据分布 归一化到了 [-1, 1]  这里  + 1 / 2 返回到 [0, 1]范围
def tensor2im(image_tensor, imtype=np.uint8):
    """
    @description  : 将tensor转化为 numpy图象
    ---------
    @param  :
    -------
    @Returns  :
    -------
    """
    
    if image_tensor.shape[1] == 1:
        image_tensor = torch.cat([image_tensor,image_tensor,image_tensor], dim=1)
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

### 保存 numpy 格式图象 
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

## 创建文件夹
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def atten2im(image_tensor, imtype=np.uint8):
#     image_tensor = image_tensor[0]
#     image_tensor = torch.cat((image_tensor, image_tensor, image_tensor), 0)
#     image_numpy = image_tensor.cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
#     image_numpy = image_numpy/(image_numpy.max()/255.0)
#     return image_numpy.astype(imtype)

# def latent2im(image_tensor, imtype=np.uint8):
#     # image_tensor = (image_tensor - torch.min(image_tensor))/(torch.max(image_tensor)-torch.min(image_tensor))
#     image_numpy = image_tensor[0].cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
#     image_numpy = np.maximum(image_numpy, 0)
#     image_numpy = np.minimum(image_numpy, 255)
#     return image_numpy.astype(imtype)

# def max2im(image_1, image_2, imtype=np.uint8):
#     image_1 = image_1[0].cpu().float().numpy()
#     image_2 = image_2[0].cpu().float().numpy()
#     image_1 = (np.transpose(image_1, (1, 2, 0)) + 1) / 2.0 * 255.0
#     image_2 = (np.transpose(image_2, (1, 2, 0))) * 255.0
#     output = np.maximum(image_1, image_2)
#     output = np.maximum(output, 0)
#     output = np.minimum(output, 255)
#     return output.astype(imtype)

# def variable2im(image_tensor, imtype=np.uint8):
#     image_numpy = image_tensor[0].data.cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
#     return image_numpy.astype(imtype)

# def diagnose_network(net, name='network'):
#     mean = 0.0
#     count = 0
#     for param in net.parameters():
#         if param.grad is not None:
#             mean += torch.mean(torch.abs(param.grad.data))
#             count += 1
#     if count > 0:
#         mean = mean / count
#     print(name)
#     print(mean)

# def varname(p):
#     for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
#         m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
#         if m:
#             return m.group(1)

# def print_numpy(x, val=True, shp=False):
#     x = x.astype(np.float64)
#     if shp:
#         print('shape,', x.shape)
#     if val:
#         x = x.flatten()
#         print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
#             np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))



# def get_model_list(dirname, key):
#     if os.path.exists(dirname) is False:
#         return None
#     gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
#                   os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
#     if gen_models is None:
#         return None
#     gen_models.sort()
#     last_model_name = gen_models[-1]
#     return last_model_name


# def get_scheduler(optimizer, hyperparameters, iterations=-1):
#     if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
#         scheduler = None # constant scheduler
#     elif hyperparameters['lr_policy'] == 'step':
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
#                                         gamma=hyperparameters['gamma'], last_epoch=iterations)
#     else:
#         return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
#     return scheduler


# def weights_init(init_type='gaussian'):
#     def init_fun(m):
#         classname = m.__class__.__name__
#         if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
#             # print m.__class__.__name__
#             if init_type == 'gaussian':
#                 init.normal(m.weight.data, 0.0, 0.02)
#             elif init_type == 'xavier':
#                 init.xavier_normal(m.weight.data, gain=math.sqrt(2))
#             elif init_type == 'kaiming':
#                 init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal(m.weight.data, gain=math.sqrt(2))
#             elif init_type == 'default':
#                 pass
#             else:
#                 assert 0, "Unsupported initialization: {}".format(init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant(m.bias.data, 0.0)

#     return init_fun
### 创建快捷方式
def symlink(src, dst):
    if os.path.islink(dst):
        os.unlink(dst)
    os.symlink(src, dst)


