#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 定义了用到的损失函数
@Date          : 2021/03/29 14:23:40
@Author        : changruowang
@version       : 1.0
'''
from .networks import Vgg16
from util.util import torch_normal_img,sum_per_image,mean_per_image
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class SSIM(object):
    ''' SSIM损失 本项目暂时没用到  参数设置为默认值即可
    '''
    def __init__(self, mean_metric=True, size=11, sigma=1.5, cs_map=False, device=torch.device("cuda:0")):
        self.window = self._fspecial_gauss(size, sigma).to(device)
        self.window.requires_grad = False
        self.mean_metric = mean_metric
        self.cs_map = cs_map
    def _fspecial_gauss(self, size, sigma):
        x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        y = torch.FloatTensor(y_data)
        x = torch.FloatTensor(x_data)

        g = torch.exp(-((x**2 + y**2)/(2.0*sigma**2))).permute(2,3,0,1)

        return g / torch.sum(g)

    def cal_one_channel(self, img1, img2):
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1*L)**2
        C2 = (K2*L)**2
        mu1 = F.conv2d(img1, self.window, padding=0)
        mu2 = F.conv2d(img2, self.window, padding=0)
        
        # print(self.window)

        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self.window, padding=0) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=0) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=0) - mu1_mu2

        if self.cs_map:
            value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2)),
                    (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2))

        if self.mean_metric:
            value = torch.mean(value)
        return value

    def __call__(self, img1, img2):
        if img1.shape[1] == 3:
            loss1 = self.cal_one_channel(img1[:,0:1,:,:], img2[:,0:1,:,:])
            loss2 = self.cal_one_channel(img1[:,1:2,:,:], img2[:,1:2,:,:])
            loss3 = self.cal_one_channel(img1[:,2:3,:,:], img2[:,2:3,:,:])
            return 1 - (loss1 + loss2 + loss3) / 3.0


def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    if opt.vgg_mean:
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean)) # subtract mean
    return batch


class PerceptualLoss(nn.Module):
    ''' VGG 损失
    调用 compute_vgg_loss 计算vgg损失即可
    '''
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.vgg = Vgg16()
        self.vgg.load_state_dict(torch.load(os.path.join("./checkpoints", 'vgg16.weight')))
        self.vgg.to(opt.device)
        if opt.use_distribute_train:
            self.vgg = nn.SyncBatchNorm.convert_sync_batchnorm(self.vgg)
            self.vgg = torch.nn.parallel.DistributedDataParallel(self.vgg,
                                                    device_ids=[opt.local_rank],
                                                    output_device=opt.local_rank)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def compute_vgg_loss(self, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = self.vgg(img_vgg, self.opt)
        target_fea = self.vgg(target_vgg, self.opt)
        if self.opt.no_vgg_instance:
            return torch.mean((img_fea - target_fea) ** 2)
        else:
            return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

class DiscLossWGANGP():
    ''' WGAN 损失 
    主要用 calc_gradient_penalty 计算正则项
    '''
    def __init__(self):
        self.LAMBDA = 10
        
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10
        
    # def get_g_loss(self, net, realA, fakeB):
    #     # First, G(A) should fake the discriminator
    #     self.D_fake = net.forward(fakeB)
    #     return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data, out_sel=None):
        '''  
        Args: 
            netD：判别器
            real_data： 真实样本  
            fake_data： 生成的假样本 
            out_sel： 用于兼容 UNET-判别器，因为他有多个输出， out_sel用于选择使用哪个输出计算正则
        '''
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)
        
        if out_sel is not None:
            disc_interpolates = netD.forward(interpolates)[out_sel]
           
        else:
            disc_interpolates = netD.forward(interpolates)
   
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

 
##############################################################################
# Classes
##############################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module): 
    ''' GAN 损失 该类支持 最小平方损失 和 BCE 损失
    Args:
        use_lsgan: 是否使用平方损失 即 LSGAN，如果不使用，损失就是 BCE 交叉熵损失
        其他参数默认即可
    '''
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.device = opt.device 
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        ''' 获取标签值，主要目的是 将 标签尺寸转为 和 input 一样的大小
        Args:
            input: 判别器 预测的 值
            target_is_real： 判别器预测的 label是真还是假
        '''
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = torch.full_like(input, self.real_label).to(self.device)
                
                self.real_label_var.requires_grad = False
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                
                self.fake_label_var = torch.full_like(input, self.fake_label).to(self.device)
                self.fake_label_var.requires_grad = False
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, mask=None):
        ''' 计算GAN损失
        Args:
            input: 判别器 预测的 值
            target_is_real： 判别器预测的 label是真还是假
            mask： 对损失加掩码 本项目没用到
        '''
        if type(target_is_real) == type(True):
            target_tensor = self.get_target_tensor(input, target_is_real)
        else:
            target_tensor = target_is_real
            
        if mask is not None:
            return torch.mean((input - target_tensor)**2 * mask)

        return self.loss(input, target_tensor)

class GetGradNormal(object):
    ''' 计算 归一化的 梯度
        Args:
            device: 设备类型
    '''
    def __init__(self, device):
        kernel_ = np.array([0,0,-1,1]).reshape(2,2)
        self.kernel_x = torch.from_numpy(kernel_).float().unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_y = torch.from_numpy(kernel_.T).float().unsqueeze(0).unsqueeze(0).to(device)
        self.kernel_x.requires_grad = False
        self.kernel_y.requires_grad = False  
    def __call__(self,  input_tensor):
        '''
        Args:
            input_tensor: b c h w 的 tensor
        Returns:
            返回 归一化的 y x 方向梯度（梯度取了abs,因此没有正负）
        '''
        x = F.pad(input_tensor, [0, 1, 0, 1])   #[0,1,0,0] 代表 输入tensor的 x方向 左 右 上 下 padding的个数 
        grad_orig_y = torch.abs(F.conv2d(x, self.kernel_y))
        grad_orig_x = torch.abs(F.conv2d(x, self.kernel_x))

        grad_min_y, grad_max_y = torch.min(grad_orig_y), torch.max(grad_orig_y)
        grad_min_x, grad_max_x = torch.min(grad_orig_x), torch.max(grad_orig_x)

        grad_normal_x = torch.div((grad_orig_x - grad_min_x), (grad_max_x-grad_min_x+0.0001))
        grad_normal_y = torch.div((grad_orig_y - grad_min_y), (grad_max_y-grad_min_y+0.0001))
        return grad_normal_y, grad_normal_x

class L_RTV(nn.Module):
    '''计算 加权变分损失，在参考图象的边缘出，给梯度较小的惩罚
        Args:
            input_tensor: b c h w 的 tensor
        Returns:
            返回 归一化的 y x 方向梯度（梯度取了abs,因此没有正负）
    '''
    def __init__(self, opt):
        super(L_RTV, self).__init__()
        device = opt.device
        self.rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140],requires_grad=False).reshape(1, -1).to(device)
        self.grad_func = GetGradNormal(device)
        self.constant = torch.tensor(0.01, requires_grad=False).to(device)

    def forward(self, L, ref_img):
        '''计算 加权变分损失，在参考图象的边缘出，给梯度较小的惩罚 
        Args:
            L: b c h w 的 tensor，待平滑的tensor
            ref_img： 参考图 如果是rgb三通道图 会自动转换为 灰度图 再计算边缘
        Returns:
            返回 计算的损失 标量
        '''
        ref_img, L = (ref_img + 1) / 2, (L + 1) / 2  ### -1 ~ 1 分布转化为 0 ~ 1分布
        if ref_img.shape[1] == 3:
            orig_gray = torch.tensordot(ref_img, self.rgb_weights, dims=([1], [-1])).permute(0,3,1,2)
        else:
            orig_gray = ref_img    
        grad_y_ref, grad_x_ref = self.grad_func(orig_gray)

        grad_y_ref[grad_y_ref > 0.9] = 1000
        grad_x_ref[grad_x_ref > 0.9] = 1000

        grad_y_w1, grad_x_w1 = self.grad_func(L)

        loss_g = torch.mean(torch.abs(grad_x_w1 / torch.max(grad_x_ref, self.constant)) + 
                        torch.abs(grad_y_w1 / torch.max(grad_y_ref, self.constant)))
        
        return loss_g

class L_TV(nn.Module):
    '''计算 变分损失
        Args:
            x: b c h w 的 tensor，待平滑的tensor
        Returns:
            返回 计算的损失 标量
    '''
    def __init__(self):
        super(L_TV,self).__init__()
       
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size


class L_MSE(nn.Module):
    '''计算 MSE 损失
        Args:
            clamp: 是否对每个像素 损失大小限幅 没用到      
    '''
    def __init__(self, clamp=False):
        super(L_MSE, self).__init__()
        self.clamp=clamp

    def forward(self, x, y, mask=None, out_vector=False, out_map=False):
        '''计算 MSE 损失
        Args:
            x, y : 要计算的两个图
            mask：不为None则使用 mask 进行区域掩码操作。
            out_vector: 主要用于 ohem 中 需要输出每个样本的损失，若为true 就不对 batch 维度做平均，
                    返回一个 bx1的向量
            out_map： 主要用于可视化 损失分布
        '''
        if out_map:
            return mask * ((x - y)**2)

        if mask is not None:  
            a = sum_per_image(mask)  ### （只对mask区域取平均）
            a[a < 0.9] = 1
            l = (sum_per_image(mask * ((x - y)**2)) / a) * 10
            
        else:
            l =  mean_per_image((x- y)**2) * 10

        return l if out_vector else torch.mean(l)

        # if mask is not None:  
        #     a = torch.sum(mask[mask > 0.2])
        #     if a >= 0.9:
        #         return (torch.sum(mask * ((x - y)**2)) / a) * 10
        #     return torch.mean(torch.mean(((x- y)**2) * mask)) * 10
        # else:
        #     return torch.mean(torch.mean(((x- y)**2))) * 10



class L_cosine(object):
    '''计算 余弦相似度 损失
        Args:
            clamp: 是否对每个像素 损失大小限幅   
    '''
    def __init__(self, clamp=True):
        super(L_cosine, self).__init__()
        self.clamp = clamp

    def __call__(self, x, y, mask=None, out_vector=False, out_map=False):
        '''计算 余弦相似度 损失
            Args:
                clamp: 对每个位置的损失限幅，为了提高对运动区域的感知能力，将损失先乘了系数20，然后限幅
                避免把噪声的损失放的过大
                mask：是否使用 mask损失
                out_vector： 输出是否保留每张图的平均损失 
                out_map： 损失可视化
        '''
        loss_map = (1-torch.cosine_similarity(x, y, dim=1).unsqueeze(1)) * 20  ##乘系数，扩大余弦损失

        if self.clamp:  ## 限幅防止部分噪声损失很大
            loss_map = torch.clamp(loss_map, 0, 1.0)

        if mask is not None: 
            a = sum_per_image(mask)  ##对mask区域求  损失的均值是mask区域的均值
            a[a < 0.9] = 1           ## 防止分母除0
            loss_map = loss_map * mask

        if out_map:
            return loss_map

        l = (sum_per_image(loss_map) / a)  if mask is not None else mean_per_image(loss_map) 

        return l if out_vector else torch.mean(l)



class L_exp(nn.Module):
    '''计算 曝光 损失
        Args:
           patch_size： 窗口尺寸
    '''
    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
    def forward(self, x, y, mask=None):

        b,c,h,w = x.shape
        # x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        # y = torch.mean(y,1,keepdim=True)
        meany = self.pool(y)

        # if mask is not None:
        #     return torch.mean(torch.mean(torch.abs(mean- meany) * mask)) * 10

        d = torch.mean(torch.mean(torch.abs(mean- meany))) * 10
        return d


class SpatialConsistencyLoss(nn.Module):
    '''计算 空间一致性损失  即梯度相似度损失 ZeroDEC原文代码
        Args:
           grad_normal ： 是否对梯度归一化 未使用
           pre_avg： 是否预先取平均后再算梯度，这样避免梯度中噪声的影响 ZeroDEC原文代码是默认使用
                预先模糊操作的，但是为了背景更好的细节效果，就将模糊关了
    '''
    def __init__(self, grad_normal=False, pre_avg=True):
        super(SpatialConsistencyLoss, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

        self.grad_normal = grad_normal
        self.pre_avg = pre_avg
    def forward(self, org , enhance, pre_avg=False, mask=None, out_vector=False, out_map=False):
        '''计算 梯度 损失
        Args:
            org , enhance：分别是目标图象 和  参考图像
            mask：是否使用 mask损失
            out_vector： 输出是否保留每张图的平均损失 
            out_map： 损失可视化
        '''
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        if pre_avg:   ## 预先 模糊一下 
            org_pool =  self.pool(org_mean)			
            enhance_pool = self.pool(enhance_mean)	
        else:
            org_pool = org_mean
            enhance_pool = enhance_mean

        if mask is not None and self.pre_avg:
            mask =  self.pool(mask)	

        # weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        if self.grad_normal:   ### 好像不能直接归一化
            D_left = (torch_normal_img(torch.abs(D_org_letf)) - torch_normal_img(torch.abs(D_enhance_letf)))**2
            D_right = D_left
            # D_right = (torch_normal_img(D_org_right) - torch_normal_img(D_enhance_right))**2
            D_up = (torch_normal_img(torch.abs(D_org_up)) - torch_normal_img(torch.abs(D_enhance_up)))**2
            D_down = D_up
            # D_down = (torch_normal_img(D_org_down) - torch_normal_img(D_enhance_down))**2
        else:
            D_left = torch.pow(D_org_letf - D_enhance_letf,2)
            D_right = torch.pow(D_org_right - D_enhance_right,2)
            D_up = torch.pow(D_org_up - D_enhance_up,2)
            D_down = torch.pow(D_org_down - D_enhance_down,2)

        ans = (D_left + D_right + D_up +D_down)
        
        if out_map:
            return mask * ans if mask is not None else ans
###########################
        if mask is not None:  
            a = sum_per_image(mask)
            a[a < 0.9] = 1
            E = (sum_per_image(mask * ans) / a)

            
            # E = (mean_per_image(mask * ans))

            # for sum_, num, avg in list(zip(sum_per_image(mask * ans), a, E)):
            #     print(sum_.item(), '    ',num.item(), '    ', avg.item())
            
        else:
            E = mean_per_image(ans)
            
        return E if out_vector else torch.mean(E)
###########################

        # if mask is not None:
        #     a = torch.sum(mask[mask > 0.2])
        #     if a >= 0.9:
        #         return torch.sum(ans * mask) / a
        #     return torch.mean(ans * mask)
            
        # return torch.mean(ans)