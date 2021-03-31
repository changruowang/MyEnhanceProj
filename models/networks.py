#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:  : 定义了所有用到的网络模型 
@Date          : 2021/03/29 14:24:47
@Author        : changruowang
@version       : 1.0
'''
import torch
import os
import math
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from torch.nn import Parameter
from util.util import pad_tensor, pad_tensor_back

# from torch.utils.serialization import load_lua
from lib.nn import SynchronizedBatchNorm2d as SynBN2d
# from torch.nn import SyncBatchNorm as SynBN2d

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        m.weight.data.normal_(0.0, 0.02)
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'synBN':
        norm_layer = functools.partial(SynBN2d, affine=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=False))]
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=False)]
        if 'norm' == 'Instance':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.LeakyReLU(inplace=False)]
        self.model = nn.Sequential(*model)
        # self.model.apply(gaussian_weights_init)
        #elif == 'Group'
    def forward(self, x):
        return self.model(x)

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=64, pad_type='zero', activation='lrelu',normal='in'):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(input_nc, ndf, 7, 1, 3, pad_type = pad_type, activation = activation, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(ndf, ndf * 2, 4, 2, 1, pad_type = pad_type, activation = activation, norm = normal, sn = True)
        self.block3 = Conv2dLayer(ndf * 2, ndf * 4, 4, 2, 1, pad_type =pad_type, activation = activation, norm = normal, sn = True)
        self.block4 = Conv2dLayer(ndf * 4, ndf * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = normal, sn = True)
        self.block5 = Conv2dLayer(ndf * 4, ndf * 4, 4, 2, 1, pad_type = pad_type, activation = activation, norm = normal, sn = True)
        self.block6 = Conv2dLayer(ndf * 4, 1, 4, 2, 1, pad_type = pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, x):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        # x = torch.cat((img, mask), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        #     return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], skip=False, opt=None):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.opt = opt
        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True, opt=opt)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout, opt=opt)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer, opt=opt)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, opt=opt)
        
        if skip == True:
            skipmodule = SkipModule(unet_block, opt)
            self.model = skipmodule
        else:
            self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SkipModule(nn.Module):
    def __init__(self, submodule, opt):
        super(SkipModule, self).__init__()
        self.submodule = submodule
        self.opt = opt

    def forward(self, x):
        latent = self.submodule(x)
        return self.opt.skip*x + latent, latent



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, opt=None):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if opt.use_norm == 0:
            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up
        else:
            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


class NoNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NoNormDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        # if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
        #     return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        # else:
        return self.model(input)


from .edvr import DCNv2Pack
class Unet_resize_conv(nn.Module):
    ''' Unet 网络
    Args： 
        skip 是否使用残差
        in_deconv： 输入是否可变形卷积  默认值即可
    '''
    def __init__(self, opt, skip, in_deconv=False):
        super(Unet_resize_conv, self).__init__()
        self.in_deconv = in_deconv
        self.opt = opt
        self.skip = skip
        p = 1

        if self.in_deconv and not opt.self_attention:
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
            self.conv1_1_offset1 = nn.Conv2d(64, 32, 3, padding=p)
            self.LReLU = nn.LeakyReLU(0.2, inplace=True)
            self.conv1_1_offset2 = nn.Conv2d(32, 32, 3, padding=p)
            self.conv1_1_cat = nn.Conv2d(64, 32, 1)
            self.dcn_pack = DCNv2Pack(
                32,
                32,
                3,
                padding=1,
                deformable_groups=4)    
        else:
            self.conv1_1 = nn.Conv2d(9 if opt.self_attention else 6, 32, 3, padding=p)

        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, opt.output_nc, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size*block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).resize(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output
    
    def forward(self, input, ref_input=None, gray=None):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1

        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        if ref_input is not None:
            ref_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ref_input)
            
        if gray is not None:
            gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)

        if self.opt.use_norm == 1:
            if self.in_deconv and not self.opt.self_attention:
                in_feat = self.conv1_1(input)
                ref_feat = self.conv1_1(ref_input)
                offset = self.LReLU(self.conv1_1_offset1(torch.cat((in_feat, ref_feat), 1)))
                offset = self.LReLU(self.conv1_1_offset2(offset))
                
                dcn_out = self.LReLU(self.dcn_pack(ref_feat, offset))
                x = self.bn1_1(self.LReLU(self.conv1_1_cat(torch.cat((dcn_out, in_feat), dim=1))))
            elif self.opt.self_attention:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, ref_input, gray), 1))))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, ref_input), 1))))

            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            # x = x*gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
            
            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            # conv4 = conv4*gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            # conv3 = conv3*gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            # conv2 = conv2*gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            # conv1 = conv1*gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent*gray

            # output = self.depth_to_space(conv10, 2)
            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    input = (input - torch.min(input))/(torch.max(input) - torch.min(input))
                    output = latent + input*self.opt.skip
                    output = output*2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent))/(torch.max(latent)-torch.min(latent))
                    output = latent + input*self.opt.skip
            else:
                output = latent

            # if self.opt.linear:
            #     output = output/torch.max(torch.abs(output))
            
        elif self.opt.use_norm == 0:
           assert(0)
        
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        relu4_1 = h
        h = F.relu(self.conv4_2(h), inplace=True)
        relu4_2 = h
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)
        relu4_3 = h

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        
        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        relu5_2 = F.relu(self.conv5_2(relu5_1), inplace=True)
        conv5_3 = self.conv5_3(relu5_2) 
        h = F.relu(conv5_3, inplace=True)
        relu5_3 = h
        if opt.vgg_choose == "conv4_3":
            return conv4_3
        elif opt.vgg_choose == "relu4_2":
            return relu4_2
        elif opt.vgg_choose == "relu4_1":
            return relu4_1
        elif opt.vgg_choose == "relu4_3":
            return relu4_3
        elif opt.vgg_choose == "conv5_3":
            return conv5_3
        elif opt.vgg_choose == "relu5_1":
            return relu5_1
        elif opt.vgg_choose == "relu5_2":
            return relu5_2
        elif opt.vgg_choose == "relu5_3" or "maxpool":
            return relu5_3

class EnhanceNet_nopool(nn.Module):
    ''' ZeroDEC 原文的模型
    '''
    def __init__(self, opt):
        super(EnhanceNet_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.e_conv1 = nn.Conv2d(6,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,opt.output_nc,3,1,1,bias=True) 

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, input_ref=None, input_gray=None):

        x1 = self.relu(self.e_conv1(torch.cat((x, input_ref),1)))
        # p1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        # p2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        # p3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        # x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))


        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        # r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


        # x = x + r1*(torch.pow(x,2)-x)
        # x = x + r2*(torch.pow(x,2)-x)
        # x = x + r3*(torch.pow(x,2)-x)
        # enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
        # x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        # x = x + r6*(torch.pow(x,2)-x)	
        # x = x + r7*(torch.pow(x,2)-x)
        # enhance_image = x + r8*(torch.pow(x,2)-x)
        # r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return x_r

class UNetDeformNet(nn.Module):
    ''' 自定义的 UNET+可变形卷积 可以用来替代本项目的unet做backbone 
    '''
    def __init__(self, opt=None):
        super(UNetDeformNet, self).__init__()
        self.opt = opt
        bilinear = True

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        # self.up1 = Up(512, 256 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)

        self.dec_conv1 = DecConv(num_feat=64, use_pyramid=True)
        self.dec_conv2 = DecConv(num_feat=128, use_pyramid=True)
        self.dec_conv3 = DecConv(num_feat=256, use_pyramid=False)

        self.out_conv = nn.Conv2d(64, 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_, ref_input=None, atten=None):
        input_, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input_)
        ref_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ref_input)

        x = torch.stack((input_, ref_input), dim=1)
        b, t, c, h, w = x.size()
      
        x1 = self.inc(x.view(-1, c, h, w))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # print(x3.shape, x2.shape, x1.shape)

        x1 = x1.view(b, t, -1, h, w)
        x2 = x2.view(b, t, -1, h//2, w//2)
        x3 = x3.view(b, t, -1, h//4, w//4)
        x4 = x4.view(b, t, -1, h//8, w//8)

        aligened_feat3, offset3 = self.dec_conv3(x3[:,0,:,:,:].clone(), x3[:,1,:,:,:].clone(), offset_lower=None)
        aligened_feat2, offset2 = self.dec_conv2(x2[:,0,:,:,:].clone(), x2[:,1,:,:,:].clone(), offset_lower=offset3)
        aligened_feat1, _ = self.dec_conv1(x1[:,0,:,:,:].clone(), x1[:,1,:,:,:].clone(), offset_lower=offset2)

        # print(aligened_feat1.shape, aligened_feat2.shape, aligened_feat3.shape)
        # print(x4.shape)
        x = self.up1(x4[:,0,:,:,:], aligened_feat3)
        x = self.up2(x, aligened_feat2)
        x = self.up3(x, aligened_feat1)

        latent = self.tanh(self.out_conv(x))
        out = latent * self.opt.skip + input_ 

        out = pad_tensor_back(out, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        return out, latent


class SpatialAt(nn.Module):
    ''' 空间注意力模块
    '''
    def __init__(self):
        super(SpatialAt, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())
    def forward(self, x):
        mean_ = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, keepdim=True, dim=1)[0]
        min_ = torch.max(x, keepdim=True, dim=1)[0]
        return self.conv(torch.cat((mean_, max_, min_), dim=1))

class ChannelAt(nn.Module):
    ''' 通道注意力模块
    '''
    def __init__(self, in_channel, out_channel):
        super(ChannelAt, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

class CBAM(nn.Module):
    '''  CBAM 模块   封装了 ChannelAt 和 SpatialAt
    Args: 
        in_channel: 输入通道数
        out_channel: 输出通道数
    ''' 
    def __init__(self, in_channel, out_channel=1):
        super(CBAM, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            )
        self.sp = SpatialAt()
        self.ct = ChannelAt(64, 64)

        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, out_channel, kernel_size=1, stride=1),   
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.ct(x1) * x1
        x3 = self.sp(x2)
        return self.conv_out(x3 * x1 + x1)

class GatedConv2d(torch.nn.Module):
    ''' 门卷积 
    Arg:
        输入通道数  输出通道数  卷积核尺寸  卷积步长  ...
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2d, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.sigmoid(mask)
        else:
            x = x * self.sigmoid(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

from .edvr import ResidualBlockNoBN, make_layer
from util.util import torch_normal_img
class AttEnhanceNet2(nn.Module):
    ''' 自定义多阶段网络 模型
    0  低光图象输入编码模块
    1  参考图象输入编码模块
    2  公用的特征 解码、融合模块
    3  公用的特征融合模块
    4  运动像素估计模块
    不同的训练阶段将对应的 模块梯度权重更新打开
    '''
    def __init__(self, opt=None, blur_win=31, skip_init=False):
        super(AttEnhanceNet2, self).__init__()
        self.opt = opt
        self.feat_enc1 = nn.Sequential(                                   # 0 低光图象输入编码模块
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            make_layer(ResidualBlockNoBN, 4, num_feat=64)
        )
        self.feat_enc2 = nn.Sequential(                                   #  1  参考图象输入编码模块
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            make_layer(ResidualBlockNoBN, 3, num_feat=64)
        )

        # self.feat_dec = nn.Sequential(          
        #     make_layer(GatedConv2d, 4, in_channels=64, out_channels=64), # 2
        #     nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Tanh()
        # )
        self.feat_dec = nn.Sequential(                                    # 2 公用的特征 解码、融合模块
            make_layer(ResidualBlockNoBN, 3, num_feat=64), # 2
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )

        self.fus_conv = nn.Sequential(                                    # 3  公用的特征融合模块
            GatedConv2d(in_channels=130, out_channels=64),   # 3 
            GatedConv2d(in_channels=64, out_channels=64)
        )
        
        self.skip_init = skip_init
        # self.feat_dec = nn.Sequential(                                 
        #     make_layer(ResidualBlockNoBN, 3, num_feat=64),
        #     nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
        #     nn.Tanh()
        # )
        # self.fus_conv = make_layer(ResidualBlockNoBN, 2, num_feat=64)        
        
        self.mask_conv = CBAM(192, 1)                                        # 4  运动像素估计模块
        self.blur_pool = nn.AvgPool2d(blur_win, 1, padding=(blur_win-1)//2, count_include_pad=False)

        self.param_dict = {0: self.feat_enc1, 1:self.feat_enc2,\
                     2:self.feat_dec, 3:self.fus_conv, 4:self.mask_conv}

    def set_grad_model(self, param_idxs=(0,), requires_grad=False):
        ''' 将对应的编号的模块的梯度 打开 or  关闭
        Args: 
            param_idxs: 模块代号
            requires_grad： 是否更新权重
        '''
        models = [self.param_dict[idx] for idx in param_idxs]
        for net in models:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def get_low_input(self, low, ref):
        v = torch.max(ref, dim=1, keepdim=True)[0]
        return torch.cat((low, self.blur_pool(v)), dim=1)

    def forward_endec_path(self, low_img, ref_img, img_b):
        ''' 训练编码解码路径调用  分别对 低光图 参考图  编解码输出
        Args: 
            low_img: 低光图象
            ref_img： 参考图像
        '''
        self.set_grad_model((0,1), True)
        self.set_grad_model((2,3,4), False)

        low_feat = self.feat_dec(self.feat_enc1(self.get_low_input(low_img, ref_img)))
        ref_feat = self.feat_dec(self.feat_enc2(img_b)) 

        return low_feat, ref_feat
    
    def init_weight(self):
        print("reinit dec and fus model...")
        for model in [self.feat_dec, self.fus_conv]:
            for m in model.modules():
                if isinstance(m,nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)  ##权重初始化方式效果影响很大
                    # nn.init.normal(m.weight.data)
                    # nn.init.xavier_normal(m.weight.data)
                    # nn.init.kaiming_normal(m.weight.data)#卷积层参数初始化
                    # m.bias.data.fill_(0)
                elif isinstance(m,nn.Linear):
                    m.weight.data.normal_()#全连接层参数初始化

    def forward_fus_path(self, input_, ref_input, mask=None):
        ''' 前向 融合 路径  
        '''
        if self.skip_init and self.train:
            self.init_weight()
            self.skip_init = False

        self.set_grad_model((1,0,4), False)
        self.set_grad_model((3,2), True)

        return self.forward(input_, ref_input, mask)

    def forward_mask_path(self, input_, ref_input):
        ''' 前向 mask 估计路径  
        '''
        self.set_grad_model((1,2,0,3), False)
        self.set_grad_model((4,), True)
        low_feat = self.feat_dec(self.feat_enc1(self.get_low_input(low_img, ref_img)))
        ref_feat = self.feat_dec(self.feat_enc2(img_b)) 

        attention = self.mask_conv(torch.cat((input_feat, ref_feat, input_feat-ref_feat), dim=1))
        return attention

    def forward(self, input_, ref_input, mask=None):
        input_feat = self.feat_enc1(self.get_low_input(input_, ref_input))
        ref_feat = self.feat_enc2(ref_input)

        if mask is None:
            mask = self.mask_conv(torch.cat((input_feat, ref_feat, input_feat-ref_feat), dim=1))
            # mask = torch_normal_img(mask)

        fus_feat = self.fus_conv(torch.cat((input_feat*mask, mask, ref_feat*(1-mask), 1-mask), 1))
        # fus_feat = self.fus_conv(input_feat * attention + ref_feat * (1 - attention))
        # fus_feat = ref_feat * (1 - attention)

        out = self.feat_dec(fus_feat)
        return out, mask

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, skip=False, opt=None):
    '''根据 模型名字 设置 生成网络 模型  

    Args： 
        input_nc 网络输入通道数（有的模型用不上这个参数）
        output_nc： 网络输出通道数
        skip: 是否残差输出  
        which_model_netG ： 生成模型的名字 目前支持的模型有
            sid_unet_resize ： unet 模型
            en_net_nopool： zeroDEC原文中的模型
            deconv_unet： 自定义的可变形卷积+UNET模型
            in_deconv_unet： 输入 可变形卷积 + UNET
            endec_atten_net: 多阶段模型 和上面的模型不通用 这个模型在 多阶段 restore_fusion_ 类方法中使用
    '''
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'sid_unet_resize':
        netG = Unet_resize_conv(opt, skip)
    elif which_model_netG == 'en_net_nopool':
        netG = EnhanceNet_nopool(opt)
    elif which_model_netG == 'deconv_unet':
        from .edvr import UNetDeformNet
        netG = UNetDeformNet(opt=opt)
    elif which_model_netG == 'in_deconv_unet':
        netG = Unet_resize_conv(opt, skip, in_deconv=True)
    elif which_model_netG == 'endec_atten_net':
        netG = AttEnhanceNet2(opt=opt, skip_init=skip)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    netG.to(opt.device)
    if opt.use_distribute_train:
        netG = torch.nn.parallel.DistributedDataParallel(netG,
                            device_ids=[opt.local_rank],
                            output_device=opt.local_rank)
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[], patch=False, opt=None):
    '''根据 模型名字 设置 判别器   

    Args： 
        input_nc 网络输入通道数（有的模型用不上这个参数）
       
        which_model_netD ： 判别模型的名字 目前支持的模型有
            no_norm_4 
            patch_dis
            unet_dis 
            不同的模型区别不大  只不过是从不同文章开源的代码中 copy 来的 其中 unet_dis是支持gan mix操作的
    '''
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    if which_model_netD == 'no_norm_4':
        netD = NoNormDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'patch_dis':
        netD = PatchDiscriminator(input_nc, ndf)
    elif which_model_netD == 'unet_dis':
        from .unet_dis import Unet_Discriminator
        netD = Unet_Discriminator(opt=opt, in_ch=input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)                          
    if opt.use_distribute_train:
        netD = nn.SyncBatchNorm.convert_sync_batchnorm(netD)
        netD.to(opt.device)
        netD = torch.nn.parallel.DistributedDataParallel(netD,
                            device_ids=[opt.local_rank],
                            output_device=opt.local_rank)
    else:
        netD.to(opt.device)
    netD.apply(weights_init)
    return netD