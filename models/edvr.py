import torch
from torch import nn as nn
from torch.nn import functional as F
# from basicsr.models.archs.arch_util import (DCNv2Pack, ResidualBlockNoBN,
#                                             make_layer)
from .networks import pad_tensor, pad_tensor_back
import math
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
# from basicsr.utils import get_root_logger

try:
    from basicsr.models.ops.dcn import (ModulatedDeformConvPack,
                                        modulated_deform_conv)
except ImportError:
    print('Cannot import dcn. Ignore this warning if dcn is not used. '
          'Otherwise install BasicSR with compiling dcn.')
    ModulatedDeformConvPack = object
    modulated_deform_conv = None

########################### 来自文章 evdr 开源代码 ###################################### 

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.
    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.
    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|
    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h).type_as(x),
        torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow,
                size_type,
                sizes,
                interp_mode='bilinear',
                align_corners=False):
    """Resize a flow according to ratio or shape.
    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.
    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(
            f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow,
        size=(output_h, output_w),
        mode=interp_mode,
        align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            # logger = get_root_logger()
            print(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)



class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.
        Args:
            nbr_feat_l (list[Tensor]): Neighboring feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
            ref_feat_l (list[Tensor]): Reference feature list. It
                contains three pyramid levels (L1, L2, L3),
                each with shape (b, c, h, w).
        Returns:
            Tensor: Aligned features.
        """
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

class DeformAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            padding=1,
            deformable_groups=deformable_groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        # Cascading
        offset = torch.cat([nbr_feat_l, ref_feat_l], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(nbr_feat_l, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class PredeblurModule(nn.Module):
    """Pre-dublur module.
    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        hr_in (bool): Whether the input has high resolution. Default: False.
    """

    def __init__(self, num_in_ch=3, num_feat=64, hr_in=False):
        super(PredeblurModule, self).__init__()
        self.hr_in = hr_in

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        if self.hr_in:
            # downsample x4 by stride conv
            self.stride_conv_hr1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
            self.stride_conv_hr2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        # generate feature pyramid
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)

        self.resblock_l3 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_1 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l2_2 = ResidualBlockNoBN(num_feat=num_feat)
        self.resblock_l1 = nn.ModuleList(
            [ResidualBlockNoBN(num_feat=num_feat) for i in range(5)])

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        if self.hr_in:
            feat_l1 = self.lrelu(self.stride_conv_hr1(feat_l1))
            feat_l1 = self.lrelu(self.stride_conv_hr2(feat_l1))

        # generate feature pyramid
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))

        feat_l3 = self.upsample(self.resblock_l3(feat_l3))
        feat_l2 = self.resblock_l2_1(feat_l2) + feat_l3
        feat_l2 = self.upsample(self.resblock_l2_2(feat_l2))

        for i in range(2):
            feat_l1 = self.resblock_l1[i](feat_l1)
        feat_l1 = feat_l1 + feat_l2
        for i in range(2, 5):
            feat_l1 = self.resblock_l1[i](feat_l1)
        return feat_l1


class EDVR(nn.Module):
    """EDVR network structure for video super-resolution.
    Now only support X4 upsampling factor.
    Paper:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks
    Args:
        num_in_ch (int): Channel number of input image. Default: 3.
        num_out_ch (int): Channel number of output image. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_frame (int): Number of input frames. Default: 5.
        deformable_groups (int): Deformable groups. Defaults: 8.
        num_extract_block (int): Number of blocks for feature extraction.
            Default: 5.
        num_reconstruct_block (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        hr_in (bool): Whether the input has high resolution. Default: False.
        with_predeblur (bool): Whether has predeblur module.
            Default: False.
        with_tsa (bool): Whether has TSA module. Default: True.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=2,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10,
                 center_frame_idx=0,
                 hr_in=False,
                 with_predeblur=False,
                 with_tsa=True):
        super(EDVR, self).__init__()
        self.center_frame_idx = center_frame_idx

        self.hr_in = hr_in
        self.with_predeblur = with_predeblur
        self.with_tsa = with_tsa

        # extract features for each frame
        # if self.with_predeblur:
        #     self.predeblur = PredeblurModule(
        #         num_feat=num_feat, hr_in=self.hr_in)
        #     self.conv_1x1 = nn.Conv2d(num_feat, num_feat, 1, 1)
        # else:

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        # self.pcd_align = PCDAlignment(
        #     num_feat=num_feat, deformable_groups=deformable_groups)
        self.pcd_align = DeformAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        if self.with_tsa:
            self.fusion = TSAFusion(
                num_feat=num_feat,
                num_frame=num_frame,
                center_frame_idx=self.center_frame_idx)
        else:
            self.fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # reconstruction
        self.reconstruction = make_layer(
            ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, low_image, ref_image):
        
        x = torch.stack((low_image, ref_image), dim=1)

        b, t, c, h, w = x.size()

        assert(t == 2)

        t = 2
        if self.hr_in:
            assert h % 16 == 0 and w % 16 == 0, (
                'The height and width must be multiple of 16.')
        else:
            assert h % 4 == 0 and w % 4 == 0, (
                'The height and width must be multiple of 4.')

        x_center = x[:, 0, :, :, :].contiguous()
        
        # extract features for each frame
        # L1
        # if self.with_predeblur:
        #     feat_l1 = self.conv_1x1(self.predeblur(x.view(-1, c, h, w)))
        #     if self.hr_in:
        #         h, w = h // 4, w // 4
        # else:

        feat_l1 = self.lrelu(self.conv_first(x.view(-1, c, h, w)))

        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [  # neighboring feature list
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        if not self.with_tsa:
            aligned_feat = aligned_feat.view(b, -1, h, w)
        feat = self.fusion(aligned_feat)

        out = self.reconstruction(feat)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)

        print(out.shape)

        if self.hr_in:
            base = x_center
        else:
            base = F.interpolate(
                x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out



class DecConv(nn.Module):
    def __init__(self, num_feat=64, use_pyramid=False,deformable_groups=8):
        super().__init__()
        self.use_pyramid = use_pyramid
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        if use_pyramid:
            self.upsample = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False)
            self.offset_conv2 = nn.Conv2d(num_feat * 2 + num_feat, num_feat, 3, 1, 1)
            self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        else:
            self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                padding=1,
                deformable_groups=deformable_groups)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x1, x_ref, offset_lower=None):

        offset = self.lrelu(self.offset_conv1(torch.cat((x1, x_ref), dim=1)))
        if offset_lower is not None and self.use_pyramid:
            offset = self.lrelu(self.offset_conv2(torch.cat((offset, self.upsample(offset_lower)*2), dim=1)))
            offset = self.lrelu(self.offset_conv3(offset))
        else:
            offset = self.lrelu(self.offset_conv2(offset))

        feat = self.lrelu(self.dcn_pack(x_ref, offset))
        
        return feat, offset


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
