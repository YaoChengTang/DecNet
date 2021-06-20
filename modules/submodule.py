import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-12
group_dim_32 = 32
pramid_dim_8 = 8
group_norm_num = 32

class Conv2dUnit(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 relu=True, lrelu=False, bn=True, bn_momentum=0.1, gn=False, gn_group=32, **kwargs):
        super(Conv2dUnit, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                              dilation=dilation, bias=(not bn and not gn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu
        self.lrelu = lrelu
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        elif self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        if self.lrelu:
            x= F.leaky_relu(x, negative_slope=0.1, inplace=True)
        return x
        

class Deconv2dUnit(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.
       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu
       Notes:
           Default momentum for batch normalization is set to be 0.01,
       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, gn=False, gn_group=32, **kwargs):
        super(Deconv2dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2, 3], "the stride({}) should be in [1,2,3]".format(stride)
        self.stride = stride
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn and not gn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu
    
    def forward(self, x):
        x = self.conv(x)
        # if self.stride == 3:
            # h, w = list(x.size())[2:]
            # y = y[:, :, :3 * h, :3 * w].contiguous()
        if self.bn is not None:
            x = self.bn(x)
        elif self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv3dUnit(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu
    Notes:
        Default momentum for batch normalization is set to be 0.01,
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, gn=False, gn_group=32, init_method="xavier", **kwargs):
        super(Conv3dUnit, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2, 3], "the stride({}) should be in [1,2,3]".format(stride)
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        elif self.gn is not None:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Deconv3dUnit(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.
       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu
       Notes:
           Default momentum for batch normalization is set to be 0.01,
       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, gn=False, gn_group=32, init_method="xavier", **kwargs):
        super(Deconv3dUnit, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2, 3], "the stride({}) should be in [1,2,3]".format(stride)
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.gn = nn.GroupNorm(gn_group, out_channels) if gn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        elif self.gn is not None:
            x = self.gn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x



class Deconv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 relu=True, bn=True, bn_momentum=0.1, gn=False, gn_group=32):
        super(Deconv2dBlock, self).__init__()
        
        self.deconv = Deconv2dUnit(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                   bn=bn, relu=relu, bn_momentum=bn_momentum, gn=gn, gn_group=gn_group)
        
        self.conv = nn.Sequential(
                                    Conv2dUnit(out_channels*2, out_channels, 3, 1, padding=1),
                                    Conv2dUnit(out_channels, out_channels, 3, 1, padding=1),
                                )
    
    def forward(self, x_pre, x):
        x_up = self.deconv(x)
        x = torch.cat((x_up, x_pre), dim=1)
        x = self.conv(x)
        return x, x_up
        
        

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, relu=True, downsample=None,
                 bn=True, bn_momentum=0.1, gn=False, gn_group=32):
        super(BasicBlock, self).__init__()
        
        self.conv1 = Conv2dUnit(in_channels, out_channels, 3, stride=3, padding=0)
        
        self.conv2 = convbn(in_channels, out_channels, 3, 1, pad, dilation)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



class ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ImagePool, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = Conv2dUnit(in_ch, out_ch, 1, stride=1, padding=0, dilation=1, bn=False)
    
    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        
        return h


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        
        self.stages = nn.Module()
        self.stages.add_module("c0", Conv2dUnit(in_ch, out_ch, 1, stride=1, padding=0, dilation=1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                Conv2dUnit(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
            )
        # self.stages.add_module("imagepool", ImagePool(in_ch, out_ch))
        # nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, groups=inplanes, bias=False),
        # nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
    
    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)



class FeatExtNetChannelPlus(nn.Module):
    def __init__(self, base_channels, num_stage=4, down_scale=3):
        super(FeatExtNetChannelPlus, self).__init__()
        print("using FeatExtNetChannelPlus")
        assert down_scale in [3,4], "down_scale should be in [3,4]"
        
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.down_scale = down_scale
        
        self.conv0 = nn.Sequential(
            Conv2dUnit(3, base_channels, 3, 1, padding=1),
            Conv2dUnit(base_channels, base_channels, 3, 1, padding=1),
        )
        self.addition_trans0 = Conv2dUnit(base_channels, base_channels, 1, stride=1, padding=0, dilation=1)
        # self.out0 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
        self.out_channels = [base_channels]
        
        if num_stage>1 :
            self.conv1 = nn.Sequential(
                                        Conv2dUnit(base_channels, base_channels * down_scale, 3, stride=down_scale, padding=1),
                                        Conv2dUnit(base_channels * down_scale, base_channels * down_scale, 3, 1, padding=1),
                                        Conv2dUnit(base_channels * down_scale, base_channels * down_scale, 3, 1, padding=1),
                                    )
            self.addition_trans1 = Conv2dUnit(base_channels * down_scale, base_channels * down_scale, 1, stride=1, padding=0, dilation=1)
            # self.out1 = nn.Conv2d(base_channels * down_scale, base_channels * down_scale, 1, bias=False)
            self.out_channels.append(base_channels * down_scale)
            self.deconv1 = Deconv2dBlock(base_channels * down_scale, base_channels, kernel_size=3, stride=3)
            
            if num_stage>2 :
                self.conv2 = nn.Sequential(
                                            Conv2dUnit(base_channels * down_scale, base_channels * down_scale**2, 3, stride=3, padding=1),
                                            Conv2dUnit(base_channels * down_scale**2, base_channels * down_scale**2, 3, 1, padding=1),
                                            Conv2dUnit(base_channels * down_scale**2, base_channels * down_scale**2, 3, 1, padding=1),
                                        )
                self.addition_trans2 = Conv2dUnit(base_channels * down_scale**2, base_channels * down_scale**2, 1, stride=1, padding=0, dilation=1)
                # self.out2 = nn.Conv2d(base_channels * down_scale**2, base_channels * down_scale**2, 1, bias=False)
                self.out_channels.append(base_channels * down_scale**2)
                self.deconv2 = Deconv2dBlock(base_channels * down_scale**2, base_channels * down_scale, kernel_size=3, stride=3)
                
                if num_stage>3 :
                    self.conv3_1 = Conv2dUnit(base_channels * down_scale**2, base_channels * down_scale**3, 3, stride=3, padding=1)
                    self.conv3_2 = nn.Sequential(
                                                Conv2dUnit(base_channels * down_scale**3, base_channels * down_scale**3, 3, 1, padding=1),
                                                Conv2dUnit(base_channels * down_scale**3, base_channels * down_scale**3, 3, 1, padding=1),
                                            )
                    self.addition_ctx_collection = nn.Sequential(
                                                                ASPP(base_channels * down_scale**3, base_channels * down_scale**3, [4,8,12]),
                                                                Conv2dUnit(4 * base_channels * down_scale**3, base_channels * down_scale**3, 1, stride=1, padding=0, dilation=1),
                                                            )
                    self.addition_fusion = Conv2dUnit(2 * base_channels * down_scale**3, base_channels * down_scale**3, 1, stride=1, padding=0, dilation=1)
                    # self.addition_fusion = nn.Sequential(
                                                        # Conv2dUnit(2 * base_channels * down_scale**3, base_channels * down_scale**3, 1, stride=1, padding=0, dilation=1),
                                                        # Conv2dUnit(base_channels * down_scale**3, 32, 1, stride=1, padding=0, dilation=1),
                                                        # )
                    # self.out3 = nn.Conv2d(base_channels * down_scale**3, base_channels * down_scale**3, 1, bias=False)
                    self.out_channels.append(base_channels * down_scale**3)
                    # self.out_channels.append(32)
                    self.deconv3 = Deconv2dBlock(base_channels * down_scale**3, base_channels * down_scale**2, kernel_size=3, stride=3)
                    # self.deconv3 = Deconv2dBlock(32, base_channels * down_scale**2, kernel_size=3, stride=3)
                    
                    if num_stage>4 :
                        raise Exception("only accept num_stage<=4")
        
        self.out_channels = self.out_channels[::-1]
        print("Feature Channel: {}".format(self.out_channels))
        
        
    def forward(self, x):
        outputs = {}
        conv0 = self.conv0(x)       # N * C * H * W
        if self.num_stage>1 :
            conv1 = self.conv1(conv0)   # N * 3C * H/3 * W/3
            if self.num_stage>2 :
                conv2 = self.conv2(conv1)   # N * 9C * H/9 * W/9
                
                if self.num_stage>3 :
                    conv3_1 = self.conv3_1(conv2)   # N * 27C * H/27 * W/27
                    conv3_2 = self.conv3_2(conv3_1)
                    conv3_ctx = self.addition_ctx_collection(conv3_1)
                    conv3 = self.addition_fusion( torch.cat((conv3_2,conv3_ctx),dim=1) )
                    outputs["stage0"] = conv3
                    
                    res, pre_up = self.deconv3( self.addition_trans2(conv2), conv3 )   # N * 9c * H/9 * W/9
                else :
                    res = conv2
                outputs["stage1"] = res
                
                res, pre_up = self.deconv2( self.addition_trans1(conv1), res )   # N * 3c * H/3 * W/3
            else :
                res = conv1
            outputs["stage2"] = res
            
            res, pre_up = self.deconv1( self.addition_trans0(conv0), res )   # N * c * H * W
        else :
            res = conv0
        outputs["stage3"] = res
        
        return outputs



class GenerateSparseMask(nn.Module):
    def __init__(self, in_channels, down_scale):
        super(GenerateSparseMask, self).__init__()
        
        self.deconv = nn.Sequential(
                                Deconv2dUnit(in_channels*down_scale, 8, kernel_size=3, stride=3, padding=0, output_padding=0,
                                   bn=False, relu=True, bn_momentum=0.01, gn=False, gn_group=32),
                                Conv2dUnit(8, 3, 3, stride=1, padding=1, relu=False, bn=True)
                                )
        self.conv_sub = nn.Sequential(
                                Conv2dUnit(in_channels, 8, kernel_size=3, stride=1, padding=1,
                                   bn=False, relu=True, bn_momentum=0.01, gn=False, gn_group=32),
                                Conv2dUnit(8, 3, 3, stride=1, padding=1, relu=False, bn=True)
                                )
        self.conv = nn.Sequential(
                                Conv2dUnit(3, 3, 3, stride=1, padding=1, relu=False, bn=True),
                                Conv2dUnit(3, 1, 1, stride=1, padding=0, relu=False, bn=True)
                                )
        
    def forward(self, cur_fea, pre_fea) :
        pre_fea = self.deconv(pre_fea)
        cur_fea = self.conv_sub(cur_fea)
        res_info = torch.pow(cur_fea-pre_fea, 2)
        # res_info = torch.abs(cur_fea - pre_fea) - 1
        detail = self.conv(res_info)
        return detail.squeeze(1), cur_fea, pre_fea



def get_disp_samples(max_dis, feature_map, stage_id=0, disprity_map=None, step=1, samp_num=9, sample_spa_size=None) :
    """function: get the sampled disparities
    args:
        max_dis: the maximum disparity;
        feature map: left or right feature map, N*C*H*W;
        disprity_map: if it is not the first stage, we need disparity map to be the sampling center in new cost volume, N*H*W;
        step: the step size between each samples, where -1 represents the traditional sampling from 0 to max_disp;
        samp_num: the total number of samples;
    return:
        the sampled disparities for each pixel, N*S*H*W;
    """
    # print("disprity_map: {}".format(disprity_map.size()))
    batch_size, channels, height, width = feature_map.size()
    if disprity_map is None or step==-1 or stage_id==0 :
        disp_samples = torch.arange(max_dis, dtype=feature_map.dtype, device=feature_map.device).expand(batch_size,height,width,-1).permute(0,3,1,2)
    else :
        # # get the range only from one pixel
        # lower_bound = disprity_map-(samp_num/2)*step
        # upper_bound = disprity_map+(samp_num/2)*step
        # lower_bound = lower_bound.clamp_(min=0.0)
        # upper_bound = upper_bound.clamp_(max=max_dis)
        
        # get the range from the pixel and its neighbors
        if sample_spa_size is None :
            kernel_size = 3 if stage_id==1 else 5 if stage_id==2 else 7
        else :
            kernel_size = sample_spa_size
        lower_bound = torch.abs( torch.max_pool2d(-disprity_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)//2)) )
        upper_bound = torch.max_pool2d(disprity_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)//2) )
        modified_disp_range = (samp_num*step - (upper_bound-lower_bound)).clamp(min=0) / 2
        lower_bound = (lower_bound - modified_disp_range).clamp(min=0, max=max_dis)
        upper_bound = (upper_bound + modified_disp_range).clamp(min=0, max=max_dis)
        
        new_step = (upper_bound-lower_bound) / (samp_num-1)
        disp_samples = lower_bound.unsqueeze(1) + (torch.arange(0, samp_num, device=disprity_map.device,
                                                                dtype=disprity_map.dtype, requires_grad=False).reshape(1, -1, 1, 1) * new_step.unsqueeze(1))
        
        # disp_samples = []
        # for i in np.arange(samp_num) :
            # disp_samples.append(lower_bound+i*step)
        # disp_samples = torch.stack(disp_samples,dim=1)
        
        # disp_samples = []
        # for i in np.arange(-(samp_num//2)*step, samp_num//2*step, step) :
            # disp_samples.append(disprity_map+i)
        # disp_samples = torch.stack(disp_samples,dim=1)
        # disp_samples = disp_samples.clamp_(min=0,max=max_dis-1)
    # print("disp_samples: {}".format(disp_samples.size()))
    return disp_samples



class GetCostVolume(nn.Module):
    """forward: compute the cost volume with warped features
    args:
        left_feature_map: feature maps from left view, N*C*H*W;
        right_feature_map: feature maps from right view, N*C*H*W;
        disp_samples: if ope_type is "homgrp", it needs the sampled disparities for each pixel, N*S*H*W;
        max_disp: if ope_type is "shift", it needs the maximum value of disparity, eg, 192;
    return:
        cost volume, N*C*S*H*W;
    """
    def __init__(self, warp_ope="homgrp", cost_func="ssd"):
        """compute the cost volume either by shifting ot by homography mapping
        args:
            warp_ope: the type of operation to warp the feature maps, "homgrp" or "shift", str;
            cost_func: the type of function to compute the cost, "ssd", "cat" or "cor", str;
        """
        super(GetCostVolume, self).__init__()
        self.warp_ope = warp_ope
        self.cost_func = cost_func
        assert self.cost_func in ["ssd","cor","cat"], "no such cost_func: {}".format(self.cost_func)
        
    def get_warped_feats_by_shift(self, left_feature_map, right_feature_map, max_disp) :
        """fucntion: build the warped feature volume. This version of the construction method takes up a little more memory.
        args:
            left_feature_map: feature maps from left view, N*C*H*W;
            right_feature_map: feature maps from right view, N*C*H*W;
            max_disp: maximum value of disparity, eg, 192;
        return:
            the warped feature volume, N*C*D*H*W;
        """
        batch_size, channels, height, width = right_feature_map.size()
        
        disp_samples = torch.arange(0.0, max_disp, device=right_feature_map.device)
        disp_samples = disp_samples.repeat(batch_size).view(batch_size, -1) # N*D
        
        x_coord = torch.arange(0.0, width, device=0).repeat(height).view(height, width)
        x_coord = torch.clamp(x_coord, min=0, max=width-1)
        x_coord = x_coord.expand(batch_szie, -1, -1) # N*H*W
        
        right_x_coord = x_coord.expand(max_disp,-1,-1,-1).permute([1,0,2,3]) # N*D*H*W
        right_x_coord = right_x_coord - disp_samples.unsqueeze(-1).unsqueeze(-1)
        right_right_x_coord_tmp = right_x_coord.unsqueeze(1) # N*1*D*H*W
        right_x_coord = torch.clamp(right_x_coord, min=0, max=max_disp-1)
        
        left_vol = left_feature_map.expand(max_disp,-1,-1,-1,-1).permute([1,2,0,3,4])
        right_vol = right_feature_map.expand(max_disp,-1,-1,-1,-1).permute([1,2,0,3,4])
        right_vol = torch.gather(right_vol, dim=4, index=right_x_coord.expand(channels,-1,-1,-1,-1).permute([1,0,2,3,4]).long()) # N*C*D*H*W
        right_vol = (1 - ((right_right_x_coord_tmp<0) + (right_right_x_coord_tmp>width-1)).float()) * (right_vol)
        
        return left_vol, right_vol
    
    def get_warped_feats_by_homgrp(self, left_feature_map, right_feature_map, disp_samples) :
        """fucntion: build the warped feature volume via homography.
        args:
            left_feature_map: feature maps from left view, N*C*H*W;
            right_feature_map: feature maps from right view, N*C*H*W;
            disp_samples: the sampled disparities for each pixel, N*S*H*W or N*S*1*1;
        return:
            the warped feature volume, N*C*S*H*W;
        """
        batch_size, channels, height, width = right_feature_map.size()
        disp_num = disp_samples.size()[1]
        
        with torch.no_grad() :
            pos_y, pos_x = torch.meshgrid([torch.arange(0, height, dtype=left_feature_map.dtype, device=left_feature_map.device),
                                           torch.arange(0, width, dtype=left_feature_map.dtype, device=left_feature_map.device)])  # (H, W)
            pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)
            pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)  # (B, S, H, W)
            
            coords_x = pos_x - disp_samples
            coords_x = coords_x / ((width - 1.0) / 2.0) - 1.0
            coords_y = pos_y / ((height - 1.0) / 2.0) - 1.0
            grid = torch.stack([coords_x, coords_y], dim=4)  # (B, S, H, W, 2)

        right_vol = F.grid_sample(right_feature_map, grid.view(batch_size, disp_num*height, width, 2), mode='bilinear',
                                  padding_mode='zeros').view(batch_size, channels, disp_num, height, width)  # (B, C, S, H, W)
        left_vol = left_feature_map.unsqueeze(2).repeat(1, 1, disp_num, 1, 1)  # (B, C, S, H, W)
        
        left_vol = left_vol.transpose(0, 1)  # (C, B, S, H, W)
        left_vol[:, pos_x < disp_samples] = 0
        left_vol = left_vol.transpose(0, 1) #(B, C, S, H, W)
        
        return left_vol, right_vol
    
    def cost_computation_cat(self, left_vol, right_vol) :
        """build the cost volume via concatenating the left volume and right volume
        """
        cost_vol = torch.cat((left_vol,right_vol), dim=1)
        return cost_vol
    
    def cost_computation_cor(self, left_vol, right_vol) :
        """build the cost volume via correlation between left volume and right volume
        """
        cost_vol = left_vol.mul_(right_vol)
        return cost_vol
    
    def cost_computation_ssd(self, left_vol, right_vol) :
        """build the cost volume via sum of square difference between left volume and right volume
        """
        volume_sum = left_vol + right_vol
        volume_sqr = left_vol.pow_(2) + right_vol.pow_(2)
        cost_vol = volume_sqr.div_(2).sub_( volume_sum.div_(2).pow_(2) )
        return cost_vol
    
    def forward(self, left_feature_map, right_feature_map, **kargs) :
        """function: compute the cost volume with warped features
        args:
            left_feature_map: feature maps from left view, N*C*H*W;
            right_feature_map: feature maps from right view, N*C*H*W;
            disp_samples: if ope_type is "homgrp", it needs the sampled disparities for each pixel, N*S*H*W;
            max_disp: if ope_type is "shift", it needs the maximum value of disparity, eg, 192;
        return:
            cost volume, N*C*S*H*W;
        """
        batch_size, channels, height, width = right_feature_map.size()
        if self.warp_ope=="homgrp" :
            disp_samples = kargs["disp_samples"]
            left_vol, right_vol = self.get_warped_feats_by_homgrp(left_feature_map, right_feature_map, disp_samples)
            
        elif self.warp_ope=="shift" :
            max_disp = kargs["max_disp"]
            left_vol, right_vol = self.get_warped_feats_by_shift(left_feature_map, right_feature_map, max_disp)
        else :
            raise Exception("No such warp operation: {}".format(self.warp_ope))
        
        if self.cost_func=="cat" :
            cost_vol = self.cost_computation_cat(left_vol, right_vol)
        elif self.cost_func=="cor" :
            cost_vol = self.cost_computation_cor(left_vol, right_vol)
        elif self.cost_func=="ssd" :
            cost_vol = self.cost_computation_ssd(left_vol, right_vol)
        else :
            raise Exception("No such cost computation function: {}".format(self.cost_func))
        
        return cost_vol



class DynamicUpsampling(nn.Module):
    def __init__(self, in_channels, down_scale) :
        super(DynamicUpsampling, self).__init__()
        self.down_scale = down_scale
        self.pad = nn.ReplicationPad2d(1)
        self.weight_learning = nn.Sequential(
                Conv2dUnit(in_channels*down_scale**2+1, down_scale**2*9, kernel_size=3, padding=1),
                Conv2dUnit(down_scale**2*9, down_scale**2*9, kernel_size=3, padding=1),
                Conv2dUnit(down_scale**2*9, down_scale**2*9, kernel_size=3, relu=False, padding=1)
                )
    
    def forward(self, disp_map, left_fea) :
        batch_size, height, width = disp_map.shape
        
        weights = torch.cat((disp_map.unsqueeze(1), F.unfold(left_fea, kernel_size=self.down_scale, stride=self.down_scale).view(batch_size,-1,height,width)), dim=1)
        weights = self.weight_learning(weights).view(batch_size,self.down_scale**2,9,height*width)
        weights = F.softmax(weights, dim=2)
        
        content = F.unfold( self.pad(disp_map.unsqueeze(1)), kernel_size=(3,3) ).unsqueeze(1)
        
        res = torch.sum(content*weights, dim=2).view(batch_size,self.down_scale**2,height,width)
        res = F.pixel_shuffle(res, self.down_scale) * self.down_scale
        
        return res.squeeze(1)



class SoftAttention(nn.Module):
    def __init__(self, in_channels, base_channels) :
        super(SoftAttention, self).__init__()
        self.conv = nn.Sequential(
                Conv2dUnit(in_channels, base_channels, kernel_size=3, padding=1),
                Conv2dUnit(base_channels, base_channels, kernel_size=3, padding=1),
                Conv2dUnit(base_channels, 1, kernel_size=3, relu=False, padding=1)
            )
    
    def forward(self, x) :
        x = self.conv(x)
        return F.sigmoid(x)



class CostRegNetNoDown(nn.Module):
    """forward: regulaize the cost volume
    args:
        x: cost volume, N*C*S*H*W;
    return:
        regularied cost volume, N*S*H*W;
    """
    def __init__(self, in_channels, base_channels, cost_func, down_scale=3):
        super(CostRegNetNoDown, self).__init__()
        self.cost_func = cost_func
        if self.cost_func == "cat" :
            self.conv_pre = nn.Conv3d(in_channels*2, in_channels, 1, stride=1, padding=0, bias=False)
        
        self.conv0 = nn.Sequential(
                                    Conv3dUnit(in_channels, in_channels, padding=1),
                                    Conv3dUnit(in_channels, in_channels, padding=1)
                                )
        
        # self.conv1 = nn.Sequential(
                                    # Deconv3dUnit(in_channels, in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                    # Conv3dUnit(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1),
                                    # Conv3dUnit(in_channels//2, in_channels, kernel_size=3, stride=2, padding=1)
                                # )
        self.conv1 = nn.Sequential(
                                    Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                                )
        
        self.conv2 = nn.Sequential(
                                    Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    Conv3dUnit(in_channels, 1, kernel_size=3, stride=1, padding=1, relu=False)
                                )
                                
        # self.conv3 = nn.Sequential(
                                    # Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    # Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                    # Conv3dUnit(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                                # )
    
    
    def forward(self, x):
        if self.cost_func=="cat" :
            x = self.conv_pre(x)
        
        output0 = self.conv0(x)
        
        output = self.conv1(output0) + output0
        
        # output = self.conv3(output) + output
        
        output = self.conv2(output)
        
        return output.squeeze(1)
        


class Refinement(nn.Module):
    """function: refienment the disparity map via regression
    args:
        left_fea: feature map from left view
        right_fea: feature map from right view after the warping
        dis_map: the initialization of disparity map
    return:
        refinement: the refiend disparity map
    """
    def __init__(self, in_channels, base_channels, stage_id=-1, down_scale=3):
        super(Refinement, self).__init__()
        if stage_id == 0 :
            self.conv = nn.Sequential(
                            Conv2dUnit(in_channels*2+1, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, 1, kernel_size=3, relu=False, bn=False, padding=1)
                        )
        elif stage_id == 1 :
            self.conv = nn.Sequential(
                            Conv2dUnit(in_channels*2+1, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, 1, kernel_size=3, relu=False, bn=False, padding=1)
                        )
        elif stage_id == 2 :
            self.conv = nn.Sequential(
                            Conv2dUnit(in_channels*2+1, in_channels, kernel_size=3, padding=2, dilation=2),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=4, dilation=4),
                            Conv2dUnit(in_channels, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=6, dilation=6),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, 1, kernel_size=3, relu=False, bn=False, padding=1)
                        )
        elif stage_id == 3 :
            self.conv = nn.Sequential(
                            Conv2dUnit(in_channels*2+1, in_channels, kernel_size=3, padding=3, dilation=3),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels, in_channels, kernel_size=3, padding=6, dilation=6),
                            Conv2dUnit(in_channels, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=9, dilation=9),
                            Conv2dUnit(in_channels//2, in_channels//2, kernel_size=3, padding=1),
                            Conv2dUnit(in_channels//2, 1, kernel_size=3, relu=False, bn=False, padding=1)
                        )
        
        
    def get_warped_feats_by_homgrp(self, right_feature_map, disp_samples) :
        """fucntion: build the warped feature volume via homography.
        args:
            left_feature_map: feature maps from left view, N*C*H*W;
            right_feature_map: feature maps from right view, N*C*H*W;
            disp_samples: the sampled disparities for each pixel, N*S*H*W or N*S*1*1;
        return:
            the warped feature volume, N*C*S*H*W;
        """
        batch_size, channels, height, width = right_feature_map.size()
        disp_num = disp_samples.size()[1]
        
        with torch.no_grad() :
            pos_y, pos_x = torch.meshgrid([torch.arange(0, height, dtype=right_feature_map.dtype, device=right_feature_map.device),
                                           torch.arange(0, width, dtype=right_feature_map.dtype, device=right_feature_map.device)])  # (H, W)
            pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)
            pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)  # (B, S, H, W)
            
            coords_x = pos_x - disp_samples
            coords_x = coords_x / ((width - 1.0) / 2.0) - 1.0
            coords_y = pos_y / ((height - 1.0) / 2.0) - 1.0
            grid = torch.stack([coords_x, coords_y], dim=4)  # (B, S, H, W, 2)
        
        warped_right_fea = F.grid_sample(right_feature_map, grid.view(batch_size, disp_num*height, width, 2), mode='bilinear',
                                         padding_mode='zeros').view(batch_size, channels, disp_num, height, width)  # (B, C, S, H, W)

        return warped_right_fea
    
    
    def forward(self, left_fea, right_fea, disp_map):
        """function: refienment the disparity map via regression
        args:
            left_fea: feature map from left view, N*C*H*W;
            right_fea: feature map from right view after the warping, N*C*H*W;
            disp_map: the initialization of disparity map, N*H*W;
        return:
            refinement: the refiend disparity map
        """
        warped_right_fea = self.get_warped_feats_by_homgrp(right_fea, disp_map.unsqueeze(1))
        warped_right_fea = warped_right_fea.squeeze(2)
        cat_input = torch.cat((left_fea,warped_right_fea,disp_map.unsqueeze(1)), dim=1)
        residual_disp = self.conv(cat_input).squeeze(1)
        res = disp_map + residual_disp
        return res, residual_disp
  


def disparity_regression(cost_vol, disp_samples):
    """function: regress the disparity according to the probability volume and the sampled disparities
    args:
        pro_vol: the probability volume, N*S*H*W;
        disp_samples: the sampled disparities, N*S*H*W;
    return :
        disparity map, N*H*W;
    """
    pro_vol = F.softmax(cost_vol, dim=1)
    pred = torch.sum(pro_vol*disp_samples, 1)

    return pred