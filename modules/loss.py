import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.module import Module

import utils.utils as utils



class Loss(nn.Module):
    def __init__(self, loss_type, if_overmask=False, stop_stage_id=4, if_train=True, thold=0.5, alpha=0.1):
        super(Loss, self).__init__()
        
        self.loss_type      = loss_type.lower()
        self.if_overmask    = if_overmask
        self.stop_stage_id  = stop_stage_id
        self.if_train       = if_train
        self.thold          = thold
        self.alpha          = alpha

        assert self.loss_type in ["chamfer", "multi_stage_regression_uploss", "LR_consistency", "multi_stage_regression_upsampleloss", "multi_stage_regression_upmaskloss"], "No such loss: {}".format(self.loss_type)
    
    
    def forward(self, pred_list=None, fusion_list=None, dense_list=None, sparse_list=None, left_mask_list=None, gt=None, weights=None, num_stage=None, down_func_name=None, down_scale=None, max_disp=None, sparse_mask_list=None, left_feature_map_all=None, right_feature_map_all=None, left_detail_list=None, right_detail_list=None, right_mask_list=None):
        """function: multiple stage chamfer loss
        args:
            pred_list:             the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            fusion_list:           the list of disparity map after fusion, N*H/s*W/s;
            dense_list:            the list of initial dense disparity map, which is the upsampled version from low layer, N*H/s*W/s;
            sparse_list:           the list of sparse disparity map, N*H/s*W/s;
            left_mask_list:        the list of binary mask representing the fine-grained areas;
            gt:                    the ground truth, N*H*W;
            weights:               weights for loss in each scale space;
            num_stage:             total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;
            down_func_name:        the name of downsampling fucntion;
            down_scale:            scale size of each downsampling;
            max_disp:              the maximum disparity value;
            sparse_mask_list:      the generated soft mask for fusion;
            left_feature_map_all:  the left feature maps;
            right_feature_map_all: the right feature maps;
            left_detail_list:      the generated details from left view;
            right_detail_list:     the generated details from right view;
            right_mask_list:       the list of binary mask representing the fine-grained areas;
        """
        if self.loss_type == "chamfer" :
            return self.multi_stage_chamfer_loss(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, gt, weights, num_stage, down_func_name, down_scale, max_disp, sparse_mask_list=sparse_mask_list)
        elif self.loss_type == "multi_stage_regression_uploss" :
            return self.multi_stage_regression_Uploss(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, gt, weights, num_stage, down_func_name, down_scale, max_disp, sparse_mask_list=sparse_mask_list)
        elif self.loss_type == "multi_stage_regression_upmaskloss" :
            return self.multi_stage_regression_UpMaskloss(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, gt, weights, num_stage, down_func_name, down_scale, max_disp, sparse_mask_list=sparse_mask_list, left_detail_list=left_detail_list, right_detail_list=right_detail_list, right_mask_list=right_mask_list,
                left_feature_map_all=left_feature_map_all, right_feature_map_all=right_feature_map_all)
        elif self.loss_type == "LR_consistency" :
            return self.LR_consistency_loss(pred_list, left_feature_map_all, right_feature_map_all, weights, num_stage, down_func_name, down_scale, max_disp)
        elif self.loss_type == "self_training" :
            return self.test_loss_func(pred_list[0], gt, max_disp)
        elif self.loss_type == "multi_stage_regression_upsampleloss" :
            return self.multi_stage_regression_UpSampleloss(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, gt, weights, num_stage, down_func_name, down_scale, max_disp, sparse_mask_list=sparse_mask_list)
        
        
    def sparseChamfer(self, x, y, down_ratio):
        """function: compute the chamfer distance
        args:
            x:              B*1*H*W;
            y:              B*1*sH*sW;
            down_ratio:     the current total downsampling size;
        return:
            chmafer distance
        """
        b,c,h,w = x.shape
        down_ratio = int(down_ratio)
        y = F.unfold(y, kernel_size=(down_ratio,down_ratio), stride=(down_ratio,down_ratio)).view( b, down_ratio*down_ratio, h, w )     # B*ss*H*W
        
        mask = y==0
        chamfer_dist = torch.pow(x-y,2)
        chamfer_dist = chamfer_dist + mask*1e6
        
        return torch.min(chamfer_dist, dim=1)[0]
        
        
    def multi_stage_chamfer_loss(self, pred_list, fusion_list, dense_list, sparse_list, left_mask_list, gt, weights, num_stage, down_func_name, down_scale, max_disp, sparse_mask_list=None):
        """function: multiple stage chamfer loss
        args:
            pred_list:      the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            gt:             the ground truth, N*H*W;
            weights:        weights for loss in each scale space;
            num_stage:      total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;;
            down_func_name: the name of downsampling fucntion
            down_scale:     scale size of each downsampling
            max_disp:       the maximum disparity value
        """
        assert (pred_list[-1].size()[-2], pred_list[-1].size()[-1]) == (gt.size()[-2],gt.size()[-1]), "the size of predcited disparity map is not equal to the size of groundtruth."
        tot_loss = 0.
        gt_list = []
        loss_list = []
        for stage_id in range(num_stage) :
            pred = pred_list[stage_id]
            
            if stage_id+1 < num_stage : # we do not need to interpolate the gt with original resolution
                down_size = down_scale**(num_stage-stage_id-1)
                if down_func_name in ["bilinear", "bicubic"] :
                    cur_gt = F.interpolate(gt.unsqueeze(1) / down_size, scale_factor=1/down_size, mode=down_func_name).squeeze(1)
                elif down_func_name=="max" :
                    cur_gt = F.max_pool2d(gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                elif down_func_name=="min" :
                    tmp_gt = gt*(gt>0) + 1e6*(gt==0)
                    cur_gt = -F.max_pool2d(-tmp_gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                else :
                    raise Exception("down_func_name should be bilinear or max, but current it is {}")
            else :
                cur_gt = gt
                down_size = 1.
            gt_list.append(cur_gt)
            
            if stage_id == 0 :
                error = self.sparseChamfer(pred.unsqueeze(1)*down_size, gt.unsqueeze(1), down_size)
                error = torch.sqrt(error+1e-6)
                mask = error<100
                disp_loss = MyHubeLoss.apply(error[mask], 1, 2)
                tot_loss += disp_loss * weights[stage_id]
                loss_list.append(disp_loss)
                
            else :
                dense = dense_list[stage_id-1]
                sparse = sparse_list[stage_id-1]
                fusion = fusion_list[stage_id-1]
                
                dense_error = self.sparseChamfer(dense.unsqueeze(1)*down_size, gt.unsqueeze(1), down_size)
                dense_error = torch.sqrt(dense_error+1e-6)
                mask = dense_error<100
                dense_disp_loss = MyHubeLoss.apply(dense_error[mask], 1, 2)
                
                sparse_error = self.sparseChamfer(sparse.unsqueeze(1)*down_size, gt.unsqueeze(1), down_size)
                sparse_error = torch.sqrt(sparse_error+1e-6)
                mask = (sparse_error<100) & (left_mask_list[stage_id-1] == 1)
                sparse_disp_loss = MyHubeLoss.apply(sparse_error[mask], 1, 2)
                
                fusion_error = self.sparseChamfer(fusion.unsqueeze(1)*down_size, gt.unsqueeze(1), down_size)
                fusion_error = torch.sqrt(fusion_error+1e-6)
                mask = fusion_error<100
                fusion_disp_loss = MyHubeLoss.apply(fusion_error[mask], 1, 2)
                
                pred_error = self.sparseChamfer(pred.unsqueeze(1)*down_size, gt.unsqueeze(1), down_size)
                pred_error = torch.sqrt(pred_error+1e-6)
                mask = pred_error<100
                pred_disp_loss = MyHubeLoss.apply(pred_error[mask], 1, 2)
                
                # print((left_mask_list[stage_id-1]).shape, sparse_mask_list[stage_id-1].shape)
                
                loss_list.append(dense_disp_loss)
                loss_list.append(sparse_disp_loss)
                loss_list.append( sparse_mask_list[stage_id-1][ left_mask_list[stage_id-1]==1 ].mean() )
                loss_list.append(fusion_disp_loss)
                loss_list.append(pred_disp_loss)
                
                tot_loss += (pred_disp_loss*0.5 + dense_disp_loss*0.1 + sparse_disp_loss*0.2*1/(10+stage_id*3.75) + fusion_disp_loss*0.2) * weights[stage_id]
                # tot_loss += (pred_loss*0.5 + dense_loss*0.1 + sparse_loss*0.2 + fusion_loss*0.2) * weights[stage_id]
                
        return gt_list, pred_list, tot_loss, loss_list
        
        
    def multi_stage_regression_Uploss(self, pred_list, fusion_list=None, dense_list=None, sparse_list=None, left_mask_list=None, gt=None, weights=None, num_stage=None, down_func_name=None, down_scale=None, max_disp=None, sparse_mask_list=None):
        """function: multiple stage loss
        args:
            pred_list:      the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            gt:             the ground truth, N*H*W;
            weights:        weights for loss in each scale space;
            num_stage:      total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;;
            down_func_name: the name of downsampling fucntion
            down_scale:     scale size of each downsampling
            max_disp:       the maximum disparity value
        """
        assert (pred_list[-1].size()[-2], pred_list[-1].size()[-1]) == (gt.size()[-2],gt.size()[-1]), "the size of predcited disparity map is not equal to the size of groundtruth."
        tot_loss = 0.
        gt_list = []
        loss_list = []
        for stage_id in range(num_stage) :
            pred = pred_list[stage_id]
            
            if stage_id+1 < num_stage : # we do not need to interpolate1 the gt with original resolution
                down_size = down_scale**(num_stage-stage_id-1)
                if down_func_name in ["bilinear", "bicubic"] :
                    cur_gt = F.interpolate(gt.unsqueeze(1) / down_size, scale_factor=1/down_size, mode=down_func_name).squeeze(1)
                elif down_func_name=="max" :
                    cur_gt = F.max_pool2d(gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                elif down_func_name=="min" :
                    tmp_gt = gt*(gt>0) + 1e6*(gt==0)
                    cur_gt = -F.max_pool2d(-tmp_gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                else :
                    raise Exception("down_func_name should be bilinear or max, but current it is {}")
            else :
                cur_gt = gt
                down_size = 1.
            gt_list.append(cur_gt)
            mask = (cur_gt < max_disp/down_size) & (cur_gt > 0)
            
            # if training set is dense disparity map after completion
            if self.if_overmask :
                mask[:, :int(108//down_size), :] = 0
            
            if stage_id == 0 or stage_id>=self.stop_stage_id :
                depth_loss = F.smooth_l1_loss(pred[mask] * down_size, cur_gt[mask] * down_size, reduction='mean')
                # depth_loss = MyLoss2Function.apply(pred[mask], cur_gt[mask], 1/down_size, 2/down_size)
                tot_loss += depth_loss * weights[stage_id]
                loss_list.append(depth_loss)
                
            else :
                dense = dense_list[stage_id-1]
                sparse = sparse_list[stage_id-1]
                fusion = fusion_list[stage_id-1]
                
                left_mask = (left_mask_list[stage_id-1] == 1)
                whole_mask = left_mask * mask
                
                # dense_loss = MyLoss2Function.apply(dense[mask], cur_gt[mask], 1/down_size, 2/down_size)
                # sparse_loss = MyLoss2Function.apply(sparse[whole_mask], cur_gt[whole_mask], 1/down_size, 2/down_size)
                # fusion_loss = MyLoss2Function.apply(fusion[mask], cur_gt[mask], 1/down_size, 2/down_size)
                # pred_loss = MyLoss2Function.apply(pred[mask], cur_gt[mask], 1/down_size, 2/down_size)
                
                dense_loss = F.smooth_l1_loss(dense[mask] * down_size, cur_gt[mask] * down_size, reduction='mean')
                sparse_loss = F.smooth_l1_loss(sparse[whole_mask] * down_size, cur_gt[whole_mask] * down_size, reduction='mean')
                fusion_loss = F.smooth_l1_loss(fusion[mask] * down_size, cur_gt[mask] * down_size, reduction='mean')
                pred_loss = F.smooth_l1_loss(pred[mask] * down_size, cur_gt[mask] * down_size, reduction='mean')
                
                
                
                loss_list.append(dense_loss)
                loss_list.append(sparse_loss)
                loss_list.append( sparse_mask_list[stage_id-1][left_mask].mean() )
                loss_list.append(fusion_loss)
                loss_list.append(pred_loss)
                
                tot_loss += (pred_loss*0.5 + dense_loss*0.1 + sparse_loss*0.2*1/(10+stage_id*3.75) + fusion_loss*0.2) * weights[stage_id]
                # tot_loss += (pred_loss*0.5 + dense_loss*0.1 + sparse_loss*0.2 + fusion_loss*0.2) * weights[stage_id]
                
        return gt_list, pred_list, tot_loss, loss_list
    
    
    def focal_loss(self, pt, gt, gamma=2, alpha=0.8):
        loss = - alpha * (1-pt)**gamma * gt * torch.log(pt+0.00001) - (1-alpha) * pt**gamma * (1-gt) * torch.log(1-pt+0.00001)
        return torch.mean(loss)
    
    def dice_loss(self, x, gt, smooth=1):
        N = gt.size(0)
        x_flat  = x.view(N, -1)
        gt_flat = gt.view(N, -1)
        
        intersection = x_flat * gt_flat
        
        loss = 2 * (intersection.sum(1) + smooth) / (x_flat.sum(1) + gt_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        
        return loss
    
    def mask_l1_loss(self, x, gt):
        valid_mask = gt>0.1
        loss = F.smooth_l1_loss(x[valid_mask], gt[valid_mask], reduction='mean')
        return loss

    def binary(self, x):
        x[x <= self.thold] = 0.
        x[x >  self.thold] = 1.
        # x = F.threshold(x, 0, 0)
        # x = -F.threshold(-x, -0.00001, -1)
        return x
        
    def multi_stage_regression_UpMaskloss(self, pred_list, fusion_list=None, dense_list=None, sparse_list=None, left_mask_list=None, gt=None, weights=None, num_stage=None, down_func_name=None, down_scale=None, max_disp=None, sparse_mask_list=None, left_feature_map_all=None, right_feature_map_all=None, left_detail_list=None, right_detail_list=None, right_mask_list=None):
        """function: multiple stage loss
        args:
            pred_list:             the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            fusion_list:           the fused result of dense prediction and sparse prediction;
            dense_list:            the result of dense prediction;
            sparse_list:           the result of sparse prediction;
            left_mask_list:        the pre-computed lost details of left view;
            gt:                    the ground truth, N*H*W;
            weights:               weights for loss in each scale space;
            num_stage:             total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;
            down_func_name:        the name of downsampling fucntion;
            down_scale:            scale size of each downsampling;
            max_disp:              the maximum disparity value;
            sparse_mask_list:      the generated soft mask for fusion;
            left_feature_map_all:  the left feature maps from adjacent layers for the computation of lost details;
            right_feature_map_all: the right feature maps from adjacent layers for the computation of lost details;
            left_detail_list:      the lost details of left view;
            right_detail_list:     the lost details of right view;
            right_mask_list:       the pre-computed lost details of right view;
        """
        assert (pred_list[-1].size()[-2], pred_list[-1].size()[-1]) == (gt.size()[-2],gt.size()[-1]), "the size of predcited disparity map is not equal to the size of groundtruth."
        tot_loss = 0.
        gt_list = []
        loss_list = []
        for stage_id in range(num_stage) :
            pred = pred_list[stage_id]
            
            if stage_id+1 < num_stage : # we do not need to interpolate1 the gt with original resolution
                down_size = down_scale**(num_stage-stage_id-1)
                if down_func_name in ["bilinear", "bicubic"] :
                    cur_gt = F.interpolate(gt.unsqueeze(1) / down_size, scale_factor=1/down_size, mode=down_func_name).squeeze(1)
                elif down_func_name=="max" :
                    cur_gt = F.max_pool2d(gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                elif down_func_name=="min" :
                    tmp_gt = gt*(gt>0) + 1e6*(gt==0)
                    cur_gt = -F.max_pool2d(-tmp_gt.unsqueeze(1) / down_size, down_size, down_size, 0, 1, False, False).squeeze(1)
                else :
                    raise Exception("down_func_name should be bilinear or max, but current it is {}")
            else :
                cur_gt = gt
                down_size = 1.
            gt_list.append(cur_gt)

            if stage_id == 0 or stage_id>=self.stop_stage_id :
                pass
                
            else :
                left_mask    = left_mask_list[stage_id-1]
                right_mask   = right_mask_list[stage_id-1]
                left_detail  = left_detail_list[stage_id-1]
                right_detail = right_detail_list[stage_id-1]
                
                # left_cur_fea  = left_feature_map_all[ (stage_id-1)*2 ].permute(0,2,3,1)
                # left_pre_fea  = left_feature_map_all[ (stage_id-1)*2+1 ].permute(0,2,3,1)
                # right_cur_fea = right_feature_map_all[ (stage_id-1)*2 ].permute(0,2,3,1)
                # right_pre_fea = right_feature_map_all[ (stage_id-1)*2+1 ].permute(0,2,3,1)
                
                # left_detail_pos = torch.zeros_like(left_detail, dtype=torch.bool)
                # left_detail_pos[ left_detail>self.thold ] = 1
                # right_detail_pos = torch.zeros_like(right_detail, dtype=torch.bool)
                # right_detail_pos[ right_detail>self.thold ] = 1
                # N,H,W = left_mask.size()
                # left_loss = torch.mean(torch.sum(left_detail, (1,2))) / (H*W) - self.alpha * F.mse_loss(left_cur_fea[left_detail_pos], left_pre_fea[left_detail_pos])
                # right_loss = torch.mean(torch.sum(right_detail, (1,2))) / (H*W) - self.alpha * F.mse_loss(right_cur_fea[right_detail_pos], right_pre_fea[right_detail_pos])
                # loss_list.append(left_loss)
                # loss_list.append(right_loss)

                # tot_loss += (left_loss+right_loss) * weights[stage_id-1]

                if self.if_train==False :
                    left_detail  = self.binary(left_detail)
                    right_detail = self.binary(right_detail)
                
                left_fl    = self.focal_loss(left_detail, left_mask, gamma=2, alpha=0.5)
                right_fl   = self.focal_loss(right_detail, right_mask, gamma=2, alpha=0.5)
                loss_list.append(left_fl)
                loss_list.append(right_fl)
                
                left_l1  = self.mask_l1_loss(left_detail, left_mask)
                right_l1 = self.mask_l1_loss(right_detail, right_mask)
                loss_list.append(left_l1)
                loss_list.append(right_l1)

                tot_loss += (left_fl + right_fl + 3*left_l1 + 3*right_l1) * weights[stage_id-1]
                
        return gt_list, pred_list, tot_loss, loss_list
    

    def multi_stage_regression_UpSampleloss(self, pred_list, fusion_list=None, dense_list=None, sparse_list=None, left_mask_list=None, gt=None, weights=None, num_stage=None, down_func_name=None, down_scale=None, max_disp=None, sparse_mask_list=None):
        """function: multiple stage loss
        args:
            pred_list:      the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            gt:             the ground truth, N*H*W;
            weights:        weights for loss in each scale space;
            num_stage:      total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;;
            down_func_name: the name of downsampling fucntion
            down_scale:     scale size of each downsampling
            max_disp:       the maximum disparity value
        """
        assert (pred_list[-1].size()[-2], pred_list[-1].size()[-1]) == (gt.size()[-2],gt.size()[-1]), "the size of predcited disparity map is not equal to the size of groundtruth."
        tot_loss = 0.
        gt_list = []
        loss_list = []
        for stage_id in range(num_stage) :
            pred = pred_list[stage_id]
            
            if stage_id+1 < num_stage :
                down_size = down_scale**(num_stage-stage_id-1)
                cur_pred = F.interpolate(pred.unsqueeze(1) * down_size, scale_factor=down_size, mode=down_func_name).squeeze(1)
            else :
                cur_pred = pred
            pred_list.append(pred)
            
            cur_gt = gt
            gt_list.append(cur_gt)
            
            mask = (cur_gt < max_disp) & (cur_gt > 0)
            loss = F.smooth_l1_loss(cur_pred[mask], cur_gt[mask], reduction='mean')
            tot_loss += loss * weights[stage_id]
            loss_list.append(loss)
            
        return gt_list, pred_list, tot_loss, loss_list
    
    
    def LR_consistency_loss(self, pred_list, left_feature_map_all, right_feature_map_all, weights, num_stage, down_func_name, down_scale, max_disp):
        """function: multiple stage loss
        args:
            pred_list:              the list of predicted multi-scale disparity maps, each size N*H/s*W/s;
            left_feature_map_all:   list of feature maps from left view,each N*Cs*H/s*W/s;
            right_feature_map_all:  list of feature maps from right view, each N*Cs*H/s*W/s;
            weights:                weights for loss in each scale space;
            num_stage:              total number of stages, the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;;
            down_func_name:         the name of downsampling fucntion
            down_scale:             scale size of each downsampling
            max_disp:               the maximum disparity value
        """
        tot_loss = 0.
        gt_list = []
        loss_list = []
        for stage_id in range(num_stage) :
            pred = pred_list[stage_id]
            
            warp_right_fea = utils.warp(pred_list[stage_id], right_feature_map_all["stage{}".format(stage_id)]).unsqueeze(2)
            
            diff = torch.pow(left_feature_map_all["stage{}".format(stage_id)] - warp_right_fea, 2)
            phmt = torch.mean(torch.sum(diff,dim=1))
            loss_list.append(phmt)
            
            tot_loss += phmt * weights[stage_id]
            
        return gt_list, pred_list, tot_loss, loss_list


def test_loss_func(pred, gt, max_disp):
    """
    """
    assert (pred.size()[-2], pred.size()[-1]) == (gt.size()[-2],gt.size()[-1]), "the size of predcited disparity map is not equal to the size of groundtruth."
    # max_disp = 192
    batch_size, height, width = pred.size()
    mask = (gt < max_disp) & (gt > 0)
    error_map = torch.where((torch.abs(pred[mask] - gt[mask])<3) | (torch.abs(pred[mask] - gt[mask])<0.05*gt[mask]), torch.ones(1,dtype=torch.float32,device=pred.device).cuda(), torch.zeros(1,dtype=torch.float32,device=pred.device).cuda())
    loss_3 = 100 - torch.sum(error_map)/torch.sum(mask)*100
    epe = torch.mean(torch.abs(pred[mask]-gt[mask]))
    return epe, loss_3








