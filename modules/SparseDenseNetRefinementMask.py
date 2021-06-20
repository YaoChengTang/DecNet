import os
import sys
import time
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .submodule import GetCostVolume, get_disp_samples, disparity_regression, SoftAttention, Refinement, DynamicUpsampling, CostRegNetNoDown, FeatExtNetChannelPlus, GenerateSparseMask
from .loss import test_loss_func, Loss
from .SparseMatching.modules.SpaMat import SpaMat
from .SparseVar.modules.SpaVar import SpaVar



class SparseDenseNetRefinementMask(nn.Module) :
    def __init__(self, max_disp=192, base_channels=8, num_stage=3, down_scale=3, step=[1,2,3], samp_num=8, sample_spa_size_list=[-1,3,3,3], down_func_name="bilinear", weights=[0.2,0.6,1.8], grad_method="detach", cost_func="cat", if_overmask=False, skip_stage_id=3, use_detail=False, thold=0.5, alpha=0.1) :
        """function: rebuild version of Baseline
        args:
            max_disp:               maximum disparity value, default 192;
            base_channels:          cardinality of following number of channels, eg, 3->base_channels->2*base_channels...;
            num_stage:              the number of pyramid's level, num_stage-1 is equal to the number of downsampling iteration;
            down_scale:             the scale size of each downsampling;
            step:                   the step size between sampled disparities at each stage, [1,2,3];
            samp_num:               the total number of sampled disparities;
            sample_spa_size_list:   the spatial size or kernel size when sampling the disparities;
            down_func_name:         the name of function that downsamples the groundtruth disparity map;
            weights:                the loss weight in each stage, [1,1,2];
            grad_method:            whether detach the predicted disparity map from last stage, "detach" or None;
            cost_func:              the function to compute the cost, "ssd", "cor", "cat";
            if_overmask:            if avoiding the influence of sky;
            skip_stage_id：         only using interpolation since the stage_id;
            use_detail:             using the lost details generated by network;
            thold:                  threshold used to obtain the lost details;
            alpha:                  the weight in the loss for lost details;
        """
        super(SparseDenseNetRefinementMask, self).__init__()
        assert len(weights) == num_stage, "the length of weights({}) should be the same as the value of num_stage({})".format(weights, num_stage)
        assert len(step) == num_stage, "the length of step should be the same as the value of num_stage, where -1 represents the traditional sampling from 0 to max_disp"
        for i, each_step in enumerate(step) :
            assert each_step*samp_num[i] <= max_disp/np.power(down_scale,num_stage-1-i), "stage{}: the total sampling range({}) whould be samller than or equal to the maximum disparity value({}), starting from stage1".format(i+1, each_step*samp_num[i], max_disp/np.power(down_scale,num_stage-1-i-1))
        assert max_disp%np.power(down_scale, num_stage-1) == 0, "the max_disp({}) should be divisible by down_scale({})^num_stage({})".format(max_disp, down_scale, num_stage)
        assert grad_method in ["detach", 'undetach', None], "grad_method should be in [\"detach\", \"undetach\", None]"
        for each_samp_num in samp_num[1:] :
            assert each_samp_num%2==0, "We constraint the number of sampled disparities({}) to be even to make the regularizer run normally".format(each_samp_num)
        
        self.max_disp = max_disp
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.down_scale = down_scale
        self.step = step
        self.samp_num = samp_num
        self.sample_spa_size_list = sample_spa_size_list
        self.down_func_name = down_func_name
        self.weights = weights
        self.grad_method = grad_method
        self.cost_func = cost_func
        self.if_overmask = if_overmask
        self.skip_stage_id = skip_stage_id
        self.use_detail = use_detail
        self.thold = thold
        self.alpha = alpha
        
        self.feature_extractor = FeatExtNetChannelPlus(base_channels=base_channels, num_stage=num_stage, down_scale=down_scale)
        
        self.get_cost_volume = GetCostVolume(warp_ope="homgrp", cost_func=self.cost_func)
        self.sparse_matching = nn.ModuleList([SpaMat() for i in np.arange(self.num_stage-1)])
        self.sparse_var = nn.ModuleList([SpaVar() for i in np.arange(self.num_stage-1)])
        
        self.cost_regularizer = CostRegNetNoDown(in_channels=self.feature_extractor.out_channels[0],
                                                 base_channels=self.feature_extractor.out_channels[0]*2,
                                                 cost_func=self.cost_func, down_scale=down_scale)
        
        self.detail_detection = nn.ModuleList( [GenerateSparseMask(in_channels=self.feature_extractor.out_channels[i+1],
                                                                        down_scale=down_scale) for i in np.arange(self.num_stage-1)] )
        self.dynamic_upsampling = nn.ModuleList( [DynamicUpsampling(in_channels=self.feature_extractor.out_channels[i+1],
                                                                    down_scale=self.down_scale) for i in np.arange(self.num_stage-1)] )
        self.soft_attention = nn.ModuleList([SoftAttention(in_channels=self.feature_extractor.out_channels[i+1]+4,
                                                           base_channels=base_channels) for i in np.arange(self.num_stage-1)])
        self.refinement = nn.ModuleList( [Refinement(in_channels=self.feature_extractor.out_channels[i+1],
                                                     base_channels=base_channels//(2**i),
                                                     stage_id=i+1, down_scale=down_scale) for i in np.arange(self.num_stage-1)] )
        
        self.train_loss_func = Loss(loss_type="multi_stage_regression_uploss", if_overmask=self.if_overmask)
        # self.train_loss_func = Loss(loss_type="multi_stage_regression_UpSampleloss", if_overmask=self.if_overmask)
        # self.train_loss_func = Loss(loss_type="multi_stage_regression_upmaskloss", if_overmask=self.if_overmask, stop_stage_id=self.skip_stage_id, thold=self.thold, alpha=0.1)
        # self.train_loss_func = Loss(loss_type="chamfer")
        # self.train_loss_func = Loss(loss_type="LR_consistency")

        self.test_loss_func = test_loss_func
        self.test_mask_loss_func = Loss(loss_type="multi_stage_regression_upmaskloss", if_overmask=self.if_overmask, stop_stage_id=self.skip_stage_id, if_train=False, thold=self.thold)
        
        self._initialize_weights()
        
        print("---------------------------------SparseDenseNetRefinementMask Network Architecture---------------------------------")
        print("max_disp: {}   base_channels: {}   num_stage: {}   down_scale: {}   grad_method: {}   cost_func: {}   if_overmask: {}".format(self.max_disp, self.base_channels, self.num_stage, self.down_scale, self.grad_method, self.cost_func, self.if_overmask))
        print("step: {}   samp_num: {}   sample_spa_size_list:{}".format(self.step, self.samp_num, self.sample_spa_size_list))
        print("down_func_name: {}   weights: {}   alpha: {}".format(self.down_func_name, self.weights, self.alpha))
        print("use_detail: {}   thold: {}".format(self.use_detail, self.thold))

        
    def forward(self, left, right, disparity, left_mask_list, right_mask_list, is_check=False, is_eval=True) :
        left_feature_map_all = self.feature_extractor(left)
        right_feature_map_all = self.feature_extractor(right)
        if self.training or is_eval :
            pred_list = []
            fusion_list = []
            dense_list = []
            sparse_list = []
            sparse_mask_list = []
            var_list = []
            residual_disp_list = []
            left_detail_list = []
            right_detail_list = []
            left_fea_list = []
            right_fea_list = []
        
        for stage_id in np.arange(self.num_stage) :
            left_feature_map = left_feature_map_all["stage{}".format(stage_id)]
            right_feature_map = right_feature_map_all["stage{}".format(stage_id)]
            
            left_mask = left_mask_list[stage_id-1]
            right_mask = right_mask_list[stage_id-1]
            cur_max_disp = self.max_disp//(self.down_scale**(self.num_stage-stage_id-1))
            
            # sampling the disparities for the following construction of cost volume
            if stage_id==0 :
                disp_samples = get_disp_samples(cur_max_disp, left_feature_map, stage_id=stage_id)
                
                # building the cost volume according to the sampled disparity values
                cost_vol = self.get_cost_volume(left_feature_map, right_feature_map, disp_samples=disp_samples, max_disp=cur_max_disp)

                # regularization for the cost volume
                cost_vol = self.cost_regularizer(cost_vol)

                # probability regression to predict the disparity map
                pred = disparity_regression(cost_vol, disp_samples)
                
                pre_left_feature_map = left_feature_map
                pre_right_feature_map = right_feature_map
                
            else :
                if stage_id >= self.skip_stage_id :
                    pred = F.interpolate(pred.unsqueeze(1) * self.down_scale, [left_feature_map.size()[-2], left_feature_map.size()[-1]], mode="bicubic").squeeze(1)
                    
                else :
                    # using the details captured by network
                    if self.use_detail :
                        left_detail, left_cur_fea, left_pre_fea = self.detail_detection[stage_id-1](left_feature_map, pre_left_feature_map)
                        right_detail, right_cur_fea, right_pre_fea = self.detail_detection[stage_id-1](right_feature_map, pre_right_feature_map)
                        if self.training or is_eval :
                            left_fea_list  += [left_cur_fea, left_pre_fea]
                            right_fea_list += [right_cur_fea, right_pre_fea]

                        pre_left_feature_map = left_feature_map
                        pre_right_feature_map = right_feature_map

                        left_detail = torch.sigmoid(left_detail)
                        right_detail = torch.sigmoid(right_detail)
                        if self.training or is_eval :
                            left_detail_list.append(left_detail)
                            right_detail_list.append(right_detail)

                        with torch.no_grad():
                            left_mask = torch.clone(left_detail)
                            right_mask = torch.clone(right_detail)
                            left_mask[left_mask <= self.thold] = 0.
                            left_mask[left_mask > self.thold] = 1.
                            right_mask[right_mask <= self.thold] = 0.
                            right_mask[right_mask > self.thold] = 1.
                    
                    if self.grad_method=="detach" :
                        cur_disprity_map = pred.detach()
                    else :
                        cur_disprity_map = pred

                    # upsampling the disparity map from last stage
                    cur_disprity_map = self.dynamic_upsampling[stage_id-1](cur_disprity_map, left_feature_map)
                    if self.training or is_eval :
                        dense_list.append(cur_disprity_map)
                    
                    # obtaining the sparse matching result
                    sparse_res = self.sparse_matching[stage_id-1](left_feature_map, right_feature_map, left_mask, right_mask, cur_max_disp)
                    if self.training or is_eval :
                        sparse_list.append(sparse_res)
                    
                    # obtaining the variance of sparse matching
                    with torch.no_grad():
                        sparse_var_res = self.sparse_var[stage_id-1](left_feature_map, right_feature_map,
                                                                     left_mask, right_mask,
                                                                     sparse_res,
                                                                     cur_max_disp)
                        if self.training or is_eval :
                            var_list.append(sparse_var_res)
                    
                    # obtaining the soft mask
                    soft_mask = self.soft_attention[stage_id-1]( torch.cat((left_feature_map,cur_disprity_map.unsqueeze(1),sparse_res.unsqueeze(1),left_mask.unsqueeze(1),-sparse_var_res.unsqueeze(1)),dim=1) ).squeeze(1)
                    if self.training or is_eval :
                        sparse_mask_list.append(soft_mask)
                    
                    # fusing the dense and sparse results
                    cur_disprity_map = cur_disprity_map*(1-soft_mask) + soft_mask*sparse_res
                    if self.training or is_eval :
                        fusion_list.append(cur_disprity_map)
                    
                    # refining the fusion results
                    pred, residual_disp = self.refinement[stage_id-1](left_feature_map, right_feature_map, cur_disprity_map)
                    if is_eval :
                        residual_disp_list.append(residual_disp)
            
            if self.training or is_eval :
                pred_list.append(pred)
        
        if self.training :
            # gt_list, pred_list, loss, loss_list = self.train_loss_func(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, disparity, self.weights, self.num_stage, self.down_func_name, self.down_scale, self.max_disp, sparse_mask_list=sparse_mask_list, left_feature_map_all=left_fea_list, right_feature_map_all=right_fea_list, left_detail_list=left_detail_list, right_detail_list=right_detail_list, right_mask_list=right_mask_list)
            gt_list, pred_list, loss, loss_list = self.train_loss_func(pred_list, fusion_list, sparse_list, left_mask_list, disparity, self.weights, self.num_stage, self.down_func_name, self.down_scale, self.max_disp)
            # gt_list, pred_list, loss, loss_list = self.train_loss_func(pred_list, left_feature_map_all=left_feature_map_all, right_feature_map_all=right_feature_map_all, weights=self.weights, num_stage=self.num_stage, down_func_name=self.down_func_name, down_scale=self.down_scale, max_disp=self.max_disp)
            if is_check :
                return gt_list, pred_list, dense_list, sparse_list, fusion_list, residual_disp_list, left_mask_list, right_mask_list, sparse_mask_list, var_list, left_feature_map_all, right_feature_map_all, loss_list, cost_vol
            
            return loss, pred_list, gt_list, loss_list, left_detail_list
        
        elif is_eval :
            if is_check :
                return pred_list, dense_list, sparse_list, fusion_list, residual_disp_list, left_mask_list, right_mask_list, sparse_mask_list, left_feature_map_all, right_feature_map_all, cost_vol
            
            loss = self.test_loss_func(pred_list[-1], disparity, self.max_disp)
            
            gt_list, pred_list, mask_loss, mask_loss_list = self.test_mask_loss_func(pred_list, fusion_list, dense_list, sparse_list, left_mask_list, disparity, self.weights, self.num_stage, self.down_func_name, self.down_scale, self.max_disp, sparse_mask_list=sparse_mask_list, left_feature_map_all=left_fea_list, right_feature_map_all=right_fea_list, left_detail_list=left_detail_list, right_detail_list=right_detail_list, right_mask_list=right_mask_list)
            
            return loss, pred_list, mask_loss, mask_loss_list, left_detail_list
        
        else :
            if is_check :
                return pred_list, dense_list, sparse_list, fusion_list, left_mask_list, right_mask_list, sparse_mask_list, left_feature_map_all, right_feature_map_all, cost_vol
            return [pred]
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()




