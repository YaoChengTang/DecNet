# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import cv2
import torch
from torch.utils import data
import torchvision.transforms as transforms


def get_occ(disparity_cuda) :
    if len(disparity_cuda.shape) == 2 :
        height, width = disparity_cuda.shape
        batch_size = 1
        disp_num = 1
        disparity_cuda = disparity_cuda.reshape((batch_size,disp_num,height,width))
    elif len(disparity_cuda.shape) == 3 :
        batch_size, height, width = disparity_cuda.shape
        disp_num = 1
        disparity_cuda = disparity_cuda.reshape((batch_size,disp_num,height,width))
    elif len(disparity_cuda.shape) == 4 :
        batch_size, disp_num, height, width = disparity_cuda.shape
    else :
        raise Exception("Only accept disparity with dim 3 or 4, but given {}".format(len(disparity_cuda.shape)))
    
    if isinstance(disparity_cuda, torch.Tensor) :
        # get the noraml position
        pos_y, pos_x = torch.meshgrid([torch.arange(0, height, dtype=disparity_cuda.dtype),
                                       torch.arange(0, width, dtype=disparity_cuda.dtype)])  # (H, W)
        pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)
        pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)        # (B, S, H, W)
        
        if disparity_cuda.is_cuda :
            pos_x = pos_x.cuda()
            pos_y = pos_y.cuda()
        
        # get the warped position
        shift_cuda = pos_x - disparity_cuda
        shift_numpy = shift_cuda.detach().cpu().numpy()
        
    elif isinstance(disparity_cuda, np.ndarray) :
        # get the noraml position
        pos_x, pos_y = np.meshgrid( np.arange(0,width), np.arange(0,height) )
        pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size,axis=0).repeat(disp_num,axis=1)
        pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size,axis=0).repeat(disp_num,axis=1)
        
        # get the warped position
        shift_cuda = pos_x - disparity_cuda
        shift_numpy = shift_cuda
    
#     print(pos_x.shape)
#     print(pos_x[0,0,100,:])
    
    # compute the minimum position from rightmost pixel to leftmost pixel
    min_shift = np.zeros_like(shift_numpy)
    min_col = np.ones((batch_size,disp_num,height)) * width
    for col in np.arange(width-1, -1, -1) :
        min_col = (min_col>shift_numpy[...,col])*shift_numpy[...,col] + (min_col<=shift_numpy[...,col])*min_col
        min_shift[...,col] = min_col
    
    # compute the position of occlusion
#     occ = shift_numpy>min_shift
    occ = (shift_numpy>min_shift) | (shift_numpy<=0)
    
    return occ



class SceneflowMask(data.Dataset):
    def __init__(self, root, split="train", is_transform=True, is_check=False, img_size=(540,960), scale=3, downsampling_iteration=3, is_training=True, is_eval=False, thold=0.5):
        """data strem for the flying3d dataset with additional preprocessed mask
        args:
            root: the root of dataset's path
            split: what kinf of data, train or test
            is_transform: whether to conduct tranformation for the data
            img_size: the size of croped image, (540,960)
        
        return:
            left, right, disparity, image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3
        """
        super(SceneflowMask, self).__init__()
        print("using SceneflowMask")
        
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (540, 960)
        self.stats = {'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]}
        self.thold = thold
        self.scale = scale
        self.downsampling_iteration = downsampling_iteration
        self.is_check = is_check
        self.is_training = is_training
        self.is_eval = is_eval
        
        if os.path.isfile(os.path.join(root, split)) :
            self.path_list = np.load(os.path.join(root, split))
        else :
            files = os.listdir(os.path.join(root, split))
            self.path_list = [ os.path.join(root, split, file) for file in files ]
        self.path_list.sort()
        
        if len(self.path_list) < 1 :
            raise Exception("No files under/in {}/{}" % (root, split))
        self.length = self.__len__()
        print("Found {} under/in {}".format(len(self.path_list), os.path.join(root, split)))
        
        
    def __len__(self):
        """__len__"""
        return len(self.path_list)
        
        
    def __getitem__(self, index):
        data = np.load(self.path_list[index])
        h,w,c = data.shape
        ori_h, ori_w, _ = data.shape
        
        # mask sure the shape of data is proper
        residual_h, residual_w = 0, 0
        interval = np.power(self.scale, self.downsampling_iteration)
        if h%interval != 0 :
            residual_h = interval - h%interval
        if w%interval != 0 :
            residual_w = interval - w%interval
        tmp_data = np.zeros((h+residual_h, w+residual_w, c), dtype=np.float32)
        tmp_data[residual_h:, residual_w:] = data
        data = np.copy(tmp_data)
        h,w,c = data.shape
        del tmp_data
        
        if self.is_training :
            th, tw = self.img_size
            th = int(np.ceil(th/interval)*interval)
            tw = int(np.ceil(tw/interval)*interval)
            
            if (th,tw) != (h,w) :
                x1 = np.random.randint(0, h-th+1)
                y1 = np.random.randint(0, w-tw+1)
                # print(x1,th,y1,tw)
                data = data[x1:x1+th, y1:y1+tw, :]
            # print(data.shape)
        
        left = data[...,0:3]
        right = data[...,3:6]
        disparity = data[...,6]
        
        if self.is_training :
            # randomly add the reflected light
            if np.random.binomial(1,0.5) :
                left, right = self.add_paralex_noise(left, right)
        
        left = left/255
        right = right/255
        
        left_image = data[...,0:3]
        left_image = transforms.ToTensor()(left_image)
        right_image = data[...,3:6]
        right_image = transforms.ToTensor()(right_image)
        
        # occ3 = get_occ(disparity).squeeze(axis=0).squeeze(axis=0)
        # down_disparity = cv2.resize(disparity/3, dsize=(disparity.shape[1]//3, disparity.shape[0]//3), interpolation=cv2.INTER_LINEAR)
        # occ2 = get_occ(down_disparity).squeeze(axis=0).squeeze(axis=0)
        # down_disparity = cv2.resize(down_disparity/3, dsize=(down_disparity.shape[1]//3, down_disparity.shape[0]//3), interpolation=cv2.INTER_LINEAR)
        # occ1 = get_occ(down_disparity).squeeze(axis=0).squeeze(axis=0)
        
        mask_path = self.path_list[index].replace( self.path_list[index].split("/")[-2], self.path_list[index].split("/")[-2]+"_mask" )
        mask_path = mask_path.split(".")[0]
        with open(mask_path, "rb") as f :
            mask_data = pickle.load(f)
        
        if self.is_training and (th,tw) != (h,w) :
            for idx in np.arange(len(mask_data)) :
                # print(self.downsampling_iteration-1-(idx%3), idx, idx%3)
                down_scale = self.scale**(idx%3)
                mask_data[idx] = mask_data[idx][x1//down_scale:(x1+th)//down_scale, y1//down_scale:(y1+tw)//down_scale]
                # print(down_scale, mask_data[idx].shape, x1//down_scale, (x1+th)//down_scale, mask_data[idx].shape)
        
        left_mask3 = torch.from_numpy(mask_data[0]).float()
        left_mask2 = torch.from_numpy(mask_data[1]).float()
        left_mask1 = torch.from_numpy(mask_data[2]).float()
        # left_mask3 = torch.from_numpy(mask_data[0]*(1-occ3)).float()
        # left_mask2 = torch.from_numpy(mask_data[1]*(1-occ2)).float()
        # left_mask1 = torch.from_numpy(mask_data[2]*(1-occ1)).float()
        
        right_mask3 = torch.from_numpy(mask_data[3]).float()
        right_mask2 = torch.from_numpy(mask_data[4]).float()
        right_mask1 = torch.from_numpy(mask_data[5]).float()
        # right_mask3 = torch.from_numpy(np.ones(mask_data[3].shape)).float()
        # right_mask2 = torch.from_numpy(np.ones(mask_data[4].shape)).float()
        # right_mask1 = torch.from_numpy(np.ones(mask_data[5].shape)).float()
        
        
        if self.is_transform :
            left, right, disparity = self.transform(left, right, disparity)
        
        if self.is_check :
            return left, right, disparity, left_image, right_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.path_list[index].split("/")[-1].split(".")[0]
        
        if self.is_training :
            return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3
            
        if self.is_eval :
            return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.path_list[index].split("/")[-1].split(".")[0], 192
        
        return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.path_list[index].split("/")[-1].split(".")[0], 192
        
        
    def transform(self, left, right, disparity) :
        """transform
        """
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**self.stats),
        ])
        
        # if self.split=='eval' or self.split=='test':
            # left = trans(left).float()
            # right = trans(right).float()
            
        # else :
            # topil = transforms.ToPILImage()
            # totensor = transforms.ToTensor()
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             # std=[0.229, 0.224, 0.225])
            
            # brightness = np.random.uniform(0, 0.2)
            # contrast = np.random.uniform(0, 0.2)
            # saturation = np.random.uniform(0, 0.2)
            # hue = np.random.uniform(0, 0.1)
            
            # left = totensor(left).float()
            # right = totensor(right).float()
            # left = topil(left)
            # right = topil(right)
            # color = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
            # left = color(left)
            # right = color(right)
            
            # left = totensor(left)
            # right = totensor(right)
            # left = left.clamp(min=0, max=1)
            # right = right.clamp(min=0, max=1)
            # left = normalize(left)
            # right = normalize(right)
            
        left = trans(left).float()
        right = trans(right).float()
        
        disparity = torch.from_numpy(disparity).float()
        
        return left, right, disparity
    
    
    def add_paralex_noise(self, left_img, right_img) :
        h,w,c = left_img.shape
        
        sel_h = np.random.randint(100, 180)
        sel_w = np.random.randint(30, 70)
        # print(sel_h, sel_w)
        
        parallel_d = np.random.randint(60,200)
        # print(parallel_d)
        
        sta_h = int(np.random.uniform(0, h-sel_h))
        sta_w = int(np.random.uniform(0, w-sel_w-parallel_d))
        # print(sta_h, sta_w)
        
        x = np.arange(sel_w)
        u = sel_w//2
        sig = 7
        noise = np.exp(-(x-u)**2 / (2*sig**2)) / (np.sqrt(2*np.pi)*sig) * 400 * np.random.uniform(0.7, 1.2)
        noise = np.repeat(noise[np.newaxis], sel_h, axis=0)
        noise = np.repeat(noise[...,np.newaxis], 3, axis=-1)
        
        right_img_noise = right_img.copy()
        right_img_noise[sta_h:sta_h+sel_h, sta_w:sta_w+sel_w] = right_img_noise[sta_h:sta_h+sel_h, sta_w:sta_w+sel_w] + noise
        right_img_noise[right_img_noise>255] = 255.
        
        left_img_noise = left_img.copy()
        left_img_noise[sta_h:sta_h+sel_h, sta_w+parallel_d:sta_w+sel_w+parallel_d] = left_img_noise[sta_h:sta_h+sel_h, sta_w+parallel_d:sta_w+sel_w+parallel_d] + noise
        left_img_noise[left_img_noise>255] = 255.
        
        return left_img_noise, right_img_noise


