# -*- coding: utf-8 -*-

import os
import pickle as pkl
import numpy as np
import cv2
import torch
from torch.utils import data
import torchvision.transforms as transforms


class MiddleburyMask(data.Dataset):
    
    def __init__(self, root, split="train", is_transform=True, img_size=(375, 1242), is_check=False, scale=3, downsampling_iteration=3, is_training=True, is_eval=False, thold=0.5):
        """__init__
        
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        """
        super(MiddleburyMask, self).__init__()
        print("using Data Loader MiddleburyMask-{}".format(is_training))
        
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (375, 1242)
        self.stats={'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}
        self.thold = thold
        self.files = {}
        self.datapath = root
        
        if split in ["train_Q", "eval_Q"] :
            self.datapath = os.path.join(self.datapath, "MiddEval3Q_processed")
            self.split = "trainingQ"
            # choose = [1,4,5,7,9,10,13,13,13]
            choose = None
        elif split in ["train_H", "eval_H"] :
            self.datapath = os.path.join(self.datapath, "MiddEval3H_processed")
            self.split = "trainingH"
            choose = None
            # if split == "eval_H" :
                # choose = [0,1,13,16,17,19,20,21,23,24,25,26,27,29,30,14]
        elif split in ["train_F"] :
            self.datapath = os.path.join(self.datapath, "MiddEval3F_processed")
            self.split = "trainingF"
            choose = None
        elif split in ["train_AG"] :
            self.split = "MiddZip_raw_split_dense"
            choose = None
        elif split in ["train_allF", "eval_allF"] :
            # self.datapath = os.path.join(self.datapath, "MiddZip_processed")
            self.split = "MiddZip_processed"
            # choose = [5,6,7,8,11,16,18]
            # choose = [0,7,8,9,14,14,14,15,16,17,19,22,22,22]
            # choose = [12,7,11,20,4,23,9,18,16,15,0,5]
            choose = None
        elif split in ["train_allF_EL", "eval_allF_EL"] :
            # self.datapath = os.path.join(self.datapath, "MiddZip_processed_EL")
            self.split = "MiddZip_processed_EL"
            choose = None
        elif split in ["train_merge"] :
            # self.datapath = os.path.join(self.datapath, "MiddMerged")
            self.split = "MiddMerged"
            choose = None
        elif split == "test_Q" :
            self.datapath = os.path.join(self.datapath, "MiddEval3Q_processed")
            self.split = "testQ"
        elif split == "test_H" :
            self.datapath = os.path.join(self.datapath, "MiddEval3H_processed")
            self.split = "testH"
        elif split == "test_F" :
            self.datapath = os.path.join(self.datapath, "MiddEval3F_processed")
            self.split = "testF"
        else :
            raise Exception("Nu such split: {}".format(split))
        
        self.files = os.listdir(os.path.join(self.datapath,self.split))
        self.files.sort()
        
        if is_training :
            tmp_files = []
            for name in self.files :
                try :
                    if float(name.replace(".pkl","").split("-")[-1]) > 0.88 :
                        tmp_files.append(name)
                except ValueError :
                    if name.find("perfect") == -1 :
                        tmp_files.append(name)
            self.files = tmp_files
        
        self.scale = scale
        self.downsampling_iteration = downsampling_iteration
        self.is_check = is_check
        self.is_training = is_training
        self.is_eval = is_eval
        # if is_training :
            # self.files = self.files * 3
        
        if len(self.files)<1:
            raise Exception("No files for ld=[%s] found in %s" % (self.split, self.datapath))
        self.length=self.__len__()
        print("Found %d in %s data" % (len(self.files), self.datapath))
    
    
    def __len__(self):
        """__len__"""
        return len(self.files)
    
    
    def __getitem__(self, index):
        """__getitem__
        
        :param index:
        """
        with open(os.path.join( os.path.join(self.datapath, self.split, self.files[index]) ), "rb") as f :
            raw_data = pkl.load(f)
        ndisp = raw_data["ndisp"]
        left = raw_data["im0"]
        right = raw_data["im1"]
        disparity = raw_data.get("disparity")
        disparity_right = raw_data.get("disparity_right")
        del raw_data
        
        if disparity is None :
            disparity = np.zeros((left.shape[0],left.shape[1]))
        else :
            disparity[disparity==np.inf] = 0
        
        if disparity_right is not None :
            disparity_right[disparity_right==np.inf] = 0
            data = np.concatenate( (left,right,disparity[...,np.newaxis],disparity_right[...,np.newaxis]), axis=2 )
        else :
            data = np.concatenate( (left,right,disparity[...,np.newaxis]), axis=2 )
        h,w,c = data.shape
        ori_h, ori_w, _ = data.shape
        # print(self.files[index], h,w,c)
        
        # make sure the shape of data is proper
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
            # horizontal flip
            if disparity_right is not None :
                if np.random.binomial(1,0.5) :
                    tmp_data = data[...,0:3]
                    data[...,0:3] = data[...,3:6][:,::-1]
                    data[...,3:6] = tmp_data[:,::-1]
                    tmp_data = data[...,6]
                    data[...,6] = data[...,7]
                    data[...,7] = tmp_data
                    data = np.ascontiguousarray(data, dtype=np.float32)
                    del tmp_data
            
            # # randomly occlude a region
            # if np.random.binomial(1,0.5) :
                # sh = int(np.random.uniform(30,80))
                # sw = int(np.random.uniform(10,80))
                # ch = int(np.random.uniform(sh,right.shape[0]-sh))
                # cw = int(np.random.uniform(sw,right.shape[1]-sw))
                # right[ch-sh:ch+sh,cw-sw:cw+sw] = np.mean(np.mean(right,0),0)[np.newaxis,np.newaxis]
            
            # randomly crop
            th, tw = self.img_size
            th = int(np.ceil(th/interval)*interval)
            tw = int(np.ceil(tw/interval)*interval)
            
            if th>h :
                tmp_data = np.zeros((th, w, c), dtype=np.float32)
                tmp_data[th-h:] = data
                data = np.copy(tmp_data)
                del tmp_data
            elif th<h :
                x1 = np.random.randint(0, h-th+1)
                data = data[x1:x1+th]
                
            if tw>w :
                tmp_data = np.zeros((data.shape[0], tw, c), dtype=np.float32)
                tmp_data[:,tw-w:] = data
                data = np.copy(tmp_data)
                del tmp_data
            elif tw<w :
                y1 = np.random.randint(0, w-tw+1)
                data = data[:, y1:y1+tw]
                
            # if (th,tw) != (h,w) :
                # x1 = np.random.randint(0, h-th+1)
                # y1 = np.random.randint(0, w-tw+1)
                # # print(x1,th,y1,tw)
                # data = data[x1:x1+th, y1:y1+tw, :]
        
        left = data[...,0:3]
        right = data[...,3:6]
        disparity = data[...,6]
        
        # # randomly add the reflected light
        # if self.is_training :
            # if np.random.binomial(1,0.8) :
                # left, right = self.add_paralex_noise(left, right)
            # if np.random.binomial(1,0.5) :
                # left, right = self.add_paralex_noise(left, right)
        
        left = left/255
        right = right/255
        
        left_image = data[...,0:3]
        left_image = transforms.ToTensor()(left_image)
        right_image = data[...,3:6]
        right_image = transforms.ToTensor()(right_image)
        
        mask_path = os.path.join(self.datapath, self.split+"_mask", self.files[index].split(".pkl")[0])
        with open(mask_path, "rb") as f :
            mask_data = pkl.load(f)
        
        if self.is_training :
            # horizontal flip
            if disparity_right is not None :
                for idx in np.arange(len(mask_data)//2) :
                    tmp_data = mask_data[idx]
                    mask_data[idx] = mask_data[idx+3][:,::-1]
                    mask_data[idx+3] = tmp_data[:,::-1]
                    mask_data[idx] = np.ascontiguousarray(mask_data[idx], dtype=np.float32)
                    mask_data[idx+3] = np.ascontiguousarray(mask_data[idx+3], dtype=np.float32)
                
            # crop
            h,w = mask_data[0].shape
            if self.is_training and (th,tw) != (h,w) :
                for idx in np.arange(len(mask_data)) :
                    # print(self.downsampling_iteration-1-(idx%3), idx, idx%3)
                    down_scale = self.scale**(idx%3)
                    if th>h :
                        tmp_data = np.zeros((th//down_scale, w//down_scale), dtype=np.float32)
                        tmp_data[(th-h)//down_scale:] = mask_data[idx]
                        mask_data[idx] = np.copy(tmp_data)
                        del tmp_data
                    elif th<h :
                        mask_data[idx] = mask_data[idx][x1//down_scale:(x1+th)//down_scale]
                    
                    if tw>w :
                        tmp_data = np.zeros((mask_data[idx].shape[0], tw//down_scale), dtype=np.float32)
                        tmp_data[:, (tw-w)//down_scale:] = mask_data[idx]
                        mask_data[idx] = np.copy(tmp_data)
                        del tmp_data
                    elif tw<w :
                        mask_data[idx] = mask_data[idx][:, y1//down_scale:(y1+tw)//down_scale]
                    
                    # mask_data[idx] = mask_data[idx][x1//down_scale:(x1+th)//down_scale, y1//down_scale:(y1+tw)//down_scale]
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
        
        
        if self.is_transform:
            left, right, disparity = self.transform(left, right, disparity)
        
        if self.is_check :
            return left, right, disparity, left_image, right_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.files[index].split(".")[0]
        
        if self.is_training :
            return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3
            
        if self.is_eval :
            return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.files[index].split(".")[0], ndisp
        
        return left, right, disparity, left_image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, self.files[index].split(".")[0], ndisp
    
    
    def transform(self, left, right, disparity):
        """transform
        """
        trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])
        
        if self.is_training == False :
            left = trans(left).float()
            right = trans(right).float()
            
        else :
            train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                RandomPhotometric(
                                    noise_stddev=0.0,
                                    min_contrast=-0.37,
                                    max_contrast=0.37,
                                    brightness_stddev=0.02,
                                    min_color=0.9,
                                    max_color=1.1,
                                    min_gamma=0.7,
                                    max_gamma=1.7),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])
            left = train_transform(left)
            right = train_transform(right)
            
        
        # left = trans(left).float()
        # right = trans(right).float()
        
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
        # noise = np.exp(-(x-u)**2 / (2*sig**2)) / (np.sqrt(2*np.pi)*sig) * 400 * np.random.uniform(0.3, 1.2)
        # noise = np.repeat(noise[np.newaxis], sel_h, axis=0)
        # noise = np.repeat(noise[...,np.newaxis], 3, axis=-1)
        noise_r = np.exp(-(x-u)**2 / (2*sig**2)) / (np.sqrt(2*np.pi)*sig) * 400
        noise_r = np.repeat(noise_r[np.newaxis], sel_h, axis=0)
        noise_g = np.exp(-(x-u)**2 / (2*sig**2)) / (np.sqrt(2*np.pi)*sig) * 300
        noise_g = np.repeat(noise_g[np.newaxis], sel_h, axis=0)
        noise_b = np.exp(-(x-u)**2 / (2*sig**2)) / (np.sqrt(2*np.pi)*sig) * 500
        noise_b = np.repeat(noise_b[np.newaxis], sel_h, axis=0)
        noise = np.stack((noise_r,noise_g,noise_b), axis=-1)
        noise = noise.reshape(-1,3)
        
        pos_w = np.arange(sta_w, sta_w+sel_w)
        pos_h = np.arange(sta_h, sta_h+sel_h)
        pos_h = np.repeat(pos_h, sel_w)
        pos_w = np.repeat(pos_w[...,np.newaxis],sel_h,axis=1).transpose().reshape(-1)
        
        step = np.random.rand() * 0.3
        pos_shift = (np.arange(sel_h) - sel_h//2) * step
        pos_shift = pos_shift.astype(np.int)
        pos_shift = np.repeat(pos_shift[...,np.newaxis],sel_w,axis=1).reshape(-1)
        
        pos_w = pos_w + pos_shift
        pos_w = np.clip(pos_w, a_min=0, a_max=w-parallel_d-1)
        
        right_img_noise = right_img.copy()
        right_img_noise[pos_h, pos_w] = right_img_noise[pos_h, pos_w] + noise
        right_img_noise[right_img_noise>255] = 255.
        
        left_img_noise = left_img.copy()
        left_img_noise[pos_h, pos_w+parallel_d] = left_img_noise[pos_h, pos_w+parallel_d] + noise
        left_img_noise[left_img_noise>255] = 255.
        
        return left_img_noise, right_img_noise





class RandomPhotometric(object):
    """Applies photometric augmentations to a list of image tensors.
    Each image in the list is augmented in the same way.

    Args:
        ims: list of 3-channel images normalized to [0, 1].

    Returns:
        normalized images with photometric augmentations. Has the same
        shape as the input.
    """

    def __init__(self,
                 noise_stddev=0.0,
                 min_contrast=0.0,
                 max_contrast=0.0,
                 brightness_stddev=0.0,
                 min_color=1.0,
                 max_color=1.0,
                 min_gamma=1.0,
                 max_gamma=1.0):
        self.noise_stddev = noise_stddev
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.brightness_stddev = brightness_stddev
        self.min_color = min_color
        self.max_color = max_color
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma

    def __call__(self, im):
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        gamma_inv = 1.0 / gamma
        color = torch.from_numpy(
            np.random.uniform(self.min_color, self.max_color, (3))).float()
        if self.noise_stddev > 0.0:
            noise = np.random.normal(scale=self.noise_stddev)
        else:
            noise = 0
        if self.brightness_stddev > 0.0:
            brightness = np.random.normal(scale=self.brightness_stddev)
        else:
            brightness = 0
        
        im_re = im.permute(1, 2, 0)
        im_re = (im_re * (contrast + 1.0) + brightness) * color
        im_re = torch.clamp(im_re, min=0.0, max=1.0)
        im_re = torch.pow(im_re, gamma_inv)
        im_re += noise
        
        im_re = im_re.permute(2, 0, 1)
        return im_re