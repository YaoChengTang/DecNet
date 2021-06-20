import os
import re
import sys
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F



def show_imgs(param, sv_img=False, save2where=None, fontsize=20, szWidth=10, szHeight=5, group=3, if_inter=False) :
    """function: visualize the input data
    args:
        paras: [(img, title, colormap), ... ] or
               [{"img":..., "title":..., "cmap":..., "point_x":..., "point_y":..., "point_s":..., "point_c":..., "point_m":..., "colorbar":...}, ... ]
        sv_img: whether to save the visualization
        fontsize : the size of font in title
        szWidth, szHeight: width and height of each subfigure
        group: the columns of the whole figure
    """
    img_num = len(param)
    cols = int(group)
    rows = int(np.ceil(img_num/group))
    sv_title = ""
    color_map = None
    plt_par_list = []
#     plt.clf()
    fig = plt.figure(figsize=(szWidth*cols, szHeight*rows))
    
    for i in np.arange(img_num) :
        if len(param[i])<2 :
            raise Exception("note, each element should be (img, title, ...)")
        
        if isinstance(param[i], list) or isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) :
            name_list = ["img", "title", "cmap", "point_x", "point_y", "point_s", "point_c", "point_m", "point_alpha"]
            plt_par = {}
            for key_id, ele in enumerate(param[i]) :
                plt_par[name_list[key_id]] = ele
        elif isinstance(param[i], dict) :
            plt_par = param[i]
        else :
            raise Exception("unrecognized type: {}, only recept element with type list, np.ndarray, tuple or dict".format(type(param[i])))
        plt_par_list.append(plt_par)
        
        plt.subplot(rows,cols,i+1)
#         plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
        plt.title(plt_par.get("title").replace("\t","   "), fontsize=fontsize)
        im = plt.imshow(plt_par.get("img"), cmap=plt_par.get("cmap"))
        
        if plt_par.get("colorbar") == True :
            plt.colorbar(im)
        
        if plt_par.get("point_x") is not None and plt_par.get("point_y") is not None :
            plt.scatter(plt_par.get("point_x"), plt_par.get("point_y"), s=plt_par.get("point_s"), c=plt_par.get("point_c"), marker=plt_par.get("point_m"), alpha=plt_par.get("point_alpha"))
        plt.axis("off")
        
#         plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
#         plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
#         plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
#         plt.margins(0,0)
        fig.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)
        
        if sv_img is True :
            if i>0 :
                sv_title += "-"
            sv_title += plt_par.get("title")
    
    if if_inter :
        from ipywidgets import Output
        output = Output()
        display(output)
        
        @output.capture()
        def onclick(event):
            if event.button == 3 and event.ydata is not None and event.xdata is not None :
                print_info = ""
                for i in np.arange(img_num) :
                    img = plt_par_list[i].get("img")
                    title = plt_par_list[i].get("title")
                    print_info += "{}:\t({},{})-{}\r\n".format(title, int(np.round(event.ydata)), int(np.round(event.xdata)), img[int(np.round(event.ydata)),int(np.round(event.xdata))])
                print(print_info)
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    
    if sv_img is True and save2where is not None :
        plt.savefig(os.path.join(save2where),dpi=600)
    plt.show(block=False)



def show_dis(param, sv_img=False, fontsize=20, szWidth=10, szHeight=5, group=3) :
    """function: visualize the input data
    args:
        paras: [([(x,y,label),(x,y,label),...], title), ... ] or
               [{"x":...shape(num_type,inter), "y":...shape(num_type,inter), "label":...shape(batch,), "title":...}, ... ]
        sv_img: whether to save the visualization
        fontsize : the size of font in title
        szWidth, szHeight: width and height of each subfigure
        group: the columns of the whole figure
    """
    fig_num = len(param)
    cols = group
    rows = np.ceil(fig_num/group)
    sv_title = ""
    color_map = None
    plt.figure(figsize=(szWidth*cols, szHeight*rows))
    
    for i in np.arange(fig_num) :
        if len(param[i])<3 :
            raise Exception("note, each element should be (x, y, title, ...)")
        
        if isinstance(param[i], list) or isinstance(param[i], np.ndarray) or isinstance(param[i], tuple) :
            name_list = ["x", "y", "title", "cmap", "point_x", "point_y", "point_s", "point_c", "point_m"]
            plt_par = {}
            for key_id, ele in enumerate(param[i]) :
                plt_par[name_list[key_id]] = ele
        elif isinstance(param[i], dict) :
            plt_par = param[i]
        else :
            raise Exception("unrecognized type: {}, only recept element with type list, np.ndarray, tuple or dict".format(type(param[i])))
        
        plt.subplot(rows,cols,i+1)
        plt.title(plt_par.get("title"), fontsize=fontsize)
        plt.bar(plt_par.get("x"), plt_par.get("y"), color=plt_par.get("cmap"))
#         plt.legend()
        
        if plt_par.get("point_x") is not None and plt_par.get("point_y") is not None :
            plt.scatter(plt_par.get("point_x"), plt_par.get("point_y"), s=plt_par.get("point_s"), c=plt_par.get("point_c"), marker=plt_par.get("point_m"))
#         plt.axis("off")
        if sv_img is True :
            if i>0 :
                sv_title += "-"
            sv_title += plt_par.get("title")
    
    if sv_img is True :
        plt.savefig(os.path.join(args.save2where,sv_title+".png"))
    plt.show(block=False)



def get_variable_name(variable, loc):
    for key in loc :
        if loc[key] is variable :
            return key

def get_variable8name(name, loc):
    for key in loc :
        if name == key :
            return loc[key]


def get_occ(disparity_cuda) :
    if len(disparity_cuda.shape) == 3 :
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



def warp(disparity_cuda, image_cuda) :
    """function: warp the image according to the disparity
        args:
            disparity_cuda: B*S*H*W or B*H*W;
            image_cuda: B*C*H*W or B*H*W;
        return:
            warped_image_cuda: B*C*S*H*W;
    """
    if len(disparity_cuda.shape) == 3 :
        batch_size, height, width = disparity_cuda.shape
        disp_num = 1
        disparity_cuda = disparity_cuda.reshape((batch_size,disp_num,height,width))
    elif len(disparity_cuda.shape) == 4 :
        batch_size, disp_num, height, width = disparity_cuda.shape
    else :
        raise Exception("Only accept disparity with dim 3 or 4, but given {}".format(len(disparity_cuda.shape)))
    
    if len(image_cuda.shape) == 3 :
        batch_size, height, width = image_cuda.shape
        chanl_num = 1
        image_cuda = image_cuda.reshape((batch_size,chanl_num,height,width))
    elif len(image_cuda.shape) == 4 :
        batch_size, chanl_num, height, width = image_cuda.shape
    else :
        raise Exception("Only accept image with dim 3 or 4, but given {}".format(len(image_cuda.shape)))
    
    # get the noraml position
    pos_y, pos_x = torch.meshgrid([torch.arange(0, height, dtype=disparity_cuda.dtype),
                                   torch.arange(0, width, dtype=disparity_cuda.dtype)])  # (H, W)
    pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)
    pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)        # (B, S, H, W)

    if disparity_cuda.is_cuda :
        pos_x = pos_x.cuda()
        pos_y = pos_y.cuda()
    
    # get the homography matrix
    proj_x = pos_x - disparity_cuda
    proj_x = proj_x / ((width - 1.0) / 2.0) - 1.0
    proj_y = pos_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([proj_x, proj_y], dim=4)
    
    # warp the image
    warped_image_cuda = F.grid_sample(image_cuda, grid.view(batch_size, disp_num*height, width, 2), mode='bilinear',
                                      padding_mode='zeros').view(batch_size, chanl_num, disp_num, height, width)  # (B, C, S, H, W)
#     warped_image_cuda = warped_image_cuda.detach().cpu().numpy()
    
    return warped_image_cuda



def padding2size(img, target_hieght, target_width) :
    """fucntion: resize the img/data to target size via padding
    args:
        img: (W,H,C)
        target_hieght: int
        target_width: int
    return:
        pad_data: data after padding
    """
    assert img.shape[0]<=target_hieght, "the height {} of img is larger than target_hieght {}".format(img.shape[0], target_hieght)
    assert img.shape[1]<=target_width, "the width {} of img is larger than target_width {}".format(img.shape[1], target_width)
    
    residual_h, residual_w = target_hieght-img.shape[0], target_width-img.shape[1]
    pad_data = np.zeros((target_hieght, target_width, img.shape[2]), dtype=np.float32)
    pad_data[residual_h:, residual_w:] = img
    return pad_data



def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())
        
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



def transform2numpy(tensor_list, local=None) :
    """tranform the tensor in list to numpy
    args:
        tensor_list: [tensor1, tensor2, ...];
        local: local=locals();
    return:
        numpy_list: [numpy1, numpy2, ...]
    """
    numpy_list = []
    if type(tensor_list) == list :
        for tensor in tensor_list :
            numpy_list.append( tensor.detach().cpu().numpy() )
    elif type(tensor_list) == dict :
        for key in tensor_list.keys() :
            numpy_list.append( tensor_list[key].detach().cpu().numpy() )
    return numpy_list

def transform2numpy4list(data_list, local=None) :
    """tranform the tensor in list to numpy
    args:
        tensor_list: [[tensor1, tensor2, ...],
                      [tensor1, tensor2, ...],...];
    return:
        numpy_list: [[numpy1, numpy2, ...],
                     [numpy1, numpy2, ...],...];
    """
    numpy_list = []
    for data in data_list :
        numpy_list.append( transform2numpy(data) )
    return numpy_list



def getError(pred, gt, min_gt=0, max_gt=192) :
    epe_list = []
    loss_3_list = []
    
    valid_mask = gt>0
    epe = np.mean(np.abs(pred[valid_mask]-gt[valid_mask]))
    epe_list.append(epe)
    
    error_pos = np.where((np.abs(pred[valid_mask] - gt[valid_mask])<3) | (np.abs(pred[valid_mask] - gt[valid_mask])<0.05*gt[valid_mask]))
    loss_3 = 100 - error_pos[0].size / np.sum(valid_mask) * 100
    loss_3_list.append(loss_3)

    return epe_list, loss_3_list

# def getError(pred_list, gt_list, min_gt=0, max_gt=192) :
#     epe_list = []
#     loss_3_list = []
#     # for pred, gt in zip(pred_list, gt_list) :
#         valid_mask = gt>0
#         epe = np.mean(np.abs(pred[valid_mask]-gt[valid_mask]))
#         epe_list.append(epe)
        
#         error_pos = np.where((np.abs(pred[valid_mask] - gt[valid_mask])<3) | (np.abs(pred[valid_mask] - gt[valid_mask])<0.05*gt[valid_mask]))
#         loss_3 = 100 - error_pos[0].size / np.sum(valid_mask) * 100
#         loss_3_list.append(loss_3)

#     return epe_list, loss_3_list

    

def Gaussian(x,sigma) :
    return np.exp(-(x**2)/(sigma**2))

def diffusion(img, iteration=10, lamda=0.1, sigma=15, show_all=False, return_list=False) :
    """function: to blur the image but preserve the edges
    args: 
        img: (W,H) or (W,H,C)
        iteration: the iteration to difffuse the image
        lamda: I = I + lamda*update
        sigma: update = dir_grad * np.exp(-x^2/sigma^2)
        show_all: whether to show all the diffused images
        return_list: whether to return all the diffused images
    return:
        if return_list is True, return the final diffused image: img;
        otherwise, return all the diffused images: [(img, name, colormap)];
    """
    if iteration==0 :
        if return_list is True :
            return [(img, "diffusion"+str(0), "gray")]
        else :
            return img
    top_data = np.copy(img[0,:])
    bottom_data = np.copy(img[-1,:])
    left_data = np.copy(img[:,0])
    right_data = np.copy(img[:,-1])

    left_grad = np.column_stack((img,right_data))-np.column_stack((left_data,img))
    left_grad = left_grad[:,:-1]
    right_grad = np.column_stack((left_data,img))-np.column_stack((img,right_data))
    right_grad = right_grad[:,1:]
    top_grad = np.row_stack((img,bottom_data))-np.row_stack((top_data,img))
    top_grad = top_grad[:-1,:]
    bottom_grad = np.row_stack((top_data,img))-np.row_stack((img,bottom_data))
    bottom_grad = bottom_grad[1:,:]
    
    img_list = []
    img_list.append((img, "diffusion"+str(0), "gray"))
    for i in np.arange(iteration) :
        img = img + lamda*( left_grad*Gaussian(left_grad,sigma) +
                            right_grad*Gaussian(right_grad,sigma) +
                            top_grad*Gaussian(top_grad,sigma) +
                            bottom_grad*Gaussian(bottom_grad,sigma))
        img_list.append((img, "diffusion"+str(i+1), "gray"))
    
    if show_all is True :
        show_imgs(img_list)
    
    if return_list is True :
#         return [img[0] for img in img_list]
        return img_list
    else :
        return img



def GaussianDown(img, ksize=(3,3), sigma=1, scale=2, anistropic=False) :
    """function: Gaussian blur and downsampling
    args:
        img: source image, W*H;
        ksize: size of kernel, eg, (5,5)
        sigma: sigma of Gaussian kernel;
        scale: the downsampling scale;
    return:
        the blured image
    """
    if anistropic is False :
        blured_img = cv2.GaussianBlur(img, ksize, sigma)
    else :
        blured_img = diffusion(img, iteration=1, lamda=0.1, sigma=15, show_all=False, return_list=False)
    res = cv2.resize(blured_img, (blured_img.shape[1]//scale, blured_img.shape[0]//scale), cv2.INTER_AREA)
    return res

def GaussianUp(img, ksize=(5,5), sigma=1, scale=2, anistropic=False) :
    """function: Gaussian blur and upsampling
    args:
        img: source image, W*H;
        ksize: size of kernel, eg, (5,5)
        sigma: sigma of Gaussian kernel;
        scale: the downsampling scale;
    return:
        the upsampled image
    """
    img = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), cv2.INTER_AREA)
    if anistropic is False :
        res = cv2.GaussianBlur(img, ksize, sigma)
    else :
        res = diffusion(img, iteration=1, lamda=0.1, sigma=15, show_all=False, return_list=False)
    return res
    
    
    
def detailDetection(img, scale, downsampling_iteration, name=None, diffusion_iteration=0, diffusion_lamda=0.1, diffusion_sigma=15, anistropic=False, thold=0.3) :
    h,w,c = img.shape
    residual_h, residual_w = 0, 0
    interval = np.power(scale,downsampling_iteration)
    if h%interval != 0 :
        residual_h = interval - h%interval
    if w%interval != 0 :
        residual_w = interval - w%interval
    tmp_img = np.zeros((h+residual_h, w+residual_w, c), dtype=np.float32)
    tmp_img[residual_h:, residual_w:] = img
    tmp_img[:residual_h, residual_w:] = img[:1]
    tmp_img[residual_h:, :residual_w] = img[:, :1]
    img = np.copy(tmp_img)
    del tmp_img
    
    diffused_img_list = diffusion(img, iteration=diffusion_iteration, lamda=diffusion_lamda, sigma=diffusion_sigma, show_all=False, return_list=True)
    
    data = diffused_img_list[-1][0]
    residual_img_list = []
    collection4image = []
    
    for i in range(downsampling_iteration) :
        down_img = GaussianDown(data, scale=scale, anistropic=anistropic)
        residual = GaussianUp(down_img, scale=scale, anistropic=anistropic)
#             print(residual.shape, data.shape, down_img.shape)
        if residual.shape != data.shape :
            residual = cv2.resize(residual, (data.shape[1],data.shape[0]))
#             print(residual.shape, data.shape)
        residual = np.abs(data - residual)
        residual_img_list.append((residual, name+"-"+str(i+1), "gray"))
        data = down_img
    
    tmp_list = []
    for i, residual in enumerate(residual_img_list) :
        residual = residual[0].sum(axis=2)
        tmp = (residual-residual.min()) / (residual.max()-residual.min())
        tmp[tmp>=thold] = 1
        tmp[tmp<thold] = 0
        
        start_h = 0
        start_w = 0
        if residual_h > 10 :
            start_h = interval//np.power(scale, i)
        if residual_w > 10 :
            start_w = interval//np.power(scale, i)
        start_h = residual_h//np.power(scale, i)
        start_w = residual_w//np.power(scale, i)
        tmp[:start_h, :] = 0
        tmp[:, :start_w] = 0
        
        collection4image.append(tmp.astype(np.bool))
    return collection4image