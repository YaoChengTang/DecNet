import os
import sys
import argparse
import numpy as np
import time
import copy
import cv2

import torch
import torchvision.transforms as transforms

from modules import get_model
from modules.sync_batchnorm import convert_model
from utils.utils import show_imgs
from utils.utils import detailDetection

import warnings
warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--seed', nargs='?', type=int, default=7,
                help='random seed (default: 7)')
parser.add_argument('--arch', nargs='?', type=str, default='SANet',
                    help='Architecture to use [\'region support network\']')
parser.add_argument('--max_disp', nargs='?', type=int, default=216,
                    help='the maximum disparity value')
parser.add_argument('--base_channels', nargs='?', type=int, default=8,
                    help='the basic number of channels')
parser.add_argument('--cost_func', nargs='?', type=str, default="ssd",
                    help='the function of cost computation')
parser.add_argument('--grad_method', nargs='?', type=str, default="detach",
                    help='whether detach the predicted disparity map from last stage, "detach" or None')
parser.add_argument('--num_stage', nargs='?', type=int, default=3,
                    help='the number of downsampling stage')
parser.add_argument('--down_scale', nargs='?', type=int, default=3,
                    help='the scale size in each downsampling stage')
parser.add_argument('--step', nargs='?', type=str, default="-1,1,1,1",
                    help='the step size between each sampling disparities')
parser.add_argument('--samp_num', nargs='?', type=str, default="-1,12,10,6",
                    help='the number of total sampled disparities')
parser.add_argument('--sample_spa_size_list', nargs='?', type=str, default="-1,3,5,7",
                    help='the spatial size or kernel size when sampling disparities')
parser.add_argument('--down_func_name', nargs='?', type=str, default="bilinear",
                    help='the name of fucntion that downsamples the groundtruth disparity map')
parser.add_argument('--loss_weights', nargs='?', type=str, default="1,1,1,1.",
                    help='the loss weights in each stage')

parser.add_argument('--skip_stage_id', nargs='?', type=int, default=100,
                    help='skip_stage_id')
parser.add_argument('--use_detail', nargs='?', type=bool, default=False,
                    help='use detail generated from fixed filter')
parser.add_argument('--is_eval', nargs='?', type=bool, default=False,
                    help='evaluation or submission')
parser.add_argument('--thold', nargs='?', type=float, default=0.5,
                    help='the threshold for detail loss detection')

parser.add_argument('--root', nargs='?', type=str, default='./InputData',
                    help='the directory to be processed')

parser.add_argument('--resume', nargs='?', type=str, default=None,
                    help='Path to previous saved model')

parser.add_argument('--save2where', nargs='?', type=str, default='./Log/FirstTry',
                    help='Path to model that need to be saved currently')
args = parser.parse_args()


torch.manual_seed(17)
torch.cuda.manual_seed_all(17)



def padding(img) :
    h,w,c = img.shape
    residual_h = int(np.ceil(h/27) * 27) - h
    residual_w = int(np.ceil(w/27) * 27) - w
    padded_img = np.zeros((h+residual_h, w+residual_w, c), dtype=np.float32)
    padded_img[residual_h:, residual_w:] = img
    return padded_img

def transform(img) :
    trans = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                            ])
    return trans(img).float().unsqueeze(0)

def transform2tensor(img_list) :
    tensor_list = []
    np2tensor = transforms.ToTensor()
    for img in img_list :
        tensor_list.append( np2tensor(img[...,np.newaxis]).float() )
    return tensor_list



def test(args):
    torch.backends.cudnn.benchmark=True
    
    # check the archive path
    if not args.is_eval :
        if not os.path.exists(args.save2where) :
            os.mkdir(args.save2where)
            print("Creating new directory : {}".format(args.save2where))
        else :
            print(args.save2where, "already exists!")
    
    # build model
    if args.arch.lower() in ["sparsedensenetrefinementmask"] :
        model = get_model(name=args.arch, max_disp=args.max_disp, base_channels=args.base_channels, cost_func=args.cost_func, grad_method=args.grad_method,
                          num_stage=args.num_stage, down_scale=args.down_scale, step=list(map(float, args.step.split(','))), samp_num=list(map(float, args.samp_num.split(','))), sample_spa_size_list=list(map(int, args.sample_spa_size_list.split(','))),
                          down_func_name=args.down_func_name, weights=list(map(float, args.loss_weights.split(','))),
                          if_overmask=False, skip_stage_id=args.skip_stage_id, use_detail=args.use_detail, thold=args.thold)
    else :
        model = get_model(name=args.arch)

    model = convert_model(model)
    model = model.cuda()
    
    # load model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = model.state_dict()
            pre_dict = {k.replace("module.",""): v for k, v in checkpoint['model_state'].items()}
            model_dict.update(pre_dict)
            model.load_state_dict(model_dict)
        else :
            raise Exception("No such model file, please check it: {}".format(args.resume))
    else:
        print('From scratch!')
        
    
    model.eval()
    dir_list = os.listdir(args.root)
    for i, name in enumerate(dir_list*10) :
        if not os.path.isdir(os.path.join(args.root,name)) :
            continue

        left_img  = cv2.imread(os.path.join(args.root,name,"im0.png"))
        right_img = cv2.imread(os.path.join(args.root,name,"im1.png"))
        left_img  = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        if os.path.exists( os.path.join(args.root,name,"calib.txt") ):
            with open(os.path.join(args.root,name,"calib.txt"), "r") as f :
                lines = f.readlines()
            n_disp = float(lines[-1].strip().split("=")[-1])
            n_disp = np.ceil(n_disp/27) * 27
        else :
            n_disp = -1
        
        ori_h, ori_w, _  = left_img.shape
        left_padded_img  = padding(left_img) / 255
        right_padded_img = padding(right_img) / 255
        
        left_mask_list  = detailDetection(left_padded_img, scale=3, downsampling_iteration=3, name=name, thold=0.3)
        right_mask_list = detailDetection(right_padded_img, scale=3, downsampling_iteration=3, name=name, thold=0.3)
        
        left  = transform(left_padded_img)
        right = transform(right_padded_img)
        left_mask1, left_mask2, left_mask3    = transform2tensor(left_mask_list[::-1])
        right_mask1, right_mask2, right_mask3 = transform2tensor(right_mask_list[::-1])
        disparity = torch.zeros((1,left.shape[2],left.shape[3]))


        with torch.no_grad():
            if n_disp>0 :
                model.max_disp = int(n_disp)

            left = left.cuda()
            right = right.cuda()
            disparity = disparity.cuda()
            left_mask1 = left_mask1.cuda()
            left_mask2 = left_mask2.cuda()
            left_mask3 = left_mask3.cuda()
            right_mask1 = right_mask1.cuda()
            right_mask2 = right_mask2.cuda()
            right_mask3 = right_mask3.cuda()
            
            torch.cuda.synchronize()
            start_time = time.time()
            pred_list = model(left, right, disparity, [left_mask1, left_mask2, left_mask3], [right_mask1, right_mask2, right_mask3], is_check=False, is_eval=False)
            torch.cuda.synchronize()
            end_time = time.time()

            output = pred_list[-1]
            output = output*256
            output[output==0] = 0
            output[output<0] = 0
            output[output>65535] = 65535
            output_np = output.data.cpu().numpy().astype('uint16')
            pre = output_np[0, -ori_h:, -ori_w:]
            cv2.imwrite(os.path.join(args.save2where, name+'.png'), pre)
            print("rebuild version, cost time: {}".format(end_time-start_time))

    print("The testing is completed: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

if __name__ == '__main__':
    test(args)
