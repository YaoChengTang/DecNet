import os
import sys
import argparse
import numpy as np
import time
import copy
import cv2

import torch
import visdom
from tqdm import tqdm

# import apex
# from apex import amp
# import torch.distributed as dist

from modules import get_model
from modules.sync_batchnorm import convert_model
from loader import get_loader, get_data_path

import warnings
warnings.filterwarnings("ignore")

# try:
    # from apex.parallel import DistributedDataParallel as DDP
    # from apex.fp16_utils import *
    # from apex import amp, optimizers
    # from apex.multi_tensor_apply import multi_tensor_applier
# except ImportError:
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")



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
parser.add_argument('--step', nargs='?', type=str, default="1,3,5",
                    help='the step size between each sampling disparities')
parser.add_argument('--samp_num', nargs='?', type=str, default="-1,9,9,9",
                    help='the number of total sampled disparities')
parser.add_argument('--sample_spa_size_list', nargs='?', type=str, default="-1,3,3,3",
                    help='the spatial size or kernel size when sampling disparities')
parser.add_argument('--down_func_name', nargs='?', type=str, default="bilinear",
                    help='the name of fucntion that downsamples the groundtruth disparity map')
parser.add_argument('--loss_weights', nargs='?', type=str, default="0.2,0.6,1.8",
                    help='the loss weights in each stage')

parser.add_argument('--skip_stage_id', nargs='?', type=int, default=100,
                    help='skip_stage_id')
parser.add_argument('--use_detail', nargs='?', type=bool, default=False,
                    help='use detail generated from fixed filter')
parser.add_argument('--is_eval', nargs='?', type=bool, default=False,
                    help='evaluation or submission')
parser.add_argument('--thold', nargs='?', type=float, default=0.5,
                    help='the threshold for detail loss detection')

parser.add_argument('--dataset', nargs='?', type=str, default='flying3d',
                    help='Dataset to use [\'sceneflow and kitti etc\']')
parser.add_argument('--img_rows', nargs='?', type=int, default=256,
                    help='Height of the input image')
parser.add_argument('--img_cols', nargs='?', type=int, default=512,
                    help='Width of the input image')
parser.add_argument('--train_split', nargs='?', type=str, default="train",
                    help='the split of training dataset')
parser.add_argument('--test_split', nargs='?', type=str, default="test",
                    help='the split of testing dataset')
parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                    help='Batch Size')
parser.add_argument('--using_apex', nargs='?', type=bool, default=False,
                    help='Whether using apex')
parser.add_argument('--is_distributed', nargs='?', type=bool, default=False,
                    help='Whether training with distributed devices')

parser.add_argument('--resume', nargs='?', type=str, default=None,
                    help='Path to previous saved model')

parser.add_argument("--local_rank", type=int, default=0)

parser.add_argument('--save2where', nargs='?', type=str, default='./Log/FirstTry',
                    help='Path to model that need to be saved currently')
parser.add_argument('--visdom', nargs='?', type=bool, default=False,
                    help='Show visualization(s) on visdom | False by default')
parser.add_argument('--visdomImg', nargs='?', type=bool, default=False,
                    help='Show visualization(s) of images on visdom | False by default')
parser.add_argument('--envVis', nargs='?', type=str, default="Yao",
                    help='the name of visdom')
args = parser.parse_args()


is_check_anomaly = False

torch.manual_seed(17)
torch.cuda.manual_seed_all(17)

if is_check_anomaly :
    torch.autograd.set_detect_anomaly(True)



def test(args):
    torch.backends.cudnn.benchmark=True
    
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    
    test_dataset = data_loader(data_path, is_transform=True, split=args.test_split, img_size=(args.img_rows, args.img_cols), is_training=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.batch_size if args.batch_size<4 else 4, shuffle=False)
    test_length = len(test_loader)
    

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
    if args.is_eval :
        model = torch.nn.DataParallel(model)
    model = model.cuda()
    
    # load model
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = model.state_dict()
            if args.is_eval :
                pre_dict = {k: v for k, v in checkpoint['model_state'].items()}
                model_dict.update(pre_dict)
            else :
                pre_dict = {k.replace("module.",""): v for k, v in checkpoint['model_state'].items()}
                model_dict.update(pre_dict)
            model.load_state_dict(model_dict)
            
        else :
            raise Exception("No such model file, please check it: {}".format(args.resume))
    else:
        print('From scratch!')
        
    
    model.eval()
    epe_rec = []
    loss_3_rec = []
    for i, (left, right, disparity, image, left_mask1, left_mask2, left_mask3, right_mask1, right_mask2, right_mask3, ori_h, ori_w, name, n_disp) in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            if args.dataset=="MiddleburyMask" :
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

            try :
                torch.cuda.synchronize()
                start_time = time.time()
                if args.is_eval :
                    loss, pred_list, mask_loss, mask_loss_list, left_detail_list = model(left, right, disparity, [left_mask1, left_mask2, left_mask3], [right_mask1, right_mask2, right_mask3], is_check=False, is_eval=args.is_eval)
                else :
                    pred_list = model(left, right, disparity, [left_mask1, left_mask2, left_mask3], [right_mask1, right_mask2, right_mask3], is_check=False, is_eval=args.is_eval)
                torch.cuda.synchronize()
                end_time = time.time()

                if args.is_eval :
                    loss_3 = loss[1].mean().item()
                    epe = loss[0].mean().item()
                    epe_rec.append(epe)
                    loss_3_rec.append(loss_3)
                    print("[{}/{}]   evaluation cost time: {} - epe: {} - loss3: {}".format( i, test_length, end_time-start_time, epe, loss_3))
                    
                else :
                    output = pred_list[-1]
                    output = output*256
                    output[output==0] = 0
                    output[output<0] = 0
                    output[output>65535] = 65535
                    output_np = output.data.cpu().numpy().astype('uint16')
                    idx = 0
                    pre = output_np[idx, -ori_h[idx]:, -ori_w[idx]:]
                    cv2.imwrite(os.path.join(args.save2where, name[idx]+'.png'), pre)
                    print("[{}/{}]   submission cost time: {}".format( i, test_length, end_time-start_time))

            except KeyError as e:
                left_np = left.cpu().detach().numpy()
                right_np = right.cpu().detach().numpy()
                disparity_np = disparity.cpu().detach().numpy()
                image_np = image.cpu().detach().numpy()
                np.savez("./Errors/testig_KeyError_data-{}-{}-{}-{}.npz".format(args.arch, epoch, i, time.strftime(' %Y-%m-%d %H-%M-%S',time.localtime(time.time()))), left=left_np, right=right_np, disparity=disparity_np, image=image_np)
                print("---{}---".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
                raise KeyError("while testing, an error occured: KeyError {}".format(e))
        
    
    print("The testing is completed: {}".format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    if args.is_eval :
        print("epe: {}, loss_3: {}".format(np.mean(epe_rec), np.mean(loss_3_rec)))

if __name__ == '__main__':
    test(args)
