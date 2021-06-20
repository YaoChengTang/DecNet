 # DecNet
This is the implementation of the paper A Decomposition Model for Stereo Matching, CVPR, 2021, Chengtang Yao, Yunde Jia, Huijun Di, Pengxiang Li, Yuwei Wu

The rebuild version of our work.   
Just using the demo.py to get the test results.   
The code is still a little messy, which will be updated later.   


# Requirements (Major Dependencies)
* Pytorch (1.6.0)
* torchvision (0.7.0)
* visdom
* tqdm
* opencv

# How to use
1. compile the CUDA code.  
   Specify the environment variables in "compile.sh" and "compile_var.sh". More examples, please refer to each shell file.
2. run demo.py  
   Specify the root path to your images in "demo.sh". More instructions, please refer to the shell file. We have provided several examples in the directory "InputData".   
   The pre-trained model is provided here. Just download it and specify the value of variable "resume" in "demo.sh".   
   + Sceneflow:   
       链接：https://pan.baidu.com/s/12G_VXaCAkfUmsIWz6MubVQ   
       提取码：tno0   
   + Kitti：  
       链接：https://pan.baidu.com/s/11txpfufr79Jrq6TusUdeOg   
       提取码：flwx   
   + Middlebury:  
       链接：https://pan.baidu.com/s/1GJktiOh2vsgvWYo2jb_wKw   
       提取码：qwv4   

# Citation
If you use our source code, or our paper, please consider citing the following:
> @inproceedings{Yao2021CVPR,  
title={A Decomposition Model for Stereo Matching},  
author={Yao, Chengtang and Jia, Yunde and Di, Huijun and Li, Pengxiang and Wu, Yuwei},  
booktitle = {CVPR},  
year = {2021}
}
