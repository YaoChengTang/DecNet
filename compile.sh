#!/bin/bash
# export LD_LIBRARY_PATH="/home/mcislab/anaconda3/lib:$LD_LIBRARY_PATH"
# export LD_INCLUDE_PATH="/home/mcislab/anaconda3/include:$LD_INCLUDE_PATH"
# export CUDA_HOME="/home/mcislab/cuda-10.2"
# export PATH="/home/mcislab/anaconda3/bin:/home/mcislab/cuda-10.2/bin:$PATH"
# export CPATH="/home/mcislab/cuda-10.2/include"
# export CUDNN_INCLUDE_DIR="/home/mcislab/cuda-10.2/include"
# export CUDNN_LIB_DIR="/home/mcislab/cuda-10.2/lib64"
# export PATH="/home/mcislab/anaconda3/envs/pytorch1.5/bin:$PATH"

export LD_LIBRARY_PATH="YOURPATH/anaconda3/lib:$LD_LIBRARY_PATH"
export LD_INCLUDE_PATH="YOURPATH/anaconda3/include:$LD_INCLUDE_PATH"
export CUDA_HOME="YOURPATH/cuda-10.2"
export PATH="YOURPATH/anaconda3/bin:YOURPATH/cuda-10.2/bin:$PATH"
export CPATH="YOURPATH/cuda-10.2/include"
export CUDNN_INCLUDE_DIR="YOURPATH/cuda-10.2/include"
export CUDNN_LIB_DIR="YOURPATH/cuda-10.2/lib64"
export PATH="YOURPATH/anaconda3/envs/pytorch1.5/bin:$PATH"


TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
echo $TORCH

cd modules/SparseMatching
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib