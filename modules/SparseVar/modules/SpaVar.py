import torch
import numpy as np
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F

from ..functions import *
from ..functions.SpaVar import SpaVarFunction



class SpaVar(Module) :
    def __init__(self) :
        super(SpaVar, self).__init__()
        
    def forward(self, ref_feas, tar_feas, ref_mask, tar_mask, disparity, max_disp) :
        """sparse matching while forwarding
        
        Args:
            ref_feas, tar_feas: feature map of left/right view, Batch*Channel*Height*Width;
            ref_mask, tar_mask: mask of left/right view, Batch*Height*Width;
            max_disp:           the maximmum disparity in current scale;
        
        Returns:
            output: the computed disparity map, Batch*Height*Width;
        """
        output = SpaVarFunction.apply(ref_feas, tar_feas, ref_mask, tar_mask, disparity, max_disp)
        return output


