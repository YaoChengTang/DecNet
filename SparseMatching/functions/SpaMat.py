import torch
from torch.autograd import Function
from torch.autograd import Variable
from ..build.lib import SpaMat



class SpaMatFunction(Function) :
    @staticmethod
    def forward(ctx, ref_feas, tar_feas, ref_mask, tar_mask, max_disp) :
        """sparse matching while forwarding
        
        Args:
            ref_feas, tar_feas: feature map of left/right view, Batch*Channel*Height*Width;
            ref_mask, tar_mask: mask of left/right view, Batch*Height*Width;
            max_disp:           the maximmum disparity in current scale;
        
        Returns:
            output: the computed disparity map, Batch*Height*Width;
        """
        assert(ref_feas.is_contiguous() == True and tar_feas.is_contiguous() == True)
        assert(ref_mask.is_contiguous() == True and tar_mask.is_contiguous() == True)
        
        with torch.cuda.device_of(ref_feas) :
            output = ref_mask.new().resize_(ref_mask.size()).zero_()
            sum_similarities = ref_mask.new().resize_(ref_mask.size()).zero_()
            max_cost = ref_mask.new().resize_(ref_mask.size()).zero_()
            SpaMat.sparse_matching_cuda_forward(ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost, max_disp)
            output = output.contiguous()
            sum_similarities = sum_similarities.contiguous()
        ctx.save_for_backward(ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost)
        ctx.max_disp = max_disp
        return output
        
    @staticmethod
    def backward(ctx, grad_output) :
        ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost = ctx.saved_tensors
        max_disp = ctx.max_disp
        # print("Backward")
        assert(grad_output.is_contiguous() == True)
        with torch.cuda.device_of(grad_output) :
            grad_ref_feas = ref_feas.new().resize_(ref_feas.size()).zero_()
            grad_tar_feas = tar_feas.new().resize_(tar_feas.size()).zero_()
            SpaMat.sparse_matching_cuda_backward(ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost,
                                                 grad_output, grad_ref_feas, grad_tar_feas, max_disp)
            # print(grad_tar_feas.max())
            grad_ref_feas = grad_ref_feas.contiguous()
            grad_tar_feas = grad_tar_feas.contiguous()
        # print(grad_output.max(), grad_ref_feas.max(), grad_tar_feas.max(), torch.isnan(grad_output).sum(), torch.isnan(grad_ref_feas).sum(), torch.isnan(grad_tar_feas).sum())
        return grad_ref_feas, grad_tar_feas, Variable(torch.Tensor([0])), Variable(torch.Tensor([0])), None


