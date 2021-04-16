//#include <torch/torch.h>
#include <torch/extension.h>
#include "SM_kernel.h"



extern "C" int
sparse_matching_cuda_forward (at::Tensor ref_feas, at::Tensor tar_feas,
                              at::Tensor ref_mask, at::Tensor tar_mask,
                              at::Tensor output,
                              at::Tensor sum_similarities, at::Tensor max_cost,
                              int max_disp){
    sparse_matching_kernel_forward (ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost, max_disp);
    return 1;
}

extern "C" int
sparse_matching_cuda_backward (at::Tensor ref_feas, at::Tensor tar_feas,
                               at::Tensor ref_mask, at::Tensor tar_mask,
                               at::Tensor output,
                               at::Tensor sum_similarities, at::Tensor max_cost,
                               at::Tensor grad_output,
                               at::Tensor grad_ref_feas, at::Tensor grad_tar_feas,
                               int max_disp){
    sparse_matching_kernel_backward (ref_feas, tar_feas, ref_mask, tar_mask, output, sum_similarities, max_cost, grad_output, grad_ref_feas, grad_tar_feas, max_disp);
    return 1;
}

PYBIND11_MODULE (TORCH_EXTENSION_NAME, SpaMat)
{
  
  SpaMat.def ("sparse_matching_cuda_forward", &sparse_matching_cuda_forward, "sparse matching forward (CUDA)");
  SpaMat.def ("sparse_matching_cuda_backward", &sparse_matching_cuda_backward, "sparse matching backward (CUDA)");
  
}

