
#include <torch/extension.h>

#ifdef __cplusplus
    extern "C" {
#endif

/**********************************************
* function: sparse matching while forwarding
* args:
*       ref_feas, tar_feas: feature map of left/right view, Batch*Channel*Height*Width;
*       ref_mask, tar_mask: mask of left/right view, Batch*Height*Width;
*       output:             the computed disparity map, Batch*Height*Width;
*       sum_similarities:   the sum of similaritis between feature maps from left and right views along disaprity dimension, Batch*Height*Width,
*                           sum_sim_data = \sum_{d=0}^{max_disp} \sum_{c=0}^{max_cha} sou(h,w,c)*ref(h,w-disp,c);
*       max_cost:           the maximum cost along disparity dimension for each pixel, Batch*Height*Width;
*       max_disp:           the maximmum disparity in current scale;
*return:
**********************************************/
void sparse_var_kernel_forward (at::Tensor ref_feas, at::Tensor tar_feas,
                                     at::Tensor ref_mask, at::Tensor tar_mask,
                                     at::Tensor disparity,
                                     at::Tensor output,
                                     at::Tensor sum_similarities, at::Tensor max_cost,
                                     const int max_disp);

/**********************************************
* function: backpropagation
* args:
*       ref_feas, tar_feas: feature map of left/right view, Batch*Channel*Height*Width;
*       ref_mask, tar_mask: mask of left/right view, Batch*Height*Width;
*       output:             the computed disparity map, Batch*Height*Width;
*       sum_similarities:   the sum of similaritis between feature maps from left and right views along disaprity dimension, Batch*Height*Width,
*                           sum_sim_data = \sum_{d=0}^{max_disp} \sum_{c=0}^{max_cha} sou(h,w,c)*ref(h,w-disp,c);
*       max_cost:           the maximum cost along disparity dimension for each pixel, Batch*Height*Width;
*       grad_output:        the gradient of disparity map, Batch*Height*Width;
*       grad_ref:           the gradient of feature map from left view, Batch*Channel*Height*Width;
*       grad_tar:           the gradient of feature map from right view, Batch*Channel*Height*Width;
*       max_disp:           the maximmum disparity in current scale;
*return:
**********************************************/
void sparse_var_kernel_backward (at::Tensor ref_feas, at::Tensor tar_feas,
                                      at::Tensor ref_mask, at::Tensor tar_mask,
                                      at::Tensor disparity,
                                      at::Tensor output,
                                      at::Tensor sum_similarities, at::Tensor max_cost,
                                      at::Tensor grad_output,
                                      at::Tensor grad_ref, at::Tensor grad_tar,
                                      at::Tensor grad_disparity,
                                      const int max_disp);

#ifdef __cplusplus
    }
#endif
