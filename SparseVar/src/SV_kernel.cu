#include <torch/extension.h>
//#include <torch/serialize/tensor.h>
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

#ifdef __cplusplus
    extern "C" {
#endif



__global__ void get_max_cost(const int num_ele, const int channel, const int height, const int width, const int max_disp, 
                             const float* ref_feas_data, const float* tar_feas_data,
                             const float* ref_mask_data, const float* tar_mask_data,
                             float* max_cost_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_ele){
        return;
    }
    // if this position does not belong to the current fine-grained areas in left view, then ignoring it.
    if(ref_mask_data[index] == 0)
        return;
    
    int id_batch = index/width/height;
    int id_height = index/width%height;
    int id_width = index%width;
    
    int step = width*height;
    int base_3d = id_batch*channel*step + id_height*width + id_width;  // built for the position of images with size of N*C*H*W;
    int mask_base = index;                                          // built for the position of mask with size of N*H*W, actually the same with output;
    int cur_max_disp = id_width-max_disp+1>=0 ? max_disp : id_width+1; 
    
    // find the maximmum cost
    float max_cost=0.000001;
    for(int disp=0; disp<cur_max_disp; disp++) {
        // if the position does not belong to the current fine-grained areas in right view, then ignoring it.
        if(tar_mask_data[mask_base-disp] == 0)
            continue;
        
        int tar_base = base_3d - disp;
        float cost = 0;
        for(int cha=0; cha<channel; cha++) {
            cost += ref_feas_data[base_3d+cha*step] * tar_feas_data[tar_base+cha*step];
        }
        if(max_cost < cost)
            max_cost = cost;
    }
    max_cost_data[index] = max_cost;
}

/**********************************************
* function: sparse matching while forwarding
* args:
*       num_ele: the number of all threads, Batch*Width*Height;
*       channel, height, width: size of channel/height/width;
*       max_disp: the maximmum disparity in current scale;
*       ref_feas_data, tar_feas_data: feature map of left/right view, Batch*Channel*Height*Width;
*       ref_mask_data, tar_mask_data: mask of left/right view, Batch*Height*Width;
*       output_data: the computed disparity map, Batch*Height*Width;
*       sum_sim_data: the sum of similaritis between feature maps from left and right views along disaprity dimension, Batch*Height*Width,
*                     sum_sim_data = \sum_{d=0}^{max_disp} \sum_{c=0}^{max_cha} sou(h,w,c)*ref(h,w-disp,c);
*       max_cost_data: the maximum cost along disparity dimension for each pixel, Batch*Height*Width;
*return:
**********************************************/
__global__ void sparse_var_forward(const int num_ele, const int channel, const int height, const int width, const int max_disp, 
                                   const float* ref_feas_data, const float* tar_feas_data,
                                   const float* ref_mask_data, const float* tar_mask_data,
                                   const float* disparity_data,
                                   const float* max_cost_data,
                                   float* output_data, float* sum_sim_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_ele){
        return;
    }
    // if this position does not belong to the current fine-grained areas in left view, then ignoring it.
    if(ref_mask_data[index] == 0)
        return;
    
    int id_batch = index/width/height;
    int id_height = index/width%height;
    int id_width = index%width;
    int step = width*height;
    
    int base_3d = id_batch*channel*step + id_height*width + id_width;  // built for the position of images with size of N*C*H*W;
    int mask_base = index;                                          // built for the position of mask with size of N*H*W, actually the same with output;
    int cur_max_disp = id_width-max_disp+1>=0 ? max_disp : id_width+1;
    
    // compute the similarity with softmax
    float sum_similarity=0.000001, sum_disp=0.000001, tmp_sim=0.000001;
    float max_cost = max_cost_data[index];
    for(int disp=0; disp<cur_max_disp; disp++) {
        // if the position does not belong to the current fine-grained areas in right view, then ignoring it.
        if(tar_mask_data[mask_base-disp] == 0)
            continue;
        
        int tar_base = base_3d - disp;
        float cost = 0;
        for(int cha=0; cha<channel; cha++) {
            cost += ref_feas_data[base_3d+cha*step] * tar_feas_data[tar_base+cha*step];
        }
        
        // if(cost>max_cost)
            // printf("cost>max_cost\n");
        // assert(cost<=max_cost);
        
        tmp_sim = expf(cost-max_cost);
        sum_disp += tmp_sim * (disp-disparity_data[index]) * (disp-disparity_data[index]);
        sum_similarity += tmp_sim;
    }
    sum_sim_data[index] = sum_similarity;
    output_data[index] = sum_disp/sum_similarity;
}

/**********************************************
* function: backpropagation over feature map from the left view
* args:
*       num_ele: the number of all threads, Batch*Channel*Height*Width;
*       channel, height, width: size of channel/height/width;
*       max_disp: the maximmum disparity in current scale;
*       ref_feas_data, tar_feas_data: feature map of left/right view, Batch*Channel*Height*Width;
*       ref_mask_data, tar_mask_data: mask of left/right view, Batch*Height*Width;
*       output_data: the computed disparity map, Batch*Height*Width;
*       sum_sim_data: the sum of similaritis between feature maps from left and right views along disaprity dimension, Batch*Height*Width,
*                     sum_sim_data = \sum_{d=0}^{max_disp} \sum_{c=0}^{max_cha} sou(h,w,c)*ref(h,w-disp,c);
*       max_cost_data: the maximum cost along disparity dimension for each pixel, Batch*Height*Width;
*       grad_output_data: the gradient of disparity map, Batch*Height*Width;
*       grad_ref_data: the gradient of feature map from the left view, Batch*Channel*Height*Width;
*return:
**********************************************/
__global__ void sparse_var_ref_backward(const int num_ele, const int channel, const int height, const int width, const int max_disp, 
                                             const float* ref_feas_data, const float* tar_feas_data,
                                             const float* ref_mask_data, const float* tar_mask_data,
                                             const float* disparity_data,
                                             const float* output_data,
                                             const float* sum_sim_data, const float* max_cost_data,
                                             const float* grad_output_data,
                                             float* grad_ref_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_ele){
        return;
    }
    
    int id_batch = index/width/height/channel;
    // int id_channel = index/width/height%channel;
    int id_height = index/width%height;
    int id_width = index%width;
    
    int step = width*height;
    int base_3D = index;
    int cost_3D = id_batch*channel*step + id_height*width + id_width;
    int base_2D = id_batch*step + id_height*width + id_width;
    
    // if this position does not belong to the current fine-grained areas in left view, then ignoring it.
    if(ref_mask_data[base_2D] == 0)
        return;
    
    int cur_max_disp = id_width-max_disp+1>=0 ? max_disp : id_width+1;
    float tmp_grad=0, tmp_sim=0;
    for(int disp=0; disp<cur_max_disp; disp++) {
        // if the position does not belong to the current fine-grained areas in right view, then ignoring it.
        if(tar_mask_data[base_2D-disp] == 0)
            continue;
        
        int tar_base_3d = cost_3D - disp;
        float cost = 0;
        for(int cha=0; cha<channel; cha++) {
            cost += ref_feas_data[cost_3D+cha*step] * tar_feas_data[tar_base_3d+cha*step];
        }
        
        // if(cost>max_cost_data[base_2D])
            // printf("in ref_backward, cost>max_cost, (%d,%d,%d)-(%d,%d,%d)\n", id_batch,id_height,id_width, channel,height,width);
        assert(cost<=max_cost_data[base_2D]);
        
        tmp_sim = cost-max_cost_data[base_2D];
        tmp_sim = expf(tmp_sim);
        // tmp_sim = 1;
        // printf("ref backward: %f\n", tmp_sim);
        tmp_grad += tmp_sim * tar_feas_data[base_3D-disp] * ((disp-disparity_data[base_2D])*(disp-disparity_data[base_2D])-output_data[base_2D]);
    }
    grad_ref_data[base_3D] = grad_output_data[base_2D] * tmp_grad / sum_sim_data[base_2D];
    // printf("ref backward: %f\n", grad_ref_data[base_3D]);
}



/**********************************************
* function: backpropagation over feature map from the left view
* args:
*       num_ele: the number of all threads, Batch*Channel*Height*Width;
*       channel, height, width: size of channel/height/width;
*       max_disp: the maximmum disparity in current scale;
*       ref_feas_data, tar_feas_data: feature map of left/right view, Batch*Channel*Height*Width;
*       ref_mask_data, tar_mask_data: mask of left/right view, Batch*Height*Width;
*       output_data: the computed disparity map, Batch*Height*Width;
*       sum_sim_data: the sum of similaritis between feature maps from left and right views along disaprity dimension, Batch*Height*Width,
*                     sum_sim_data = \sum_{d=0}^{max_disp} \sum_{c=0}^{max_cha} sou(h,w,c)*ref(h,w-disp,c);
*       max_cost_data: the maximum cost along disparity dimension for each pixel, Batch*Height*Width;
*       grad_output_data: the gradient of disparity map, Batch*Height*Width;
*       grad_tar_data: the gradient of feature map from the right view, Batch*Channel*Height*Width;
* return:
**********************************************/
__global__ void sparse_var_tar_backward(const int num_ele, const int channel, const int height, const int width, const int max_disp, 
                                             const float* ref_feas_data, const float* tar_feas_data,
                                             const float* ref_mask_data, const float* tar_mask_data,
                                             const float* disparity_data,
                                             const float* output_data,
                                             const float* sum_sim_data, const float* max_cost_data,
                                             const float* grad_output_data,
                                             float* grad_tar_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_ele){
        return;
    }
    
    int id_batch = index/width/height/channel;
    // int id_channel = index/width/height%channel;
    int id_height = index/width%height;
    int id_width = index%width;
    
    int step = width*height;
    int base_3D = index;
    int cost_3D = id_batch*channel*step + id_height*width + id_width;
    int base_2D = id_batch*step + id_height*width + id_width;
    
    // if this position does not belong to the current fine-grained areas in left view, then ignoring it.
    if(tar_mask_data[base_2D] == 0)
        return;
    
    int cur_max_disp = id_width+max_disp<=width ? max_disp : width-id_width;
    float tmp_grad=0, tmp_sim=0;
    for(int disp=0; disp<cur_max_disp; disp++) {
        // if the position does not belong to the current fine-grained areas in right view, then ignoring it.
        int base_2D_shift = base_2D+disp;
        int base_3D_shift = base_3D+disp;
        int cost_3D_shift = cost_3D+disp;
        if(ref_mask_data[base_2D_shift] == 0)
            continue;
        
        float cost = 0;
        for(int cha=0; cha<channel; cha++) {
            cost += ref_feas_data[cost_3D_shift+cha*step] * tar_feas_data[cost_3D+cha*step];
        }
        tmp_sim = cost-max_cost_data[base_2D_shift];
        // if(tmp_sim>1)
            // printf("tar backward: %f   %f   %f\n", tmp_sim, cost, max_cost_data[base_2D_shift]);
        tmp_sim = expf(tmp_sim);
        // tmp_sim = 1;
        
        tmp_grad += grad_output_data[base_2D_shift] * tmp_sim * ref_feas_data[base_3D_shift] * ((disp-disparity_data[base_2D_shift])*(disp-disparity_data[base_2D_shift])-output_data[base_2D_shift]) / sum_sim_data[base_2D_shift];
        
        // if(tmp_grad>1000)
        // {
            // printf("tar backward: %f - %f, %f, %f, %f, %f\n", tmp_grad, grad_output_data[base_2D_shift], tmp_sim, ref_feas_data[base_3D_shift], disp-output_data[base_2D_shift], sum_sim_data[base_2D_shift]);
        // }
    }
    grad_tar_data[index] = tmp_grad;
}



__global__ void sparse_var_dis_backward(const int num_ele, const int channel, const int height, const int width, const int max_disp, 
                                         const float* ref_feas_data, const float* tar_feas_data,
                                         const float* ref_mask_data, const float* tar_mask_data,
                                         const float* disparity_data,
                                         const float* sum_sim_data, const float* max_cost_data,
                                         const float* grad_output_data,
                                         float* grad_disparity_data) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= num_ele){
        return;
    }
    
    // if this position does not belong to the current fine-grained areas in left view, then ignoring it.
    if(ref_mask_data[index] == 0)
        return;
    
    int id_batch = index/width/height;
    int id_height = index/width%height;
    int id_width = index%width;
    int step = width*height;
    
    int base_3d = id_batch*channel*step + id_height*width + id_width;  // built for the position of images with size of N*C*H*W;
    int base_2d = index;                                          // built for the position of mask with size of N*H*W, actually the same with output;
    
    int cur_max_disp = id_width-max_disp+1>=0 ? max_disp : id_width+1;
    float tmp_grad=0, tmp_sim=0;
    for(int disp=0; disp<cur_max_disp; disp++) {
        // if the position does not belong to the current fine-grained areas in right view, then ignoring it.
        if(tar_mask_data[base_2d-disp] == 0)
            continue;
        
        int tar_base_3d = base_3d - disp;
        float cost = 0;
        for(int cha=0; cha<channel; cha++) {
            cost += ref_feas_data[base_3d+cha*step] * tar_feas_data[tar_base_3d+cha*step];
        }
        
        // if(cost>max_cost_data[base_2D])
            // printf("in ref_backward, cost>max_cost, (%d,%d,%d)-(%d,%d,%d)\n", id_batch,id_height,id_width, channel,height,width);
        assert(cost<=max_cost_data[base_2d]);
        
        tmp_sim = cost-max_cost_data[base_2d];
        tmp_sim = expf(tmp_sim);
        // tmp_sim = 1;
        // printf("ref backward: %f\n", tmp_sim);
        tmp_grad += tmp_sim * (disp-disparity_data[base_2d]);
    }
    grad_disparity_data[base_2d] = -2 * grad_output_data[base_2d] * tmp_grad / sum_sim_data[base_2d];
    // printf("ref backward: %f\n", grad_ref_data[base_3D]);
}



void sparse_var_kernel_forward (at::Tensor ref_feas, at::Tensor tar_feas,
                                     at::Tensor ref_mask, at::Tensor tar_mask,
                                     at::Tensor disparity,
                                     at::Tensor output,
                                     at::Tensor sum_similarities, at::Tensor max_cost,
                                     const int max_disp){
	const int batch = ref_feas.size(0);
	const int channel = ref_feas.size(1);
	const int height = ref_feas.size(2);
	const int width = ref_feas.size(3);
    
	float *output_data = output.data_ptr<float>();
    float *sum_sim_data = sum_similarities.data_ptr<float>();
    float *max_cost_data = max_cost.data_ptr<float>();
    
	const float *ref_feas_data = ref_feas.data_ptr<float>();
	const float *tar_feas_data = tar_feas.data_ptr<float>();
    const float *ref_mask_data = ref_mask.data_ptr<float>();
    const float *tar_mask_data = tar_mask.data_ptr<float>();
    const float *disparity_data = disparity.data_ptr<float>();
    
	const int num_ele = batch * height * width;
	int blocks = (num_ele + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
	// int N = output.numel();
	// cudaMemset(output_data, 0, sizeof (float) * N);
    
    get_max_cost <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, max_cost_data);
    // cudaDeviceSynchronize();
	sparse_var_forward <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, disparity_data, max_cost_data, output_data, sum_sim_data);
}

void sparse_var_kernel_backward (at::Tensor ref_feas, at::Tensor tar_feas,
                                  at::Tensor ref_mask, at::Tensor tar_mask,
                                  at::Tensor disparity,
                                  at::Tensor output,
                                  at::Tensor sum_similarities, at::Tensor max_cost,
                                  at::Tensor grad_output,
                                  at::Tensor grad_ref, at::Tensor grad_tar,
                                  at::Tensor grad_disparity,
                                  const int max_disp){
	const int batch = ref_feas.size(0);
	const int channel = ref_feas.size(1);
	const int height = ref_feas.size(2);
	const int width = ref_feas.size(3);
    
    const float *ref_feas_data = ref_feas.data_ptr<float>();
	const float *tar_feas_data = tar_feas.data_ptr<float>();
    const float *ref_mask_data = ref_mask.data_ptr<float>();
    const float *tar_mask_data = tar_mask.data_ptr<float>();
    const float *disparity_data = disparity.data_ptr<float>();
    const float *output_data = output.data_ptr<float>();
    const float *sum_sim_data = sum_similarities.data_ptr<float>();
    const float *max_cost_data = max_cost.data_ptr<float>();
    
    const float *grad_output_data = grad_output.data_ptr<float>();
    
	float *grad_ref_data = grad_ref.data_ptr<float>();
    float *grad_tar_data = grad_tar.data_ptr<float>();
    float* grad_disparity_data = grad_disparity.data_ptr<float>();
    
    
    // int num_ele = batch * height * width;
	// int blocks = (num_ele + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    // check_max_cost1 <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, max_cost_data);
    
    
    int num_ele = grad_ref.numel();
	int blocks = (num_ele + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
    // check_max_cost2 <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, output_data, sum_sim_data, max_cost_data, grad_output_data, grad_tar_data);
    // printf("batch, channel, height, width - num_ele: %d %d %d %d - %d\n", batch, channel, height, width, num_ele);
    
	sparse_var_ref_backward <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, disparity_data, output_data, sum_sim_data, max_cost_data, grad_output_data, grad_ref_data);
    
	sparse_var_tar_backward <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, disparity_data, output_data, sum_sim_data, max_cost_data, grad_output_data, grad_tar_data);
    
    num_ele = grad_disparity.numel();
	blocks = (num_ele + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    
    sparse_var_dis_backward <<< blocks, CUDA_NUM_THREADS >>> (num_ele, channel, height, width, max_disp, ref_feas_data, tar_feas_data, ref_mask_data, tar_mask_data, disparity_data, sum_sim_data, max_cost_data, grad_output_data, grad_disparity_data);
}


 
#ifdef __cplusplus
    }
#endif
