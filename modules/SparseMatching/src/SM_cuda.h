
int sparse_matching_cuda_forward (at::Tensor ref_feas, at::Tensor tar_feas,
                                  at::Tensor ref_mask, at::Tensor tar_mask,
                                  at::Tensor output,
                                  at::Tensor sum_similarities, at::Tensor max_cost,
                                  int max_disp);
int sparse_matching_cuda_backward (at::Tensor ref_feas, at::Tensor tar_feas,
                                   at::Tensor ref_mask, at::Tensor tar_mask,
                                   at::Tensor output,
                                   at::Tensor sum_similarities, at::Tensor max_cost,
                                   at::Tensor grad_output,
                                   at::Tensor grad_ref_feas, at::Tensor grad_tar_feas,
                                   int max_disp);
