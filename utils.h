#include <torch/extension.h>

// #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define CHECK_LESS(x, y) TORCH_CHECK(x < y, #x " should be less than " #y)

std::vector<torch::Tensor> ka_convolution_fwd(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  int group,
  int out_c,
  int kernel_size);

torch::Tensor rational_fwd_cuda_1dgroup(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  int group);


std::vector<torch::Tensor> ka_convolution_bwd(
  torch::Tensor dy, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  int group,
  int numerator,
  int denominator);

std::vector<torch::Tensor> rational_bwd_cuda_1dgroup(
  torch::Tensor dy, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  int group,
  int numerator,
  int denominator);



