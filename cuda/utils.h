#include <torch/extension.h>

torch::Tensor ka_convolution_fwd(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d,
  torch::Tensor bias, 
  int out_channels, int kernel_size, int stride, bool bias_enabled);

torch::Tensor ka_convolution_fwd_cuda(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  torch::Tensor bias,
  int out_channels, 
  int kernel_size, 
  int stride, bool bias_enabled);


std::vector<torch::Tensor> ka_convolution_bwd(
  torch::Tensor dy, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d,
  torch::Tensor bias, 
  int out_channels, 
  int kernel_size, 
  int stride,
  int numerator,
  int denominator, bool bias_enabled);

std::vector<torch::Tensor> ka_convolution_bwd_cuda(
  torch::Tensor dy, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d, 
  torch::Tensor bias,
  int out_channels, 
  int kernel_size, 
  int stride,
  int numerator,
  int denominator, bool bias_enabled);
