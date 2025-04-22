#include <torch/extension.h>
#include "utils.h"

torch::Tensor ka_convolution_fwd(
    torch::Tensor input, 
    torch::Tensor n_comb, 
    torch::Tensor d_comb, 
    torch::Tensor bias,
    int out_channels, int kernel_size, int stride, bool bias_enabled) {

        /*
        input is the padded image
        n_comb are all the numerators -> shape [out_c * in_c * kernel_size * kernel_size, 6]
        d_comb are all the denominators -> shape [out_c * in_c * kernel_size * kernel_size, 4]
        bias is the bias term,

        returns the result after applying the rational function to the input for every single weight
        */

        return ka_convolution_fwd_cuda(input, n_comb, d_comb, bias, out_channels, kernel_size, stride, bias_enabled);
    
}


std::vector<torch::Tensor> ka_convolution_bwd(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d, 
    torch::Tensor bias,
    int out_channels, int kernel_size, int stride, int numerator, int denominator, bool bias_enabled) {

        /*
        grad_output is passed from previous layer
        x is the padded image
        n are all numerators
        d are all denominators
        bias is the bias term
        numerator and denominator are passed so that a dynamic order calculation in the kernel is possible if numerator < 6 or denominator < 4
       

        returns d_input, d_weight_numerator, d_weight_denominator (and d_bias if bias enabled)
        */

        return ka_convolution_bwd_cuda(grad_output, x, n, d, bias, out_channels, kernel_size, stride, numerator, denominator, bias_enabled);
        
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ka_convolution_fwd", &ka_convolution_fwd, "Kolmogorov-Arnold Convolution forward");
    m.def("ka_convolution_bwd", &ka_convolution_bwd, "Kolmogorov-Arnold Convolution backward");
}