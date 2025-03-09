#include <torch/extension.h>
#include "utils.h"

torch::Tensor ka_convolution_fwd(
    torch::Tensor input, 
    torch::Tensor n_comb, 
    torch::Tensor d_comb, 
    int group) {

        /*
        input is the relevant input, that each weight sees -> shape [num_weights, B, L, D]
        n_comb are all the numerators -> shape [num_weight, 6]
        d_comb are all the denominators -> shape [num_weight, 4]
        group is the group index -> how many channels should be computed together

        returns the result after applying the rational function to the relative input for every single weight -> summed up later in python shape[num_weights, B, L, D]
        */

        return rational_fwd_cuda_1dgroup(input, n_comb, d_comb, group);
    
}


std::vector<torch::Tensor> ka_convolution_bwd(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d, 
    int group, 
    int numerator, 
    int denominator) {

        /*
        grad_output is the relevant grad_out for every weight
        x is the relavant input for every weight
        n are all numerators
        d are all denominators
        group is grouping
        numerator and denominator are for the order calculations in the Kernel

        returns d_input, d_weight_numerator, d_weight_denominator for all weights
        */

        return rational_bwd_cuda_1dgroup(grad_output, x, n, d, group, numerator, denominator);
        
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ka_convolution_fwd", &ka_convolution_fwd, "Kolmogorov-Arnold Convolution forward");
    m.def("ka_convolution_bwd", &ka_convolution_bwd, "Kolmogorov-Arnold Convolution backward");
}