#include <torch/extension.h>
#include "utils.h"


std::vector<torch::Tensor> ka_convolution_fwd(
    torch::Tensor input, 
    torch::Tensor n_comb, 
    torch::Tensor d_comb, 
    int group,
    int out_c,
    int kernel_size) {


    std::vector<torch::Tensor> relevant_inp;
    std::vector<torch::Tensor> relevant_out;

    auto batch = input.size(0);
    auto height = input.size(2);
    auto width = input.size(3);
    auto in_c = input.size(1);
    
    int pad_size = (kernel_size - 1) / 2;

    auto padded = torch::constant_pad_nd(input, {pad_size, pad_size, pad_size, pad_size}, 0);
    padded = padded.permute({0, 2, 3, 1});

    int step = in_c / group;

    for (int f = 0; f < out_c; ++f) {
        
        for (int c = 0; c < group; ++c) {
            
            for (int a = 0; a < kernel_size; ++a) {
                for (int b = 0; b < kernel_size; ++b) {
                    
                    auto idx = f * group * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b;
                    
                    torch::Tensor n = n_comb[idx];
                    torch::Tensor d = d_comb[idx];
                   

                    auto slicee = padded.slice(1, a, a + height).slice(2, b, b + width).slice(3, c , c +  step);
                    
                    slicee = slicee.reshape({batch, -1, step});
                    relevant_inp.push_back(slicee);
                    
                    
                    if (!slicee.is_contiguous()) {
                        
                        slicee = slicee.contiguous();
                    }

                    torch::Tensor res = rational_fwd_cuda_1dgroup(slicee, n, d, step);
                    relevant_out.push_back(res);

                }
            }
        }
    }

    return {torch::stack(relevant_inp), torch::stack(relevant_out)};
}

std::vector<torch::Tensor> ka_convolution_bwd(
    torch::Tensor grad_output, 
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d, 
    int group, 
    int numerator, 
    int denominator) {

    std::vector<torch::Tensor> dxs;
    std::vector<torch::Tensor> dns;
    std::vector<torch::Tensor> dds;


    auto length = x.size(0);
    auto step = x.size(3);

    
    for (int idx = 0; idx < length; ++idx) {
   
        auto out = grad_output[idx];
        auto inp = x[idx];
        auto num = n[idx];
        auto denom = d[idx];
        
        
        auto res = rational_bwd_cuda_1dgroup(out, inp, num, denom, step, numerator, denominator);
        

        auto dx = res[0];
        auto dn = res[1];
        auto dd = res[2];

        dxs.push_back(dx);
        dns.push_back(dn);
        dds.push_back(dd);
        
    }
    
    
    return {torch::stack(dxs), torch::stack(dns), torch::stack(dds)};

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ka_convolution_fwd", &ka_convolution_fwd, "Kolmogorov-Arnold Convolution forward");
    m.def("ka_convolution_bwd", &ka_convolution_bwd, "Kolmogorov-Arnold Convolution backward");
}