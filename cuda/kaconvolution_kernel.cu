#include <torch/extension.h>

//default order is (5, 4), kernel should work with higher orders, but this has not been tested yet
constexpr int n_order = 5;
constexpr int d_order = 4;


template <typename scalar_t>
__global__ void ka_convolution_fwd_cuda_kernel(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    const scalar_t* __restrict__ bias,
    float* __restrict__ result, 
    int B, int H, int W, int out_channels, int in_channels, int k, int padded_h, int padded_w, int stride,
    int x_size, int num_weights, int padded_size, bool bias_enabled) {


    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y;
    
    int tile_x = blockIdx.x * tile_width;
    int tile_y = blockIdx.y * tile_height;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    int w_index = tile_x + thread_x; 
    int h_index = tile_y + thread_y; 
    
    if (w_index < W && h_index < H) { 

        const int total = B * out_channels;

        for (int flat_idx = blockIdx.z; flat_idx < total; flat_idx += gridDim.z) {

            int out_c_idx = flat_idx % out_channels;
            int batch_idx = flat_idx / out_channels;

            scalar_t accum_value = 0;

            for (int in_c = 0; in_c < in_channels; ++in_c) {

                #pragma unroll
                for (int kh = 0; kh < k; ++kh) {

                    #pragma unroll
                    for (int kw = 0; kw < k; ++kw) {

                        int h_in = h_index * stride + kh;
                        int w_in = w_index * stride + kw;
                
                        int weight_idx = in_c  *k * k + kh * k + kw;

                        int x_idx = ((batch_idx * in_channels * padded_size) + in_c * padded_size + h_in * padded_w + w_in);

                        int w_full_idx = out_c_idx * num_weights + weight_idx;

                        int a_idx = w_full_idx * 6;
                        int b_idx = w_full_idx * 4;

                        scalar_t s_a[n_order + 1], s_b[d_order];
                        #pragma unroll
                        for (int i = 0; i < n_order + 1; ++i) {
                            s_a[i] = a[a_idx + i];
                        }
                        #pragma unroll
                        for (int i = 0; i < d_order; ++i) {
                            s_b[i] = abs(b[b_idx + i]);  
                        }

                        scalar_t xp1 = x[x_idx];
                        scalar_t abs_xp1 = abs(xp1);

                        // Compute the polynomial for P using Horner's method
                        scalar_t P = s_a[n_order];
                        #pragma unroll
                        for (int i = n_order - 1; i >= 0; --i) {
                            P = fmaf(P, xp1, s_a[i]);
                        }
                        
                        // Compute the polynomial for Q using Horner's method
                        scalar_t Q = s_b[d_order - 1];
                        #pragma unroll
                        for (int i = d_order - 2; i >= 0; --i) {
                            Q = fmaf(Q, abs_xp1, s_b[i]);
                        }
                        Q = fmaf(Q, abs_xp1, 1.0);

                        scalar_t one_div_Q = __frcp_rn(Q);

                        accum_value = fmaf(P, one_div_Q, accum_value); // += P/Q

                    }
                }
            }
            
            if (bias_enabled) {
                accum_value += bias[out_c_idx];
            }

            int result_idx = ((batch_idx * out_channels + out_c_idx) * H + h_index) * W + w_index;

            result[result_idx] = accum_value;
    
        }
    }
}


torch::Tensor ka_convolution_fwd_cuda(
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d,
    torch::Tensor bias,
    int out_channels, int kernel_size, int stride, bool bias_enabled
    ){
    
        
    int B = x.size(0);
    int in_channels = x.size(1);
    int padded_h = x.size(2);
    int padded_w = x.size(3);

    int H = (padded_h - kernel_size) / stride + 1;
    int W = (padded_w - kernel_size) / stride + 1;
    int num_weights = in_channels * kernel_size * kernel_size;

    auto result = at::zeros({B, out_channels, H, W}, x.options()).toType(at::kFloat);
    
    const int x_size = B * out_channels * H * W;

    int blockSize = 16;
    int maxGridZ = 32768; //ensure that blockDim.z doesn't get to large
    int gridDimz = (B * out_channels <= maxGridZ) ? B * out_channels : maxGridZ;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((W + blockSize - 1) / blockSize, (H + blockSize - 1) / blockSize, gridDimz);
    

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ka_convolution_fwd_cuda_kernel", ([&] {
    ka_convolution_fwd_cuda_kernel<scalar_t>
        <<<gridDim, blockDim>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
            B, H, W, out_channels, in_channels, kernel_size, padded_h, padded_w, stride, x_size, num_weights, padded_h * padded_w, bias_enabled);
        }));

    return result;
}


__forceinline__ __device__ void warp_reduce_sum(float* vals, int num_values) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0; i < num_values; i++) { 
            vals[i] += __shfl_down_sync(0xFFFFFFFF, vals[i], offset);
        }
    }
}


__forceinline__ __device__ void warp_reduce_sum_float(float &val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    
}


template <typename scalar_t>
__global__ void ka_convolution_bwd_cuda_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    float* __restrict__ d_x,
    float* __restrict__ d_a,
    float* __restrict__ d_b, 
    float* __restrict__ d_bias,
    int numerator, 
    int denominator,
    int B, int H, int W, int out_channels, int in_channels, int k, int padded_h, int padded_w, int stride,
    int num_weights, int img_size, int padded_size, bool bias_enabled) {
    

    const int tile_width = blockDim.x;   
    const int tile_height = blockDim.y;  

    int tile_x = blockIdx.x * tile_width;
    int tile_y = blockIdx.y * tile_height;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    int lane = (threadIdx.y * blockDim.x + threadIdx.x) % 32; 

    int w_index = tile_x + thread_x;
    int h_index = tile_y + thread_y; 
    int num_check = numerator;


    if (w_index < W && h_index < H)  {  

        const int total = B * out_channels;

        for (int flat_idx = blockIdx.z; flat_idx < total; flat_idx += gridDim.z) {

            int out_c_idx = flat_idx % out_channels;
            int batch_idx = flat_idx / out_channels;

            int grad_idx = batch_idx * (img_size * out_channels) + out_c_idx * img_size + h_index * W + w_index;
            float grad_o = grad_output[grad_idx];


            for (int in_c = 0; in_c < in_channels; ++in_c) {
                
                for (int kh = 0; kh < k; ++kh) {

                    for (int kw = 0; kw < k; ++kw) {

                        int h_in = h_index * stride + kh;
                        int w_in = w_index * stride + kw;
                
                        int weight_idx = in_c  *k * k + kh * k + kw;
                
                        int x_idx = ((batch_idx * in_channels * padded_size) + in_c * padded_size + h_in * padded_w + w_in);
                        scalar_t xp = x[x_idx];
                        int w_full_idx = out_c_idx * num_weights + weight_idx;
                
                        int a_idx = w_full_idx * (n_order + 1);
                        int b_idx = w_full_idx * d_order;
                
                        
                        scalar_t shared_a[n_order + 1], shared_b[d_order];
                        #pragma unroll
                        for (int i = 0; i < n_order + 1; ++i) {
                            shared_a[i] = a[a_idx + i];
                        }
                        #pragma unroll
                        for (int i = 0; i < d_order; ++i) {
                            shared_b[i] = b[b_idx + i];
                            
                        }
                
                        scalar_t axp = abs(xp);

        
                        scalar_t P = shared_a[n_order];
                        #pragma unroll
                        for (int i = n_order - 1; i >= 0; --i) {
                            P = fmaf(P, xp, shared_a[i]);
                        }
                
                        scalar_t Q = abs(shared_b[d_order - 1]);
                        #pragma unroll
                        for (int i = d_order - 2; i >= 0; --i) {
                            Q = fmaf(Q, axp, abs(shared_b[i]));
                        }
                        Q = fmaf(Q, axp, 1.0);
                
                
                        scalar_t R = scalar_t(n_order) * shared_a[n_order];
                        #pragma unroll
                        for (int i = n_order - 1; i >= 1; --i) {
                            R = fmaf(R, xp, scalar_t(i) * shared_a[i]);
                        }


                        scalar_t S = scalar_t(0);

                        #pragma unroll
                        for (int i = d_order; i >= 1; --i) {
                            S = fmaf(S, axp, scalar_t(i) * abs(shared_b[i - 1]));
                        }

                        S = copysign(S, xp);


                        float local_da[n_order + 1] = {0}; // Local accumulation arrays
                        float local_db[d_order] = {0};
                

                        scalar_t one_div_Q = __frcp_rn(Q); //use frcp to compute 1 / Q so that P/Q can be rewritten as P * (1/Q)
                
                        scalar_t mpq2 = -P * (one_div_Q*one_div_Q); // -P / (Q**2)
                
                        scalar_t d_i_x = (R * one_div_Q + S * mpq2) * grad_o;
                
                        int d_x_idx = batch_idx * in_channels * img_size + in_c * img_size + h_index * W + w_index;

                        atomicAdd(&d_x[d_x_idx], d_i_x);
                
                        if (num_check == -1) { //if the numerator is set to -1 it means that only a_1 should be used as a numerator and the gradient of a_0 would be set to zero
                            numerator = 2;

                        } else {    
                            local_da[0] = one_div_Q * grad_o;
                        }
                
                        scalar_t xp_pow = xp;
                        #pragma unroll
                        for (int i = 0; i < numerator - 1; ++i) {
                            local_da[i + 1] = (xp_pow * one_div_Q) * grad_o;
                            xp_pow *= xp;
                        }
                    
                        scalar_t axp_pow = axp;
                        #pragma unroll
                        for (int i = 0; i < denominator; ++i) {
                            local_db[i] = mpq2 * copysign(scalar_t(1.0), shared_b[i]) * axp_pow * grad_o;
                            axp_pow *= axp;
                        }
                        
                        warp_reduce_sum(local_da, n_order + 1);
                        warp_reduce_sum(local_db, d_order);
                        
                        __syncwarp();

                        if (lane == 0) {
                            #pragma unroll
                            for(int i = 0; i < n_order + 1; ++i) {
                                atomicAdd(&d_a[a_idx + i], local_da[i]);
                            }
                            #pragma unroll
                            for(int i = 0; i < d_order; ++i) {
                                atomicAdd(&d_b[b_idx + i], local_db[i]);
                            }
                            
                        }
            
                    }   
                }
            } 
                
            if (bias_enabled) {
                warp_reduce_sum_float(grad_o);

                if (lane == 0) { 
                    atomicAdd(&d_bias[out_c_idx], grad_o); 
                }
            }

        }  
    }
}

std::vector<torch::Tensor> ka_convolution_bwd_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d, torch::Tensor bias, int out_channels, int kernel_size, int stride, int numerator, int denominator, bool bias_enabled) {

    int B = x.size(0);
    int in_channels = x.size(1);
    int padded_h = x.size(2);
    int padded_w = x.size(3);

    int H = (padded_h - kernel_size) / stride + 1;
    int W = (padded_w - kernel_size) / stride + 1;
    int num_weights = in_channels * kernel_size * kernel_size;

    const int x_size = B * out_channels * H * W;
    const int n_size = n.numel();
    const int d_size = d.numel();

    auto d_x = at::zeros({B, in_channels, H, W}, x.options()).toType(at::kFloat);
    auto d_n = at::zeros_like(n).toType(at::kFloat);
    auto d_d = at::zeros_like(d).toType(at::kFloat);
    auto d_bias = at::zeros_like(bias).toType(at::kFloat);

    int maxGridZ = 32768;
    int gridDimz = (B * out_channels <= maxGridZ) ? B * out_channels : maxGridZ;

    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16, gridDimz);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ka_convolution_bwd_cuda_kernel", ([&] {
    ka_convolution_bwd_cuda_kernel<scalar_t>
        <<<gridDim, blockDim>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<float>(),
            d_n.data_ptr<float>(),
            d_d.data_ptr<float>(),
            d_bias.data_ptr<float>(),
            numerator, denominator,
            B, H, W, out_channels, in_channels, kernel_size, padded_h, padded_w, stride , num_weights, H* W, padded_h * padded_w, bias_enabled);
    }));

    if (bias_enabled) {
        return {d_x, d_n, d_d, d_bias};
    }

    return {d_x, d_n, d_d};
}