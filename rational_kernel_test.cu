#include <torch/extension.h>

template <typename scalar_t>
__global__ void rational_fwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    float* __restrict__ result, 
    int B, int H, int W, int out_channels, int in_channels, int k, int padded_h, int padded_w, int stride,
    int x_size, int num_weights) {


    const int tile_width = blockDim.x;   // e.g. 16
    const int tile_height = blockDim.y;  // e
    
    int tile_x = blockIdx.x * tile_width;
    int tile_y = blockIdx.y * tile_height;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;
    
    int w_index = tile_x + thread_x;  // output width position
    int h_index = tile_y + thread_y; 
    
    int out_c_idx = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;
    
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (w_index >= W || h_index >= H) return;  

    //int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //if (idx >= x_size) return;  // Prevent out-of-bounds memory access

    // int w_index = idx % W;
    // int h_index = (idx / W) % W;
    
    // int out_channel = (idx / (W * H)) % out_channels;
    // int batch_idx = idx / (W * H * out_channels);

    scalar_t accum_value = 0;

    for (int weight_idx = 0; weight_idx < num_weights; ++weight_idx) {

        int in_c = weight_idx / (k * k);
        int kernel_offset = weight_idx % (k * k);
        int kh = kernel_offset / k;
        int kw = kernel_offset % k;

        int h_in = h_index * stride + kh;
        int w_in = w_index * stride + kw;

        int x_idx = ((batch_idx * in_channels + in_c) * padded_h + h_in) * padded_w + w_in; //not coalesced and could load the same thing multiple times, since iterating over out_C

        int w_full_idx = out_c_idx * num_weights + weight_idx;

        int a_idx = w_full_idx * 6;
        int b_idx = w_full_idx * 4;

        scalar_t s_a[6], s_b[4];
        #pragma unroll
        for (int i = 0; i < 6; ++i) {
            s_a[i] = a[a_idx + i];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s_b[i] = abs(b[b_idx + i]);  // Store absolute values directly if needed
        }

        // Obtain the input value from the tensor
        scalar_t xp1 = x[x_idx];
        scalar_t abs_xp1 = abs(xp1);

        // Compute the polynomial for P using Horner's method
        scalar_t P = s_a[5];
        #pragma unroll
        for (int i = 4; i >= 0; --i) {
            P = fmaf(P, xp1, s_a[i]);
        }
        
        // Compute the polynomial for Q using Horner's method
        scalar_t Q = s_b[3];
        #pragma unroll
        for (int i = 2; i >= 0; --i) {
            Q = fmaf(Q, abs_xp1, s_b[i]);
        }
        Q = fmaf(Q, abs_xp1, 1.0);

        scalar_t one_div_Q = __frcp_rn(Q);

        accum_value = fmaf(P, one_div_Q, accum_value); // += P/Q

    }

    
    // Write the result of P / Q
    
    int result_idx = ((batch_idx * out_channels + out_c_idx) * H + h_index) * W + w_index;

    result[result_idx] = accum_value;
    
}


torch::Tensor rational_fwd_cuda_1dgroup(
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d,
    int out_channels, int kernel_size, int stride
    ){
    
        
    int B = x.size(0);
    int in_channels = x.size(1);
    int padded_h = x.size(2);
    int padded_w = x.size(3);

    int H = (padded_h - kernel_size) / stride + 1;
    int W = (padded_w - kernel_size) / stride + 1;
    int num_weights = in_channels * kernel_size * kernel_size;

    
    auto result = at::zeros({B, out_channels, H, W}, x.options()).toType(at::kFloat);
    

    //x = x.repeat({out_channels, 1, 1, 1});
    const int x_size = B * out_channels * H * W;

    int blockSize = 16;
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim((W + blockSize - 1) / blockSize, (H + blockSize - 1) / blockSize, B * out_channels);
    //x = x.repeat({out_channels, 1, 1, 1});

    int threads_per_block = 256;  // Adjust as needed based on device capabilities
    int num_blocks = (x_size + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_fwd_cuda_1dgroup", ([&] {
    rational_fwd_cuda_kernel_1dgroup<scalar_t>
        <<<gridDim, blockDim>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<float>(),
            B, H, W, out_channels, in_channels, kernel_size, padded_h, padded_w, stride, x_size, num_weights);
        }));

    return result;
}

//P(X) = a_0 + a_1*X + a_2*X^2 ...
//Q(X) = 1 + |b_0||X| + |b_1||X|^2 + |b_2||X|^3
//R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
//S(X) = sign(X) * ( |b_0| + 2|b_1||X| + 3|b_2||X|^2 ...)
//dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
//dF/da_i = x^i/Q(X), i \in {0,5}
//dF/db_i = (-P(X)/Q(X)^2) * sign(b_i) * |X^{i+1}| , i \in {0,4}


//Try NHWC layout instead of channels second
template <typename scalar_t>
__global__ void rational_bwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    float* __restrict__ d_x,
    float* __restrict__ d_a,
    float* __restrict__ d_b,
    int x_size, 
    const int n_size, 
    const int d_size, 
    int numerator, 
    int denominator,
    int B, int H, int W, int out_channels, int in_channels, int k, int padded_h, int padded_w, int stride,
    int num_weights) {
    

    const int tile_width = blockDim.x;   // e.g. 16
    const int tile_height = blockDim.y;  // e

    int tile_x = blockIdx.x * tile_width;
    int tile_y = blockIdx.y * tile_height;
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int w_index = tile_x + thread_x;  // output width position
    int h_index = tile_y + thread_y; 

    int out_c_idx = blockIdx.z % out_channels;
    int batch_idx = blockIdx.z / out_channels;


    

    if (w_index >= W || h_index >= H) return;  // Prevent out-of-bounds memory access

    int grad_idx = batch_idx * (H * W * out_channels) + out_c_idx * H * W + h_index * W + w_index;
    scalar_t grad_o = grad_output[grad_idx];

    for (int in_c = 0; in_c < in_channels; ++in_c) {

        for (int kh = 0; kh < k; ++kh) {

            for (int kw = 0; kw < k; ++kw) {

                int h_in = h_index * stride + kh;
                int w_in = w_index * stride + kw;
        
                int weight_idx = in_c  *k * k + kh * k + kw;
        
                int x_idx = ((batch_idx * in_channels * padded_h * padded_w) + in_c * padded_h * padded_w + h_in * padded_w + w_in);
                int w_full_idx = out_c_idx * num_weights + weight_idx;
        
                int a_idx = w_full_idx * 6;
                int b_idx = w_full_idx * 4;
        
                // Load coefficients into registers
                scalar_t shared_a[6], shared_b_abs[4], shared_b[4];
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    shared_a[i] = a[a_idx + i];
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    shared_b[i] = b[b_idx + i];
        
                    shared_b_abs[i] = abs(shared_b[i]);  // Store absolute values directly if needed
                    
                }
        
                scalar_t local_da[6] = {0}; // Local accumulation arrays
                scalar_t local_db[4] = {0};
                
                scalar_t xp = x[x_idx];
                scalar_t axp = abs(xp);
                // Compute powers of xp
                scalar_t xp_powers[5];
                xp_powers[0] = xp;
                xp_powers[1] = xp * xp_powers[0]; // xp^2
                xp_powers[2] = xp * xp_powers[1]; // xp^3
                xp_powers[3] = xp * xp_powers[2]; // xp^4
                xp_powers[4] = xp * xp_powers[3]; // xp^5
        
                // Compute powers of axp
                scalar_t axp_powers[4];
                axp_powers[0] = axp;
                axp_powers[1] = axp * axp_powers[0]; // axp^2
                axp_powers[2] = axp * axp_powers[1]; // axp^3
                axp_powers[3] = axp * axp_powers[2]; // axp^4
        
                // Compute absolute values once
        
                scalar_t P = shared_a[0] 
                + shared_a[1] * xp_powers[0] 
                + shared_a[2] * xp_powers[1] 
                + shared_a[3] * xp_powers[2] 
                + shared_a[4] * xp_powers[3] 
                + shared_a[5] * xp_powers[4];
        
                scalar_t Q = scalar_t(1.0)
                + shared_b_abs[0] * axp_powers[0] 
                + shared_b_abs[1] * axp_powers[1] 
                + shared_b_abs[2] * axp_powers[2] 
                + shared_b_abs[3] * axp_powers[3];
        
        
                scalar_t R = shared_a[1] 
                + scalar_t(2.0) * shared_a[2] * xp_powers[0] 
                + scalar_t(3.0) * shared_a[3] * xp_powers[1] 
                + scalar_t(4.0) * shared_a[4] * xp_powers[2] 
                + scalar_t(5.0) * shared_a[5] * xp_powers[3];
        
                scalar_t S = copysign(scalar_t(1.0), xp) * (shared_b_abs[0] 
                + scalar_t(2.0) * shared_b_abs[1] * axp_powers[0] 
                + scalar_t(3.0) * shared_b_abs[2] * axp_powers[1] 
                + scalar_t(4.0) * shared_b_abs[3] * axp_powers[2]);
                
                
                
        
                scalar_t one_div_Q = __frcp_rn(Q); //use frcp to compute 1 / Q so that P/Q can be rewritten as P * (1/Q)
        
                scalar_t mpq2 = -P * (one_div_Q*one_div_Q);
        
                scalar_t d_i_x = (R * one_div_Q + S * mpq2) * grad_o;
        
                
                int d_x_idx = batch_idx * in_channels * H * W + in_c * H * W + h_index * W + w_index;
                
        
                atomicAdd(&d_x[d_x_idx], d_i_x); //accumulate for in_c first for every in_c there are out_c adds -> add all out_c in shared and then in global
        
                local_da[0] = (numerator != -1) ? (scalar_t(1.0) * one_div_Q * grad_o) : local_da[0];
                numerator = (numerator == -1) ? 2 : numerator;
        
                #pragma unroll
                for (int i = 0; i < numerator - 1; ++i) {
                    local_da[i + 1] = (xp_powers[i] * one_div_Q) * grad_o;
                }
            
                // Loop for computing d_b contributions
                #pragma unroll
                for (int i = 0; i < denominator; ++i) {
                    local_db[i] = mpq2 * copysign(scalar_t(1.0), shared_b[i]) * axp_powers[i] * grad_o;
                }
        
                #pragma unroll
                for (int i = 0; i < 6; ++i) {
                    atomicAdd(&d_a[a_idx + i], local_da[i]);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    atomicAdd(&d_b[b_idx + i], local_db[i]);
                }
        
            }   
            }
        }

        // int in_c = weight_idx / (k * k);
        // int kernel_offset = weight_idx % (k * k);
        // int kh = kernel_offset / k;
        // int kw = kernel_offset % k;

        
    
}

std::vector<torch::Tensor> rational_bwd_cuda_1dgroup(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d, int out_channels, int kernel_size, int stride, int numerator, int denominator) {

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


    //int blockSize = 256;  // You might want to experiment with this value
    dim3 blockDim(16, 16);
    dim3 gridDim((W + 15) / 16, (H + 15) / 16, B * out_channels);
    //int numBlocks = (x_size + blockSize - 1) / blockSize;
    
    //int shared_mem_size = (out_channels * num_weights * 10) * sizeof(float);
    //int shared_mem_elements = out_channels * num_weights * 10;


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_bwd_cuda_1dgroup", ([&] {
    rational_bwd_cuda_kernel_1dgroup<scalar_t>
        <<<gridDim, blockDim>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<float>(),
            d_n.data_ptr<float>(),
            d_d.data_ptr<float>(),
            x_size, n_size, d_size, numerator, denominator,
            B, H, W, out_channels, in_channels, kernel_size, padded_h, padded_w, stride , num_weights);
    }));


    return {d_x, d_n, d_d};
}


// template <typename scalar_t>
// __global__ void imtocol_kernel(const scalar_t* input, scalar_t* output, int batch_size, int channels, int height, int width, int out_size, int kernel_size) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= batch_size * channels * out_size * out_size * kernel_size * kernel_size) return;

//     int wh = width * height;
//     int owh = out_size * out_size;

//     int batch = idx / (channels * owh * kernel_size * kernel_size);
//     int ch_idx = (idx % (channels * owh * kernel_size * kernel_size)) / (owh * kernel_size * kernel_size);
//     int oh_idx = (idx % (owh * kernel_size * kernel_size)) / (out_size * kernel_size * kernel_size);
//     int ow_idx = (idx % (out_size * kernel_size * kernel_size)) / (kernel_size * kernel_size);
//     int kh_idx = (idx % (kernel_size * kernel_size)) / kernel_size;
//     int kw_idx = idx % kernel_size;

//     int h_offset = oh_idx + kh_idx;
//     int w_offset = ow_idx + kw_idx;

//     output[idx] = input[batch * channels * wh + ch_idx * wh + h_offset * width + w_offset];

// }

// torch::Tensor imtocol_cuda(torch::Tensor input, int kernel_size) {

//     int batch_size = input.size(0);
//     int channels = input.size(1);
//     int height = input.size(2); //padded
//     int width = input.size(3); //padded

//     //printf("%d", height);

//     int out_h = height - kernel_size + 1;
//     int out_w = width - kernel_size + 1;

//     auto output = torch::zeros({batch_size, channels, out_h, out_w, kernel_size, kernel_size}, torch::device(input.device()).dtype(input.dtype()));

//     int num_kernels = batch_size * channels * out_h * out_w * kernel_size * kernel_size;
//     int block_size = 256;
//     int num_blocks = (num_kernels + block_size - 1) / block_size;

//     AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "imtocol_kernel", ([&] {
//         imtocol_kernel<scalar_t><<<num_blocks, block_size>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), batch_size, channels, height, width, out_h, kernel_size);}));

//     return output;

// }