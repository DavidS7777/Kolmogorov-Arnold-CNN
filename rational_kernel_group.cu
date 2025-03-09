#include <torch/extension.h>

template <typename scalar_t>
__global__ void rational_fwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b, 
    scalar_t* __restrict__ result, 
    int B, int L, int D, int group, 
    int x_size, int D_per_group, int num_weights) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= x_size) return;  // Prevent out-of-bounds memory access

    int w_index = idx / (B * L * D);
    int local_idx = idx % (B * L * D);
    // Calculate the index within the dimension D
    int d_index = local_idx % D;
    // Calculate the group index based on the position within dimension D
    int g_index = floor(d_index / D_per_group);

    
    

    // Calculate specific indices for a and b based on group
    int a_idx = w_index * 6;
    int b_idx = w_index * 4;
    // Load coefficients into registers
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
    scalar_t xp1 = x[idx];
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

    // Write the result of P / Q
    result[idx] = P / Q;
}


torch::Tensor rational_fwd_cuda_1dgroup(
    torch::Tensor x, 
    torch::Tensor n, 
    torch::Tensor d,
    int group
    ){
    auto result = at::empty_like(x);
    const int x_size = x.numel();
    int num_weights = x.size(0);
    int B = x.size(1);
    int L = x.size(2);
    int D = x.size(3);

    int threads_per_block = 256;  // Adjust as needed based on device capabilities
    int num_blocks = (x_size + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_fwd_cuda_1dgroup", ([&] {
    rational_fwd_cuda_kernel_1dgroup<scalar_t>
        <<<num_blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            B, L, D, group, x_size, D / group, num_weights);
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



template <typename scalar_t>
__global__ void rational_bwd_cuda_kernel_1dgroup(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ d_x,
    float* __restrict__ d_a,
    float* __restrict__ d_b,
    int B, int L, int D, int group, 
    int x_size, 
    const int n_size, 
    const int d_size,
    int D_per_group, 
    int numerator, 
    int denominator, 
    int num_weights) {
    
    // Shared memory for accumulation
    // group < 32, so we can use 192 and 128 shared memory
    
    //hardcoded for OneConvfor now -> biggest layer has 1 in 6 out 9 kernel weights
    // __shared__ float sda[32*6];
    // __shared__ float sdb[32*4];
    extern __shared__ float shared_mem[];
    float* sda = shared_mem;
    float* sdb = sda + (num_weights * 6);

    // initialize shared memory to zero
    if ( threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < num_weights * 6; ++i) {
            sda[i] = 0;
        }
        #pragma unroll
        for (int i = 0; i < num_weights * 4; ++i) {
            sdb[i] = 0;
        }
    }


    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= x_size) return;  // Prevent out-of-bounds memory access

    // Calculate the index within the dimension D
    int w_index = idx / (B * L * D);
    int local_idx = idx % (B * L * D);
    int d_index = local_idx % D;
    // Calculate the group index based on the position within dimension D
    int g_index = floor(d_index / D_per_group);

    //int w_index = (idx / D) % num_weights;

    // Calculate specific indices for a and b based on group
    int a_idx = w_index * 6;
    int b_idx = w_index * 4;

    // Load coefficients into registers
    scalar_t shared_a[6], shared_b_abs[4], shared_b[4];
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        shared_a[i] = a[a_idx + i];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        shared_b_abs[i] = abs(b[b_idx + i]);  // Store absolute values directly if needed
        shared_b[i] = b[b_idx + i];
    }

    scalar_t local_da[6] = {0}; // Local accumulation arrays
    scalar_t local_db[4] = {0};
    
    scalar_t xp = x[idx];
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
    

    scalar_t grad_o = grad_output[idx];
    
    scalar_t mpq2 = -P/(Q*Q);

    scalar_t d_i_x = (R / Q + S * mpq2) * grad_o;
    d_x[idx] = d_i_x;

    // Loop for computing d_a contributions
    
    if (numerator == -1) {
        numerator = 2;
    } else {    
        local_da[0] = scalar_t(1.0) / Q * grad_o;
    }

    //local_da[0] = scalar_t(1.0) / Q * grad_o;
    #pragma unroll
    for (int i = 1; i < numerator; ++i) {
        local_da[i] = (xp_powers[i-1] / Q) * grad_o;
    }

    // Loop for computing d_b contributions
    #pragma unroll
    for (int i = 0; i < denominator; ++i) {
        local_db[i] = mpq2 * copysign(scalar_t(1.0), shared_b[i]) * axp_powers[i] * grad_o;
    }

    // Reduce local arrays to shared memory
    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        atomicAdd(&sda[a_idx + i], local_da[i]);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        atomicAdd(&sdb[b_idx + i], local_db[i]);
    }

    __syncthreads();

    // Only one thread writes back to global memory
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < n_size; ++i) {
            atomicAdd(&d_a[i], sda[i]);
        }
        #pragma unroll
        for (int i = 0; i < d_size; ++i) {
            atomicAdd(&d_b[i], sdb[i]);
        }
    }
    
}

std::vector<torch::Tensor> rational_bwd_cuda_1dgroup(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d, int group, int numerator, int denominator) {
    const int x_size = x.numel();
    const int n_size = n.numel();
    const int d_size = d.numel();

    auto d_x = at::empty_like(x);
    auto d_n = at::zeros_like(n).toType(at::kFloat);
    auto d_d = at::zeros_like(d).toType(at::kFloat);

    int num_weights = x.size(0);
    int B = x.size(1);
    int L = x.size(2);
    int D = x.size(3);

    int blockSize = 256;  // You might want to experiment with this value
    int numBlocks = (x_size + blockSize - 1) / blockSize;
    int shared_mem_size = (num_weights * 10) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "rational_bwd_cuda_1dgroup", ([&] {
    rational_bwd_cuda_kernel_1dgroup<scalar_t>
        <<<numBlocks, blockSize, shared_mem_size>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            n.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(),
            d_x.data_ptr<scalar_t>(),
            d_n.data_ptr<float>(),
            d_d.data_ptr<float>(),
            B, L, D, group, x_size, n_size, d_size, D / group, numerator, denominator, num_weights);
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