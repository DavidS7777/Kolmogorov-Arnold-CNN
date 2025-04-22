import torch
import torch.nn as nn
import kaconvolution_cuda


class KaConvolution(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, nums, denoms, bias, out_channels, kernel_size, stride, numerator, denominator, bias_enabled):

        pad_size = (kernel_size - 1) // 2

        padded = torch.nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)  

        ctx.out_c = out_channels
        ctx.numerator = numerator
        ctx.denominator = denominator
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.bias_enabled = bias_enabled
    
        ctx.save_for_backward(padded, nums, denoms, bias)

        result = kaconvolution_cuda.ka_convolution_fwd(padded, nums, denoms, bias, out_channels, kernel_size, stride, bias_enabled)
       

        return result
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        
        
        padded, nums, denoms, bias = ctx.saved_tensors
        
        out_c = ctx.out_c
        numerator = ctx.numerator
        denominator = ctx.denominator
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        bias_enabled = ctx.bias_enabled

        
        if bias_enabled:
            d_input, d_weight_numerator, d_weight_denominator, d_bias = kaconvolution_cuda.ka_convolution_bwd(grad_output, padded, nums, denoms, bias, out_c, kernel_size, stride, numerator, denominator, bias_enabled)
            return d_input, d_weight_numerator, d_weight_denominator, d_bias, None, None, None, None, None, None
        else:
            d_input, d_weight_numerator, d_weight_denominator = kaconvolution_cuda.ka_convolution_bwd(grad_output, padded, nums, denoms, bias, out_c, kernel_size, stride, numerator, denominator, bias_enabled) 
            
        
        return d_input, d_weight_numerator, d_weight_denominator, None, None, None, None, None, None, None


class KAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order=(5, 4), bias=False):
        
        """
        
        Args:

        in_channels(int) = number of input channels

        out_channels(int) = number of output channels / filter
        
        kernel_size(int) = 1d shape of kernel, kernel will be quadratic

        order(int, int) = order of Polynomials P and Q. Kernel is hardcoded for (5, 4) so this parameter is only relevant for decreasing the order, Default=(5,4)

        bias = enable a bias term or not, Default=False


        """

        super(KAConv, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.groups = in_channels #grouping hasn't been implemented yet
        self.stride = 1
        self.bias_enabled = bias

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        total_size = out_channels * self.groups * kernel_size * kernel_size

        self.nums = nn.Parameter(num.repeat(total_size, 1))
        
        self.denoms = nn.Parameter(denom.repeat(total_size, 1))

        self.bias = nn.Parameter(torch.zeros(out_channels))


        torch.nn.init.xavier_uniform_(self.nums)
        torch.nn.init.xavier_uniform_(self.denoms)

        with torch.no_grad():
            self.denoms.mul_(2) #keeps numerator slighty regulated, which helps with early training stability
            if self.numerator == -1:
                self.nums[:, 0] = 0
                self.numerator = 2
            for i in range(self.numerator, 6): #sets not needed coefficients to zero when order != (5, 4)
                self.nums[:, i] = 0

            for i in range(self.denominator, 4):
                self.denoms[:, i] = 0



    def forward(self, input):

        result = KaConvolution.apply(input, self.nums, self.denoms, self.bias, self.out_channels, self.kernel_size, self.stride, self.numerator, self.denominator, self.bias_enabled)

        return result