import torch
import torch.nn as nn
from kat_rational import KAT_Group
import kaconvolution_cu
import kaconvolutiontest
from kat_rational import rational_1dgroup
from rational_triton import RationalTriton1DGroup
import rational_triton_vectorized
from rational_triton_vectorized import RationalTritonVectorized
import time
import math


class KaConvolution(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, nums, denoms, group, numerator, denominator, x):

        

        ctx.save_for_backward(input, nums, denoms)
        ctx.group = group
        ctx.numerator = numerator
        ctx.denominator = denominator
        
        return x
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        
        
        torch.cuda.empty_cache()
        
        relevant_input, nums, denoms = ctx.saved_tensors
        
        group = ctx.group
        numerator = ctx.numerator
        denominator = ctx.denominator     

        max_weight = 800 #32 KB shared max
        if grad_output.shape[0] > max_weight:

            num_weights = grad_output.shape[0]
            partitions = math.ceil(num_weights / max_weight)
            
            d_x_p = []
            d_n_p = []
            d_d_p = []

            for i in range(partitions):
                #torch.cuda.synchronize()
                torch.cuda.empty_cache() #everything previously used by the kernel is irrelevant for the calculation of the next batch
                start_idx = i * max_weight
                end_idx = min((i + 1) * max_weight, num_weights)

                grad_out_part = grad_output[start_idx:end_idx]
                x_part = relevant_input[start_idx:end_idx]
                n_part = nums[start_idx:end_idx]
                d_part = denoms[start_idx:end_idx]
                
                res = kaconvolution_cu.ka_convolution_bwd(grad_out_part, x_part, n_part, d_part, group, numerator, denominator)
                #torch.cuda.synchronize()

                
                
                d_x_p.append(res[0])
                d_n_p.append(res[1])
                d_d_p.append(res[2])
        
            
            
            d_input = torch.cat(d_x_p, 0)
            d_weight_numerator = torch.cat(d_n_p, 0)
            d_weight_denominator = torch.cat(d_d_p, 0)
            

        else:
        
            d_input, d_weight_numerator, d_weight_denominator = kaconvolution_cu.ka_convolution_bwd(grad_output, relevant_input, nums, denoms, group, numerator, denominator)       
            #torch.cuda.synchronize()

        torch.cuda.empty_cache()
        
        # H = int(math.sqrt(d_input.shape[2]))
        # B = grad_output.shape[1]
        # d_input = d_input.view(2, 3, -1, B, H, H).sum(dim=(0, 2)).permute(1, 0, 2, 3).contiguous()
        # print(d_input)
        # print(d_weight_numerator)
        # print(d_weight_denominator)
        #exit()

        return d_input, d_weight_numerator, d_weight_denominator, None, None, None, None

def helper_method(input, nums, denoms, group, out_c, kernel_size, numerator, denominator):

    
    torch.cuda.empty_cache()

    batch = input.size(0)
    in_c = input.size(1)
    height = input.size(2)
    width = input.size(3)

    pad_size = (kernel_size - 1) // 2

    padded = torch.nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
    
    step = in_c // group
    
    #perform im2col tranformation to get part of the image that is relevant for each weight

    relevant_inp = padded.unfold(2, height, 1).unfold(3, width, 1).contiguous().view(batch, -1, height * width, step).permute(1, 0, 2, 3).repeat(out_c, 1, 1, 1).contiguous() #<- this is too big already

    result = kaconvolution_cu.ka_convolution_fwd(relevant_inp, nums, denoms, group)
    
    result = KaConvolution.apply(relevant_inp, nums, denoms, group, numerator, denominator, result) #needs to be saved for bwd -> rel_out needs to be returned by forward method for grad_output

    torch.cuda.empty_cache()

    return result



class KaConvolutionComparision(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, nums, denoms, out_channels, kernel_size, stride, numerator, denominator):

        pad_size = (kernel_size - 1) // 2

        padded = torch.nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)

        #ctx.save_for_backward(input, nums, denoms)
        ctx.out_c = out_channels
        ctx.numerator = numerator
        ctx.denominator = denominator
        ctx.kernel_size = kernel_size
        ctx.stride = stride
    
        ctx.save_for_backward(padded, nums, denoms)

        result = kaconvolutiontest.ka_convolution_fwd(padded, nums, denoms, out_channels, kernel_size, stride)
       

        return result
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        
       
        torch.cuda.empty_cache()
        
        padded, nums, denoms = ctx.saved_tensors

        out_c = ctx.out_c
        numerator = ctx.numerator
        denominator = ctx.denominator
        kernel_size = ctx.kernel_size
        stride = ctx.stride

        num_weights = padded.shape[1] * kernel_size * kernel_size
        
        #TODO slicing not 100% accurate results -> floats? -> with partitions faster to ignore shared memory -> if blockSize then 8*8 since 256*9*10*4 to big

        # max_weight = 800 #32 KB shared max
        # if nums.shape[0] > max_weight:


        #     parallel_out_c = max_weight // num_weights
        #     parallel_out_c = min(parallel_out_c, out_c)

        #     partitions = math.ceil(out_c / parallel_out_c)

        #     pad_size = (kernel_size - 1) // 2

        #     d_x_p = torch.zeros(padded.shape[0], padded.shape[1], padded.shape[2] - 2 * pad_size, padded.shape[3] - 2 * pad_size, device="cuda")
            
        #     d_n_p = []
        #     d_d_p = []

        #     for i in range(partitions):
                
        #         torch.cuda.empty_cache() #everything previously used by the kernel is irrelevant for the calculation of the next batch
        #         start_idx = i * parallel_out_c
        #         end_idx = min((i + 1) * parallel_out_c, out_c)

        #         out_c_part = end_idx - start_idx

        #         grad_out_part = grad_output[:, start_idx:end_idx, :, :].contiguous()

        #         x_part = padded

        #         n_part = nums[start_idx * num_weights:end_idx * num_weights].contiguous()
        #         d_part = denoms[start_idx * num_weights:end_idx * num_weights].contiguous()

                
        #         res = kaconvolutiontest.ka_convolution_bwd(grad_out_part, padded, n_part, d_part, out_c_part, kernel_size, stride, numerator, denominator)
                
        #         #print(grad_out_part)

        #         d_x_p.add_(res[0])
        #         d_n_p.append(res[1])
        #         d_d_p.append(res[2])

        #     #torch.cuda.empty_cache()
            
        #     #print(res[1])
            
        #     d_weight_numerator = torch.cat(d_n_p, 0)
        #     d_weight_denominator = torch.cat(d_d_p, 0)
        #     d_input = d_x_p

        # else:
        
        d_input, d_weight_numerator, d_weight_denominator = kaconvolutiontest.ka_convolution_bwd(grad_output, padded, nums, denoms, out_c, kernel_size, stride, numerator, denominator)       
            
        torch.cuda.empty_cache()
        
        #torch.set_printoptions(sci_mode=False)
        
        # print(d_input)
        # print(d_weight_numerator.sum())
        # print(d_weight_denominator.sum())
        # exit()

        return d_input, d_weight_numerator, d_weight_denominator, None, None, None, None, None, None



def helper_method_comp(input, nums, denoms, out_c, kernel_size, stride, numerator, denominator):


    pad_size = (kernel_size - 1) // 2

    padded = torch.nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
    # step = in_c // group

    # #perform im2col tranformation to get part of the image that is relevant for each weight
    # relevant_inp = padded.unfold(2, height, 1).unfold(3, width, 1).contiguous().view(batch, -1, height * width, step)
    
    # relevant_inp = relevant_inp.permute(1, 0, 2, 3).contiguous()
    

    result = KaConvolutionComparision.apply(padded, nums, denoms, out_c, kernel_size, stride, numerator, denominator) #needs to be saved for bwd -> rel_out needs to be returned by forward method for grad_output
    
    #result = result.view(batch, -1, height, width)
    

    return result


class KAConvC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order=(5, 4)):
        
        super(KAConvC, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        #no grouping for now -> can't dynamically allocate in CUDA
        self.groups = in_channels 


        step = in_channels // self.groups
        self.step = step

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda")

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * self.groups * kernel_size * kernel_size)], dim=0))
        
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * self.groups * kernel_size * kernel_size)], dim=0))
    
    def forward(self, input):

        result = helper_method(input, self.nums, self.denoms, self.groups, self.out_channels, self.kernel_size, self.numerator, self.denominator)
        
        # og = torch.zeros((input.shape[0], self.out_channels, input.shape[2] * input.shape[3]), device="cuda")
        
        # for idx, res in enumerate(result):
        #     f = idx // (self.kernel_size * self.kernel_size * self.in_channels) 
        #     og[:, f] += torch.sum(res, dim=-1)
        
        # og = og.view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

        
        # og2 = torch.zeros((input.shape[0], self.out_channels, input.shape[2] * input.shape[3]), device="cuda")
        
        # for idx, res in enumerate(result):
        #     f = idx // (self.kernel_size * self.kernel_size * self.in_channels) 
        #     og2[:, f].add_(torch.sum(res, dim=-1))
            
        
        # og2= og2.view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

        og = result.view(self.out_channels, -1, result.shape[1], result.shape[2], result.shape[3])
        og = og.sum(1)
        og = og.permute(1,0,2,3).view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

        
        # print(og[0, 0, :, :])
        # exit()
        
        return og

class KAConvCComparision(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order=(5, 4), stride=1):
        
        super(KAConvCComparision, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        #no grouping for now -> can't dynamically allocate in CUDA
        self.groups = in_channels


        step = in_channels // self.groups
        self.step = step
        self.stride = stride

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0], device="cuda")

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * self.groups * kernel_size * kernel_size)], dim=0))
        
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * self.groups * kernel_size * kernel_size)], dim=0))
    
    def forward(self, input):

        result = KaConvolutionComparision.apply(input, self.nums, self.denoms, self.out_channels, self.kernel_size, self.stride, self.numerator, self.denominator)

        return result

class KAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode=["swish", "swish"], num_groups=None, order=(5, 4)):
        """
        
        Args:

        in_channels(int) = number of input channels

        out_channels(int) = number of output channels / filter
        
        kernel_size(int) = 1d shape of kernel, kernel will be quadratic

        device(string) = device

        mode(list of strings) = initial activation function

        num_groups(int) = the number of groups to divide in_channel with


        """
        super(KAConv, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        if num_groups != None and in_channels >= num_groups and in_channels % num_groups == 0:
            self.groups = num_groups
        else:
            #no grouping
            self.groups = in_channels

        #for indexing if kernel_shape == even
        if type(mode) == str:
            mode = [mode]

        functions = []

        step = in_channels // self.groups
        
        functions = [KAT_Group(num_groups=step, mode=mode[0], device='cuda', order=order) for _ in range(out_channels * kernel_size * kernel_size * self.groups)]
        
        self.kernel = nn.ModuleList(functions)
    

    def forward(self, input):
        
        """
        Args:

        input: Image of shape [Batch, Channels, Height, Width]

        """

        batch = input.shape[0]
        height = input.shape[2]
        width = input.shape[3]

        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        kernel = self.kernel
        groups = self.groups

        step = in_c // groups

        pad_size = (kernel_size - 1) // 2
        
        padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        padded = padded.permute(0, 2, 3, 1)
        result = torch.zeros((batch, out_c, height * width), device='cuda') #same dim because of padding

        
        for f in range(out_c):
            for c in range(groups):
                for a in range(kernel_size):
                    for b in range(kernel_size):
                    
                        idx = f * groups * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b #get corresponding Weight in "Matrix" from List
                        """
                        - for each img in batch -> compute one feature map at a time
                        - for every feature map -> get corresponding weight and use it on every relevant pixel
                        - kernel uses [B, L, C] and pixel of shape [B, H, W, C]
                        - reshape [B, H, W, Step] to [B, H * W, Step] (channel is step long -> group computation)
                        - input now [B, H * W, Step]
                        - sum up channels to remove channel dim
                        - result[batch, feature_map] += img after conv with shape [H * W]

                        """
                        slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1, step)

                        if not slicee.is_contiguous():
                            slicee = slicee.contiguous()
                        
                        #process step channels at a time -> needs channel dim, since problems with backprop when mutliple channel dims
                                                
                        result[:, f] += torch.sum(kernel[idx](slicee), dim=-1)
                        
        
        return result.view(batch, out_c, height, width)
    
class KAConvComparision(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=None, order=(5, 4)):
        
        super(KAConvComparision, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        if num_groups != None and in_channels >= num_groups and in_channels % num_groups == 0:
            self.groups = num_groups
        else:
            #no grouping
            self.groups = in_channels


        step = in_channels // self.groups
        self.step = step

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).to('cuda')
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0]).to('cuda')

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0).view(out_channels*in_channels*kernel_size*kernel_size, 6))
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0).view(out_channels*in_channels*kernel_size*kernel_size, 4))
    
    def forward(self, input):

        batch = input.shape[0]
        height = input.shape[2]
        width = input.shape[3]

        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        groups = self.groups

        step = in_c // groups

        pad_size = (kernel_size - 1) // 2
        
        padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        padded = padded.permute(0, 2, 3, 1)
        result = torch.zeros((batch, out_c, height * width), device='cuda') #same dim because of padding

        
        for f in range(out_c):
            for c in range(groups):
                for a in range(kernel_size):
                    for b in range(kernel_size):

                        idx = f * groups * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b

                        slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1, step)

                        n = self.nums[idx]
                        d = self.denoms[idx]

                        if not slicee.is_contiguous():
                            slicee = slicee.contiguous()
                                                 
                        result[:, f] += torch.sum(rational_1dgroup.apply(slicee, n, d, self.step, self.numerator, self.denominator), dim=-1)
             
        return result.view(batch, out_c, height, width)
    
class KAConvTriton(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=None, order=(5, 4)):
        
        super(KAConvTriton, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        if num_groups != None and in_channels >= num_groups and in_channels % num_groups == 0:
            self.groups = num_groups
        else:
            #no grouping
            self.groups = in_channels


        step = in_channels // self.groups
        self.step = step

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).to('cuda')
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0]).to('cuda')

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))
    
    
    def forward(self, input):

        batch = input.shape[0]
        height = input.shape[2]
        width = input.shape[3]

        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        groups = self.groups

        step = in_c // groups

        pad_size = (kernel_size - 1) // 2
        
        padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        padded = padded.permute(0, 2, 3, 1)
        result = torch.zeros((batch, out_c, height * width), device='cuda') #same dim because of padding

        
        for f in range(out_c):
            for c in range(groups):
                for a in range(kernel_size):
                    for b in range(kernel_size):
                    
                        idx = f * groups * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b #get corresponding Weight in "Matrix" from List
                        
                        slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1, step)

                        n = self.nums[idx]
                        d = self.denoms[idx]

                        if not slicee.is_contiguous():
                            slicee = slicee.contiguous()
                        
                                                 
                        result[:, f] += torch.sum(RationalTriton1DGroup.apply(slicee, n, d, self.step), dim=-1)
             
        return result.view(batch, out_c, height, width)

class KAConvTritonVectorized(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=None, order=(5, 4)):
        
        super(KAConvTritonVectorized, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels


        if num_groups != None and in_channels >= num_groups and in_channels % num_groups == 0:
            self.groups = num_groups
        else:
            #no grouping
            self.groups = in_channels


        step = in_channels // self.groups
        self.step = step

        self.numerator = order[0] + 1 if order[0] != -1 else -1
        self.denominator = order[1]

        num = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).to('cuda')
        denom = torch.tensor([0.0, 0.0, 0.0, 0.0]).to('cuda')

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))

        self.num_weights = out_channels * kernel_size * kernel_size * self.groups
    
    
    def forward(self, input):

        result = rational_triton_vectorized.helper_method(input, self.nums, self.denoms, self.step, self.numerator, self.denominator, self.out_channels, self.kernel_size)
        
        #print(result.shape)
        #exit()
        og = torch.zeros((input.shape[0], self.out_channels, input.shape[2] * input.shape[3])).to("cuda")

        for idx, res in enumerate(result):
            f = idx // (self.kernel_size * self.kernel_size * self.in_channels)
            og[:, f] += torch.sum(res, dim=-1)
        
        og = og.view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

        return og
             
        return result