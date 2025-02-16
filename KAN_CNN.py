import torch
import torch.nn as nn
from kat_rational import KAT_Group
import kaconvolution_cu
from kat_rational import rational_1dgroup
from rational_triton import RationalTriton1DGroup

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

        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        
        relevant_input, nums, denoms = ctx.saved_tensors

        group = ctx.group
        numerator = ctx.numerator
        denominator = ctx.denominator

        d_input, d_weight_numerator, d_weight_denominator = kaconvolution_cu.ka_convolution_bwd(grad_output, relevant_input, nums, denoms, group, numerator, denominator)       


        return d_input, d_weight_numerator, d_weight_denominator, None, None, None, None, None

def helper_method(input, nums, denoms, group, numerator, denominator, out_c, kernel_size):

    
    relevant_inp, out = kaconvolution_cu.ka_convolution_fwd(input, nums, denoms, group, out_c, kernel_size)

    result = KaConvolution.apply(relevant_inp, nums, denoms, group, numerator, denominator, out) #needs to be saved for bwd -> rel_out needs to be returned by forward method for grad_output
    
    return result


class KAConvC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=None, order=(5, 4)):
        
        super(KAConvC, self).__init__()

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

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).unsqueeze(0).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))
        
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).unsqueeze(0).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0))
    
    
    def forward(self, input):

        result = helper_method(input, self.nums, self.denoms, self.groups, self.numerator, self.denominator, self.out_channels, self.kernel_size)
        
        og = torch.zeros((input.shape[0], self.out_channels, input.shape[2] * input.shape[3])).to("cuda")

        for idx, res in enumerate(result):
            f = idx // (self.kernel_size * self.kernel_size * self.in_channels)
            og[:, f] += torch.sum(res, dim=-1)
        
        og = og.view(input.shape[0], self.out_channels, input.shape[2], input.shape[3])

        return og
        
#just one Layer -> Pooling is later
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

        self.nums = nn.Parameter(torch.cat([nn.Parameter(num.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0).view(out_channels, in_channels, kernel_size, kernel_size, 6))
        self.denoms = nn.Parameter(torch.cat([nn.Parameter(denom.view(1, -1).float(), requires_grad=True) for _ in range(out_channels * kernel_size * kernel_size * self.groups)], dim=0).view(out_channels, in_channels, kernel_size, kernel_size, 4))
    
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
                    
                        slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1, step)

                        n = self.nums[f,c,a,b]
                        d = self.denoms[f,c,a,b]

                        if not slicee.is_contiguous():
                            slicee = slicee.contiguous()
                                                 
                        result[:, f] += torch.sum(rational_1dgroup.apply(slicee, n, d, self.step, 6, 4), dim=-1)
             
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

