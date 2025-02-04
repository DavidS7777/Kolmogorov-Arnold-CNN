import torch
import torch.nn as nn
import numpy as np
from kat_rational import KAT_Group

#just one Layer -> Pooling is later
class KAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mode=["swish", "swish"], num_groups=None, kernel_shape="even", order=(5, 4)):
        """
        
        Args:

        in_channels(int) = number of input channels

        out_channels(int) = number of output channels / filter
        
        kernel_size(int) = 1d shape of kernel, kernel will be quadratic

        device(string) = device

        mode(list of strings) = initial activation function

        num_groups(int) = the number of groups to divide in_channel with

        kernel_shape(string) = the shape of the different activiation functions in the kernel matrices.
                "even": use the first activation function for the entire kernel
                "plus": use the first activation function just in the middle rows/cols and the second on the rest
                "x": use the first activation function just on the diagonals and the second on the rest
                "alternating": same as even for one 1 in_c, otherwise alternative through dimensions


        """
        super(KAConv, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        #assert in_channels % num_groups == 0, "Input Channels should be divisible by the number of groups"

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
        if kernel_shape=="even":
            functions = [KAT_Group(num_groups=step, mode=mode[0], device='cuda', order=order) for _ in range(out_channels * kernel_size * kernel_size * self.groups)]

        # elif kernel_shape=="plus":
        # #for every filter there is a kernel matrix with num_groups depth, since every kernel weight operatres on in_c // groups channels at the same time
            
        #     for i in range(out_channels):
        #         for g in range(self.groups):
        #             for j in range(kernel_size):
        #                 for k in range(kernel_size):
                            
        #                     if j == kernel_size // 2 or k == kernel_size // 2:
        #                         m = mode[0]
        #                     else:
        #                         m = mode[1]    

        #                     group = KAT_Group(num_groups=1, mode=m, device='cuda')
        #                     functions.append(group)

        # elif kernel_shape=="x":
            
        #     for i in range(out_channels):
        #         for g in range(self.groups):
        #             for j in range(kernel_size):
        #                 for k in range(kernel_size):
                            
        #                     if j == k or ((j + k) == (kernel_size - 1)) :
        #                         m = mode[0]
        #                     else:
        #                         m = mode[1]    

        #                     group = KAT_Group(num_groups=1, mode=m, device='cuda')
        #                     functions.append(group)
        
        # elif kernel_shape=="alternating":
        #     for i in range(out_channels):
        #         for g in range(self.groups):
        #             for j in range(kernel_size):
        #                 for k in range(kernel_size):
                            
        #                     idx = g
        #                     if idx >= len(mode):
        #                         idx %= len(mode)

        #                     m = mode[idx]
                                
        #                     group = KAT_Group(num_groups=1, mode=m, device='cuda')
        #                     functions.append(group)
            


        # else:
        #     raise ValueError("Kernel_shape not implemented")
        
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
    
    # def different_forward(self, input):
        
    #     """
    #     Args:

    #     input: Image of shape [Batch, Channels, Height, Width]

    #     """

    #     batch = input.shape[0]
    #     height = input.shape[2]
    #     width = input.shape[3]

    #     in_c = self.in_channels
    #     out_c = self.out_channels
    #     kernel_size = self.kernel_size
    #     kernel = self.kernel
    #     device = self.device
    #     groups = self.groups

    #     step = in_c // groups

    #     pad_size = (kernel_size - 1) // 2
        
    #     padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
    #     padded = padded.permute(0, 2, 3, 1)
    #     result = torch.zeros((batch, out_c, height * width), device=device) #same dim because of padding

        
    #     for f in range(out_c):
    #         for c in range(groups):
    #             for a in range(kernel_size):
    #                 for b in range(kernel_size):
                    
    #                     idx = f * groups * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b #get corresponding Weight in "Matrix" from List
                    
    #                     """
    #                     - for each img in batch -> compute one feature map at a time
    #                     - for every feature map -> get corresponding weight and use it on every relevant pixel
    #                     - kernel uses [B, L, C] and pixel of shape [B, H, W, C]
    #                     - reshape [B, H, W, Step] to [B, H * W, Step] (channel is step long -> group computation)
    #                     - input now [B, H * W, Step]
    #                     - sum up channels to remove channel dim
    #                     - result[batch, feature_map] += img after conv with shape [H * W]

    #                     """
    #                     slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1)

    #                     if not slicee.is_contiguous():
    #                         slicee = slicee.contiguous()
                        
    #                     #process step channels at a time -> needs channel dim, since problems with backprop when mutliple channel dims
                                                 
    #                     result[:, f] += torch.sum(kernel[idx](slicee.unsqueeze(-1)).view(batch, -1, step), dim=-1)
        
        
    #     return result.view(batch, out_c, height, width)

    
    

class SimpleModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order=(5, 4)):
        super(SimpleModel, self).__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1 #for hydra

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=["identity"], order=order)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*14 * out_channels, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        
        out = nn.functional.relu(self.fc1(out))

        out = self.fc2(out)
        return out

#just for testing
class CustomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        
        super(CustomConv, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        functions = []
        functions = [nn.Parameter(torch.tensor(0.5, dtype=torch.float32)) for _ in range(out_channels * kernel_size * kernel_size * in_channels)]

        self.kernel = functions
        
    
    def forward(self, input):
        

        batch = input.shape[0]
        height = input.shape[2]
        width = input.shape[3]

        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        kernel = self.kernel

        step = 1

        pad_size = (kernel_size - 1) // 2
        
        padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        padded = padded.permute(0, 2, 3, 1)
        result = torch.zeros((batch, out_c, height * width), device='cuda') #same dim because of padding
        
        for f in range(out_c):
            for c in range(in_c):
                for a in range(kernel_size):
                    for b in range(kernel_size):
                    
                        idx = f * in_c * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b #get corresponding Weight in "Matrix" from List
                    
                        slicee = padded[:, a:a + height, b:b + width, c:c + step].reshape(batch, -1, step)

                        if not slicee.is_contiguous():
                            slicee = slicee.contiguous()
                        
                        
                        
                        #process step channels at a time -> needs channel dim, since problems with backprop when mutliple channel dims
                        
                        result[:, f] += torch.sum(kernel[idx] * (slicee), dim=-1)
        
        
        return result.view(batch, out_c, height, width)

    
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleCNN, self).__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1

        self.conv1 = CustomConv(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*14 * out_channels, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        
        out = nn.functional.relu(self.fc1(out))

        out = self.fc2(out)
        return out


class KANet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, kernel_size=3, mode=["swish", "swish"], num_groups=None, kernel_shape="even", second_out = None, order=(5, 4)):

        super(KANet5, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out

        self.groups = num_groups
        self.modes = mode
        
        self.kernel_shape = kernel_shape
        
        if second_out == None:
            second_out = out_channels * 2

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=mode, num_groups=num_groups, kernel_shape=kernel_shape, order=order)
        self.conv2 = KAConv(out_channels, second_out, kernel_size, mode=mode, num_groups=num_groups, kernel_shape=kernel_shape, order=order)
        

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        #self.different = different

    def forward(self, x):

        #if self.different:
            #x = self.conv1.different_forward(x)
        #else:
        x = self.conv1(x)
        x = self.pool(x)
        
        #if self.different:
            #x = self.conv2.different_forward(x)
        #else:
        x = self.conv2(x)
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        
        return x


# class KANet5_KAN(nn.Module):
#     def __init__(self, in_channels=1, out_channels=6, kernel_size=3, mode="swish"):

#         super(KANet5_KAN, self).__init__()

#         self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=mode)
#         self.conv2 = KAConv(6, 16, 3, mode=mode)

#         self.pool = nn.MaxPool2d((2, 2))

#         self.act1 = KAT_Group(1, "relu", "cuda")
#         self.fc1 = nn.Linear(16*7*7, 120)
#         self.act2 = KAT_Group(1, "relu", "cuda")
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):

        
#         x = self.conv1(x)
#         x = self.pool(x)

#         x = self.conv2(x)
#         x = self.pool(x)     

#         #flatten  
#         x = x.view(-1, 16*7*7)
        
#         x = self.fc1(x).unsqueeze(-1)
        
#         x = self.act1(x).squeeze()
#         #x = self.fc1(x)
        
#         x = self.act2(self.fc2(x).unsqueeze(-1)).squeeze()
#         #x = self.fc2(x)
#         x = self.fc3(x)

#         return x
