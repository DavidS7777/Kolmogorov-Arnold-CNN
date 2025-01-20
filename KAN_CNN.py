import torch
import torch.nn as nn
import numpy as np
from kat_rational import KAT_Group

#just one Layer -> Pooling is later
class KANCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device="cuda", mode=["swish", "swish"], num_groups=None):
        """
        
        Args:

        in_channels(int) = number of input channels

        out_channels(int) = number of output channels / filter
        
        kernel_size(int) = 1d shape of kernel, kernel will be quadratic

        device(string) = device

        mode(list of strings) = initial activation function

        num_groups(int) = the number of groups to divide in_channel with


        """
        super(KANCNN, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        #assert in_channels % num_groups == 0, "Input Channels should be divisible by the number of groups"

        if num_groups != None and in_channels >= num_groups and in_channels % num_groups == 0:
            self.groups = num_groups
        else:
            #no grouping
            self.groups = in_channels

        self.modes = mode
        

        #for every filter there is a kernel matrix with num_groups depth, since every kernel weight operatres on in_c // groups channels at the same time
        # functions = []
        # for i in range(out_channels):
        #     for j in range(kernel_size):
        #         for k in range(kernel_size):
                    
        #             if j == kernel_size // 2 or k == kernel_size // 2:
        #                 m = mode[1]
        #             else:
        #                 m = mode[0]    

        #             group = KAT_Group(num_groups=1, mode=m, device='cuda')
        #             functions.append(group)

        functions = [KAT_Group(num_groups=1, mode=mode, device='cuda') for _ in range(out_channels * kernel_size * kernel_size * self.groups)]
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
        device = self.device
        groups = self.groups

        step = in_c // groups

        pad_size = (kernel_size - 1) // 2
        
        padded = nn.functional.pad(input, pad=[pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        padded = padded.permute(0, 2, 3, 1)
        result = torch.zeros((batch, out_c, height * width), device=device) #same dim because of padding
        
        for f in range(out_c):
            for a in range(kernel_size):
                for b in range(kernel_size):
                    for c in range(groups):
                
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
                        
                        #process step channels at a time -> needs channel dim, since problems with backprop when mutliple channel dims
                        result[:, f] += torch.sum(kernel[idx]((padded[:, a:a + height, b:b + width, c:c+step].reshape(batch, -1)).unsqueeze(-1)).view(batch, -1, step), dim=-1)
        
        return result.view(batch, out_c, height, width)
    
    
    # def different_fwd(self, input):
        
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
    #         for a in range(kernel_size):
    #             for b in range(kernel_size):
    #                 idx = f * kernel_size * kernel_size + a * kernel_size + b #get corresponding Weight in "Matrix" from List

    #                 # res = padded[:, a:a+height, b:b+width, :].reshape(batch, -1)
    #                 # print(res.shape)
    #                 # res = kernel[idx](res.unsqueeze(-1))
    #                 # print(res.shape)
    #                 # res = res.view(batch, -1, in_c)
    #                 # print(res.shape)

    #                 result[:, f] += torch.sum(kernel[idx]((padded[:, a:a+height, b:b+width, :].reshape(batch, -1)).unsqueeze(-1)).view(batch, -1, in_c), dim=-1)

    #     #print("hier", result.shape)
    #     return result.view(batch, out_c, height, width)

    
    

# class SimpleModel(nn.Module):
#     def __init__(self, in_channels, num_classes, kernel_size, height, width):
#         super(SimpleModel, self).__init__()
#         self.layer1 = KANCNN(in_channels, 3, kernel_size)
#         self.pool = nn.MaxPool2d(kernel_size=(2, 2))
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear((height // 2) * (width // 2) * 3, num_classes)
    
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.pool(out)

#         out = self.flatten(out)
        
#         out = self.fc1(out)
#         return out



class KANet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, kernel_size=3, mode=["swish", "swish"], num_groups=None):

        super(KANet5, self).__init__()

        self.groups = num_groups
        
        self.conv1 = KANCNN(in_channels, out_channels, kernel_size, mode=mode, num_groups=num_groups)
        self.conv2 = KANCNN(out_channels, out_channels * 2, 3, mode=mode, num_groups=num_groups)
        

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((out_channels*2)*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class KANet5_KAN(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, mode="swish"):

        super(KANet5_KAN, self).__init__()

        self.conv1 = KANCNN(in_channels, out_channels, kernel_size, mode=mode)
        self.conv2 = KANCNN(6, 16, 3, mode=mode)

        self.pool = nn.MaxPool2d((2, 2))

        self.act1 = KAT_Group(1, "relu", "cuda")
        self.fc1 = nn.Linear(16*7*7, 120)
        self.act2 = KAT_Group(1, "relu", "cuda")
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)     

        #flatten  
        x = x.view(-1, 16*7*7)
        
        x = self.fc1(x).unsqueeze(-1)
        
        x = self.act1(x).squeeze()
        #x = self.fc1(x)
        
        x = self.act2(self.fc2(x).unsqueeze(-1)).squeeze()
        #x = self.fc2(x)
        x = self.fc3(x)

        return x
