import torch
import torch.nn as nn
import numpy as np

from kat_rational import KAT_Group

#just one Layer -> Pooling is later
class KANCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device="cuda"):
        """
        shape needs to be [B, L, C] from KAT

        
        Args:

        in_channels(int) = number of input channels

        out_channels(int) = number of output channels / filter
        
        kernel_size(int) = 1d shape of kernel, kernel will be quadratic

        filter(int) = number of filters

        channels(int) = number of channels of the picture (1=black and white, 3=rbg)


        """
        super(KANCNN, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device


        
        #for every filter there is a kernel matrix with in_channel depth
        
        functions = [KAT_Group(num_groups=1, mode="swish", device='cuda') for _ in range(out_channels * kernel_size * kernel_size * in_channels)]

        # for f in range(out_channels): #number of filters
        #     for a in range(kernel_size):
        #         for b in range(kernel_size):
        #             for c in range(in_channels):

        #                 #according to paper swish produces best results, start without group KANS -> num_groups = kernel * kernel
        #                 #3 x 3 Kernel means 9 groups
        #                 functions.append(KAT_Group(num_groups=kernel_size*kernel_size, mode="swish", device='cuda'))
        
        self.kernel = nn.ModuleList(functions)
        

    def forward(self, input):
        
        """
        
        Args:
        
        input: Image of Shape[batch, height, width, in_channels]

        Returns:

        a gpu tensor of shape [out_channels][height][width] with out_channels feature maps
        
        """

        #make sure input is always in "batches"

        batch = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        kernel = self.kernel


        if in_c != input.shape[3]:
            raise ValueError("image channels do not match in_channels of Layer")
        

        pad_size = (kernel_size - 1) // 2
        padded = nn.functional.pad(input, pad=[0, 0, pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        

        result = torch.zeros((out_c, height, width), device=self.device) #same dim because of padding
        
        for f in range(out_c): #for every filter

            #go over every pixel -> even though KAN takes flattened sequence it's easier to move kernel matrix -> TODO can be optimized 
            for i in range(height): #padding

                for j in range(width): #padding

                    for a in range(kernel_size):

                        for b in range(kernel_size):

                            for c in range(in_c): #one in channel at a time
                                
                                """
                                first without batching 
                                input needs to be (Tensor): 3D input tensor with shape (batch, length, channels)

                                batch is of size 1
                                length of size 1 since one pixel at a time
                                channels of size 1 since one channel at a time (for now) -> NOTE: All channels at the same time produces the same result, but there will be less weights in Kernel (could be good or bad)

                                """
                                idx = f * in_c * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b
                                result[f, i, j] += kernel[idx](padded[:, i + a, j + b, c].unsqueeze(0).unsqueeze(0)).squeeze()

                                

        return result 
    
    def fast_forward(self, input):

        batch = input.shape[0]
        height = input.shape[1]
        width = input.shape[2]
        in_c = self.in_channels
        out_c = self.out_channels
        kernel_size = self.kernel_size
        kernel = self.kernel

        pad_size = (kernel_size - 1) // 2
        padded = nn.functional.pad(input, pad=[0, 0, pad_size, pad_size, pad_size, pad_size], mode="constant", value = 0)
        

        result = torch.zeros((out_c, height * width), device=self.device) #same dim because of padding

        for f in range(out_c):
            for a in range(kernel_size):
                for b in range(kernel_size):
                    for c in range(in_c):
                    
                        idx = f * in_c * kernel_size * kernel_size + c * kernel_size * kernel_size + a * kernel_size + b

                        #print(result[f].shape)
                        #res = kernel[idx](padded[:, a:a + height, b:b + width, c].reshape(batch, -1, 1))
                        #print(res.shape)

                        result[f] += kernel[idx](padded[:, a:a + height, b:b + width, c].reshape(batch, -1, 1)).squeeze()
        
        return result.reshape(out_c, height, width)
        



        
    

class SimpleModel(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size, height, width):
        super(SimpleModel, self).__init__()
        self.layer1 = KANCNN(in_channels, 5, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear((height // 2) * (width // 2) * 5, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out




def main():
    original_image = torch.tensor([ [1, 2, 0, 1, 2], [3, 1, 1, 0, 0], [2, 0, 2, 3, 1], [0, 1, 3, 1, 0], [1, 2, 1, 0, 3] ], dtype=torch.float32)
    padded_image = nn.functional.pad(original_image, pad=[1, 1, 1, 1], mode='constant', value=0)
    # print(padded_image)

    # print(padded_image.shape)

    # print(padded_image[1][1].shape)

    # Angenommen, img ist dein Bildtensor mit der Form (Höhe, Breite, Kanal)
    img = torch.randn(256, 256, 3)  # Beispielbild mit zufälligen Werten

    print(img.shape)
    print(img[2][4])
    print(img[2][4].unsqueeze(0).unsqueeze(0))
    print(img[2][4].unsqueeze(0).unsqueeze(0).shape)

    # Die Kanäle aufsummieren
    summed_img = torch.sum(img, dim=2)

    print(summed_img.shape)  # Sollte (256, 256) ausgeben


#main()