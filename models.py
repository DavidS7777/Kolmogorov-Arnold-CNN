import torch
import torch.nn as nn
from KAN_CNN import *


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

class SuperSimpleModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order=(5, 4)):
        super(SuperSimpleModel, self).__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1 #for hydra

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=["identity"], order=order)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(14*14 * out_channels, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        out = self.fc1(out)

        return out




class KANet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, mode="identity", num_groups=None, second_out = 16, order=(5, 4)):

        super(KANet5, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out

        self.groups = num_groups
        self.modes = mode
        
        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=mode, num_groups=num_groups, order=order)
        self.conv2 = KAConv(out_channels, second_out, kernel_size, mode=mode, num_groups=num_groups, order=order)
        
        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

        

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        
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
    
class KANet5OneFc(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28):

        super(KANet5OneFc, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=['identity'], order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = KAConv(out_channels, second_out, kernel_size, mode=['identity'], order=order)
        self.bn2 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*height // 4 * width // 4, 10)
        
        self.h = height // 4
        self.w = width // 4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x
    
class KANet5OneFcC(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28):

        super(KANet5OneFcC, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out


        self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = KAConvC(out_channels, second_out, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*height // 4 * width // 4, 10)

        self.h = height // 4
        self.w = width // 4
        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x 

class KANet5ConvConvPool(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28):

        super(KANet5ConvConvPool, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out

        self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = KAConvC(out_channels, second_out, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(second_out)
        self.pool = nn.MaxPool2d((4, 4))
        self.fc1 = nn.Linear((second_out)*height // 4 * width // 4, 10)
        self.h = height // 4
        self.w = width // 4
        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x 

class KANet5OneFcComparision(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28):

        super(KANet5OneFcComparision, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out


        self.conv1 = KAConvComparision(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = KAConvComparision(out_channels, second_out, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*height // 4* width // 4, 10)

        self.h = height // 4
        self.w = width // 4
        #self.different = different

    def forward(self, x):

        
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x 
    
class KANet5ConvConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4)):

        super(KANet5ConvConv, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out
        self.order = order

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode=['identity'], order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = KAConv(out_channels, out_channels, kernel_size, "identity", order=order)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = KAConv(out_channels, second_out, kernel_size, mode=['identity'], order=order)
        self.bn3 = nn.BatchNorm2d(second_out)

        self.conv4 = KAConv(second_out, second_out, kernel_size, "identity", order=order)
        self.bn4 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 10)

        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        
        x = self.fc1(x)
        
        return x
    
class KANet5ConvConvC(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), halfsteps=False, height=28, width=28):

        super(KANet5ConvConvC, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out
        self.order = order

        if not halfsteps:

            self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = KAConvC(out_channels, out_channels, kernel_size, order=order)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = KAConvC(out_channels, second_out, kernel_size, order=order)
            self.bn3 = nn.BatchNorm2d(second_out)

            self.conv4 = KAConvC(second_out, second_out, kernel_size, order=order)
            self.bn4 = nn.BatchNorm2d(second_out)
        
        else:
            self.conv1 = KAConvC(in_channels, out_channels//2, kernel_size, order=order)
            self.bn1 = nn.BatchNorm2d(out_channels//2)

            self.conv2 = KAConvC(out_channels//2, out_channels, kernel_size, order=order)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = KAConvC(out_channels, second_out//2, kernel_size, order=order)
            self.bn3 = nn.BatchNorm2d(second_out//2)

            self.conv4 = KAConvC(second_out//2, second_out, kernel_size, order=order)
            self.bn4 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear((second_out)*height // 4 * width // 4, 10)

        self.h = height // 4
        self.w = width // 4
        #self.fc1 = nn.Linear((second_out)*7*7, 10)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x

class KANet5ConvConvCDescending(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), halfsteps=False):

        super(KANet5ConvConvCDescending, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out
        self.order = order

        if not halfsteps:

            self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = KAConvC(out_channels, out_channels, kernel_size, order=order)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = KAConvC(out_channels, second_out, kernel_size, order=order)
            self.bn3 = nn.BatchNorm2d(second_out)

            self.conv4 = KAConvC(second_out, second_out, kernel_size, order=order)
            self.bn4 = nn.BatchNorm2d(second_out)
        
        else:
            self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = KAConvC(out_channels, out_channels//2, kernel_size, order=order)
            self.bn2 = nn.BatchNorm2d(out_channels//2)

            self.conv3 = KAConvC(out_channels//2, second_out*2, kernel_size, order=order)
            self.bn3 = nn.BatchNorm2d(second_out*2)

            self.conv4 = KAConvC(second_out*2, second_out, kernel_size, order=order)
            self.bn4 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 10)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        
        x = self.fc1(x)
        
        return x
    
class LeNet5ConvConvDescending(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False, halfsteps=False):
        super(LeNet5ConvConvDescending, self).__init__()

        if not halfsteps:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d((2, 2))

            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn3 = nn.BatchNorm2d(second_out)
            self.conv4 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn4 = nn.BatchNorm2d(second_out)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels//2, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels//2)
            self.pool = nn.MaxPool2d((2, 2))

            self.conv3 = nn.Conv2d(in_channels=out_channels//2, out_channels=second_out*2, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn3 = nn.BatchNorm2d(second_out*2)
            self.conv4 = nn.Conv2d(in_channels=second_out*2, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn4 = nn.BatchNorm2d(second_out)

        self.fc1 = nn.Linear(second_out*7*7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*7*7)

        x = self.fc1(x)

        return x


class KANet5ConvConvComparision(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4)):

        super(KANet5ConvConvComparision, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out
        self.order = order

        self.conv1 = KAConvComparision(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = KAConvComparision(out_channels, out_channels, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = KAConvComparision(out_channels, second_out, kernel_size, order=order)
        self.bn3 = nn.BatchNorm2d(second_out)

        self.conv4 = KAConvComparision(second_out, second_out, kernel_size, order=order)
        self.bn4 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 10)

        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        
        x = self.fc1(x)
        
        return x
    
class KANet5OneFcTriton(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4)):

        super(KANet5OneFcTriton, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out


        self.conv1 = KAConvTriton(in_channels, out_channels, kernel_size, order=order)
        self.conv2 = KAConvTriton(out_channels, second_out, kernel_size, order=order)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*7*7, 10)

        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*7*7)
        
        x = self.fc1(x)
        
        return x

class KANet5OneFcTritonVectorized(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28):

        super(KANet5OneFcTritonVectorized, self).__init__()

        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out


        self.conv1 = KAConvTritonVectorized(in_channels, out_channels, kernel_size, order=order)
        self.conv2 = KAConvTritonVectorized(out_channels, second_out, kernel_size, order=order)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear((second_out)*height//4*width//4, 10)

        self.h = height // 4
        self.w = width // 4
        #self.different = different

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.pool(x)    
        
        #flatten  
        x = x.view(-1, x.shape[1]*self.h*self.w)
        
        x = self.fc1(x)
        
        return x

class LeNet5(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)

        self.fc1 = nn.Linear(second_out*7*7, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*7*7)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x
    
class LeNet5OneFc(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False, height=28, width=28):
        super(LeNet5OneFc, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn2 = nn.BatchNorm2d(second_out)
        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear((second_out)*height // 4 * width // 4, 10)

        self.h = height // 4
        self.w = width // 4
        
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*self.w*self.h)

        x = self.fc1(x)

        return x

class LeNet5CCPool(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False):
        super(LeNet5CCPool, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn2 = nn.BatchNorm2d(second_out)
        self.pool = nn.MaxPool2d((4, 4))
        self.fc1 = nn.Linear(second_out*7*7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*7*7)

        x = self.fc1(x)

        return x

class LeNet5ConvConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False, halfsteps=False, height=28, width=28):
        super(LeNet5ConvConv, self).__init__()

        self.h = height //4
        self.w = width // 4

        if not halfsteps:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d((2, 2))

            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn3 = nn.BatchNorm2d(second_out)
            self.conv4 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn4 = nn.BatchNorm2d(second_out)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels//2)
            
            self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d((2, 2))

            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=second_out//2, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn3 = nn.BatchNorm2d(second_out//2)
            self.conv4 = nn.Conv2d(in_channels=second_out//2, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
            self.bn4 = nn.BatchNorm2d(second_out)

        self.fc1 = nn.Linear(second_out*self.h*self.w, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*self.h*self.w)

        x = self.fc1(x)

        return x

class ConvConvKANStarter(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False):
        super(ConvConvKANStarter, self).__init__()
        self.in_c = in_channels
        self.hidden_c = out_channels
        self.ker_size = kernel_size
        self.out_c = second_out
        self.conv1 = KAConvC(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn3 = nn.BatchNorm2d(second_out)
        self.conv4 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn4 = nn.BatchNorm2d(second_out)

        self.fc1 = nn.Linear(second_out*7*7, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.pool(x)

        x = x.view(-1, x.shape[1]*7*7)

        x = self.fc1(x)

        return x

class SimpleLeNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SimpleLeNet, self).__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
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

class SuperSimpleLeNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SuperSimpleLeNet, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same", bias=bias)
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7 * out_channels, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        
        out = self.fc1(out)

        return out

class KANetOneConv(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 6, kernel_size = 3, order=(5, 4), height=28, width=28):
        super(KANetOneConv, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1

        self.conv1 = KAConvC(in_channels, out_channels, kernel_size, order=order)
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.flatten = nn.Flatten()
        
        self.h = height // 4
        self.w = width // 4
        self.fc1 = nn.Linear(self.h*self.w * out_channels, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        
        out = self.fc1(out)

        return out

class KANetOneConvComparision(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 6, kernel_size = 3):
        super(KANetOneConvComparision, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.ker_size = kernel_size
        self.hidden_c = -1

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, mode="identity")
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7* out_channels, 10)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)

        out = self.flatten(out)
        
        out = self.fc1(out)

        return out


class JustFc(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, height=28, width=28):
        super(JustFc, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, "same", bias=False)
        self.pool = nn.MaxPool2d(kernel_size=(4, 4))
        self.flatten = nn.Flatten()

        self.h = height // 4
        self.w = width // 4
        self.fc1 = nn.Linear(self.h*self.w * out_channels, 10)
        

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out