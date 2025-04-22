import torch
import torch.nn as nn
from KAN_CNN import *

class KANet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28, num_classes=10):

        super(KANet, self).__init__()

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = KAConv(out_channels, second_out, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(second_out * height//4 * width//4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, num_classes)

        self.h = height // 4
        self.w = width // 4
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
    

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)    
        
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    
class KAVGG(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=3, second_out = 16, order=(5, 4), height=28, width=28, num_classes=10):

        super(KAVGG, self).__init__()

        self.conv1 = KAConv(in_channels, out_channels, kernel_size, order=order)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = KAConv(out_channels, out_channels, kernel_size, order=order)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = KAConv(out_channels, second_out, kernel_size, order=order)
        self.bn3 = nn.BatchNorm2d(second_out)

        self.conv4 = KAConv(second_out, second_out, kernel_size, order=order)
        self.bn4 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(second_out * height // 4 * width // 4, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()


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

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.fc2(x)
        
        return x

    
class LeNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False, height=28, width=28, num_classes=10):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn2 = nn.BatchNorm2d(second_out)

        self.pool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(second_out * height//4 * width // 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, num_classes)
        
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)
        
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class VGG(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, kernel_size=5, second_out=16, bias=False, height=28, width=28, num_classes=10):
        super(VGG, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn3 = nn.BatchNorm2d(second_out)
        self.conv4 = nn.Conv2d(in_channels=second_out, out_channels=second_out, kernel_size=kernel_size, stride=1, padding="same", bias=bias)
        self.bn4 = nn.BatchNorm2d(second_out)
        

        self.fc1 = nn.Linear(second_out *height // 4 * width // 4, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

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

        x = self.flatten(x)

        x = self.fc1(x)

        x = self.bn5(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x

class WideCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels2, out_channels, height=32, width=32, num_classes=10, kernel_size=3):
        super(WideCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels2, kernel_size=kernel_size, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels2)

        self.conv3 = nn.Conv2d(hidden_channels2, out_channels, kernel_size=kernel_size, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        
        self.fc1 = nn.Linear(out_channels * width // 8 * height // 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        
        self.bn4= nn.BatchNorm1d(1024)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x

class KAWideCNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels2, out_channels, height=32, width=32, num_classes=10, kernel_size=3):
        super(KAWideCNN, self).__init__()

        self.conv1 = KAConv(in_channels, hidden_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = KAConv(hidden_channels, hidden_channels2, kernel_size)
        self.bn2 = nn.BatchNorm2d(hidden_channels2)
        
        self.conv3 = KAConv(hidden_channels2, out_channels, kernel_size)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(out_channels * height // 8 * width // 8 , 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.adaptpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn4= nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(

            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),      
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),

            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),

            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class KAConvBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super(KAConvBlock, self).__init__()

        self.block = nn.Sequential(
            KAConv(in_channels, bottleneck_channels, kernel_size=1),   
            nn.BatchNorm2d(bottleneck_channels),
            

            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),

            KAConv(bottleneck_channels, out_channels, kernel_size=1), 
            nn.BatchNorm2d(out_channels),
            
        )

    def forward(self, x):
        return self.block(x)

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = ConvBlock(32, 16, 64)

        self.pool = nn.MaxPool2d(2, 2)

        self.block2 = ConvBlock(64, 32, 128)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class KASqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(KASqueezeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.block1 = KAConvBlock(32, 16, 64)

        self.pool = nn.MaxPool2d(2)

        self.block2 = KAConvBlock(64, 32, 128)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        
        x = self.block1(x)
        x = self.pool(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

