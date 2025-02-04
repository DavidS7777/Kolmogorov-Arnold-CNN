import torch
import torch.nn as nn


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
