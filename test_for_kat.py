import torch.nn as nn
import torch

from kat_rational import KAT_Group
class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)

        x = x.view(x.shape[0], -1)
        
        #x = self.drop1(x)
        
        x = self.fc1(x)

        x = x.unsqueeze(-1)
        
        
        x = self.act2(x)
        #x = self.drop2(x)
        
        x = x.view(x.shape[0], - 1)

        x = self.fc2(x)
        return x


import keras
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from KAN_CNN import KANCNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SimpleModel(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size, height, width):
        super(SimpleModel, self).__init__()
        self.layer1 = KANCNN(in_channels, 3, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((height // 2) * (width // 2) * 3, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool(out)
        
        out = out.unsqueeze(0) #no batch processing as of now

        out = self.flatten(out)
        
        out = self.fc1(out)
        return out


# Modell instanziieren

#model = SimpleModel(in_channels=1, num_classes=10, kernel_size=3, height=28, width=28)

#model = KAN(28 * 28, 128, 10)

# # Parameter des Modells anzeigen
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)

model = KANCNN(1, 10, 3, device)

model.to(device)
print(device)

original_image = torch.tensor([ [1, 2, 0, 1, 2], [3, 1, 1, 0, 0], [2, 0, 2, 3, 1], [0, 1, 3, 1, 0], [1, 2, 1, 0, 3] ], dtype=torch.float32).to(device)
#padded_image = nn.functional.pad(original_image, pad=[1, 1, 1, 1], mode='constant', value=0)

img = original_image.unsqueeze(0).unsqueeze(-1)

# print(padded_image.shape)
# padded = padded_image.unsqueeze(0)
# # padded = padded.permute(0, 1, 2)
# # print(padded.shape)
# # padded = padded.view(padded.size(0), -1)
# # print(padded.shape)
# padded = padded.unsqueeze(-1)
# print(padded.shape)

#part = padded[:, 0:3, 0:3, :]

#print(part.reshape(part.size(0), -1, part.size(3)).shape)

import time

start = time.time()

res1 = model(img)
end = time.time()
print(end - start)

start = time.time()
res2 = model.fast_forward(img)
end = time.time()
print(end - start)

print(torch.equal(res1, res2))

exit()

1/0

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

#normalize
x_train = x_train.astype(np.float32) / 255.0 
x_test = x_test.astype(np.float32) / 255.0

# x_train = np.reshape(x_train, (x_train.shape[0], 28 * 28))
# x_test = np.reshape(x_test, (x_test.shape[0], 28 * 28))

x_train = torch.tensor(x_train[:10]).unsqueeze(-1)
y_train = torch.tensor(y_train[:10], dtype=torch.long)
x_test = torch.tensor(x_test[:5]).unsqueeze(-1)
y_test = torch.tensor(y_test[:5], dtype=torch.long)

train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

batches = 1
lr = 0.001
num_epochs = 1

# import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

train_loader = DataLoader(dataset=train_data, batch_size=batches, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batches, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#print(model(x_train[1].unsqueeze(0).to(device)))

model.train()
k = 0

import time

start = time.time()



for epoch in range(num_epochs):
    for (imgs, labels) in train_loader:
        k += 1
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        pred = model(imgs)
        loss = criterion(pred, labels)

        torch.cuda.synchronize()

        
        if torch.isnan(loss).any():
            print("NaN-Wert in Loss gefunden!")
        if torch.isinf(loss).any():
            print("Inf-Wert in Loss gefunden!")

        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


end = time.time()
print("Time: ", end - start)
print("Schleifen Durchläfe:", k)
exit()

model.eval()

correct = 0

with torch.no_grad():
    for (imgs, labels) in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        

        scores = model(imgs)
        pred = scores.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)

print(accuracy)



# [B, L, C]
#print(model(x_train[1]).shape)