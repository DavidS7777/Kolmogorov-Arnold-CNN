import torch.nn as nn
import torch

from tensorflow import keras

import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    torch.cuda.set_device(0)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

def plot_activations(curr_channel, kernel_size, axs, kernel):
        
        
        for i in range(kernel_size):
            for j in range(kernel_size):

                if kernel_size == 1:
                    axs[i].cla()
                else:
                    axs[i, j].cla()

                x = torch.linspace(-2, 2, 100).to(device)
                x = x.unsqueeze(0).unsqueeze(0)
                idx = curr_channel * kernel_size * kernel_size + i * kernel_size + j

                y = kernel[idx](x)
                
                x = x.squeeze(0).squeeze(0)
                y = y.squeeze(0).squeeze(0)

                if kernel_size == 1:
                    axs[i].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
                else:
                    axs[i, j].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        
        
        return


def plot_pre(model, layer):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker = model.ker_size

    figures = []

    if layer == 1:

        for i in range(in_c):
            fig, axs = plt.subplots(ker, ker * 2, figsize=(20, 6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")

            if ker == 1:
                plot_activations(i, ker, axs[:ker], model.conv1.kernel)
            else:

                plot_activations(i, ker, axs[:, :ker], model.conv1.kernel)

    elif layer == 2:

        for i in range(hidden_c):
            fig, axs = plt.subplots(ker, ker * 2, figsize=(20, 6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")
            if ker == 1:
                plot_activations(i, ker, axs[:ker], model.conv2.kernel)
            else:
                plot_activations(i, ker, axs[:, :ker], model.conv2.kernel)
    
    return figures
    
def plot_post(model, layer, figures, path=""):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker = model.ker_size

    if layer == 1:
        for i in range(in_c):
            fig, axs = figures[i]
            if ker == 1:
                plot_activations(i, ker, axs[ker:], model.conv1.kernel)
            else:
                plot_activations(i, ker, axs[:, ker:], model.conv1.kernel)
            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv1" + str(i + 1) + ".png")

    elif layer == 2:
        for i in range(hidden_c):
            fig, axs = figures[i]
            if ker == 1:
                plot_activations(i, ker, axs[ker:], model.conv2.kernel)
            else:
                plot_activations(i, ker, axs[:, ker:], model.conv2.kernel)

            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv2" + str(i + 1) + ".png")


def init_weights(model, two_conv=False, C=False):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker_size = model.ker_size
    out_c = model.out_c

    skip = False

    if hidden_c == -1:
        hidden_c = out_c
        skip = True

    if not two_conv:
        inits = LeNet5(in_c, hidden_c, ker_size, out_c, False).to(device)
        params = []
        for f in range(hidden_c):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv1.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

        if C:
            model.conv1.nums= nn.Parameter(torch.cat(params, dim=0))

        if skip:
            return

        params = []
        for f in range(out_c):
            for c in range(hidden_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv2.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
        if C:
            model.conv2.nums= nn.Parameter(torch.cat(params, dim=0))

    else:
        inits = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False).to(device)

        params = []

        for f in range(hidden_c // 2):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv1.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

        if C:
            model.conv1.nums= nn.Parameter(torch.cat(params, dim=0))                         
        
        params = []
        for f in range(hidden_c):
            for c in range(hidden_c // 2):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c // 2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv2.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

        if C:
            model.conv2.nums= nn.Parameter(torch.cat(params, dim=0))

        params = []
        for f in range(out_c // 2):
            for c in range(hidden_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv3.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv3.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

        if C:
            model.conv3.nums= nn.Parameter(torch.cat(params, dim=0))
        params = []
        for f in range(out_c):
            for c in range(out_c // 2):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * out_c // 2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv4.weight[f, c, i, j]
                        if C:
                            params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                        else:
                            model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
        if C:
            model.conv4.nums= nn.Parameter(torch.cat(params, dim=0))
    return

#not yet implemented for C
def double_init(model, two_conv=False):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker_size = model.ker_size
    out_c = model.out_c

    skip = False

    if hidden_c == -1:
        hidden_c = out_c
        skip = True

    if not two_conv:
        inits = LeNet5(in_c, hidden_c, ker_size, out_c, False).to(device)

        for f in range(hidden_c):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv1.weight[f, c, i, j]
                        model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

        if skip:
            return
    
        for f in range(out_c):
            for c in range(hidden_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv2.weight[f, c, i, j]
                        model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
    else:
        inits_num = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False).to(device)
        inits_denom = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False).to(device)

        for f in range(hidden_c // 2):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        n = inits_num.conv1.weight[f, c, i, j]
                        d = inits_denom.conv1.weight[f, c, i, j]
                        model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, n, 0.0, 0.0, 0.0, 0.0]).to(device))
                        model.conv1.kernel[idx].weight_denominator = nn.Parameter(torch.tensor([d, 0.0, 0.0, 0.0]).to(device))
        
        for f in range(hidden_c):
            for c in range(hidden_c // 2):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c // 2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        n = inits_num.conv2.weight[f, c, i, j]
                        d = inits_denom.conv2.weight[f, c, i, j]
                        model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, n, 0.0, 0.0, 0.0, 0.0]).to(device))
                        model.conv2.kernel[idx].weight_denominator = nn.Parameter(torch.tensor([d, 0.0, 0.0, 0.0]).to(device))
        
        for f in range(out_c // 2):
            for c in range(hidden_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        n = inits_num.conv3.weight[f, c, i, j]
                        d = inits_denom.conv3.weight[f, c, i, j]
                        model.conv3.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, n, 0.0, 0.0, 0.0, 0.0]).to(device))
                        model.conv3.kernel[idx].weight_denominator = nn.Parameter(torch.tensor([d, 0.0, 0.0, 0.0]).to(device))

        for f in range(out_c):
            for c in range(out_c // 2):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * out_c // 2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        n = inits_num.conv4.weight[f, c, i, j]
                        d = inits_denom.conv4.weight[f, c, i, j]
                        model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, n, 0.0, 0.0, 0.0, 0.0]).to(device))
                        model.conv4.kernel[idx].weight_denominator = nn.Parameter(torch.tensor([d, 0.0, 0.0, 0.0]).to(device))

    return


import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['HYDRA_FULL_ERROR'] = '1'

    #set_seed(42)

    model = cfg.model

    name = model.name

    
    batches = model.batch_size
    
    in_c = model.in_channels
    hidden_c = model.hidden_channels
    out_c = model.out_channels
    ker_size = model.kernel_size

    lr = cfg.training.learning_rate
    num_epochs = cfg.training.epochs
    log = cfg.training.log
    optimizer = cfg.training.optimizer
    train_size = cfg.training.size
    plot = False

    figuresc1 = []
    figuresc2 = []

    if name == "KANet":
        
        plot = model.plot
        name += f"({model.num}, {model.denom})" 
        
        model = KANet5(in_c, hidden_c, ker_size, mode=['identity'], second_out=out_c, order=(model.num, model.denom)).to(device)
        init_weights(model)
               

        if plot:
            figuresc1 = plot_pre(model, 1)
            figuresc2 = plot_pre(model, 2)


    elif name == "LeNet":
        model = LeNet5(in_c, hidden_c, ker_size, out_c, model.bias).to(device)
    
    elif name == "SimpleModel":
        name += f"({model.num}, {model.denom})"
        model = SimpleModel(in_c, out_c, ker_size, order=(model.num, model.denom)).to(device)
        init_weights(model)
    
    elif name == "SimpleLeNet":
        model = SimpleLeNet(in_c, out_c, ker_size, model.bias).to(device)
        

    elif name == "KANetOneFc":
        name += f"({model.num}, {model.denom})"
        model = KANet5OneFc(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom)).to(device)
        init_weights(model)
    
    elif name == "LeNetOneFc":
        model = LeNet5OneFc(in_c, hidden_c, ker_size, out_c).to(device)
    
    elif name == "SuperSimpleModel":
        model = SuperSimpleModel(in_c, out_c, ker_size, order=(model.num, model.denom)).to(device)
        init_weights(model)
    
    elif name == "SuperSimpleLeNet":
        model = SuperSimpleLeNet(in_c, out_c, ker_size).to(device)
    
    elif name == "KANetOneFcC":
        name += f"({model.num}, {model.denom})"
        model = KANet5OneFcC(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom)).to(device)
        init_weights(model, C=True)
        
    elif name == "LeNetConvConv":
        model = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c).to(device)
    
    elif name == "KANetConvConv":
        
        name += f"({model.num}, {model.denom})"
        
        init_mode = model.init
        name += f"({init_mode})"

        model = KANet5ConvConv(in_c, hidden_c, ker_size, out_c, (model.num, model.denom)).to(device)

        if init_mode == "n":
            init_weights(model, True)
        elif init_mode == "both":
            double_init(model, True)
        else:
            raise NotImplementedError("InitMode not defined")

    elif name == "KANetOneFcTriton":
        model = KANet5OneFcTriton(in_c, hidden_c, ker_size, out_c).to(device)
    
    elif name == "KANetConvConvC":

        name += f"({model.num}, {model.denom})"
        model = KANet5ConvConvC(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom)).to(device)
        init_weights(model, True, True)
    
    elif name == "KANetOneFcComparision":
        model = KANet5OneFcComparision(in_c, hidden_c, ker_size, out_c).to(device)

    else:
        raise NotImplementedError("Model not defined")
    
    

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    #normalize
    x_train = x_train.astype(np.float32) / 255.0 
    x_test = x_test.astype(np.float32) / 255.0
 
    x_train = torch.tensor(x_train[:train_size + train_size // 10]).unsqueeze(1) #add channel dim
    y_train = torch.tensor(y_train[:train_size + train_size // 10], dtype=torch.long)
    x_val = x_train[train_size:]
    y_val = y_train[train_size:]
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    x_test = torch.tensor(x_test[:train_size // 5]).unsqueeze(1) #add channel dim
    y_test = torch.tensor(y_test[:train_size // 5], dtype=torch.long)


    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)
    val_data = TensorDataset(x_val, y_val)

    train_loader = DataLoader(dataset=train_data, batch_size=batches, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=batches, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=batches, shuffle=False, num_workers=4)


    if log:
        if cfg.training.name != "None":
            name += f"{cfg.training.name}"
        wandb.init(
            # set the wandb project where this run will be logged
            project="KA-CNN",
            name= f"{name}, {in_c}, {hidden_c}, {ker_size}, {out_c}, {batches}, {lr}, {cfg.training.size}, {num_epochs}, {optimizer}",
            tags = [],
            

            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": "Fashion-MNIST",
            "epochs": num_epochs,
            }
        )
        if cfg.training.tag != "None":
            wandb.run.tags = [str(cfg.training.tag)]
        

    criterion = nn.CrossEntropyLoss()

    if optimizer == "adam":

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


    accuracy = -1
    
    start = time.time()
            
    for epoch in tqdm(range(num_epochs), desc="Training"):

        model.train()
        running_loss = 0.0

        correct = 0
        total = 0

        for (imgs, labels) in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)
                
            pred = model(imgs)

            loss = criterion(pred, labels)


            optimizer.zero_grad()

            loss.backward()
                
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total

        if log:
            wandb.log({"train_loss": avg_loss, "train_accuracy" : accuracy}, commit=False)
            
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (imgs, labels) in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                scores = model(imgs)
                loss = criterion(scores, labels)
                test_loss += loss.item()
                _, pred = torch.max(scores.data, 1)
                total += labels.size(0)
                    
                correct += (pred == labels).sum().item()

        avg_loss = test_loss / len(val_loader)
        accuracy = correct / total

                

        if log:
            wandb.log({f"val_loss" : avg_loss, "val_accuracy" : accuracy})
            
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (imgs, labels) in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            scores = model(imgs)
            _, pred = torch.max(scores.data, 1)
            total += labels.size(0)
                    
            correct += (pred == labels).sum().item()

    accuracy = correct / total      

    if log:
        wandb.summary["test_accuracy"] = accuracy
        wandb.finish()

            
    end = time.time()
    print() #new line


    print("Training time:", (end - start) // 60, "m", "%.2f" % ((end - start) % 60),  "s")
    print(accuracy)


    if plot:
        plot_post(model, 1, figuresc1, "pics/")
        plot_post(model, 2, figuresc2, "pics/")


   
if __name__ == "__main__":
    main()