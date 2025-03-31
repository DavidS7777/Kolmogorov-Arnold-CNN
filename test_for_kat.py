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

from plotting import *
from weight_inits import *

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


def cleanup(sig, frame):
    print("cleanup")
    torch.cuda.empty_cache()
    exit()

import random
import signal
from torchviz import make_dot

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

    
import gc
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    signal.signal(signal.SIGTSTP, cleanup)

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['HYDRA_FULL_ERROR'] = '1'

    #set_seed(42)

    # batch = 64
    # in_c = 4
    # height = 28
    # width = 28
    # ker_size = 3
    # out_c = 12

    # set_seed(42)

    # data = torch.randn(batch, in_c, height, width).to(device)

    # set_seed(42)
    
    # m1 = KAConvC(in_c, out_c, ker_size).to(device)
    
    # set_seed(42)

    # m2 = KAConvCComparision(in_c, out_c, ker_size).to(device)

    # inits = nn.Conv2d(in_c, out_c, ker_size).to(device)
    
    # params = []
    # for f in range(out_c):
    #     for c in range(in_c):
    #         for i in range(ker_size):
    #             for j in range(ker_size):
    #                 idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
    #                 w = inits.weight[f, c, i, j]
    #                 params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
    #                 #m1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
    #                 #m1.kernel[idx].weight_denominator = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0]).to(device))

    # m1.nums= nn.Parameter(torch.cat(params, dim=0))
    
    # m2.nums= nn.Parameter(torch.cat(params, dim=0))

    # # print(m1.kernel[0].weight_numerator)
    # # print(m2.nums[0])

    # start = time.time()
    # res1 = m1(data)
    # print(time.time() - start)

    # start = time.time()
    # res2= m2(data)
    # print(time.time() - start)
    
    # # print(res1.shape)
    # # print(res2.shape)
    # # #print(torch.equal(res1, res2))
    # print((res1 - res2).abs().max())
    # exit()

    

    model = cfg.model
    name = model.name
    batches = model.batch_size
    
    in_c = model.in_channels
    hidden_c = model.hidden_channels
    out_c = model.out_channels
    ker_size = model.kernel_size

    lr = cfg.training.learning_rate
    weight_decay = cfg.training.decay
    num_epochs = cfg.training.epochs
    log = cfg.training.log
    optimizer = cfg.training.optimizer
    train_size = cfg.training.size
    plot = False

    if cfg.training.dataset == "Fashion-MNIST":
        height, width = 28, 28
    elif cfg.training.dataset == "CIFAR10":
        height, width = 32, 32
    else:
        raise NotImplementedError("Datset not implemented")
    
    figuresc1 = []
    figuresc2 = []
    figuresc3 = []
    figuresc4 = []

    if name == "KANet":
        
        plot = model.plot
        twoconv = False
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
        model = LeNet5OneFc(in_c, hidden_c, ker_size, out_c, model.bias, height, width).to(device)
        
    
    elif name == "SuperSimpleModel":
        model = SuperSimpleModel(in_c, out_c, ker_size, order=(model.num, model.denom)).to(device)
        init_weights(model)
    
    elif name == "SuperSimpleLeNet":
        model = SuperSimpleLeNet(in_c, out_c, ker_size).to(device)
    
    elif name == "KANetOneFcC":
        name += f"({model.num}, {model.denom})"
        init_mode = model.init
        init_strength = model.init_strength

        model = KANet5OneFcC(in_c, hidden_c, ker_size, out_c, (model.num, model.denom), height, width).to(device)
        if init_mode == "n":
            
            init_weights(model, False, True)
        elif init_mode == "all":
            all_init(model, True, init_strength, False)
            
        else:
            raise NotImplementedError("InitMode not defined")
        
        
        
        
    elif name == "LeNetConvConv":
        model = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, model.bias, model.halfsteps, height, width).to(device)
    
    elif name == "KANetConvConv":
        
        name += f"({model.num}, {model.denom})"
        #name += f"AugmentedData"
        plot = model.plot
        twoconv = True
        
        init_mode = model.init
        name += f"({init_mode})"

        model = KANet5ConvConv(in_c, hidden_c, ker_size, out_c, (model.num, model.denom)).to(device)

        if init_mode == "n":
            
            init_weights(model, True)
        elif init_mode == "both":
            double_init(model, True)
        elif init_mode == "noinit":
            pass

        else:
            raise NotImplementedError("InitMode not defined")
            

        if plot:
            figuresc1 = plot_pre(model, 1, True)
            figuresc2 = plot_pre(model, 2, True)
            figuresc3 = plot_pre(model, 3, True)
            figuresc4 = plot_pre(model, 4, True)

    elif name == "KANetOneFcTriton":
        model = KANet5OneFcTriton(in_c, hidden_c, ker_size, out_c).to(device)
    
    elif name == "KANetConvConvC":

        name += f"({model.num}, {model.denom})"
        init_mode = model.init
        init_strength = model.strength
        name += f"({init_mode})"
    
        halfsteps = model.halfsteps

        model = KANet5ConvConvC(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom), halfsteps=halfsteps, height=height, width=width).to(device)
        if init_mode == "n":
            init_weights(model, True, True, False, halfsteps)
        elif init_mode == "all":
            all_init(model, True, init_strength)            
        elif init_mode == "noinit":
            pass
        else:
            raise NotImplementedError("InitMode not defined")
        
    elif name == "KANetOneFcComparision":
        model = KANet5OneFcComparision(in_c, hidden_c, ker_size, out_c).to(device)
        init_weights(model, False, True)
    
    elif name == "KANetOneFcTritonVectorized":
        model = KANet5OneFcTritonVectorized(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom)).to(device)

    elif name == "KANetOneConv":
        plot = model.plot
        model = KANetOneConv(in_c, out_c, ker_size, (model.num, model.denom), height, width).to(device)
        init_weights(model, C=True, KANStarter=True)

        if plot:
            oneconv = True
            twoconv = False
            figuresc1 = plot_pre(model, 1, oneconv=True)

    elif name == "KANetOneConvComparision":
        model = KANetOneConvComparision(in_c, out_c, ker_size).to(device)
    
    elif name == "JustFc":
        model = JustFc(in_c, out_c, ker_size, height, width).to(device)
        
    elif name=="ConvConvKANStarter":
        model = ConvConvKANStarter(in_c, hidden_c, ker_size, out_c).to(device)
        init_weights(model, C=True, KANStarter=True)
    
    elif name == "KANetConvConvComparision":
        model = KANet5ConvConvComparision(in_c, hidden_c, ker_size, out_c).to(device)
        init_weights(model, True, True)

    elif name == "KANetConvConvCDescending":
        model = KANet5ConvConvCDescending(in_c, hidden_c, ker_size, out_c, (model.num, model.denom), model.halfsteps).to(device)
        init_weights(model, True, True, False, True, True)
    
    elif name == "ConvConvPool":
        model = KANet5ConvConvPool(in_c, hidden_c, ker_size, out_c, (model.num, model.denom)).to(device)
        init_weights(model, False, True)
    elif name == "LeNetConvConvPool":
        model = LeNet5CCPool(in_c, hidden_c, ker_size, out_c).to(device)
    elif name == "KANetC":
        #name += f"({model.num}, {model.denom})"
        model = KANet5C(in_c, hidden_c, ker_size, out_c, (model.num, model.denom)).to(device)
    
    elif name == "KANetOneFcCComparision":

        model = KANet5OneFcCComparision(in_c, hidden_c, ker_size, out_c).to(device)
        init_weights(model, False, True)

    else:
        raise NotImplementedError("Model not defined")
    

    from torchvision import transforms
    from sklearn.model_selection import train_test_split

    # Define data augmentation
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(28)
])

    if cfg.training.dataset == "Fashion-MNIST":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    elif cfg.training.dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError("Dataset not implemented")


    #normalize
    x_train = x_train.astype(np.float32) / 255.0 
    x_test = x_test.astype(np.float32) / 255.0
 
    x_train = torch.tensor(x_train[:train_size + train_size // 10])
        
    #print(len(x_train))
    y_train = torch.tensor(y_train[:train_size + train_size // 10], dtype=torch.long)

    if cfg.training.dataset == "Fashion-MNIST":
        x_train = x_train.unsqueeze(1) #add channel dim
    elif cfg.training.dataset == "CIFAR10":
        x_train = x_train.permute(0, 3, 1, 2)
        y_train = y_train.squeeze()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_test = torch.tensor(x_test[:train_size // 5])

    y_test = torch.tensor(y_test[:train_size // 5], dtype=torch.long)

    if cfg.training.dataset == "Fashion-MNIST":
        x_test = x_test.unsqueeze(1) #add channel dim
    elif cfg.training.dataset == "CIFAR10":
        x_test = x_test.permute(0, 3, 1, 2)
        y_test = y_test.squeeze()

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
            name= f"{name}, {in_c}, {hidden_c}, {ker_size}, {out_c}, {batches}, {lr}, {cfg.training.size}, {num_epochs}, {optimizer}, {weight_decay}",
            tags = [],
                
            # track hyperparameters and run metadata
            config={
            "learning_rate": lr,
            "architecture": "CNN",
            "dataset": cfg.training.dataset,
            "epochs": num_epochs,
            }
        )
        if cfg.training.tag != "None":
            wandb.run.tags = [str(cfg.training.tag)]
            

    criterion = nn.CrossEntropyLoss()

    if optimizer == "adam":

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    import torch.autograd.profiler as profiler

    accuracy = -1
    
    start = time.time()
                
    for epoch in tqdm(range(num_epochs), desc="Training"):

        model.train()
        running_loss = 0.0

        correct = 0
        total = 0

        for (imgs, labels) in train_loader:

            #print("batch")
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            pred = model(imgs)
            #make_dot(pred, params=dict(model.named_parameters())).render("graph", format="png")
            #exit()
            loss = criterion(pred, labels)
            
            #start = time.time()
            with profiler.profile(with_stack=True, use_cuda=True) as prof:
                loss.backward()
            
            #exit()

            print(prof.key_averages().table(sort_by="cuda_time_total"))
            exit()
            
            #print("whole bwd:", time.time() - start)

            optimizer.step()

            with torch.no_grad():
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
        
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        # exit()
        
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
    
    #print(prof.key_averages().table(sort_by="cuda_time_total"))

    print("Training time:", (end - start) // 60, "m", "%.2f" % ((end - start) % 60),  "s")
    print(accuracy)
    
    
    if plot:
        if not twoconv and not oneconv:
            plot_post(model, 1, figuresc1, "pics/")
            plot_post(model, 2, figuresc2, "pics/")
        
        elif oneconv:
            plot_post(model, 1, figuresc1, "pics/OneConv43/", False, True)
            
        else:
            plot_post(model, 1, figuresc1, "pics/ConvConv/", True)
            plot_post(model, 2, figuresc2, "pics/ConvConv/", True)
            plot_post(model, 3, figuresc3, "pics/ConvConv/", True)
            plot_post(model, 4, figuresc4, "pics/ConvConv/", True)


import cProfile
import pstats
import io
   
if __name__ == "__main__":

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()

    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats(30)
    # print(s.getvalue())