import torch.nn as nn
import torch

import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.nn.utils import clip_grad_norm_

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == "cuda", "device needs to be cuda"
torch.cuda.set_device(0)


import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, dataloader, device, class_names=None, get_top5=False):
    model.eval()
    all_preds = []
    all_labels = []
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)

            preds = torch.argmax(out, dim=1)

            top5 = torch.topk(out, k=5, dim=1).indices
            matches_top5 = top5.eq(labels.view(-1, 1))

            top5_correct += matches_top5.any(dim=1).sum().item()
            total += labels.size(0)
        
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    top5_acc = top5_correct / total

    print(f"Accuracy (Top-1):  {acc}")
    if get_top5:
        print(f"Top-5 Accuracy: {top5_acc}")

    print(f"Precision: {prec}")
    print(f"Recall:    {rec}")
    print(f"F1 Score:  {f1}")
    if class_names != None:
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))



    
import gc
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:


    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    #os.environ['HYDRA_FULL_ERROR'] = '1'

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
    num_classes = 10
    get_top5 = False

    if cfg.training.dataset == "Fashion-MNIST":
        height, width = 28, 28
    elif cfg.training.dataset == "CIFAR10":
        height, width = 32, 32
    elif cfg.training.dataset == "TinyImageNet":
        height, width, num_classes = 64, 64, 200
        get_top5 = True
    else:
        raise NotImplementedError("Datset not implemented")
    

    if name == "LeNet":
        name = "LeNet"
        model = LeNet(in_c, hidden_c, ker_size, out_c, model.bias, height, width).to(device)
        
    elif name == "VGG":
        model = VGG(in_c, hidden_c, ker_size, out_c, model.bias, height, width).to(device)

    elif name == "KAVGG":
        model = KAVGG(in_c, hidden_c, ker_size, out_c, order=(model.num, model.denom),height=height, width=width)
        model.to(device)
 
    elif name == "KANet":
        model = KANet(in_c, hidden_c, ker_size, out_c, (model.num, model.denom), height=height, width=width, num_classes=num_classes)
        model.to(device)

    elif name == "SqueezeNet":
        model = SqueezeNet(num_classes).to(device)

    elif name == "KASqueezeNet":
        model = KASqueezeNet(num_classes).to(device)

    elif name == "WideCNN":
        model = WideCNN(in_c, hidden_c, model.hidden_channels2, out_c, height, width, num_classes, ker_size).to(device)

    elif name == "KAWideCNN":
        model = KAWideCNN(in_c, hidden_c, model.hidden_channels2, out_c, height, width, num_classes, ker_size)
        model.to(device)
        
    else:
        raise NotImplementedError("Model not defined")
    

    from sklearn.model_selection import StratifiedShuffleSplit

    if cfg.training.dataset == "TinyImageNet":
        transform = transforms.Compose([transforms.Resize((64, 64)), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(64, padding=4),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])
        test_transform = transforms.Compose([transforms.Resize((64, 64)), 
                                             transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                                 ])
        
        train_data = datasets.ImageFolder(root='tiny-imagenet-200/train', transform=transform)
        train_data_no_transform = datasets.ImageFolder(root='tiny-imagenet-200/train', transform=test_transform)
        targets = np.array([label for _, label in train_data.samples])

        if cfg.training.size < len(train_data):
            ratio = cfg.training.size / len(train_data)
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
            train_idx, _ = next(splitter.split(np.zeros(len(targets)), targets))

            train_data = Subset(train_data, train_idx)
            train_data_no_transform = Subset(train_data_no_transform, train_idx)

        targets = [train_data[i][1] for i in range(len(train_data))]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))


        train_data = Subset(train_data, train_idx)
        val_data = Subset(train_data_no_transform, val_idx)
        
        test_data = datasets.ImageFolder(root='tiny-imagenet-200/val/organized', transform=test_transform)
        #test_data = datasets.ImageFolder(root='tiny-imagenet-200/test', transform=transform) -> no labels

    else:

        if cfg.training.dataset == "Fashion-MNIST":

            train_transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),          
                transforms.RandomHorizontalFlip(p=0.5),         
                transforms.ToTensor(),                          
                transforms.Normalize((0.5,), (0.5,))             
            ])

            test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
            ])

            train_data = datasets.FashionMNIST(root="./data", train=True, download=True, transform=train_transform)
            test_data = datasets.FashionMNIST(root="./data", train=False, download=True, transform=test_transform)

            class_names = [
            "T-shirt/top", 
            "Trouser", 
            "Pullover", 
            "Dress", 
            "Coat", 
            "Sandal", 
            "Shirt", 
            "Sneaker", 
            "Bag", 
            "Ankle boot"
            ]


        elif cfg.training.dataset == "CIFAR10":

            transform = transforms.Compose([
             
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
            test_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                                 ])
            
            train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
            test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)


            class_names = [
            "airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"
            ]

        elif cfg.training.dataset == "CIFAR100":
            
            transform = transforms.Compose([
  
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
            test_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                                                 ])
            
            train_data = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
            test_data = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transform)
            
        else:
            raise NotImplementedError("Dataset not implemented")

        

        targets = np.array([label for _, label in train_data])

        ratio = cfg.training.size / len(train_data)
        

        if cfg.training.size < len(train_data):
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - ratio, random_state=42)
            train_idx, _ = next(splitter.split(np.zeros(len(targets)), targets))

            train_data = Subset(train_data, train_idx)

            targets = np.array([label for _, label in train_data])
        

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
        val_idx, train_idx = next(splitter.split(np.zeros(len(targets)), targets))

        val_data = Subset(train_data, val_idx)
        train_data = Subset(train_data, train_idx)

        

    train_loader = DataLoader(dataset=train_data, batch_size=batches, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=batches, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=val_data, batch_size=batches, shuffle=False, num_workers=4)

    
    if log:
        wandb.init(
                project="KA-CNN",
                name= f"{name}, {batches}, {lr}, {cfg.training.size}, {num_epochs}, {optimizer}, {weight_decay}",
                tags = [],
                    
                config={
                "learning_rate": lr,
                "architecture": "CNN",
                "dataset": cfg.training.dataset,
                "epochs": num_epochs,
                }
            )
        if cfg.training.tag != "None":
            wandb.run.tags = [str(cfg.training.tag)]
            

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    if optimizer == "adam":

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
    elif optimizer == "adamw":
        decay = set()
        no_decay = set()
        for n, param in model.named_parameters():
            if 'bn' in n or 'bias' in n:
                no_decay.add(n)
            else:
                decay.add(n)

        param_groups = [
            {"params": [p for n, p in model.named_parameters() if n in decay], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= num_epochs)


    accuracy = -1
    test_loss = 0.0
    start = time.time()           
    
    for epoch in tqdm(range(num_epochs), desc="Training"):

        model.train()
        running_loss = 0.0

        correct = 0
        total = 0

        for (imgs, labels) in train_loader:

            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)

            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)

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
        scheduler.step()

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
    print() 
    
    print("Training time:", (end - start) // 60, "m", "%.2f" % ((end - start) % 60),  "s")
    print(accuracy)

    evaluate_model(model, test_loader, device, class_names, get_top5=get_top5)



if __name__ == "__main__":
    main()