import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
device = "cuda"


def plot_activations(curr_channel, kernel_size, axs, kernel, C=False):
        
        
        for i in range(kernel_size):
            for j in range(kernel_size):

                if kernel_size == 1:
                    axs[i].cla()
                else:
                    axs[i, j].cla()

                x = torch.linspace(-2, 2, 100).to(device)
                x = x.unsqueeze(0).unsqueeze(0)
                idx = curr_channel * kernel_size * kernel_size + i * kernel_size + j
                
                if C:
                    n = kernel.nums[idx]
                    d = kernel.denoms[idx]

                    y = 0
                    n_total = 0
                    d_total = 1
                    for k in range(6):
                        n_total += n[k] * (x ** k) 
                    for k in range(4):
                        d_total += d[k] * (x ** (k + 1))
                    
                    y = n_total / d_total
                else:
                    y = kernel[idx](x)
                
                x = x.squeeze(0).squeeze(0)
                y = y.squeeze(0).squeeze(0)

                if kernel_size == 1:
                    axs[i].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
                else:
                    axs[i, j].plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        
        
        return


def plot_pre(model, layer, twoconv=False, oneconv=False):

    out1 = model.in_c
    out2 = model.hidden_c
    ker = model.ker_size
    out3 = 0
    out4 = 0

    figures = []

    if twoconv:
        out1 = model.in_c
        out2 = model.hidden_c // 2
        out3 = model.hidden_c
        out4 = model.out_c // 2
    elif oneconv:
        out1 = model.out_c
        for i in range(out1):
            fig, axs = plt.subplots(ker, ker * 2, figsize=(20, 6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")

            plot_activations(i, ker, axs[:, :ker], model.conv1, True)
        
        return figures

    figures = []

    if layer == 1:

        for i in range(out1):
            fig, axs = plt.subplots(ker, ker * 2, figsize=(20, 6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")

            if ker == 1:
                plot_activations(i, ker, axs[:ker], model.conv1.kernel)
            else:

                plot_activations(i, ker, axs[:, :ker], model.conv1.kernel)

    elif layer == 2:

        for i in range(out2):
            fig, axs = plt.subplots(ker, ker * 2, figsize=(20, 6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")
            if ker == 1:
                plot_activations(i, ker, axs[:ker], model.conv2.kernel)
            else:
                plot_activations(i, ker, axs[:, :ker], model.conv2.kernel)
    
    elif layer == 3:
        
        for i in range(out3):
            fig, axs = plt.subplots(ker, ker*2, figsize=(20,6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")
            plot_activations(i, ker, axs[:, :ker], model.conv3.kernel)
        
    
    elif layer == 4:
        
        for i in range(out4):
            fig, axs = plt.subplots(ker, ker*2, figsize=(20,6))
            figures.append((fig, axs))
            fig.text(0.3, 0.95, "Inital Kernel Functions " + str(i + 1), ha="center")
            plot_activations(i, ker, axs[:, :ker], model.conv4.kernel)
        
    return figures
    
def plot_post(model, layer, figures, path="", twoconv=False, oneconv=False):

    out1 = model.in_c
    out2 = model.hidden_c
    ker = model.ker_size
    out3 = 0
    out4 = 0

    if twoconv:
        out1 = model.in_c
        out2 = model.hidden_c // 2
        out3 = model.hidden_c
        out4 = model.out_c // 2

    elif oneconv:
        out1 = model.out_c
        for i in range(out1):
            fig, axs = figures[i]
            plot_activations(i, ker, axs[:, ker:], model.conv1, True)
            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv1" + str(i + 1) + ".png")

        return


    if layer == 1:
        for i in range(out1):
            fig, axs = figures[i]
            if ker == 1:
                plot_activations(i, ker, axs[ker:], model.conv1.kernel)
            else:
                plot_activations(i, ker, axs[:, ker:], model.conv1.kernel)
            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv1" + str(i + 1) + ".png")

    elif layer == 2:
        for i in range(out2):
            fig, axs = figures[i]
            if ker == 1:
                plot_activations(i, ker, axs[ker:], model.conv2.kernel)
            else:
                plot_activations(i, ker, axs[:, ker:], model.conv2.kernel)

            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv2" + str(i + 1) + ".png")
    
    elif layer == 3:
        for i in range(out3):
            fig, axs = figures[i]
            plot_activations(i, ker, axs[:, ker:], model.conv2.kernel)

            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv3" + str(i + 1) + ".png")
    
    elif layer == 4:
        for i in range(out4):
            fig, axs = figures[i]
            plot_activations(i, ker, axs[:, ker:], model.conv2.kernel)

            fig.text(0.7, 0.95, "After train", ha="center")
            fig.savefig(path + "PostConv4" + str(i + 1) + ".png")