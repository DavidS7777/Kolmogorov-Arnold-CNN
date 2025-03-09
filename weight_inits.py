import torch
import numpy as np
from models import *

device = 'cuda'

def init_weights(model, two_conv=False, C=False, KANStarter=False, halfsteps=False, descending=False):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker_size = model.ker_size
    out_c = model.out_c

    skip = False

    if hidden_c == -1:
        hidden_c = out_c
        skip = True

    if not two_conv:
        inits = LeNet5OneFc(in_c, hidden_c, ker_size, out_c, False).to(device)
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

        if skip or KANStarter:
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
        if not descending or not halfsteps:
            if halfsteps:
                inits = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False, halfsteps=True).to(device)

                params = []

                for f in range(hidden_c//2):
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
                    for c in range(hidden_c//2):
                        for i in range(ker_size):
                            for j in range(ker_size):
                                idx = f * hidden_c//2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                                w = inits.conv2.weight[f, c, i, j]
                                if C:
                                    params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                                else:
                                    model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

                if C:
                    model.conv2.nums= nn.Parameter(torch.cat(params, dim=0))

                params = []
                for f in range(out_c//2):
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
                    for c in range(out_c//2):
                        for i in range(ker_size):
                            for j in range(ker_size):
                                idx = f * out_c//2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                                w = inits.conv4.weight[f, c, i, j]
                                if C:
                                    params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                                else:
                                    model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
                if C:
                    model.conv4.nums= nn.Parameter(torch.cat(params, dim=0))
            else:
                inits = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False).to(device)

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
                
                params = []
                for f in range(hidden_c):
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

                params = []
                for f in range(out_c):
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
                    for c in range(out_c):
                        for i in range(ker_size):
                            for j in range(ker_size):
                                idx = f * out_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                                w = inits.conv4.weight[f, c, i, j]
                                if C:
                                    params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                                else:
                                    model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
                if C:
                    model.conv4.nums= nn.Parameter(torch.cat(params, dim=0))
        else:
            inits = LeNet5ConvConvDescending(in_c, hidden_c, ker_size, out_c, False, halfsteps=True).to(device)

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
                
            params = []
            for f in range(hidden_c//2):
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

            params = []
            for f in range(out_c*2):
                for c in range(hidden_c//2):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * hidden_c//2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv3.weight[f, c, i, j]
                            if C:
                                params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                            else:
                                model.conv3.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv3.nums= nn.Parameter(torch.cat(params, dim=0))
            params = []
            for f in range(out_c):
                for c in range(out_c*2):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * out_c*2 * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv4.weight[f, c, i, j]
                            if C:
                                params.append(nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).view(1, -1).to(device)))
                            else:
                                model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
            if C:
                model.conv4.nums= nn.Parameter(torch.cat(params, dim=0))

    return

def all_init(model, C=False, factor=0.01, twoconv=True):

    in_c = model.in_c
    hidden_c = model.hidden_c
    ker_size = model.ker_size
    out_c = model.out_c

    skip = False

    if hidden_c == -1:
        hidden_c = out_c
        skip = True
    
    if twoconv:
        inits = LeNet5ConvConv(in_c, hidden_c, ker_size, out_c, False).to(device)

        params_n = []
        params_d = []

        for f in range(hidden_c // 2):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv1.weight[f, c, i, j]
                        
                        num = torch.tensor(np.random.randn(6)*factor).float()
                        num[1] = w
                        
                        if C:
                            params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                            params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                        else:
                            model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv1.nums= nn.Parameter(torch.cat(params_n, dim=0))        
                model.conv1.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 
            
            params_n = []
            params_d = []
            for f in range(hidden_c):
                for c in range(hidden_c // 2):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * (hidden_c // 2) * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv2.weight[f, c, i, j]
                            num = torch.tensor(np.random.randn(6)*factor).float()
                            num[1] = w
                            
                            if C:
                                params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                                params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                            else:
                                model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv2.nums= nn.Parameter(torch.cat(params_n, dim=0))
                model.conv2.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 

            params_n = []
            params_d = []
            for f in range(out_c // 2):
                for c in range(hidden_c):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv3.weight[f, c, i, j]
                            num = torch.tensor(np.random.randn(6)*factor).float()
                            num[1] = w
                            if C:
                                params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                                params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                            else:
                                model.conv3.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv3.nums= nn.Parameter(torch.cat(params_n, dim=0))
                model.conv3.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 

            params_n = []
            params_d = []
            for f in range(out_c):
                for c in range(out_c // 2):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * (out_c // 2) * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv4.weight[f, c, i, j]
                            num = torch.tensor(np.random.randn(6)*factor).float()
                            num[1] = w
                            if C:
                                params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                                params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                            else:
                                model.conv4.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))
            if C:
                model.conv4.nums= nn.Parameter(torch.cat(params_n, dim=0))
                model.conv4.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 
    else:
        inits = LeNet5OneFc(in_c, hidden_c, ker_size, out_c, False).to(device)

        params_n = []
        params_d = []

        for f in range(hidden_c):
            for c in range(in_c):
                for i in range(ker_size):
                    for j in range(ker_size):
                        idx = f * in_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                        w = inits.conv1.weight[f, c, i, j]
                        num = torch.tensor(np.random.randn(6)*factor).float()
                        num[1] = w
                        
                        if C:
                            params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                            params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                        else:
                            model.conv1.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv1.nums= nn.Parameter(torch.cat(params_n, dim=0))        
                model.conv1.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 
            
            params_n = []
            params_d = []
            for f in range(out_c):
                for c in range(hidden_c):
                    for i in range(ker_size):
                        for j in range(ker_size):
                            idx = f * hidden_c * ker_size * ker_size + c * ker_size * ker_size + i * ker_size + j
                            w = inits.conv2.weight[f, c, i, j]
                            num = torch.tensor(np.random.randn(6)*factor).float()
                            num[1] = w
                            
                            if C:
                                params_n.append(nn.Parameter(num.view(1, -1).to(device)))
                                params_d.append(nn.Parameter(torch.tensor(np.random.randn(4)*factor).float().view(1, -1).to(device)))
                            else:
                                model.conv2.kernel[idx].weight_numerator = nn.Parameter(torch.tensor([0.0, w, 0.0, 0.0, 0.0, 0.0]).to(device))

            if C:
                model.conv2.nums= nn.Parameter(torch.cat(params_n, dim=0))
                model.conv2.denoms= nn.Parameter(torch.cat(params_d, dim=0)) 
        

            
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