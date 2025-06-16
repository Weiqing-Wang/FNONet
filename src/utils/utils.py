import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce
from functools import partial




def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

class LpLoss_rel(object):
    def __init__(self, p=2, size_average=True, reduction=True):
        super(LpLoss_rel, self).__init__()

        assert p > 0

        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms
    def __call__(self, x,y):
        return self.rel(x,y)

class LpLoss_abs(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss_abs, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def __call__(self, x, y):
        return self.abs(x, y)
class LpLoss(object):
    def __init__(self):
        super(LpLoss, self).__init__()

    def compute_gradient_ft(self, x):
        grad_x_ft = (torch.fft.rfft2(x[:, :, 1:, :]) - torch.fft.rfft2(x[:, :, :-1, :]))
        grad_y_ft = (torch.fft.rfft2(x[:, :, :, 1:]) - torch.fft.rfft2(x[:, :, :, :-1]))
        return grad_x_ft, grad_y_ft
    def loss_function(self, out,y,channel_weights):

        lossU = ((out[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        lossV = ((out[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        lossP = torch.abs((out[:, 2, :, :] - y[:, 2, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        loss = (lossU + lossV + lossP)/channel_weights

        out_grad_x_ft_p, out_grad_y_ft_p = self.compute_gradient_ft(out[:, 2, :, :].unsqueeze(1))
        y_grad_x_ft_p, y_grad_y_ft_p = self.compute_gradient_ft(y[:, 2, :, :].unsqueeze(1))

        loss_grad_ft_p_x = torch.abs(out_grad_x_ft_p - y_grad_x_ft_p)
        loss_grad_ft_p_y = torch.abs(out_grad_y_ft_p - y_grad_y_ft_p)

        gamma = 0.1

        loss = (lossU + lossV + lossP) / channel_weights + gamma * (loss_grad_ft_p_x.mean() + loss_grad_ft_p_y.mean())
        return torch.sum(loss)
    def __call__(self, x, y,channel_weights):
        return self.loss_function(x, y,channel_weights)




def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2



class Unet_loss(object):
    def __init__(self):
        super(Unet_loss, self).__init__()
    def loss_function(self, out,y,channel_weights):

        lossU = ((out[:, 0, :, :] - y[:, 0, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        lossV = ((out[:, 1, :, :] - y[:, 1, :, :]) ** 2).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        lossP = torch.abs((out[:, 2, :, :] - y[:, 2, :, :])).reshape(y.shape[0], 1, y.shape[2], y.shape[3])
        loss = (lossU + lossV + lossP)/channel_weights

        return torch.sum(loss)
    def __call__(self, x, y,channel_weights):
        return self.loss_function(x, y,channel_weights)

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)



