from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


@torch.jit.script
def lsq_forward(data, bit, alpha, sym: bool):
    if sym:
        n_lv = 2 ** (bit.detach() - 1) - 1
        data_q = F.hardtanh(data / alpha, -1., 1.) * n_lv
    else:
        n_lv = 2 ** bit.detach() - 1
        data_q = F.hardtanh(data / alpha, 0., 1.) * n_lv

    out = (data_q.round() + (data_q - data_q.detach())) * (alpha / n_lv)
    # out = RoundQuant.apply(data_q) * (alpha / n_lv)
    return out


# sampling based
@torch.jit.script
def noise_quant(data, bit, alpha, is_training: bool, noise: bool, sym: bool, is_stochastic: bool, is_discretize: bool):
    N_BIN = 256
    bit = 2 + torch.sigmoid(bit) * 12

    # Stochastic Rounding
    if is_training and noise and is_stochastic:
        bit += (torch.rand_like(bit) - 0.5)

    if not is_training or is_discretize:
        bit = bit.round() + (bit - bit.detach())

    alpha = F.softplus(alpha)
    lsq = lsq_forward(data, bit.round(), alpha, sym)

    if is_training and noise:
        if sym:
            c1 = data >= alpha
            c2 = data <= -alpha
            delta = alpha / (2 ** (bit - 1) - 1)

            with torch.no_grad():
                diff = (lsq - data) / delta
                sel = diff[torch.logical_not(torch.logical_or(c1, c2))]
                hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)

                noise_here = torch.multinomial(hist, data.numel(), True) + torch.rand_like(data.view(-1))
                noise_here = (noise_here / N_BIN - 0.5).view(data.shape)
            return torch.where(c1, alpha, torch.where(c2, -alpha, data + noise_here * delta))
        else:
            c1 = data >= alpha
            c2 = data <= 0
            delta = alpha / (2 ** bit - 1)

            with torch.no_grad():
                diff = (lsq - data) / delta
                sel = diff[torch.logical_not(torch.logical_or(c1, c2))]
                hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)

                noise_here = torch.multinomial(hist, data.numel(), True) + torch.rand_like(data.view(-1))
                noise_here = (noise_here / N_BIN - 0.5).view(data.shape)
            return torch.where(c1, alpha, torch.where(c2, 0, data + noise_here * delta))
    else:
        return lsq


class Q_Act(nn.Module):
    # xuan: todo: change fixed_bit = 8 for ablation
    def __init__(self):
        super(Q_Act, self).__init__()
        self.quant = True
        self.noise = True

        fixed_bit = 8
        self.bit = Parameter(torch.Tensor(1).zero_())
        bit = (fixed_bit + 0.00001 - 2) / 12
        bit = np.log(bit / (1 - bit))
        self.bit.data.fill_(bit)
        self.bit.requires_grad = False

        self.alpha = Parameter(torch.Tensor(1).fill_(6))

        self.is_stochastic = True
        self.is_discretize = True

    def forward(self, x):
        if self.quant is False:
            return x
        return noise_quant(x, self.bit, self.alpha, self.training, self.noise, False, self.is_stochastic,
                           self.is_discretize)


class Q_Linear(nn.Linear):
    # xuan: todo: change fixed_bit = 4 for ablation
    def __init__(self, *args, act_func=Q_Act, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func()
        self.quant = True
        self.noise = True

        fixed_bit = 4
        self.bit = Parameter(torch.Tensor(1).zero_())
        bit = (fixed_bit + 0.00001 - 2) / 12
        bit = np.log(bit / (1 - bit))
        self.bit.data.fill_(bit)
        self.bit.requires_grad = False

        self.alpha = Parameter(torch.Tensor(1).fill_(1))

        self.is_stochastic = True
        self.is_discretize = True

    def _weight_quant(self):
        if self.quant is False:
            return self.weight
        return noise_quant(self.weight, self.bit, self.alpha, self.training, self.noise, True, self.is_stochastic,
                           self.is_discretize)

    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)

        try:
            return F.linear(x, self._weight_quant(), self.bias)
        except:
            return F.linear(x, torch.transpose(self._weight_quant(), 0, 1), self.bias)   # for gpt training

