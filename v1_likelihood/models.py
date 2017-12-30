import datajoint as dj
import numpy as np
import torch
from torch.nn.init import normal, constant
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=51, dropout=0.9, std=0.01):
        super().__init__()
        self.n_channel = n_channel
        self.std = std
        self.n_output = n_output
        self.dropout = dropout

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)

        self.n_hidden = n_hidden

        n_prev = n_channel
        hiddens = []

        for n in n_hidden:
            hiddens.append(nn.Linear(n_prev, n))
            hiddens.append(nn.ReLU())
            if dropout > 0.0:
                hiddens.append(nn.Dropout(p=dropout, inplace=True))
            n_prev = n
        if len(hiddens) > 0:
            self.hiddens = nn.Sequential(*hiddens)
        else:
            self.hiddens = lambda x: x
        self.ro_layer = nn.Linear(n_prev, n_output)

        self.initialize()

    def forward(self, x):
        x = self.hiddens(x)
        x = self.ro_layer(x)
        return x

    def initialize(self):
        def fn(mod):
            if isinstance(mod, nn.Linear):
                normal(mod.weight, std=self.std)
                constant(mod.bias, 0)
        self.apply(fn)
