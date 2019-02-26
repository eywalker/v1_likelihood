import datajoint as dj
import numpy as np
import torch
from torch.nn.init import normal, constant
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=51, dropout=0.9, std=0.01):
        super().__init__()
        self.n_channel = n_channel
        self.std = std
        self.n_output = n_output
        self.dropout = dropout

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)


        # prune out 0
        n_hidden = [i for i in n_hidden if i != 0]

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


class CombinedNet(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=91, dropout=0.1, std=0.01, nonlin='relu'):
        super().__init__()
        self.n_channel = n_channel
        self.nonlin = nonlin.lower()

        self.std = std
        self.n_output = n_output
        self.dropout = dropout

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)

        # prune out 0
        n_hidden = [i for i in n_hidden if i != 0]

        self.n_hidden = n_hidden

        n_prev = n_channel
        hiddens = OrderedDict()

        for i, n in enumerate(n_hidden):
            prefix = 'layer{}'.format(i)
            hiddens[prefix] = nn.Linear(n_prev, n)
            if self.nonlin != 'none':
                if self.nonlin == 'relu':
                    hiddens[prefix + '_nonlin'] = nn.ReLU()
                else:
                    raise ValueError('Nonlin {} is not yet supported'.format(self.nonlin))
            if dropout > 0.0:
                hiddens[prefix + '_dropout'] = nn.Dropout(p=dropout, inplace=False)
            n_prev = n
        self.hiddens = nn.Sequential(hiddens)

        self.ro_layer = nn.Linear(n_prev, n_output)

        self.initialize()

    def forward(self, x):
        x = self.hiddens(x)
        x = self.ro_layer(x)
        return x

    def l2_weights(self):
        reg = dict(weight=0.0)
        def accum(mod):
            if isinstance(mod, nn.Linear):
                reg['weight'] = reg['weight'] + mod.weight.pow(2).sum()

        self.apply(accum)
        return reg['weight']

    def initialize(self):
        def fn(mod):
            if isinstance(mod, nn.Linear):
                normal(mod.weight, std=self.std)
                constant(mod.bias, 0)
        self.apply(fn)


class PoissonLike(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=51, dropout=0.9, std=0.01):
        super().__init__()
        self.n_channel = n_channel
        self.std = std
        self.n_output = n_output
        self.dropout = dropout

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)

        # prune out 0
        n_hidden = [i for i in n_hidden if i != 0]

        self.n_hidden = n_hidden

        n_prev = n_channel
        hiddens = []

        for n in n_hidden:
            hiddens.append(nn.Linear(n_prev, n))
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


class FixedLikelihoodNet(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=91, dropout=0.5, sigma_init=3, std=0.01):
        super().__init__()
        self.n_channel = n_channel
        self.std = std
        self.n_output = n_output
        self.n_likelihood = n_output
        self.dropout = dropout
        self.sigma_init = sigma_init

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)

        # prune out 0
        n_hidden = [i for i in n_hidden if i != 0]

        self.n_hidden = n_hidden

        self.likelihood = nn.Parameter(torch.ones(1, 1, 1, self.n_likelihood))

        grid_x = torch.linspace(-1, 1, n_output)
        grid_y = torch.ones(n_output)

        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

        n_prev = n_channel

        hiddens = OrderedDict()

        for i, n in enumerate(n_hidden):
            prefix = 'layer{}'.format(i)
            hiddens[prefix] = nn.Linear(n_prev, n)
            hiddens[prefix + '_nonlin'] = nn.ReLU()
            if dropout > 0.0:
                hiddens[prefix + '_dropout'] = nn.Dropout(p=dropout, inplace=False)
            n_prev = n
        self.hiddens = nn.Sequential(hiddens)

        self.mu_ro = nn.Linear(n_prev, 1)
        self.register_buffer('bins', (torch.arange(n_output) - n_output // 2).unsqueeze(0))

        self.initialize()


    def forward(self, x):
        x = self.hiddens(x)
        mus = self.mu_ro(x)

        grid_x = Variable(self.grid_x) + mus

        likelihood = self.likelihood

        grid = torch.stack([grid_x, Variable(self.grid_y).expand_as(grid_x)], dim=-1).unsqueeze(1)
        shifted_likelihood = F.grid_sample(likelihood.expand(x.shape[0], -1, -1, -1), grid, padding_mode='border')

        # remove unneeded dimensions
        return shifted_likelihood.view([-1, shifted_likelihood.shape[-1]])

    def l2_weights(self):
        reg = dict(weight=0.0)

        def accum(mod):
            if isinstance(mod, nn.Linear):
                reg['weight'] = reg['weight'] + mod.weight.pow(2).sum()

        self.apply(accum)

        return reg['weight']

    def initialize(self):
        def fn(mod):
            if isinstance(mod, nn.Linear):
                normal(mod.weight, std=self.std)
                constant(mod.bias, 0)

        self.apply(fn)
        # normal(self.likelihood, std=self.std * 0.1)
        self.likelihood.data.copy_(
            -(torch.arange(self.n_likelihood).view(1, 1, 1, -1) - self.n_likelihood // 2).pow(2) / 2 / self.sigma_init ** 2)



class FlexiNet(nn.Module):
    def __init__(self, n_channel=96, n_hidden=100, n_output=91, dropout=0.5, sigma_init=3, std=0.01):
        super().__init__()
        self.n_channel = n_channel
        self.std = std
        self.n_output = n_output
        self.n_likelihood = n_output
        self.dropout = dropout
        self.sigma_init = sigma_init

        if not isinstance(n_hidden, (list, tuple)):
            n_hidden = (n_hidden,)

        # prune out 0
        n_hidden = [i for i in n_hidden if i != 0]

        self.n_hidden = n_hidden

        self.likelihood = nn.Parameter(torch.ones(1, 1, 1, self.n_likelihood))

        grid_x = torch.linspace(-1, 1, n_output)
        grid_y = torch.ones(n_output)

        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)

        n_prev = n_channel

        hiddens = []

        for n in n_hidden:
            hiddens.append(nn.Linear(n_prev, n))
            hiddens.append(nn.ReLU())
            if dropout > 0.0:
                hiddens.append(nn.Dropout(p=dropout, inplace=False))
            n_prev = n
        if len(hiddens) > 0:
            self.hiddens = nn.Sequential(*hiddens)
        else:
            self.hiddens = lambda x: x
        self.mu_ro = nn.Linear(n_prev, 1)

        self.register_buffer('bins', (torch.arange(n_output) - n_output // 2).unsqueeze(0))

        self.initialize()

        self.detach_center = False
        self.detach_likelihood = False

    def forward(self, x):
        x = self.hiddens(x)
        mus = self.mu_ro(x)
        if self.detach_center:
            mus = mus.detach()
        grid_x = Variable(self.grid_x) + mus

        likelihood = self.likelihood
        if self.detach_likelihood:
            likelihood = likelihood.detach()
        grid = torch.stack([grid_x, Variable(self.grid_y).expand_as(grid_x)], dim=-1).unsqueeze(1)
        shifted_likelihood = F.grid_sample(likelihood.expand(x.shape[0], -1, -1, -1), grid, padding_mode='border')

        # remove unneeded dimensions
        return shifted_likelihood.view([-1, shifted_likelihood.shape[-1]])

    def l2_weights(self):
        reg = dict(weight=0.0)

        def accum(mod):
            if isinstance(mod, nn.Linear):
                reg['weight'] = reg['weight'] + mod.weight.pow(2).sum()

        self.apply(accum)

        return reg['weight']

    def initialize(self):
        def fn(mod):
            if isinstance(mod, nn.Linear):
                normal(mod.weight, std=self.std)
                constant(mod.bias, 0)

        self.apply(fn)
        # normal(self.likelihood, std=self.std * 0.1)
        self.likelihood.data.copy_(
            -(torch.arange(self.n_likelihood).view(1, 1, 1, -1) - self.n_likelihood // 2).pow(2) / 2 / self.sigma_init ** 2)
        # noise = (torch.randn(prior.shape) * 0.).type_as(prior.data)
        # self.likelihood.data.copy_((prior.data + noise).view(1, 1, 1, -1))
        # normal(self.likelihood, std=0.001)
        # self.likelihood.data.zero_()