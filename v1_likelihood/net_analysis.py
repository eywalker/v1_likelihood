from v1_likelihood import train3
from v1_likelihood.analysis import class_discrimination, cd_dataset, cd_dlset
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
import seaborn as sns

from torch.optim import SGD
import datajoint as dj

schema = dj.schema('edgar_cd_net_analysis')


def make_attr_relu(base_input, base_output):
    class AttrReLU(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

        @staticmethod
        def forward(ctx, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            output = input.clamp(min=0)
            ctx.save_for_backward(input, output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, output = ctx.saved_variables

            return grad_output * (output - base_output) / (input - base_input + 1e-12)

    return AttrReLU


class AttrReLUModule(nn.Module):
    def __init__(self, base_input, base_output):
        super().__init__()
        self.attr_relu = make_attr_relu(base_input, base_output)

    def forward(self, x):
        return self.attr_relu.apply(x)


@schema
class Attribution(dj.Computed):
    definition = """
    -> train3.CVTrainedModel
    ---
    grad_attr_scores: longblob
    gi_attr_scores: longblob
    gd_attr_scores: longblob
    lift_attr_scores: longblob
    grad_attr_corr = null: float
    gi_attr_corr = null: float
    gd_attr_corr = null: float
    lift_attr_corr = null: float
    """


    @property
    def key_source(self):
        datasets = class_discrimination.CSCLookup & 'count_start = 0 and count_stop = 500'
        return train3.CVTrainedModel & (train3.BestNonlin() & datasets & 'selection_objective = "mse"').proj()

    def make(self, key):

        net = (train3.CVTrainedModel & key).load_model()

        net.cuda()
        net.eval()

        train_x, train_t, valid_x, valid_t, prior, objective = train3.CVTrainedModel().prepare_parts(key)

        tx = Variable(train_x.cuda(), requires_grad=True)

        mu = tx.mean(dim=0).unsqueeze(0)
        input0 = net.hiddens.layer0(mu)
        output0 = net.hiddens.layer0_nonlin(input0)
        input1 = net.hiddens.layer1(output0)
        output1 = net.hiddens.layer1_nonlin(input1)

        # get grad, gi, and gd first
        optim = SGD([tx], lr=0.01)

        # compute for mean
        optim.zero_grad()

        logL = net(tx)

        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)

        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        mu.sum().backward()
        mp = tx.grad.data.cpu().numpy()

        # compute for variance
        optim.zero_grad()
        logL = net(tx)
        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)
        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        var = (L * v.pow(2)).sum(dim=1) - mu.pow(2)
        var.sum().backward()
        vp = tx.grad.data.cpu().numpy()

        delta_x = (train_x - train_x.mean(dim=0)).numpy()


        # Gradient based
        attr_mu = mp
        attr_var = vp
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_var_avg = np.abs(attr_var).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_var_avg)[0, 1]
        key['grad_attr_scores'] = np.stack([attr_mu_avg, attr_var_avg])
        key['grad_attr_corr'] = corr

        # Gradient * Input
        attr_mu = mp * train_x.numpy()
        attr_var = vp * train_x.numpy()
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_var_avg = np.abs(attr_var).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_var_avg)[0, 1]
        key['gi_attr_scores'] = np.stack([attr_mu_avg, attr_var_avg])
        key['gi_attr_corr'] = corr

        # Gradient * Delta Input
        attr_mu = mp * delta_x
        attr_var = vp * delta_x
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_var_avg = np.abs(attr_var).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_var_avg)[0, 1]
        key['gd_attr_scores'] = np.stack([attr_mu_avg, attr_var_avg])
        key['gd_attr_corr'] = corr

        ## Now compute DeepLIFT
        net.hiddens.layer0_nonlin = AttrReLUModule(input0, output0)
        net.hiddens.layer1_nonlin = AttrReLUModule(input1, output1)

        # get grad, gi, and gd first
        optim = SGD([tx], lr=0.01)

        # compute for mean
        optim.zero_grad()

        logL = net(tx)

        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)

        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        mu.sum().backward()
        mp = tx.grad.data.cpu().numpy()

        # compute for variance
        optim.zero_grad()
        logL = net(tx)
        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)
        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        var = (L * v.pow(2)).sum(dim=1) - mu.pow(2)
        var.sum().backward()
        vp = tx.grad.data.cpu().numpy()

        delta_x = (train_x - train_x.mean(dim=0)).numpy()


        attr_mu = mp * delta_x
        attr_var = vp * delta_x
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_var_avg = np.abs(attr_var).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_var_avg)[0, 1]
        key['lift_attr_scores'] = np.stack([attr_mu_avg, attr_var_avg])
        key['lift_attr_corr'] = corr

        self.insert1(key)


@schema
class Attribution2(dj.Computed):
    """
    Same as Attribution but performing attribution on the standard deviation instead of the variance
    of the likelihood function.
    """
    definition = """
    -> train3.CVTrainedModel
    ---
    grad_attr_scores: longblob
    gi_attr_scores: longblob
    gd_attr_scores: longblob
    lift_attr_scores: longblob
    grad_attr_corr = null: float
    gi_attr_corr = null: float
    gd_attr_corr = null: float
    lift_attr_corr = null: float
    """


    @property
    def key_source(self):
        datasets = class_discrimination.CSCLookup & 'count_start = 0 and count_stop = 500'
        return train3.CVTrainedModel & (train3.BestNonlin() & datasets & 'selection_objective = "mse"').proj()

    def make(self, key):

        net = (train3.CVTrainedModel & key).load_model()

        net.cuda()
        net.eval()

        train_x, train_t, valid_x, valid_t, prior, objective = train3.CVTrainedModel().prepare_parts(key)

        tx = Variable(train_x.cuda(), requires_grad=True)

        mu = tx.mean(dim=0).unsqueeze(0)
        input0 = net.hiddens.layer0(mu)
        output0 = net.hiddens.layer0_nonlin(input0)
        input1 = net.hiddens.layer1(output0)
        output1 = net.hiddens.layer1_nonlin(input1)

        # get grad, gi, and gd first
        optim = SGD([tx], lr=0.01)

        # compute for mean
        optim.zero_grad()

        logL = net(tx)

        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)

        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        mu.sum().backward()
        mp = tx.grad.data.cpu().numpy()

        # compute for variance
        optim.zero_grad()
        logL = net(tx)
        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)
        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        sigma = torch.sqrt((L * v.pow(2)).sum(dim=1) - mu.pow(2))
        sigma.sum().backward()
        sp = tx.grad.data.cpu().numpy()

        delta_x = (train_x - train_x.mean(dim=0)).numpy()


        # Gradient based
        attr_mu = mp
        attr_sigma = sp
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_sigma_avg = np.abs(attr_sigma).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_sigma_avg)[0, 1]
        key['grad_attr_scores'] = np.stack([attr_mu_avg, attr_sigma_avg])
        key['grad_attr_corr'] = corr

        # Gradient * Input
        attr_mu = mp * train_x.numpy()
        attr_sigma = sp * train_x.numpy()
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_sigma_avg = np.abs(attr_sigma).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_sigma_avg)[0, 1]
        key['gi_attr_scores'] = np.stack([attr_mu_avg, attr_sigma_avg])
        key['gi_attr_corr'] = corr

        # Gradient * Delta Input
        attr_mu = mp * delta_x
        attr_sigma = sp * delta_x
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_sigma_avg = np.abs(attr_sigma).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_sigma_avg)[0, 1]
        key['gd_attr_scores'] = np.stack([attr_mu_avg, attr_sigma_avg])
        key['gd_attr_corr'] = corr

        ## Now compute DeepLIFT
        net.hiddens.layer0_nonlin = AttrReLUModule(input0, output0)
        net.hiddens.layer1_nonlin = AttrReLUModule(input1, output1)

        # get grad, gi, and gd first
        optim = SGD([tx], lr=0.01)

        # compute for mean
        optim.zero_grad()

        logL = net(tx)

        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)

        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        mu.sum().backward()
        mp = tx.grad.data.cpu().numpy()

        # compute for variance
        optim.zero_grad()
        logL = net(tx)
        logL = logL - logL.max(dim=1)[0].unsqueeze(-1)
        L = torch.exp(logL)
        L = L / L.sum(dim=1).unsqueeze(-1)
        v = Variable(torch.arange(91)).cuda()
        mu = (L * v).sum(dim=1)
        sigma = torch.sqrt((L * v.pow(2)).sum(dim=1) - mu.pow(2))
        sigma.sum().backward()
        sp = tx.grad.data.cpu().numpy()

        delta_x = (train_x - train_x.mean(dim=0)).numpy()


        attr_mu = mp * delta_x
        attr_sigma = sp * delta_x
        attr_mu_avg = np.abs(attr_mu).mean(axis=0)
        attr_sigma_avg = np.abs(attr_sigma).mean(axis=0)
        corr = np.corrcoef(attr_mu_avg, attr_sigma_avg)[0, 1]
        key['lift_attr_scores'] = np.stack([attr_mu_avg, attr_sigma_avg])
        key['lift_attr_corr'] = corr

        self.insert1(key)