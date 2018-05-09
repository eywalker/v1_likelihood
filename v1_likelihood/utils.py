import numpy as np
from numpy.linalg import inv
import hashlib
import random
import torch
from collections import namedtuple


def bin_loc(x, bin_edges, clip=True, include_edge=True):
    assign = np.digitize(x, bin_edges)
    if clip and include_edge:
        assign[x == bin_edges[-1]] = len(bin_edges) - 1
    assign_matrix = np.empty((len(bin_edges) + 1, len(x)))
    assign_matrix.fill(np.nan)
    assign_matrix[np.arange(len(bin_edges) + 1)[:, None] == assign] = 1
    if clip:
        assign_matrix = assign_matrix[1:-1, :]
    return assign_matrix


def binned_group(x, bin_edges, *args, **kwargs):
    assign = bin_loc(x, bin_edges, *args, **kwargs)
    return list(np.where(r == 1) for r in assign)


Stats = namedtuple('stats', ('binc', 'mu', 'sigma', 'n', 'sem'))


def binned_stats(x, y, bin_edges, clip=True):
    assign = bin_loc(x, bin_edges, clip=clip)
    binc = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    mus = np.nanmean(assign * y, axis=1)
    ns = np.nansum(assign, axis=1)
    stds = np.nanstd(assign * y, axis=1)
    sem = stds / np.sqrt(ns)
    return Stats(binc, mus, stds, ns, sem)


def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def lin_reg(train_counts, train_ori, valid_counts):
    tc = extend_ones(train_counts)

    w = inv(tc.T @ tc + np.diag(np.ones(tc.shape[1]) * 0.0001)) @ tc.T @ train_ori

    t_hat = extend_ones(valid_counts) @ w

    return np.sqrt(np.mean((t_hat - valid_ori)**2)) * delta


def list_hash(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def set_seed(seed, cuda=True):
    print('Setting numpy and torch seed to', seed, flush=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed(int(seed))
