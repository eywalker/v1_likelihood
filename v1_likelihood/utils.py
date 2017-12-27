import numpy as np
from numpy.linalg import inv
import hashlib
import random
import torch


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
