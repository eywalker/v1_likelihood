import numpy as np
from numpy.linalg import inv


def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def lin_reg(train_counts, train_ori, valid_counts):
    tc = extend_ones(train_counts)

    w = inv(tc.T @ tc + np.diag(np.ones(tc.shape[1]) * 0.0001)) @ tc.T @ train_ori

    t_hat = extend_ones(valid_counts) @ w

    return np.sqrt(np.mean((t_hat - valid_ori)**2)) * delta