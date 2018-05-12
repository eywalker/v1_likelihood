#!/usr/bin/env python3
from v1_likelihood import train
import datajoint as dj

#train.CVSet().populate(order='random', reserve_jobs=True)
restr = train.BinConfig & 'bin_counts = 81'
train.LinearRegression().populate(order='random', reserve_jobs=True)
train.CVTrainedModel().populate(restr, order='random', reserve_jobs=True)
#train.KernelRegression().populate(order='random', reserve_jobs=True)
#train.CVTrainedModel3().populate(order='random', reserve_jobs=True, suppress_errors=True)
