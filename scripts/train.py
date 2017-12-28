#!/usr/bin/env python3
from v1_likelihood import train
import datajoint as dj

train.CVTrainedModel().populate(order='random', reserve_jobs=True)
