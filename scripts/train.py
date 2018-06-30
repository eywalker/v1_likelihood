#!/usr/bin/env python3
from v1_likelihood import train, analysis, simulate_poisson
import datajoint as dj

simulate_poisson.GaussTuningSet().fill()
simulate_poisson.PoissonSimulation().populate(order='random', reserve_jobs=True)
simulate_poisson.GTScores().populate(order='random', reserve_jobs=True)
simulate_poisson.LinearRegression().populate(order='random', reserve_jobs=True)
simulate_poisson.FitTuningCurves().populate(order='random', reserve_jobs=True)
simulate_poisson.FittedPoissonScores().populate(order='random', reserve_jobs=True)
simulate_poisson.FittedPoissonKL().populate(order='random', reserve_jobs=True)
simulate_poisson.PoissonTrainedModel().populate(order='random', reserve_jobs=True)
#simulate_poisson.TrainedNetKL().populate(order='random', reserve_jobs=True)


# #train.CVSet().populate(order='random', reserve_jobs=True)
# restr = train.BinConfig & 'bin_counts = 91'
# train.LinearRegression().populate(order='random', reserve_jobs=True)
# analysis.LikelihoodStats.populate(order='random', reserve_jobs=True)
# #train.RefinedCVTrainedModel().populate(restr, order='random', reserve_jobs=True)
#
# train.BestRecoveredModel().populate(restr, order='random', reserve_jobs=True)
#
# #train.KernelRegression().populate(order='random', reserve_jobs=True)
# #train.CVTrainedModel3().populate(order='random', reserve_jobs=True, suppress_errors=True)
