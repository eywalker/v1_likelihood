#!/usr/bin/env python3
from v1_likelihood import train3, analysis, simulate_poisson, simulate_gaussian, decision
import datajoint as dj

#simulate_poisson.GaussTuningSet().fill()
#simulate_poisson.PoissonSimulation().populate(order='random', reserve_jobs=True)
#simulate_poisson.GTScores().populate(order='random', reserve_jobs=True)
#simulate_poisson.LinearRegression().populate(order='random', reserve_jobs=True)
#simulate_poisson.FitTuningCurves().populate(order='random', reserve_jobs=True)
#simulate_poisson.FittedPoissonScores().populate(order='random', reserve_jobs=True)
#simulate_poisson.FittedPoissonKL().populate(order='random', reserve_jobs=True)
#simulate_poisson.PoissonTrainedModel().populate(order='random', reserve_jobs=True)
#
#
#simulate_gaussian.GaussTuningSet().fill()
#simulate_gaussian.CorrelationMatrix().populate(order='random', reserve_jobs=True)
#simulate_gaussian.GaussianSimulation().populate(order='random', reserve_jobs=True)
#simulate_gaussian.GTScores().populate(order='random', reserve_jobs=True)
#simulate_gaussian.LinearRegression().populate(order='random', reserve_jobs=True)
#simulate_gaussian.FitTuningCurves().populate(order='random', reserve_jobs=True)
#simulate_gaussian.FittedPoissonScores().populate(order='random', reserve_jobs=True)
#simulate_gaussian.FittedPoissonKL().populate(order='random', reserve_jobs=True)
#simulate_gaussian.OptimalPoissonScores().populate(order='random', reserve_jobs=True)
#simulate_gaussian.OptimalPoissonKL().populate(order='random', reserve_jobs=True)
#simulate_gaussian.GaussianTrainedModel().populate(order='random', reserve_jobs=True)

#simulate_poisson.TrainedNetKL().populate(order='random', reserve_jobs=True)

# simulate_gaussian.GaussianTrainedModelCE().populate(order='random', reserve_jobs=True)
# simulate_poisson.PoissonTrainedModelCE().populate(order='random', reserve_jobs=True)


# train.CVSet().populate(order='random', reserve_jobs=True)
# restr = train.BinConfig & 'bin_counts = 91'
# train.CVTrainedModelWithState().populate(restr, order='random', reserve_jobs=True)
# train.LinearRegression().populate(order='random', reserve_jobs=True)
analysis.LikelihoodStats.populate(order='random', reserve_jobs=True)
analysis.LikelihoodSummary.populate(order='random', reserve_jobs=True)

# Manual step of selecting out the best model should occur here

# now recover the model
# train.BestRecoveredModel().populate(order='random', reserve_jobs=True)

#
#
# targets = analysis.class_discrimination.CSCLookup() & 'count_start=250 or count_stop=250'
#
# train3.CVTrainedModel().populate('objective="mse"', targets, order='random', reserve_jobs=True)
#
#train3.CVTrainedFixedLikelihood().populate('objective="mse"', targets, order='random', reserve_jobs=True)

#decision.CVTrainedLtoD.populate(order='random', reserve_jobs=True)

#
#train.CVTrainedFixedLikelihoodAlt().populate(targets, bin_config, order='random', reserve_jobs=True)
# #
# train.BestRecoveredModel().populate(restr, order='random', reserve_jobs=True)
#
# #train.KernelRegression().populate(order='random', reserve_jobs=True)
# #train.CVTrainedModel3().populate(order='random', reserve_jobs=True, suppress_errors=True)
