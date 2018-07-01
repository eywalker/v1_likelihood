import datajoint as dj
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from attorch.train import early_stopping
from torch.utils.data import TensorDataset, DataLoader
from numpy.linalg import inv
from .models import Net
from .utils import list_hash, set_seed
from itertools import chain, product, count
from tqdm import tqdm
from numpy.linalg import matrix_rank, inv, svd

schema = dj.schema('edgar_cd_ml_sim_gauss')


def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def binnify(x, center=0, delta=1, nbins=61, clip=True):
    """
    Bin the dat into bins, with center bin at `center`. Each bin has width `delta`
    and you will have equal number of bins to the left and to the right of the center bin.
    The left most bin starts at bin number and the last bin at `nbins`-1. If `clip`=True,
    then data falling out of the bins would be assigned bin number `-1` to indicate that it
    is out of the range. Otherwise, the data would be assigned to the nearest edge bin. A data point
    x would fall into bin i if  bin_i_left <= x < bin_i_right

    Args:
        x: data to bin
        center: center of the bins
        delta: width of each bin
        nbins: number of bins
        clip: whether to clip data falling out of bin range. Defaults to True

    Returns:
        (xv, p) - xv is an array of bin centers and thus has length nbins. p is the bin assignment of
            each data point in x and thus len(p) == len(x).
    """
    p = np.round((x - center) / delta) + (nbins // 2)
    if clip:
        out = (p < 0) | (p >= nbins)
        p[out] = -1
    else:
        p[p < 0] = 0
        p[p >= nbins] = nbins - 1

    xv = (np.arange(nbins) - nbins // 2) * delta + center
    return xv, p


@schema
class StimulusSession(dj.Lookup):
    definition = """
    stim_id: int   # simulation id
    stim_seed: int # simulation seed
    ---
    train_size: int    # trainset size
    valid_size: int    # validset size
    test_size: int     # testset size
    """
    contents = [(0, 12345, 800, 200, 200)]

    sigma_a = 3
    sigma_b = 15

    def get_stimulus(self, key=None):
        key = key or {}
        key = (self & key).fetch1()

        np.random.seed(key['stim_seed'])
        train_size, valid_size, test_size = key['train_size'], key['valid_size'], key['test_size']
        total_trials = train_size + valid_size + test_size
        sv = np.random.choice([self.sigma_a, self.sigma_b], size=total_trials)
        ori = np.random.randn(total_trials) * sv

        # split them into training set, validation set, and test set
        return ori[:train_size], ori[train_size:(train_size + valid_size)], ori[(train_size + valid_size):]

    @classmethod
    def log_prior(cls, pv):
        sigmaA = cls.sigma_a
        sigmaB = cls.sigma_b
        return np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)


@schema
class GaussTuningSet(dj.Manual):
    definition = """
    tuning_set_id: int    # tuning set id
    ---
    n_cells: int  # number of cells
    centers: longblob # ceneters of tuning curves
    amps: longblob   # amplitudes of tuning curves
    widths: longblob # widths of tuning curves
    """

    def fill(self):
        # add first set of tuned neurons
        key = {
            'tuning_set_id': 0,
            'n_cells': 96,
        }
        key['centers'] = np.linspace(-40, 40, key['n_cells'])
        key['amps'] = 6 * np.ones(key['n_cells'])
        key['widths'] = 21 * np.ones(key['n_cells'])
        self.insert1(key, skip_duplicates=True)

        key = {
            'tuning_set_id': 1,
            'n_cells': 96,
        }
        key['centers'] = np.linspace(-40, 40, key['n_cells'])
        key['amps'] = np.ones(key['n_cells'])
        key['widths'] = 30 * np.ones(key['n_cells'])
        self.insert1(key, skip_duplicates=True)


    def get_f(self, key=None):
        """
        Return a function that would return mean population response for stimuli.
        Returns n_neurons x n_stimuli matrix
        """
        key = key or {}
        centers, amps, widths = (self & key).fetch1('centers', 'amps', 'widths')
        centers = centers[:, None]
        amps = amps[:, None]
        widths = widths[:, None]

        def f(stim):
            return np.exp(-(stim - centers) ** 2 / 2 / widths ** 2) * amps

        return f


@schema
class ResponseGain(dj.Lookup):
    definition = """
    gain_id: int  # gain id
    ---
    gain: float   # gain value
    """
    contents = [(0, 1)]


@schema
class SimulationSeed(dj.Lookup):
    definition = """
    sim_seed: int   # simulation seed
    """
    contents = list(zip([114514]))


@schema
class CorrSeed(dj.Lookup):
    definition = """
    corr_seed: int  # correlation seed
    """
    contents = list(zip([34839]))

@schema
class CorrelationMatrix(dj.Computed):
    definition = """
    -> GaussTuningSet
    -> CorrSeed
    ---
    corr_matrix: longblob  # correlation matrix
    avg_corr: float        # average correlation in the matrix
    max_corr: float        # maximum correlation in the matrix
    """

    def make(self, key):
        n_cells = (GaussTuningSet() & key).fetch1('n_cells')
        corr_seed = (CorrSeed() & key).fetch1('corr_seed')

        np.random.seed(corr_seed)

        a = np.random.rand(n_cells, 10) - 0.5
        s = a @ a.T + np.eye(n_cells) * 0.1

        f = np.diag(1 / np.sqrt(np.diag(s)))

        ss = f @ s @ f  # normalize the marginals

        # ensure that the corr matrix is full rank
        assert matrix_rank(ss) == n_cells
        off_ss = ss - np.eye(n_cells)
        key['avg_corr'] = np.abs(off_ss).mean()
        key['max_corr'] = np.abs(off_ss).max()
        key['corr_matrix'] = ss

        self.insert1(key)


def sample_normal(N, sigma):
    u, sv, ht = svd(sigma)
    x = np.random.randn(sigma.shape[0], N)
    return u @ np.diag(np.sqrt(sv)) @ x

@schema
class GaussianSimulation(dj.Computed):
    definition = """
    -> StimulusSession
    -> GaussTuningSet
    -> CorrelationMatrix
    -> ResponseGain
    -> SimulationSeed
    """

    def make(self, key):
        self.insert1(key)

    def simulate_responses(self, key=None):
        key = key or {}
        key = (self & key).fetch1('KEY')
        gain = (ResponseGain() & key).fetch1('gain')
        sim_seed = (SimulationSeed() & key).fetch1('sim_seed')
        stims = (StimulusSession() & key).get_stimulus()
        sigma = (CorrelationMatrix() & key).fetch1('corr_matrix')
        sigma_inv = inv(sigma)
        stim_keys = ['train', 'valid', 'test']

        # this is the expected value of the spike counts for each stimulus
        f = (GaussTuningSet() & key).get_f()

        responses = {
            'mean_f': f
        }

        np.random.seed(sim_seed)
        for k, stim in zip(stim_keys, stims):
            resp_set = {}
            mus = (f(stim) * gain).T  # make it into trials x units
            counts = mus + sample_normal(mus.shape[0], sigma).T * np.sqrt(mus)

            def closure(counts):
                def ll(decode_stim):
                    mu_hat = (gain * f(decode_stim))[None, ...]  # 1 x units x dec_stim
                    v = (counts[..., None] - mu_hat) / np.sqrt(mu_hat)
                    vv = np.transpose(v, [1, 2, 0])
                    shape = vv.shape
                    vv = vv.reshape(shape[0], -1)

                    logl = (-(vv * (sigma_inv @ vv)).sum(axis=0) / 2).reshape(*shape[1:]).T
                    logl = logl - logl.max(axis=1, keepdims=True)
                    return logl

                return ll

            resp_set['stimulus'] = stim
            resp_set['expected_responses'] = mus
            resp_set['counts'] = counts
            resp_set['logl_f'] = closure(counts)

            responses[k] = resp_set

        return responses


@schema
class BinConfig(dj.Lookup):
    definition = """
    bin_config_id  : varchar(128)   # id
    ---
    bin_width: decimal(3, 2)
    bin_counts: int  # number of bins
    clip_outside: bool   # whether to clip outside
    """
    contents = [
        (list_hash(x),) + x for x in [
            (1.0, 91, True)
        ]
    ]

    def get_binc(self, key=None):
        if key is None:
            key = self.fetch1('KEY')
        width, counts = (self & key).fetch1('bin_width', 'bin_counts')
        return (np.arange(counts) - counts // 2) * width


@schema
class GTScores(dj.Computed):
    definition = """
    -> GaussianSimulation
    -> BinConfig
    ---
    gt_trainset_score: float 
    gt_validset_score: float
    gt_testset_score: float
    """

    def make(self, key):
        resp = (GaussianSimulation() & key).simulate_responses()

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)

        sigmaA = 3
        sigmaB = 15

        set_types = ['train', 'valid', 'test']

        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)

        for st in set_types:
            logl = resp[st]['logl_f'](pv)
            ori = resp[st]['stimulus']
            xv, ori_bins = binnify(ori, delta=delta, nbins=nbins, clip=clip_outside)

            y = logl + prior
            t_hat = np.argmax(y, 1)

            key['gt_{}set_score'.format(st)] = np.sqrt(np.mean((t_hat.ravel() - ori_bins) ** 2)) * delta

        self.insert1(key)


def mse(y, t):
    return np.sqrt(np.mean((y - t) ** 2))


@schema
class LinearRegression(dj.Computed):
    definition = """
    -> BinConfig
    -> GaussianSimulation
    ---
    lr_weights : longblob        # learned weights
    lr_trainset_score:  float    # score on trainset 
    lr_validset_score:  float    # score on validation set
    lr_testset_score:   float    # score on testset
    """

    def make(self, key):
        resp = (GaussianSimulation() & key).simulate_responses()

        bin_width, bin_counts, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        bin_width = float(bin_width)

        train_counts, train_ori = resp['train']['counts'], resp['train']['stimulus']
        valid_counts, valid_ori = resp['valid']['counts'], resp['valid']['stimulus']

        test_counts, test_ori = resp['test']['counts'], resp['test']['stimulus']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, valid_bins = binnify(valid_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, test_bins = binnify(test_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_bins = train_bins[good_pos]

        good_pos = valid_bins >= 0
        valid_counts = valid_counts[good_pos]
        valid_bins = valid_bins[good_pos]

        good_pos = test_bins >= 0
        test_counts = test_counts[good_pos]
        test_bins = test_bins[good_pos]

        grouped_counts = np.concatenate([train_counts, valid_counts])
        grouped_bins = np.concatenate([train_bins, valid_bins])

        gc = extend_ones(grouped_counts)
        # plain old ridge linear regression
        w = inv(gc.T @ gc + np.eye(gc.shape[1]) * 0.0001) @ gc.T @ grouped_bins

        t_hat_train = extend_ones(train_counts) @ w
        t_hat_valid = extend_ones(valid_counts) @ w
        t_hat_test = extend_ones(test_counts) @ w

        train_score = mse(t_hat_train, train_bins) * bin_width
        valid_score = mse(t_hat_valid, valid_bins) * bin_width
        test_score = mse(t_hat_test, test_bins) * bin_width

        key['lr_weights'] = w
        key['lr_trainset_score'] = train_score
        key['lr_validset_score'] = valid_score
        key['lr_testset_score'] = test_score

        self.insert1(key)


@schema
class FitTuningCurves(dj.Computed):
    definition = """
    -> GaussianSimulation
    ---
    fit_amps: longblob            # amplitude of tuning curves
    fit_centers: longblob        # center of tuning curves
    fit_widths: longblob         # width of tuning curves
    """

    def make(self, key):
        resp = (GaussianSimulation() & key).simulate_responses()

        counts = np.concatenate([resp['train']['counts'], resp['valid']['counts']])
        ori = np.concatenate([resp['train']['stimulus'], resp['valid']['stimulus']])
        ct = counts.T

        mu_counts = (ct * ori).sum(axis=1, keepdims=True) / ct.sum(axis=1, keepdims=True)
        sigma_counts = np.sqrt(
            (ct * ori ** 2).sum(axis=1, keepdims=True) / ct.sum(axis=1, keepdims=True) - mu_counts ** 2)
        max_counts = counts.max(axis=0)

        from scipy.optimize import curve_fit

        def gaus(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / 2 / (sigma ** 2))

        amps = []
        centers = []
        widths = []

        for i in range(counts.shape[1]):
            pp, po = curve_fit(gaus, ori, counts[:, i], p0=[max_counts[i], mu_counts[i], sigma_counts[i]])
            amps.append(pp[0])
            centers.append(pp[1])
            widths.append(pp[2])

        key['fit_amps'] = np.array(amps)
        key['fit_centers'] = np.array(centers)
        key['fit_widths'] = np.array(widths)

        self.insert1(key)

    def get_f(self, key=None):
        """
        Return a function that would return mean population response for stimuli.
        Returns n_neurons x n_stimuli matrix
        """
        key = key or {}
        centers, amps, widths = (self & key).fetch1('fit_centers', 'fit_amps', 'fit_widths')
        centers = centers[:, None]
        amps = amps[:, None]
        widths = widths[:, None]

        def f(stim):
            return np.exp(-(stim - centers) ** 2 / 2 / widths ** 2) * amps

        return f


@schema
class FittedPoissonScores(dj.Computed):
    definition = """
    -> FitTuningCurves
    -> BinConfig
    ---
    fit_trainset_score: float 
    fit_validset_score: float
    fit_testset_score: float
    """

    def response_summary(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        stim_keys = ['train', 'valid', 'test']

        # this is the expected value of the spike counts for each stimulus
        f = (FitTuningCurves() & key).get_f()
        responses = (GaussianSimulation() & key).simulate_responses()

        responses['mean_f'] = f

        for k in stim_keys:
            resp_set = responses[k]
            mus = f(resp_set['stimulus'])
            counts = resp_set['counts']

            def closure(counts):
                def ll(decode_stim):
                    mu_hat = f(decode_stim)[None, ...]  # 1 x units x dec_stim
                    logl = (counts[..., None] * np.log(mu_hat)).sum(axis=1) - mu_hat.sum(axis=1)  # trials x dec_stim
                    return logl

                return ll

            resp_set['expected_responses'] = mus
            resp_set['logl_f'] = closure(counts)

        return responses

    def make(self, key):
        resp = self.response_summary(key)

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)

        sigmaA = 3
        sigmaB = 15

        set_types = ['train', 'valid', 'test']

        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)

        for st in set_types:
            logl = resp[st]['logl_f'](pv)
            ori = resp[st]['stimulus']
            xv, ori_bins = binnify(ori, delta=delta, nbins=nbins, clip=clip_outside)

            y = logl + prior
            t_hat = np.argmax(y, 1)

            key['fit_{}set_score'.format(st)] = np.sqrt(np.mean((t_hat.ravel() - ori_bins) ** 2)) * delta

        self.insert1(key)


@schema
class FittedPoissonKL(dj.Computed):
    definition = """
    -> FittedPoissonScores
    ---
    fit_train_med_kl: float  # med KL
    fit_valid_med_kl: float  # med KL
    fit_test_med_kl: float  # med KL
    fit_train_kl: longblob   # KL values
    fit_valid_kl: longblob   # KL values
    fit_test_kl: longblob   # KL values
    """

    def make(self, key):
        gt_resp = (GaussianSimulation() & key).simulate_responses()
        fit_resp = (FittedPoissonScores() & key).response_summary()

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)
        set_types = ['train', 'valid', 'test']
        pv = (np.arange(nbins) - nbins // 2) * delta

        for st in set_types:
            gt_logl = gt_resp[st]['logl_f'](pv)
            gt_nl = np.exp(gt_logl)
            gt_nl = gt_nl / np.sum(gt_nl, axis=1, keepdims=True)

            fit_logl = fit_resp[st]['logl_f'](pv)
            fit_nl = np.exp(fit_logl)
            fit_nl = fit_nl / np.sum(fit_nl, axis=1, keepdims=True)

            eps = 1e-15
            KL = ((np.log(gt_nl + eps) - np.log(fit_nl + eps)) * gt_nl).sum(axis=1)
            key['fit_{}_med_kl'.format(st)] = np.median(KL)
            key['fit_{}_kl'.format(st)] = KL

        self.insert1(key)


@schema
class OptimalPoissonScores(dj.Computed):
    definition = """
    -> GaussianSimulation
    -> BinConfig
    ---
    fit_trainset_score: float 
    fit_validset_score: float
    fit_testset_score: float
    """

    def response_summary(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        stim_keys = ['train', 'valid', 'test']

        # this is the expected value of the spike counts for each stimulus
        responses = (GaussianSimulation() & key).simulate_responses()

        f = responses['mean_f']

        for k in stim_keys:
            resp_set = responses[k]
            mus = f(resp_set['stimulus'])
            counts = resp_set['counts']

            def closure(counts):
                def ll(decode_stim):
                    mu_hat = f(decode_stim)[None, ...]  # 1 x units x dec_stim
                    logl = (counts[..., None] * np.log(mu_hat)).sum(axis=1) - mu_hat.sum(axis=1)  # trials x dec_stim
                    return logl

                return ll

            resp_set['expected_responses'] = mus
            resp_set['logl_f'] = closure(counts)

        return responses

    def make(self, key):
        resp = self.response_summary(key)

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)

        sigmaA = 3
        sigmaB = 15

        set_types = ['train', 'valid', 'test']

        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)

        for st in set_types:
            logl = resp[st]['logl_f'](pv)
            ori = resp[st]['stimulus']
            xv, ori_bins = binnify(ori, delta=delta, nbins=nbins, clip=clip_outside)

            y = logl + prior
            t_hat = np.argmax(y, 1)

            key['fit_{}set_score'.format(st)] = np.sqrt(np.mean((t_hat.ravel() - ori_bins) ** 2)) * delta

        self.insert1(key)


@schema
class OptimalPoissonKL(dj.Computed):
    definition = """
    -> OptimalPoissonScores
    ---
    fit_train_med_kl: float  # med KL
    fit_valid_med_kl: float  # med KL
    fit_test_med_kl: float  # med KL
    fit_train_kl: longblob   # KL values
    fit_valid_kl: longblob   # KL values
    fit_test_kl: longblob   # KL values
    """

    def make(self, key):
        gt_resp = (GaussianSimulation() & key).simulate_responses()
        fit_resp = (OptimalPoissonScores() & key).response_summary()

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)
        set_types = ['train', 'valid', 'test']
        pv = (np.arange(nbins) - nbins // 2) * delta

        for st in set_types:
            gt_logl = gt_resp[st]['logl_f'](pv)
            gt_nl = np.exp(gt_logl)
            gt_nl = gt_nl / np.sum(gt_nl, axis=1, keepdims=True)

            fit_logl = fit_resp[st]['logl_f'](pv)
            fit_nl = np.exp(fit_logl)
            fit_nl = fit_nl / np.sum(fit_nl, axis=1, keepdims=True)

            eps = 1e-15
            KL = ((np.log(gt_nl + eps) - np.log(fit_nl + eps)) * gt_nl).sum(axis=1)
            key['fit_{}_med_kl'.format(st)] = np.median(KL)
            key['fit_{}_kl'.format(st)] = KL

        self.insert1(key)


@schema
class TrainSeed(dj.Lookup):
    definition = """
    # training seed
    train_seed:   int       # training seed
    """
    contents = zip((8, 92, 123, 823))


@schema
class ModelDesign(dj.Lookup):
    definition = """
    model_id: varchar(128)   # model id
    ---
    hidden1:  int      # size of first hidden layer
    hidden2:  int      # size of second hidden layer
    """
    contents = [(list_hash(x),) + x for x in [
        (400, 400),
        (600, 600),
        (800, 800),
        (1000, 1000)
    ]]


@schema
class TrainParam(dj.Lookup):
    definition = """
    param_id: varchar(128)    # ID of parameter
    ---
    learning_rate:  float     # initial learning rate
    dropout:       float     # dropout rate
    init_std:       float     # standard deviation for weight initialization
    smoothness:     float     # regularizer on Laplace smoothness
    """
    contents = [(list_hash(x),) + x for x in product(
        (0.01, 0.03, 0.3),  # learning rate
        (0.1, 0.4, 0.5, 0.9),  # dropout rate
        (0.01, 0.001, 0.0001),  # initialization std
        (0.3, 3, 30, 300, 3000)  # smoothness
    )]


@schema
class GaussianTrainedModel(dj.Computed):
    definition = """
    -> GaussianSimulation
    -> BinConfig
    -> ModelDesign
    -> TrainParam
    -> TrainSeed
    ---
    cnn_train_score: float   # score on train set
    cnn_valid_score:  float   # score on validation set
    cnn_test_score: float     # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob  # saved model
    """

    def load_model(self, key=None):
        if key is None:
            key = {}

        rel = self & key

        state_dict = rel.fetch1('model')
        state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}

        init_std = float((TrainParam() & rel).fetch1('init_std'))
        dropout = float((TrainParam() & rel).fetch1('dropout'))
        h1, h2 = [int(x) for x in (ModelDesign() & rel).fetch1('hidden1', 'hidden2')]
        nbins = int((BinConfig() & rel).fetch1('bin_counts'))

        net = Net(n_output=nbins, n_hidden=[h1, h2], std=init_std, dropout=dropout)
        net.load_state_dict(state_dict)
        return net

    def get_dataset(self, key=None, keep_all=False):
        if key is None:
            key = self.fetch1(dj.key)

        resp = (GaussianSimulation() & key).simulate_responses()
        bin_width = float((BinConfig() & key).fetch1('bin_width'))
        bin_counts = int((BinConfig() & key).fetch1('bin_counts'))
        clip_outside = bool((BinConfig() & key).fetch1('clip_outside')) and not keep_all

        train_counts, train_ori = resp['train']['counts'], resp['train']['stimulus']
        valid_counts, valid_ori = resp['valid']['counts'], resp['valid']['stimulus']
        test_counts, test_ori = resp['test']['counts'], resp['test']['stimulus']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, valid_bins = binnify(valid_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, test_bins = binnify(test_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_ori = train_bins[good_pos]

        good_pos = valid_bins >= 0
        valid_counts = valid_counts[good_pos]
        valid_ori = valid_bins[good_pos]

        good_pos = test_bins >= 0
        test_counts = test_counts[good_pos]
        test_ori = test_bins[good_pos]

        train_x = torch.Tensor(train_counts)
        train_t = torch.Tensor(train_ori).type(torch.LongTensor)

        valid_x = Variable(torch.Tensor(valid_counts))
        valid_t = Variable(torch.Tensor(valid_ori).type(torch.LongTensor))

        test_x = Variable(torch.Tensor(test_counts))
        test_t = Variable(torch.Tensor(test_ori).type(torch.LongTensor))

        return train_x, train_t, valid_x, valid_t, test_x, test_t

    def make_objective(self, valid_x, valid_t, prior, delta):
        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            posterior = y + prior
            _, loc = torch.max(posterior, dim=1)
            v = (t.double() - loc.double()).pow(2).mean().sqrt() * delta
            return v.data.cpu().numpy()[0]

        return objective

    def train(self, net, loss, objective, train_dataset, prior, alpha, init_lr):
        learning_rates = init_lr * 3.0 ** (-np.arange(4))
        for lr in learning_rates:
            print('\n\n\n\n LEARNING RATE: {}'.format(lr))
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            for epoch, valid_score in early_stopping(net, objective, interval=20, start=100, patience=20,
                                                     max_iter=300000, maximize=False):
                data_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
                for x_, t_ in data_loader:
                    x, t = Variable(x_).cuda(), Variable(t_).cuda()
                    net.train()
                    optimizer.zero_grad()
                    y = net(x)
                    post = y + prior
                    val, _ = post.max(1, keepdim=True)
                    post = post - val
                    conv_filter = Variable(
                        torch.from_numpy(np.array([-0.25, 0.5, -0.25])[None, None, :]).type(y.data.type()))
                    try:
                        smoothness = nn.functional.conv1d(y.unsqueeze(1), conv_filter).pow(2).mean()
                    except:
                        # if smoothness computation overflows, then don't bother with it
                        smoothness = 0
                    score = loss(post, t)
                    score = score + alpha * smoothness
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))
                    # scheduler.step()

    def test_model(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        net = self.load_model(key)
        net.cuda()
        net.eval()

        objective = self.prepare_objective(key)
        train_x, train_t, valid_x, valid_t, test_x, test_t = self.get_dataset(key)
        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()
        test_x, test_t = test_x.cuda(), test_t.cuda()
        train_score = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        valid_score = objective(net, x=valid_x, t=valid_t)
        test_score = objective(net, x=test_x, t=test_t)

        return train_score, valid_score, test_score

    def prepare_objective(self, key):
        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        sigmaA = 3
        sigmaB = 15
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x, train_t, valid_x, valid_t, test_x, test_t = self.get_dataset(key)

        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()

        return self.make_objective(valid_x, valid_t, prior, delta)

    def make(self, key):
        # train_counts, train_ori, valid_counts, valid_ori = self.get_dataset(key)

        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        sigmaA = 3
        sigmaB = 15
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x, train_t, valid_x, valid_t, test_x, test_t = self.get_dataset(key)

        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()
        test_x, test_t = test_x.cuda(), test_t.cuda()

        train_dataset = TensorDataset(train_x, train_t)
        # valid_dataset = TensorDataset(valid_x, valid_t)

        objective = self.make_objective(valid_x, valid_t, prior, delta)

        init_lr = float((TrainParam() & key).fetch1('learning_rate'))
        alpha = float((TrainParam() & key).fetch1('smoothness'))
        init_std = float((TrainParam() & key).fetch1('init_std'))
        dropout = float((TrainParam() & key).fetch1('dropout'))
        h1, h2 = [int(x) for x in (ModelDesign() & key).fetch1('hidden1', 'hidden2')]
        seed = key['train_seed']

        net = Net(n_output=nbins, n_hidden=[h1, h2], std=init_std, dropout=dropout)
        net.cuda()
        loss = nn.CrossEntropyLoss().cuda()

        net.std = init_std
        set_seed(seed)
        net.initialize()

        self.train(net, loss, objective, train_dataset, prior, alpha, init_lr)

        print('Evaluating...')
        net.eval()

        key['cnn_train_score'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['cnn_valid_score'] = objective(net, x=valid_x, t=valid_t)
        key['cnn_test_score'] = objective(net, x=test_x, t=test_t)

        y = net(test_x)
        yd = y.data.cpu().numpy()
        yd = np.exp(yd)
        yd = yd / yd.sum(axis=1, keepdims=True)

        loc = yd.argmax(axis=1)
        ds = (np.arange(nbins) - loc[:, None]) ** 2
        avg_sigma = np.mean(np.sqrt(np.sum(yd * ds, axis=1))) * delta
        if np.isnan(avg_sigma):
            avg_sigma = -1

        key['avg_sigma'] = avg_sigma
        key['model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

        self.insert1(key)


@schema
class TrainedNetKL(dj.Computed):
    definition = """
    -> GaussianTrainedModel
    ---
    cnn_train_med_kl: float  # med KL
    cnn_valid_med_kl: float  # med KL
    cnn_test_med_kl: float  # med KL
    cnn_train_kl: longblob   # KL values
    cnn_valid_kl: longblob   # KL values
    cnn_test_kl: longblob   # KL values
    """

    def make(self, key):
        gt_resp = (GaussianSimulation() & key).simulate_responses()
        rel = (GaussianTrainedModel() & key)
        model = rel.load_model()
        train_x, train_t, valid_x, valid_t, test_x, test_t = rel.get_dataset(keep_all=True)
        train_x, train_t = Variable(train_x), Variable(train_t)

        model.cuda()
        model.eval()

        delta, nbins, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        delta = float(delta)
        pv = (np.arange(nbins) - nbins // 2) * delta

        set_types = ['train', 'valid', 'test']
        set_x = [train_x, valid_x, test_x]

        for st, x in zip(set_types, set_x):
            gt_logl = gt_resp[st]['logl_f'](pv)
            gt_nl = np.exp(gt_logl)
            gt_nl = gt_nl / np.sum(gt_nl, axis=1, keepdims=True)

            cnn_logl = model(x.cuda()).data.cpu().numpy()
            cnn_nl = np.exp(cnn_logl)
            cnn_nl = cnn_nl / np.sum(cnn_nl, axis=1, keepdims=True)

            eps = 1e-15
            KL = ((np.log(gt_nl + eps) - np.log(cnn_nl + eps)) * gt_nl).sum(axis=1)
            key['cnn_{}_med_kl'.format(st)] = np.median(KL)
            key['cnn_{}_kl'.format(st)] = KL

        self.insert1(key)