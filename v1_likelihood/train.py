import datajoint as dj
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from matplotlib import pyplot as plt
from attorch.train import early_stopping
from torch.utils.data import TensorDataset, DataLoader
from numpy.linalg import inv
from .models import Net
from .utils import list_hash, set_seed
from itertools import chain, product, count
from tqdm import tqdm

from .cd_dataset import CleanContrastSessionDataSet

schema = dj.schema('edgar_cd_ml', locals())


def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def binnify(x, center=270, delta=1, nbins=61, clip=True):
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
        p[p >= nbins] = nbins -1

    xv = (np.arange(nbins) - nbins//2) * delta + center
    return xv, p



@schema
class CVSeed(dj.Lookup):
    definition = """
    cv_seed: int    # seed for the cv set
    """
    contents = zip((35,))


@schema
class CVConfig(dj.Lookup):
    definition = """
    cv_config_id: varchar(128)  # id for config
    ---
    cv_fraction: float   # fraction
    """
    contents = [
        (list_hash(x),) + x for x in [
            (0.8,)
        ]
    ]


@schema
class CVSet(dj.Computed):
    """
    Separates the CV set into the training and the test sets.
    """
    definition = """
    -> CleanContrastSessionDataSet
    -> CVSeed
    -> CVConfig
    ---
    train_index: longblob    # training indices
    test_index:  longblob    # testing indices
    """

    def _make_tuples(self, key):
        print('Working on ', key)
        seed = key['cv_seed']
        np.random.seed(seed)
        fraction = float((CVConfig() & key).fetch1('cv_fraction'))
        dataset = (CleanContrastSessionDataSet() & key).fetch_dataset()
        N = len(dataset)
        pos = np.arange(N)
        split = round(N * fraction)
        np.random.shuffle(pos)
        key['train_index'] = pos[:split]
        key['test_index'] = pos[split:]
        self.insert1(key)

    def fetch_datasets(self):
        assert len(self) == 1, 'Only can fetch one dataset at a time'
        dataset = (CleanContrastSessionDataSet() & self).fetch_dataset()
        train_index, test_index = self.fetch1('train_index', 'test_index')
        train_set = dataset[train_index]
        test_set = dataset[test_index]
        return train_set, test_set


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
            (1.0, 61, True),
            (1.0, 81, False)
        ]
    ]


def mse(y, t):
    return np.sqrt(np.mean((y - t)**2))



def kernel(x, y, sigma):
    return torch.exp(-(x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1) / 2 / sigma**2)


class KernelReg:
    def __init__(self, lam=0.01, sigma=300):
        self.sigma = sigma
        self.lam = lam
        self.alpha = None
        self.mu = None
        self.x_train = None

    def fit(self, x, y):
        self.mu = x.mean(0, keepdim=True)
        xc = x - self.mu
        self.x_train = xc
        K = kernel(xc, xc, self.sigma)
        reg = torch.eye(K.size(0)) * self.lam
        reg = reg.type(K.type())
        self.alpha = torch.inverse(K + reg) @ y

    def __call__(self, x):
        return kernel(x - self.mu, self.x_train, self.sigma) @ self.alpha

    def rmse(self, x, y):
        yhat = self(x)
        return np.sqrt((yhat - y).pow(2).mean())

@schema
class KernelRegression(dj.Computed):
    definition = """
    -> CVSet
    ---
    kr_lambda: float         # optimal lambda
    kr_sigma: float          # optimal sigma
    kr_alpha: longblob       # learned kernel weight
    kr_mu: longblob          # mean of training samples
    kr_trainset_score: float # score on trainset
    kr_testset_score: float  # score on testset
    """
    def _make_tuples(self, key):
        train_set, test_set = (CVSet() & key).fetch_datasets()
        train_counts, train_ori = np.concatenate(train_set['counts'], 1).T, train_set['orientation']-270
        test_counts, test_ori = np.concatenate(test_set['counts'], 1).T, test_set['orientation']-270

        train_counts = torch.Tensor(train_counts).cuda()
        train_ori = torch.Tensor(train_ori).cuda()
        test_counts = torch.Tensor(test_counts).cuda()
        test_ori = torch.Tensor(test_ori).cuda()

        kr = KernelReg()
        lams = 10.0 ** np.arange(-3, 4)
        sigmas = np.logspace(-1, 3, 300)
        rmse = np.empty((len(lams), len(sigmas)))
        for (i, l), (j, s) in tqdm(product(enumerate(lams), enumerate(sigmas))):
            kr.sigma = s
            kr.lam = l
            kr.fit(train_counts, train_ori)
            rmse[i, j] = kr.rmse(test_counts, test_ori)

        lmin_pos, smin_pos = np.where(rmse == rmse.min())
        lam = lams[lmin_pos[0]]
        sigma = sigmas[smin_pos[0]]

        kr.lam = lam
        kr.sigma = sigma
        kr.fit(train_counts, train_ori)
        train_score = kr.rmse(train_counts, train_ori)
        test_score = kr.rmse(test_counts, test_ori)

        key['kr_lambda'] = lam
        key['kr_sigma'] = sigma
        key['kr_alpha'] = kr.alpha.cpu().numpy()
        key['kr_mu'] = kr.mu.cpu().numpy()
        key['kr_trainset_score'] = train_score
        key['kr_testset_score'] = test_score

        self.insert1(key)


@schema
class LinearRegression(dj.Computed):
    definition = """
    -> BinConfig
    -> CVSet
    ---
    lr_weights : longblob        # learned weights
    lr_trainset_score:  float    # score on trainset 
    lr_testset_score:   float    # score on testset
    """

    def _make_tuples(self, key):
        train_set, test_set = (CVSet() & key).fetch_datasets()
        bin_width, bin_counts, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        bin_width = float(bin_width)

        train_counts, train_ori = np.concatenate(train_set['counts'], 1).T, train_set['orientation']
        test_counts, test_ori = np.concatenate(test_set['counts'], 1).T, test_set['orientation']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, test_bins = binnify(test_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_bins = train_bins[good_pos]

        good_pos = test_bins >= 0
        test_counts = test_counts[good_pos]
        test_bins = test_bins[good_pos]

        tc = extend_ones(train_counts)
        w = inv(tc.T @ tc + np.diag(np.ones(tc.shape[1]) * 0.0001)) @ tc.T @ train_bins

        t_hat_train = tc @ w
        t_hat_test = extend_ones(test_counts) @ w

        train_score = mse(t_hat_train, train_bins) * bin_width
        test_score = mse(t_hat_test, test_bins) * bin_width

        key['lr_weights'] = w
        key['lr_trainset_score'] = train_score
        key['lr_testset_score'] = test_score

        self.insert1(key)

@schema
class TrainSeed(dj.Lookup):
    definition = """
    # training seed
    train_seed:   int       # training seed
    """
    contents = zip((8, 92, 123))

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
    contents = [(list_hash(x), ) + x for x in product(
        (0.01, 0.03, 0.6),     # learning rate
        (0.5, 0.9, 0.99),      # dropout rate
        (0.01, 0.001, 0.0001),    # initialization std
        (3, 30, 300, 3000, 30000)  # smoothness
    )]

@schema
class CVTrainedModel(dj.Computed):
    definition = """
    -> CVSet
    -> BinConfig
    -> ModelDesign
    -> TrainParam
    -> TrainSeed
    ---
    cnn_train_score: float   # score on train set
    cnn_valid_score:  float   # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob      # trained model
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

    def get_dataset(self, key=None):
        if key is None:
            key = self.fetch1(dj.key)

        train_set, valid_set = (CVSet() & key).fetch_datasets()
        bin_width = float((BinConfig() & key).fetch1('bin_width'))
        bin_counts = int((BinConfig() & key).fetch1('bin_counts'))
        clip_outside = bool((BinConfig() & key).fetch1('clip_outside'))

        train_counts, train_ori = np.concatenate(train_set['counts'], 1).T, train_set['orientation']
        valid_counts, valid_ori = np.concatenate(valid_set['counts'], 1).T, valid_set['orientation']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, valid_bins = binnify(valid_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_ori = train_bins[good_pos]

        good_pos = valid_bins >= 0
        valid_counts = valid_counts[good_pos]
        valid_ori = valid_bins[good_pos]

        train_x = torch.Tensor(train_counts)
        train_t = torch.Tensor(train_ori).type(torch.LongTensor)

        valid_x = Variable(torch.Tensor(valid_counts))
        valid_t = Variable(torch.Tensor(valid_ori).type(torch.LongTensor))

        return train_x, train_t, valid_x, valid_t

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
                    smoothness = nn.functional.conv1d(y.unsqueeze(1), conv_filter).pow(2).mean()
                    score = loss(post, t)
                    score = score + alpha * smoothness
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))
                    # scheduler.step()

    def _make_tuples(self, key):
        #train_counts, train_ori, valid_counts, valid_ori = self.get_dataset(key)

        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        sigmaA = 3
        sigmaB = 15
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x, train_t, valid_x, valid_t = self.get_dataset(key)

        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()

        train_dataset = TensorDataset(train_x, train_t)
        #valid_dataset = TensorDataset(valid_x, valid_t)

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

        y = net(valid_x)
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



def mean_post(lp):
    nbins = lp.size(1)
    v = lp - lp.max(dim=1, keepdim=True)[0]
    post = torch.exp(v)
    ro_pos = Variable(torch.arange(nbins).type(post.data.type()))
    return (ro_pos*post).sum(dim=1) / post.sum(dim=1)


def stat_logp(lp):
    nbins = lp.size(1)
    v = lp - lp.max(dim=1, keepdim=True)[0]
    post = torch.exp(v)
    ro_pos = Variable(torch.arange(nbins).type(post.data.type()))
    mu = (ro_pos*post).sum(dim=1, keepdim=True) / post.sum(dim=1, keepdim=True)
    sigma = torch.sqrt(((ro_pos - mu).pow(2)*post).sum(dim=1, keepdim=True) / post.sum(dim=1, keepdim=True))
    return mu, sigma


@schema
class CVTrainedModel2(dj.Computed):
    """
    Same thing as CVTrainedModel but using the mean of posterior instead of the maximum a posteriori
    when assessing the score.
    """
    definition = """
    -> CVSet
    -> BinConfig
    -> ModelDesign
    -> TrainParam
    -> TrainSeed
    ---
    cnn_train_score: float   # score on train set
    cnn_valid_score:  float   # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob      # trained model
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

    def get_dataset(self, key=None):
        if key is None:
            key = self.fetch1(dj.key)

        train_set, valid_set = (CVSet() & key).fetch_datasets()
        bin_width = float((BinConfig() & key).fetch1('bin_width'))
        bin_counts = int((BinConfig() & key).fetch1('bin_counts'))
        clip_outside = bool((BinConfig() & key).fetch1('clip_outside'))

        train_counts, train_ori = np.concatenate(train_set['counts'], 1).T, train_set['orientation']
        valid_counts, valid_ori = np.concatenate(valid_set['counts'], 1).T, valid_set['orientation']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, valid_bins = binnify(valid_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_ori = train_bins[good_pos]

        good_pos = valid_bins >= 0
        valid_counts = valid_counts[good_pos]
        valid_ori = valid_bins[good_pos]

        train_x = torch.Tensor(train_counts)
        train_t = torch.Tensor(train_ori).type(torch.LongTensor)

        valid_x = Variable(torch.Tensor(valid_counts))
        valid_t = Variable(torch.Tensor(valid_ori).type(torch.LongTensor))

        return train_x, train_t, valid_x, valid_t


    def _make_tuples(self, key):
        #train_counts, train_ori, valid_counts, valid_ori = self.get_dataset(key)

        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        sigmaA = 3
        sigmaB = 15
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x, train_t, valid_x, valid_t = self.get_dataset(key)

        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()

        train_dataset = TensorDataset(train_x, train_t)
        #valid_dataset = TensorDataset(valid_x, valid_t)

        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            posterior = y + prior
            # _, loc = torch.max(posterior, dim=1)
            loc, _ = stat_logp(posterior)
            v = (t.double() - loc.double()).pow(2).mean().sqrt() * delta
            return v.data.cpu().numpy()[0]

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
                    smoothness = nn.functional.conv1d(y.unsqueeze(1), conv_filter).pow(2).mean()
                    score = loss(post, t)
                    score = score + alpha * smoothness
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))

        print('Evaluating...')
        net.eval()

        key['cnn_train_score'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['cnn_valid_score'] = objective(net, x=valid_x, t=valid_t)

        y = net(valid_x)
        _, sigmas = stat_logp(y)
        avg_sigma = sigmas.data.cpu().numpy().mean() * delta
        if np.isnan(avg_sigma):
          avg_sigma = -1

        key['avg_sigma'] = avg_sigma
        key['model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

        self.insert1(key)

@schema
class CVTrainedModel3(CVTrainedModel):
    """
    Same thing as CVTrainedModel but using the mean of posterior instead of the maximum a posteriori
    when assessing the score.
    """
    def make_objective(self, valid_x, valid_t, prior, delta):
        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            posterior = y + prior
            _, loc = torch.max(posterior, dim=1)
            loc, _ = stat_logp(posterior)
            v = (t.double() - loc.double()).pow(2).mean().sqrt() * delta
            return v.data.cpu().numpy()[0]
        return objective

    def _make_tuples(self, key):
        #train_counts, train_ori, valid_counts, valid_ori = self.get_dataset(key)

        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        sigmaA = 3
        sigmaB = 15
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x, train_t, valid_x, valid_t = self.get_dataset(key)

        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()

        train_dataset = TensorDataset(train_x, train_t)
        #valid_dataset = TensorDataset(valid_x, valid_t)

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

        y = net(valid_x)
        _, sigmas = stat_logp(y)
        avg_sigma = sigmas.data.cpu().numpy().mean() * delta
        if np.isnan(avg_sigma):
            avg_sigma = -1

        key['avg_sigma'] = avg_sigma
        key['model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

        self.insert1(key)


@schema
class BestModel(dj.Computed):
    definition = """
    -> CVTrainedModel
    ---
    cnn_train_score: float   # score on train set
    cnn_valid_score:  float   # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob      # trained model
    """

    @property
    def key_source(self):
        return CVSet()

    def _make_tuples(self, key):
        targets = CVTrainedModel() * ModelDesign() & key
        best = targets & CVSet().aggr(targets, cnn_valid_score='min(cnn_valid_score)')

        # if duplicate score happens to occur, pick the model with the largest hidden layer
        best_model = best.fetch(dj.key, order_by='hidden1 DESC')[0]

        self.insert(CVTrainedModel() & best_model)


# saving state dict {k: v.cpu().numpy() for k, v in model.state_dict().items()})
# loading: state_dict = (self & key).fetch1('model')
# state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}
