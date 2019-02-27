import datajoint as dj
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from attorch.train import early_stopping
from torch.utils.data import TensorDataset, DataLoader
from numpy.linalg import inv
from .models import CombinedNet, FixedLikelihoodNet
from .utils import list_hash, set_seed
from itertools import chain, product, count
from tqdm import tqdm

from .cd_dataset import CleanContrastSessionDataSet

schema = dj.schema('edgar_cd_ml2')

dj.config['external-model'] = dict(
    protocol='file',
    location='/external/state_dicts/')


# Check for access to external!
external_access = False
with open('/external/pass.dat', 'r'):
    external_access = True


def best_model(model, extra=None, key=None):
    if key is None:
        key = {}
    targets = model & key

    aggr_targets = CVSet * BinConfig * EvalObjective
    if extra is not None:
        aggr_targets = aggr_targets * extra
    return targets * aggr_targets.aggr(targets, min_loss='min(valid_loss)') & 'valid_loss = min_loss'

def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def binnify(x, center=270, delta=1, nbins=91, clip=True):
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

    def make(self, key):
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
            (1.0, 91, True)
        ]
    ]


def mse(y, t):
    return np.sqrt(np.mean((y - t)**2))


def kernel(x, y, sigma):
    return torch.exp(-(x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1) / 2 / sigma**2)


@schema
class TrainSeed(dj.Lookup):
    definition = """
    # training seed
    train_seed:   int       # training seed
    """
    contents = zip((8, 92, 123))


@schema
class NonLinearity(dj.Lookup):
    definition = """
    nonlin: varchar(16)  # type of nonlinearity
    """
    contents = zip(['none', 'relu'])

@schema
class EvalObjective(dj.Lookup):
    definition = """
    objective: varchar(16)  # type of objective
    """
    contents = zip(['ce'])


@schema
class ModelDesign(dj.Lookup):
    definition = """
    model_id: varchar(32)   # model id
    ---
    hidden1:  int      # size of first hidden layer
    hidden2:  int      # size of second hidden layer
    """
    contents = [(list_hash(x),) + x for x in [
        (0, 0),
        (1000, 0),
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
    l2_reg:         float     # regularizer on L2 of linear layer weights      
    """
    contents = [(list_hash(x), ) + x for x in product(
        (1e-4, 1e-3, 1e-2),     # learning rate
        (0.2, 0.5),      # dropout rate
        (1e-4, 1e-3),    # initialization std
        (3, 30, 300),  # smoothness
        (0.5, 1)        # l2_reg
    )]


class BaseModel(dj.Computed):
    extra_deps = ""

    @property
    def definition(self):
        def_str = """
        -> CVSet
        -> BinConfig
        -> ModelDesign
        -> TrainParam
        -> TrainSeed
        -> EvalObjective
        {}
        ---
        train_loss: float   # loss on train set
        valid_loss:  float   # loss on test set
        train_ce:    float   # ce on train set
        valid_ce:    float   # ce on test set
        train_mse:   float   # mse on train set
        valid_mse:   float   # mse on test set
        avg_sigma:   float   # average width of the likelihood functions
        model_saved: bool   # whether model was saved
        model: external-model  # saved model
        """
        return def_str.format(self.extra_deps)

    def get_dataset(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        train_set, valid_set = (CVSet() & key).fetch_datasets()
        bin_width = float((BinConfig() & key).fetch1('bin_width'))
        bin_counts = int((BinConfig() & key).fetch1('bin_counts'))
        clip_outside = bool((BinConfig() & key).fetch1('clip_outside'))

        train_counts, train_ori = np.concatenate(train_set['counts'], 1).T, train_set['orientation']
        valid_counts, valid_ori = np.concatenate(valid_set['counts'], 1).T, valid_set['orientation']

        xv, train_bins = binnify(train_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)
        _, valid_bins = binnify(valid_ori, delta=bin_width, nbins=bin_counts, clip=clip_outside)

        # remove bins falling outside of valid range
        good_pos = train_bins >= 0
        train_counts = train_counts[good_pos]
        train_ori = train_bins[good_pos]

        # remove bins falling outside of valid range
        good_pos = valid_bins >= 0
        valid_counts = valid_counts[good_pos]
        valid_ori = valid_bins[good_pos]

        train_x = torch.Tensor(train_counts)
        train_t = torch.Tensor(train_ori).type(torch.LongTensor)

        # wrap validation dataset in variable already
        valid_x = Variable(torch.Tensor(valid_counts))
        valid_t = Variable(torch.Tensor(valid_ori).type(torch.LongTensor))

        return train_x, train_t, valid_x, valid_t

    def make_objective(self, valid_x, valid_t, prior, delta, obj_type='ce'):
        def objective(net, x=None, t=None, obj=None):
            if obj is None:
                obj = obj_type
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            posterior = y + prior
            if obj == 'ce':
                v = F.cross_entropy(posterior, t)
            elif obj == 'mse':
                _, loc = torch.max(posterior, dim=1)
                v = (t.double() - loc.double()).pow(2).mean().sqrt() * delta
            return v.data.cpu().numpy()[0]

        return objective

    @staticmethod
    def make_prior(nbins, delta, sigmaA=3, sigmaB=15):
        pv = (np.arange(nbins) - nbins // 2) * delta
        prior = np.log(np.exp(- pv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- pv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        return Variable(torch.from_numpy(prior)).cuda().float()

    def prepare_parts(self, key):
        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))
        obj_type = key['objective']

        prior = self.make_prior(nbins, delta)
        train_x, train_t, valid_x, valid_t = self.get_dataset(key)
        valid_x, valid_t = valid_x.cuda(), valid_t.cuda()

        return train_x, train_t, valid_x, valid_t, prior, self.make_objective(valid_x, valid_t, prior, delta, obj_type)

    @staticmethod
    def train(net, loss, objective, train_dataset, prior, alpha, beta, init_lr):
        learning_rates = init_lr * 3.0 ** (-np.arange(4))
        for lr in learning_rates:
            print('\n\n\n\n LEARNING RATE: {}'.format(lr))
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
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
                    score = score + alpha * smoothness + beta * net.l2_weights()
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))
                    # scheduler.step()



    def load_model(self, key=None):
        if key is None:
            key = {}

        key = (self & key).fetch1('KEY')

        net = self.prepare_model(key)

        rel = self & key
        state_dict = rel.fetch1('model')
        state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}

        net.load_state_dict(state_dict)
        return net

    def test_model(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        net = self.load_model(key)
        net.cuda()
        net.eval()

        train_x, train_t, valid_x, valid_t, prior, objective = self.prepare_parts(key)
        train_score = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        valid_score = objective(net, x=valid_x, t=valid_t)
        return train_score, valid_score

    def check_to_save(self, key, valid_loss):
        raise NotImplementedError


    def make(self, key):
        if not external_access:
            raise ValueError('No access to external! Will not be able to save model!')

        delta = float((BinConfig() & key).fetch1('bin_width'))
        nbins = int((BinConfig() & key).fetch1('bin_counts'))

        train_x, train_t, valid_x, valid_t, prior, objective = self.prepare_parts(key)
        train_dataset = TensorDataset(train_x, train_t)

        seed = key['train_seed']
        set_seed(seed)

        net = self.prepare_model(key)
        net.cuda()
        loss = nn.CrossEntropyLoss().cuda()

        init_lr = float((TrainParam() & key).fetch1('learning_rate'))
        alpha = float((TrainParam() & key).fetch1('smoothness'))
        beta = float((TrainParam() & key).fetch1('l2_reg'))
        self.train(net, loss, objective,
                   train_dataset=train_dataset, prior=prior, alpha=alpha, beta=beta, init_lr=init_lr)

        print('Evaluating...')
        net.eval()

        key['train_loss'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['valid_loss'] = objective(net, x=valid_x, t=valid_t)

        key['train_ce'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda(), obj='ce')
        key['valid_ce'] = objective(net, x=valid_x, t=valid_t, obj='ce')

        key['train_mse'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda(), obj='mse')
        key['valid_mse'] = objective(net, x=valid_x, t=valid_t, obj='mse')

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

        # determine if the state should be saved
        key['model_saved'] = self.check_to_save(key, key['valid_loss'])

        if key['model_saved']:
            print('Better model achieved! Updating...')
            blob = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        else:
            blob = {}

        key['model'] = blob
        key['model_saved'] = int(key['model_saved'])

        self.insert1(key)


@schema
class CVTrainedModel(BaseModel):
    extra_deps = """
    -> NonLinearity
    """
    def prepare_model(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        nonlin = key['nonlin']
        nbins = int((BinConfig() & key).fetch1('bin_counts'))
        init_std = float((TrainParam() & key).fetch1('init_std'))
        dropout = float((TrainParam() & key).fetch1('dropout'))
        h1, h2 = [int(x) for x in (ModelDesign() & key).fetch1('hidden1', 'hidden2')]

        net = CombinedNet(n_output=nbins, n_hidden=[h1, h2], std=init_std, dropout=dropout, nonlin=nonlin)
        return net

    def check_to_save(self, key, valid_loss):
        scores = (self & (CVSet * BinConfig * EvalObjective * NonLinearity & key)).fetch('valid_loss')
        return int(len(scores) == 0 or valid_loss < scores.min())



@schema
class FixedLikelihoodTrainParam(dj.Lookup):
    definition = """
    fl_param_id: varchar(32)    # ID of parameter
    ---
    sigma_init:      float   # standard deviation for initializing Gaussian likelihood
    """
    contents = [(list_hash(x), ) + x for x in product(
        (3, 5),  # sigma_init
    )]


@schema
class CVTrainedFixedLikelihood(BaseModel):
    extra_deps = """
    -> FixedLikelihoodTrainParam
    """
    def prepare_model(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        nbins = int((BinConfig() & key).fetch1('bin_counts'))
        init_std = float((TrainParam() & key).fetch1('init_std'))
        sigma_init = float((FixedLikelihoodTrainParam() & key).fetch1('sigma_init'))
        dropout = float((TrainParam() & key).fetch1('dropout'))
        h1, h2 = [int(x) for x in (ModelDesign() & key).fetch1('hidden1', 'hidden2')]

        net = FixedLikelihoodNet(n_output=nbins, n_hidden=[h1, h2], std=init_std, dropout=dropout, sigma_init=sigma_init)
        return net

    def check_to_save(self, key, valid_loss):
        scores = (self & (CVSet * BinConfig * EvalObjective & key)).fetch('valid_loss')
        return int(len(scores) == 0 or valid_loss < scores.min())


#### Aggregator tables
@schema
class BestPoissonLike(dj.Computed):
    definition = """
    -> CVTrainedModel
    ---
    train_loss:  float   # score on train set
    valid_loss:  float   # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob      # saved model state
    """

    @property
    def key_source(self):
        return CVSet() * BinConfig() * EvalObjective() & (CVTrainedModel & 'nonlin="none"')

    def make(self, key):
        best = best_model(CVTrainedModel & 'nonlin="none"' & 'model_saved = True', key=key) * ModelDesign
        # if duplicate score happens to occur, pick the model with the largest hidden layer
        selected = best.fetch('KEY', order_by='hidden1 DESC')[0]
        data = (CVTrainedModel & selected).fetch1()
        assert data['model_saved'], 'Model was not saved despite being the best!!'
        data['model'] = {k: data['model'][k][0] for k in data['model'].dtype.fields}
        self.insert1(data, ignore_extra_fields=True)