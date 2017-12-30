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

from .cd_dataset import CleanContrastSessionDataSet

schema = dj.schema('edgar_cd_ml', locals())


def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)


def binnify(x, center=270, delta=1, nbins=61, clip=True):
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
class TrainSeed(dj.Lookup):
    definition = """
    # training seed
    train_seed:   int       # training seed
    """
    contents = zip((8, 92, 123))


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
        assert len(self)==1, 'Only can fetch one dataset at a time'
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
            (1.0, 61, True)
        ]
    ]


def mse(y, t):
    return np.sqrt(np.mean((y - t)**2))


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
class ModelDesign(dj.Lookup):
    definition = """
    model_id: varchar(128)   # model id
    ---
    hidden1:  int      # size of first hidden layer
    hidden2:  int      # size of second hidden layer
    """
    contents = [(list_hash(x),) + x for x in [
        (400, 400),
        (600, 600)
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
        (0.03,),     # learning rate
        (0.5,),      # dropout rate
        (0.0001,),         # initialization std
        (0.3, 30)  # smoothness
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
    cnn_trainset_score: float   # score on train set
    cnn_testset_score:  float   # score on test set
    avg_sigma:   float   # average width of the likelihood functions
    model: longblob      # trained model
    """

    def _make_tuples(self, key):
        print('Working!')

        train_set, test_set = (CVSet() & key).fetch_datasets()
        bin_width, bin_counts, clip_outside = (BinConfig() & key).fetch1('bin_width', 'bin_counts', 'clip_outside')
        bin_counts = int(bin_counts)
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

        sigmaA = 3
        sigmaB = 15

        prior = np.log(np.exp(- xv ** 2 / 2 / sigmaA ** 2) / sigmaA + np.exp(- xv ** 2 / 2 / sigmaB ** 2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        train_x = torch.Tensor(train_counts)
        train_t = torch.Tensor(train_bins).type(torch.LongTensor)

        test_x = Variable(torch.Tensor(test_counts)).cuda()
        test_t = Variable(torch.Tensor(test_bins).type(torch.LongTensor)).cuda()

        train_dataset = TensorDataset(train_x, train_t)
        test_dataset = TensorDataset(test_x, test_t)

        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = test_x
                t = test_t
            net.eval()
            y = net(x)
            posterior = y + prior
            _, loc = torch.max(posterior, dim=1)
            v = (t.double() - loc.double()).pow(2).mean().sqrt() * bin_width
            return v.data.cpu().numpy()[0]

        ## setup model
        h1, h2 = (ModelDesign() & key).fetch1('hidden1', 'hidden2')
        h1, h2 = int(h1), int(h2)
        lr, dropout, init_std, alpha = (TrainParam() & key).fetch1('learning_rate', 'dropout', 'init_std', 'smoothness')
        lr, dropout, init_std, alpha = float(lr), float(dropout), float(init_std), float(alpha)

        ## train the network
        set_seed(key['train_seed'])
        net = Net(n_output=bin_counts, n_hidden=[h1, h2], std=init_std, dropout=dropout)
        net.cuda()
        loss = nn.CrossEntropyLoss().cuda()

        learning_rates = lr * 3.0**(-np.arange(4))
        for lr in learning_rates:
            print('\n\n\n\n LEARNING RATE: {}'.format(lr))
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            for epoch, valid_score in early_stopping(net, objective, interval=20, start=100, patience=20, max_iter=300000, maximize=False):
                data_loader = DataLoader(train_dataset, shuffle=True, batch_size=128)
                for x_, t_ in data_loader:
                    x, t = Variable(x_).cuda(), Variable(t_).cuda()
                    net.train()
                    optimizer.zero_grad()
                    y = net(x)
                    post = y + prior
                    val, _ = post.max(1, keepdim=True)
                    post = post - val
                    #sparcity = y.abs().sum(1).mean()
                    conv_filter = Variable(torch.from_numpy(np.array([-0.25, 0.5, -0.25])[None, None, :]).type(y.data.type()))
                    smoothness = nn.functional.conv1d(y.unsqueeze(1), conv_filter).pow(2).mean()
                    score = loss(post, t)
                    score = score + alpha * smoothness
                    score.backward()
                    optimizer.step()
                # if epoch % 10 == 0:
                #     print('Score: {}'.format(score.data.cpu().numpy()[0]))


        net.eval()

        key['cnn_trainset_score'] = objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['cnn_testset_score'] = objective(net, x=test_x, t=test_t)

        y = net(test_x)
        yd = y.data.cpu().numpy()
        yd = np.exp(yd)
        yd = yd / yd.sum(axis=1, keepdims=True)

        loc = yd.argmax(axis=1)
        ds = (np.arange(bin_counts) - loc[:, None]) ** 2
        avg_sigma = np.mean(np.sqrt(np.sum(yd * ds, axis=1))) * bin_width
        if np.isnan(avg_sigma):
          avg_sigma = -1

        key['avg_sigma'] = avg_sigma
        key['model'] = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

        self.insert1(key)



# saving state dict {k: v.cpu().numpy() for k, v in model.state_dict().items()})
# loading: state_dict = (self & key).fetch1('model')
# state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}
