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
from .models import CombinedNet
from .utils import list_hash, set_seed
from itertools import chain, product, count
from tqdm import tqdm
from .analysis import cd_dlset
from . import train3

from .cd_dataset import CleanContrastSessionDataSet

schema = dj.schema('edgar_cd_mldecision')

dj.config['external-model'] = dict(
    protocol='file',
    location='/external/state_dicts/')


# Check for access to external by checking for precense of indicator file `pass.dat`
external_access = False

try:
    with open('/external/pass.dat', 'r'):
        external_access = True
except:
    pass


def best_model(model, extra=None, key=None, field='valid_loss'):
    if key is None:
        key = {}
    targets = model & key

    aggr_targets = CVSet * BinConfig
    if extra is not None:
        aggr_targets = aggr_targets * extra
    return targets * aggr_targets.aggr(targets, min_loss='min({})'.format(field)) & '{} = min_loss'.format(field)

def extend_ones(x):
    return np.concatenate([x, np.ones([x.shape[0], 1])], axis=1)






@schema
class ValSeed(dj.Lookup):
    definition = """
    val_seed: int    # seed for the cv set
    """
    contents = zip((35,))


@schema
class ValConfig(dj.Lookup):
    definition = """
    val_config_id: varchar(128)  # id for config
    ---
    val_fraction: float   # fraction
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
    -> cd_dlset.CVSet.Member
    -> ValSeed
    -> ValConfig
    ---
    train_index: longblob    # training indices
    valid_index: longblob    # validation indicies
    test_index:  longblob    # testing indices
    """

    def make(self, key):
        train_index, test_index = (cd_dlset.CVSet.Member & key).fetch1('train_indices', 'test_indices')

        # adjust from MATLAB indexing into Python indexing
        train_index, test_index = train_index.squeeze().astype('int') - 1, test_index.squeeze().astype('int') - 1
        seed = key['val_seed']
        np.random.seed(seed)
        fraction = float((ValConfig() & key).fetch1('val_fraction'))
        N = len(train_index)
        split = round(N * fraction)
        np.random.shuffle(train_index)
        key['train_index'] = train_index[:split]
        key['valid_index'] = train_index[split:]
        key['test_index'] = test_index
        self.insert1(key)

    def fetch_datasets(self):
        assert len(self) == 1, 'Only can fetch one dataset at a time'
        dataset = (CleanContrastSessionDataSet & self.proj(dataset_hash='dec_trainset_hash')).fetch_dataset()
        train_index, valid_index, test_index = self.fetch1('train_index', 'valid_index', 'test_index')
        return dataset, train_index, valid_index, test_index



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
    model_id: varchar(32)   # model id
    ---
    hidden1:  int      # size of first hidden layer
    hidden2:  int      # size of second hidden layer
    """
    contents = [(list_hash(x),) + x for x in [
        (800, 800),
        (1000, 1000)
    ]]


@schema
class TrainParam(dj.Lookup):
    definition = """
    param_id: varchar(128)    # ID of parameter
    ---
    optim_method: varchar(16)  # optimizer method
    learning_rate:  float     # initial learning rate
    dropout:       float     # dropout rate
    init_std:       float     # standard deviation for weight initialization
    l2_reg:         float     # regularizer on L2 of linear layer weights      
    """
    contents = [(list_hash(x), ) + x for x in product(
        ('SGD',),
        (0.01, 0.03),     # learning rate
        (0.2, 0.5),      # dropout rate
        (1e-4, 1e-3, 1e-2, 1e-1),    # initialization std
        (0, 0.5)        # l2_reg
    )]

@schema
class LikelihoodDecoder(dj.Lookup):
    definition = """
    decoder: varchar(32)     # type of likelihood decoder
    """
    contents = zip(['full', 'fixed'])

    def get_decoder(self, decoder, session_key):
        """
        Args:
            key: Must uniquely identify an entry for this table AND an entry in train3.CleanContrastSessionDataSet

        Returns:
            Likelihood decoder corresponding to the key
        """
        if decoder == 'full':
            nonlin_entry = train3.BestNonlin() & session_key & 'selection_objective="mse"'
            net = (train3.CVTrainedModel & nonlin_entry.proj()).load_model()
        elif decoder == 'fixed':
            fl_entry = train3.BestFixedLikelihood & session_key & 'selection_objective="mse"'
            net = (train3.CVTrainedFixedLikelihood & fl_entry.proj()).load_model()

        net.cuda()
        net.eval()
        return net



class BaseModel(dj.Computed):
    extra_deps = ""

    @property
    def definition(self):
        def_str = """
        -> CVSet
        -> LikelihoodDecoder
        -> ModelDesign
        -> TrainParam
        -> TrainSeed
        {}
        ---
        nbins:   int       # number of input likelihood bins
        train_logl: float    # avg logl on train set
        valid_logl:  float   # avg logl on validation set
        test_logl:   float   # avg logl on test set
        train_acc:   float   # accuracy on train set
        valid_acc:   float   # accuracy on validation set
        test_acc:    float   # accuracy on test set
        model_saved: bool   # whether model was saved
        model: external-model  # saved model
        """
        return def_str.format(self.extra_deps)


    def get_dataset(self, key=None):
        if key is None:
            key = self.fetch1('KEY')

        # find the corresponding entry in CleanContrastSessionDataSet
        member_key =dict(key)
        member_key['dataset_hash'] = member_key['dec_trainset_hash']
        session_key = (train3.CleanContrastSessionDataSet & member_key).fetch1('KEY')


        ds, train_pos, valid_pos, test_pos = (CVSet() & key).fetch_datasets()


        net = LikelihoodDecoder().get_decoder(key['decoder'], session_key)

        # extract counts and classification
        counts = np.concatenate(ds['counts'], 1).T
        class_id = (ds['selected_class'] == 'A').astype('int')  # A = 1 B = 0

        # compute the likelihood functions
        Ls = net(Variable(torch.Tensor(counts)).cuda()).data.cpu().numpy()

        train_Ls = Ls[train_pos]
        # train_counts = counts[train_pos]
        train_class = class_id[train_pos]

        valid_Ls = Ls[valid_pos]
        # valid_counts = counts[valid_pos]
        valid_class = class_id[valid_pos]

        test_Ls = Ls[test_pos]
        # test_counts = counts[test_pos]
        test_class = class_id[test_pos]

        train_x = torch.Tensor(train_Ls)
        train_t = torch.Tensor(train_class).type(torch.LongTensor)

        valid_x = Variable(torch.Tensor(valid_Ls)).cuda()
        valid_t = Variable(torch.Tensor(valid_class).type(torch.LongTensor)).cuda()

        test_x = Variable(torch.Tensor(test_Ls)).cuda()
        test_t = Variable(torch.Tensor(test_class).type(torch.LongTensor)).cuda()


        return train_x, train_t, valid_x, valid_t, test_x, test_t


    @staticmethod
    def make_accuracy_objective(valid_x, valid_t):
        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            _, loc = torch.max(y, dim=1)
            v = (t.double() - loc.double()).pow(2).mean()
            return v.data.cpu().numpy()[0]

        return objective

    @staticmethod
    def make_ce_objective(valid_x, valid_t):
        def objective(net, x=None, t=None):
            if x is None and t is None:
                x = valid_x
                t = valid_t
            net.eval()
            y = net(x)
            v = F.cross_entropy(y, t)
            return v.data.cpu().numpy()[0]

        return objective


    @staticmethod
    def train(net, loss, objective, train_dataset, beta, init_lr, optim_method):
        learning_rates = init_lr * 3.0 ** (-np.arange(4))
        optim_lut = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam
        }
        optim_choice = optim_lut[optim_method]
        for lr in learning_rates:
            print('\n\n\n\n LEARNING RATE: {}'.format(lr))
            optimizer = optim_choice(net.parameters(), lr=lr)
            for epoch, valid_score in early_stopping(net, objective, interval=20, start=100, patience=30,
                                                     max_iter=300000, maximize=False):
                data_loader = DataLoader(train_dataset, shuffle=True, batch_size=256)
                for x_, t_ in data_loader:
                    x, t = Variable(x_).cuda(), Variable(t_).cuda()
                    net.train()
                    optimizer.zero_grad()
                    y = net(x)
                    score = loss(y, t)
                    if beta > 0:
                        try:
                            l2 = net.l2_weights_per_layer()
                        except:
                            # if L2 weight overflows, skip it
                            l2 = 0
                        score = score + beta * l2
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))
                    # scheduler.step()



    def load_model(self, key=None):
        if key is None:
            key = {}

        key = (self & key).fetch1('KEY')

        nbins = (self & key).fetch1('nbins')

        net = self.prepare_model(key, nbins)

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

        train_x, train_t, valid_x, valid_t, test_x, test_t = self.get_dataset(key)
        ce_objective = self.make_ce_objective(valid_x, valid_t)
        acc_objective = self.make_accuracy_objective(valid_x, valid_t)

        train_dataset = TensorDataset(train_x, train_t)

        seed = key['train_seed']
        set_seed(seed)

        key['nbins'] = train_x.shape[1]


        net = self.prepare_model(key, nbins=key['nbins'])
        net.cuda()
        loss = nn.CrossEntropyLoss().cuda()

        optim_method = (TrainParam() & key).fetch1('optim_method')
        init_lr = float((TrainParam() & key).fetch1('learning_rate'))
        beta = float((TrainParam() & key).fetch1('l2_reg'))

        self.train(net, loss, ce_objective,
                   train_dataset=train_dataset, beta=beta, init_lr=init_lr,
                   optim_method=optim_method)

        print('Evaluating...')
        net.eval()

        key['train_logl'] = -ce_objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['valid_logl'] = -ce_objective(net, x=valid_x, t=valid_t)
        key['test_logl'] = -ce_objective(net, x=test_x, t=test_t)

        key['train_acc'] = 1 - acc_objective(net, x=Variable(train_x).cuda(), t=Variable(train_t).cuda())
        key['valid_acc'] = 1 - acc_objective(net, x=valid_x, t=valid_t)
        key['test_acc'] = 1 - acc_objective(net, x=test_x, t=test_t)


        # determine if the state should be saved
        key['model_saved'] = self.check_to_save(key, key['valid_logl'])

        if key['model_saved']:
            print('Better model achieved! Updating...')
            blob = {k: v.cpu().numpy() for k, v in net.state_dict().items()}
        else:
            blob = {}

        key['model'] = blob
        key['model_saved'] = int(key['model_saved'])

        self.insert1(key)



@schema
class CVTrainedLtoD(BaseModel):
    def prepare_model(self, key=None, nbins=91):
        if key is None:
            key = self.fetch1('KEY')

        init_std = float((TrainParam() & key).fetch1('init_std'))
        dropout = float((TrainParam() & key).fetch1('dropout'))
        h1, h2 = [int(x) for x in (ModelDesign() & key).fetch1('hidden1', 'hidden2')]

        net = CombinedNet(n_channel=nbins, n_output=2, n_hidden=[h1, h2], std=init_std, dropout=dropout, nonlin='relu')
        return net

    def check_to_save(self, key, valid_ce):
        scores_ce = (self & (CVSet * LikelihoodDecoder & key)).fetch('valid_logl')
        return int(len(scores_ce) == 0 or key['valid_logl'] > scores_ce.max())


