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


cd_dataset = dj.create_virtual_module('cd_dataset', 'edgar_cd_dataset')
class_discrimination = dj.create_virtual_module('class_discrimination', 'edgar_class_discrimination')

schema = dj.schema('edgar_cd_ml', locals())


@schema
class TrainSeed(dj.Lookup):
    definition = """
    # training seed
    train_seed:   int       # training seed
    """
    contents = zip((8, 92, 123))


#@schema
class CVSeed(dj.Lookup):
    definition = """
    cv_seed: int    # seed for the cv set
    """
    contents = zip((35,))


# @schema
class CVConfig(dj.Computed):
    pass

#@schema
class CVSet(dj.Computed):
    definition = """
    -> datasource
    -> CVSeed
    -> CVConfig
    """

    class DataSet(dj.Part):
        definition = """
        -> master
        -> cv_set_id:   int   # set id
        ---
        train_index: longblob    # training indices
        test_index:  longblob    # testing indices
        """

#@schema
class LinearRegression(dj.Computed):
    definition = """
    -> CVSet.DataSet
    ---
    lr_trainset_score:  float    # score on trainset 
    lr_testset_score:   float    # score on testset
    """

    def _make_tuples(self, key):
        pass


# @schema
class ModelDesign(dj.Lookup):
    definition = """
    model_id: varchar(128)   # model id
    ---
    hidden1:  int      # size of first hidden layer
    hidden2:  int      # size of second hidden layer
    """
    contents = [(list_hash(x),) + x for x in [
        (400, 400),
        (500, 500),
        (600, 600)
    ]]


# @schema
class TrainParam(dj.Lookup):
    definition = """
    param_id: varchar(128)    # ID of parameter
    ---
    learning_rate:  float     # initial learning rate
    droptout:       float     # dropout rate
    init_std:       float     # standard deviation for weight initialization
    smoothness:     float     # regularizer on Laplace smoothness
    """

#@schema
class CVTrainedModel(dj.Computed):
    definition = """
    -> CVSet.DataSet
    -> ModelDesign
    -> TrainParam
    -> TrainSeed
    ---
    
    """

    def _make_tuples(self, key):

        # Get all sessions for `subject_id=21` (Tom)

        sessions = (class_discrimination.SpikeCountSet() & 'subject_id = 21').fetch(dj.key)

        # Select a single session

        key = sessions[4]

        # Get the data

        all_data = (class_discrimination.ClassDiscriminationTrial() * class_discrimination.SpikeCountTrials()
                    & class_discrimination.CleanSpikeCountTrials() & key).fetch()

        # Select out contrast

        np.unique(all_data['contrast'])

        contrast = 0.02
        f = all_data['contrast'] == contrast

        data = all_data[f]

        counts = np.concatenate(data['counts'], 1).T
        ori = data['orientation']

        nbins = 51
        delta = 1
        p = np.round((ori - 270)/ delta) + (nbins//2)

        sigmaA = 3
        sigmaB = 15

        pos = (p >= 0) & (p < nbins)

        p = p[pos]
        counts = counts[pos]

        # split into train and validation set
        fraction = 0.8
        N = len(counts)
        pos = np.arange(N)
        split = round(N * fraction)
        np.random.shuffle(pos)
        train_pos = pos[:split]
        valid_pos = pos[split:]

        train_counts = counts[train_pos]
        train_ori = p[train_pos]

        valid_counts = counts[valid_pos]
        valid_ori = p[valid_pos]

        pv = (np.arange(nbins) - nbins//2) * delta
        prior = np.log(np.exp(- pv**2 / 2 / sigmaA**2) / sigmaA + np.exp(- pv**2 / 2 / sigmaB**2) / sigmaB)
        prior = Variable(torch.from_numpy(prior)).cuda().float()

        x = torch.Tensor(train_counts)
        t = torch.Tensor(train_ori).type(torch.LongTensor)

        valid_x = Variable(torch.Tensor(valid_counts)).cuda()
        valid_t = Variable(torch.Tensor(valid_ori).type(torch.LongTensor)).cuda()

        train_dataset = TensorDataset(x, t)
        valid_dataset = TensorDataset(valid_x, valid_t)

        def objective(net):
            net.eval()
            valid_y = net(valid_x)
            posterior = valid_y + prior
            _, loc = torch.max(posterior, dim=1)
            v =(valid_t.double() - loc.double()).pow(2).mean().sqrt() * delta
            return v.data.cpu().numpy()[0]
            #return loss(valid_y, valid_t).data.cpu().numpy()[0]


        net = Net(n_hidden=[400, 400], std=0.001, dropout=0.5)
        net.cuda()
        loss = nn.CrossEntropyLoss().cuda()

        net.eval()
        y = net(valid_x)

        y = y + prior
        val, idx = torch.max(y, 1)
        yd = y.data.cpu().numpy()

        plt.subplot(211)
        t_hat = idx.data.cpu().numpy()
        plt.scatter(t_hat, valid_ori)

        print(np.sqrt(np.mean((t_hat - valid_ori)**2)) * delta)

        plt.subplot(212)

        plt.plot(yd[47])

        net.std = 1e-2
        #net.initialize()
        alpha = 0.000
        beta = 3e-2 #7e-3 #1e-3




        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.3)
        best_score = None

        learning_rates = 0.03 * 3.0**(-np.arange(4))
        beta = 30
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
                    sparcity = y.abs().sum(1).mean()
                    conv_filter = Variable(torch.from_numpy(np.array([-0.25, 0.5, -0.25])[None, None, :]).type(y.data.type()))
                    smoothness = nn.functional.conv1d(y.unsqueeze(1), conv_filter).pow(2).mean()
                    score = loss(post, t)
                    score = score + alpha * sparcity + beta * smoothness
                    score.backward()
                    optimizer.step()
                if epoch % 10 == 0:
                    print('Score: {}'.format(score.data.cpu().numpy()[0]))
        #scheduler.step()


# saving state dict {k: v.cpu().numpy() for k, v in model.state_dict().items()})
# loading: state_dict = (self & key).fetch1('model')
# state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}