import datajoint as dj
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

cd_dataset = dj.create_virtual_module('cd_dataset', 'edgar_cd_dataset')
class_discrimination = dj.create_virtual_module('class_discrimination', 'edgar_class_discrimination')
cd_lc = dj.create_virtual_module('cd_lc', 'edgar_cd_lc')
cd_dlset = dj.create_virtual_module('cd_dlset', 'edgar_cd_dlset');
ephys = dj.create_virtual_module('ephys', 'ephys')


schema = dj.schema('edgar_cd_analysis')

@schema
class LikelihoodStats(dj.Computed):
    definition = """
    -> cd_dlset.DLSetInfo
    ---
    contrasts: longblob
    orientation: longblob
    mu_likelihood: longblob
    sigma_likelihood: longblob
    mean_sigma: float
    max_ori: longblob
    """

    def make(self, key):
        path = (cd_dlset.DLSetInfo & key).fetch1('dataset_path')
        data = loadmat(path)['dataSet'][0, 0]

        cont = data['contrast']
        s = data['decodeOri'].T
        L = data['likelihood']
        ori = data['orientation']

        mu_L = (s * L).sum(axis=0) / L.sum(axis=0)
        sigma_L = np.sqrt((s ** 2 * L).sum(axis=0) / L.sum(axis=0) - mu_L ** 2)

        max_pos = np.argmax(L, axis=0)
        max_ori = s[max_pos].squeeze()

        key['contrasts'] = cont.squeeze()
        key['orientation'] = ori.squeeze()
        key['mu_likelihood'] = mu_L.squeeze()
        key['max_ori'] = max_ori
        key['sigma_likelihood'] = sigma_L.squeeze()
        key['mean_sigma'] = sigma_L.mean()

        self.insert1(key)


@schema
class SummaryBinConfig(dj.Lookup):
    definition = """
    summary_bin_id: int
    ---
    bin_extent: float
    nbins: int
    """
    contents = [
        (0, 50, 101)
    ]


@schema
class LikelihoodSummary(dj.Computed):
    definition = """
    -> cd_dlset.DLSetInfo
    -> SummaryBinConfig
    ---
    contrast: float
    samples: longblob
    likelihoods: longblob
    ori_centered_likelihoods: longblob
    mean_centered_likelihoods: longblob
    max_centered_likelihoods: longblob
    """

    def make(self, key):
        bin_extent, nbins = (SummaryBinConfig & key).fetch1('bin_extent', 'nbins')
        samples = np.linspace(-bin_extent, bin_extent, nbins)
        mu_oris, max_oris = (LikelihoodStats & key).fetch1('mu_likelihood', 'max_ori')

        path = (cd_dlset.DLSetInfo & key).fetch1('dataset_path')
        data = loadmat(path)['dataSet'][0, 0]

        conts = data['contrast'].squeeze()
        assert np.all(conts == conts[0]), 'More than one contrast present!'
        contrast = conts[0]
        s = data['decodeOri'].squeeze() - 270
        Ls = data['likelihood'].T # trials x decodeOri
        oris = data['orientation'].squeeze() - 270

        likelihoods = []
        ori_centered_likelihoods = []
        mean_centered_likelihoods = []
        max_centered_likelihoods = []
        for L, ori, mu_ori, max_ori in zip(Ls, oris, mu_oris, max_oris):
            f = interp1d(s, L, kind='quadratic', bounds_error=False, fill_value=0)
            likelihoods.append(f(samples))
            ori_centered_likelihoods.append(f(samples + ori))
            mean_centered_likelihoods.append(f(samples + mu_ori))
            max_centered_likelihoods.append(f(samples + max_ori))

        likelihoods = np.stack(likelihoods)
        ori_centered_likelihoods = np.stack(ori_centered_likelihoods)
        mean_centered_likelihoods = np.stack(mean_centered_likelihoods)
        max_centered_likelihoods = np.stack(max_centered_likelihoods)

        key['contrast'] = contrast
        key['samples'] = samples
        key['likelihoods'] = likelihoods
        key['ori_centered_likelihoods'] = ori_centered_likelihoods
        key['mean_centered_likelihoods'] = mean_centered_likelihoods
        key['max_centered_likelihoods'] = max_centered_likelihoods

        self.insert1(key)



