import datajoint as dj
import numpy as np
from scipy.io import loadmat

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

        key['contrasts'] = cont
        key['orientation'] = ori
        key['mu_likelihood'] = mu_L
        key['sigma_likelihood'] = sigma_L

        self.insert1(key)
