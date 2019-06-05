
import datajoint as dj
schema = dj.schema('edgar_cd_dataset', locals())
class_discrimination = dj.create_virtual_module('class_discrimination', 'edgar_class_discrimination')


@schema
class CleanContrastSessionDataSet(dj.Computed):
    def fetch_dataset(self):
        assert len(self)==1, 'Only can fetch one dataset at a time!'
        data = (class_discrimination.ClassDiscriminationTrial() * class_discrimination.SpikeCountTrials() * class_discrimination.CSCLookup() & class_discrimination.CleanSpikeCountTrials() & self).fetch(order_by='trial_num')
        contrast = float(self.fetch1('dataset_contrast'))
        f = data['contrast'] == contrast
        return data[f]



@schema
class CSInfo(dj.Computed):
    definition = """
    -> CleanContrastSessionDataSet
    ---
    subject_id: int
    n_trials: int
    """

    @property
    def key_source(self):
        return CleanContrastSessionDataSet & (class_discrimination.CSCLookup & 'count_start = 0 and count_stop = 500')


    def make(self, key):
        dset = (CleanContrastSessionDataSet & key).fetch_dataset()
        key['n_trials'] = len(dset['contrast'])
        key['subject_id'] = dset['subject_id'][0]

        self.insert1(key)