
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