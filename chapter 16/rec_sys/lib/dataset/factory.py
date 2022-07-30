# -*- coding: utf-8 -*-
from lib.dataset import DatasetType
from lib.dataset.datasets.tf_dataset import TFDataset
from lib.feature.feature_factory import FeatureFactory


class DatasetFactory:
    def __init__(self, mode, conf_factory):
        self._mode = mode
        self.conf_factory = conf_factory
        self._dataset = self._get_dataset()

    def _get_dataset(self):
        dataset_conf = self.conf_factory.dataset_conf
        set_type = dataset_conf.get('set_type')
        if set_type == DatasetType.TF_RECORD:
            dataset = TFDataset(self._mode, self.conf_factory)
        elif set_type == DatasetType.CSV:
            raise NotImplementedError(f'DatasetFactory not implements {set_type}')
        else:
            raise NotImplementedError(f'DatasetFactory not implements {set_type}')
        return dataset

    def preview(self):
        return self._dataset.preview()

    def input_fn(self):
        return self._dataset.input_fn()


def input_fn(conf_factory, mode='train', _type=None):
    factory = DatasetFactory(mode, conf_factory)
    return factory.preview() if _type == 'preview' else factory.input_fn()
