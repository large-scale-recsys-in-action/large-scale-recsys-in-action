from lib.conf.conf import ModelConf, DatasetConf, FeatureConf, LoggerConf


class ConfFactory:
    def __init__(self, flags):
        self._flags = flags
        self._model_conf = ModelConf(flags).conf
        self._dataset_conf = DatasetConf(flags).conf
        self._feature_conf = FeatureConf(flags).conf
        self._logger = LoggerConf(flags).get_logger()

    @property
    def flags(self):
        return self._flags

    @property
    def model_conf(self):
        return self._model_conf

    @property
    def dataset_conf(self):
        return self._dataset_conf

    @property
    def feature_conf(self):
        return self._feature_conf

    @property
    def logger(self):
        return self._logger
