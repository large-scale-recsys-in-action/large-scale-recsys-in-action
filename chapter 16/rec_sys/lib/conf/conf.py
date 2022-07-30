from abc import ABC
import os
import json
import logging.config


class Conf(ABC):
    def __init__(self, flags):
        self._flags = flags
        self._model_name = self._flags.model_name
        self._d_root_conf = self._root_conf_path()

    def _root_conf_path(self):
        project_dir = self._flags.project_dir
        return project_dir.joinpath('conf')

    def _parse(self):
        raise NotImplementedError('Conf not implement parse.')


class ModelConf(Conf):
    def __init__(self, flags):
        super().__init__(flags)
        self._f_conf = (self._d_root_conf
                        .joinpath('model')
                        .joinpath(self._model_name)
                        .joinpath('model.conf'))
        if not os.path.exists(self._f_conf):
            raise FileNotFoundError(f'model {self._model_name} '
                                    f'missing model conf.')
        self._f_default_conf = self._d_root_conf.joinpath('model.conf')
        self._conf = self._parse()

    def _parse(self):
        conf = {}
        if os.path.exists(self._f_default_conf):
            with open(self._f_default_conf) as _f_default_conf:
                default_model_conf = self._file_parse(_f_default_conf)
            conf.update(default_model_conf)
        with open(self._f_conf) as _f_conf:
            model_conf = self._file_parse(_f_conf)
        conf.update(model_conf)
        conf['save_summary_steps'] = int(conf['save_summary_steps'])
        conf['save_checkpoints_steps'] = int(conf['save_checkpoints_steps'])
        conf['keep_checkpoint_max'] = int(conf['keep_checkpoint_max'])
        conf['log_step_count_steps'] = int(conf['log_step_count_steps'])
        conf['eval_steps'] = int(conf['eval_steps'])
        conf['eval_throttle_secs'] = int(conf['eval_throttle_secs'])
        conf['max_steps'] = int(conf['max_steps']) if 'max_steps' in conf else None

        for k in conf:
            if k in self._flags.__dict__:
                conf[k] = self._flags.__dict__[k]

        return conf

    @property
    def conf(self):
        return self._conf

    @staticmethod
    def _file_parse(f):
        conf = {}
        for line in f:
            if len(line.strip()) == 0 or line.strip().startswith('#'):
                continue
            num = line.count('=')
            if 0 == num:
                continue
            elif 1 == num:
                k, v = line.split('=')
                k = k.strip()
                v = v.strip()
                if k == 'owners':
                    v = v.split(',')
                if k == 'slots' or k == 'serving_slots':
                    v = list(map(int, v.split(',')))
                conf[k] = v
        return conf


class FeatureConf(Conf):
    def __init__(self, flags):
        super().__init__(flags)
        self._f_conf = (self._d_root_conf
                        .joinpath('model')
                        .joinpath(self._model_name)
                        .joinpath('features.conf'))
        self._f_default_conf = self._d_root_conf.joinpath('features.conf')
        if not os.path.exists(self._f_default_conf):
            raise FileNotFoundError(f'model {self._model_name} '
                                    f'missing feature conf.')

        self._conf = self._parse()

    def _parse(self):
        conf = {}

        with open(self._f_default_conf) as _f_default_conf:
            default_feature_conf = self._file_parse(_f_default_conf)
        conf.update(default_feature_conf)

        if os.path.exists(self._f_conf):
            with open(self._f_conf) as _f_conf:
                model_feature_conf = self._file_parse(_f_conf)
            if model_feature_conf:
                for slot, slot_conf in model_feature_conf.items():
                    conf.setdefault(slot, {}).update(slot_conf)

        for k in conf:
            if k in self._flags.__dict__:
                conf[k] = self._flags.__dict__[k]

        return conf

    @property
    def conf(self):
        return self._conf

    @staticmethod
    def _file_parse(f):
        conf = {}
        for raw_line in f:
            if len(raw_line.strip()) == 0 or raw_line.strip().startswith('#'):
                continue
            slot_conf = {}
            for kv in raw_line.split(','):
                kv = kv.split('=')
                if len(kv) != 2:
                    raise RuntimeError(f'FeatureConf parse error.'
                                       f'kv: {kv}\n raw_line: {raw_line}')
                k, v = kv
                k = k.strip()
                v = v.strip()
                if k in ('slot', 'depend'):
                    v = int(v)
                slot_conf[k] = v

            slot = slot_conf['slot']
            if slot in conf:
                raise RuntimeError(f'FeatureConf duplicated slot: {slot}')
            conf[slot] = slot_conf
        return conf


class DatasetConf(Conf):
    def __init__(self, flags):
        super().__init__(flags)
        self._f_conf = (self._d_root_conf
                        .joinpath('dataset')
                        .joinpath(f'{self._flags.dataset}'))
        if not os.path.exists(self._f_conf):
            raise FileNotFoundError(f'model {self._model_name} '
                                    f'missing dataset conf.')
        self._conf = self._parse()

    def _parse(self):
        conf = {}
        with open(self._f_conf) as f:
            for raw_line in f:
                if (not raw_line.strip() or
                        raw_line.strip().startswith('#')):
                    continue
                key, value = raw_line.split('=')
                key = key.strip()
                value = value.strip()
                conf[key] = value
        return conf

    @property
    def conf(self):
        return self._conf


class LoggerConf(Conf):
    def __init__(self, flags):
        super().__init__(flags)
        self._f_conf = self._d_root_conf.joinpath('logger.conf')
        self._logger_dir = self._flags.project_dir.joinpath('logs')
        if not os.path.exists(self._logger_dir):
            os.mkdir(path=self._logger_dir)

        config = self._parse()

        if config:
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=logging.DEBUG)

    def _parse(self):
        if not os.path.exists(self._f_conf):
            return None

        with open(self._f_conf) as log:
            conf = json.load(log)

        for handler in conf['handlers']:
            h_conf = conf['handlers'][handler]
            if 'filename' not in h_conf:
                continue
            h_conf['filename'] = str(self._logger_dir
                                     .joinpath(h_conf['filename']))

        return conf

    @staticmethod
    def get_logger(name='tensorflow'):
        return logging.getLogger(name)
