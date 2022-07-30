# -*- coding: utf-8 -*-
import os
import math
import random
from abc import ABC
import tensorflow as tf
from datetime import datetime, timedelta
from lib.dataset import DatasetType
from lib.common.io import local
from tensorflow.data import experimental
from lib.feature.feature_factory import FeatureFactory


class Dataset(ABC):
    def __init__(self, mode, conf_factory):
        self._mode = mode
        self._flags = conf_factory.flags
        self._model_conf = conf_factory.model_conf
        self._feature_conf = conf_factory.feature_conf
        self._dataset_conf = conf_factory.dataset_conf
        self._logger = conf_factory.logger
        self._is_training = mode == 'train'
        self._start = self._flags.start
        self._end = self._flags.end
        self._slots = self._get_slots()
        self._num_parallel_calls = self._flags.num_parallel_calls or os.cpu_count()
        self._fs = self._file_system()
        self._data_files = self._get_files()
        self._slot_feature_map = FeatureFactory.parse(conf_factory.feature_conf)
        self._label_name, self._label_type = self._get_label()
        self._dataset_fn = self._get_dataset_fn()
        self._padded_shapes, self._padding_values = self._padded_shapes_and_padding_values()
        self._assert()

    def _padded_shapes_and_padding_values(self):
        raise NotImplementedError('Dataset not implement _padded_shapes_and_padding_values.')

    def input_fn(self):
        files = tf.data.Dataset.from_tensor_slices(self._data_files)
        data_set = files.apply(experimental.parallel_interleave(self._dataset_fn,
                                                                cycle_length=self._num_parallel_calls,
                                                                sloppy=True))

        data_set = data_set.apply(experimental.ignore_errors())
        data_set = data_set.map(map_func=self.parse_fn,
                                num_parallel_calls=self._num_parallel_calls)
        if self._mode == 'train':
            data_set = data_set.shuffle(buffer_size=self._flags.buffer_size)
            data_set = data_set.repeat(self._flags.num_epochs)
        data_set = data_set.padded_batch(self._flags.batch_size,
                                         padded_shapes=self._padded_shapes,
                                         padding_values=self._padding_values)
        data_set = data_set.prefetch(buffer_size=experimental.AUTOTUNE)
        return data_set

    @staticmethod
    def _default_value(d_type):
        if d_type == 'string':
            return tf.constant('0')
        elif d_type == 'int64':
            return tf.constant(0, tf.int64)
        elif d_type == 'int32':
            return tf.constant(0, tf.int32)
        elif d_type == 'float32':
            return tf.constant(0.0)
        else:
            raise NotImplementedError('d_type {} error'.format(d_type))

    @staticmethod
    def _tf_type(d_type):
        if d_type == 'string':
            tf_type = tf.string
        elif d_type == 'int64':
            tf_type = tf.int64
        elif d_type == 'float32':
            tf_type = tf.float32
        else:
            raise NotImplementedError('data type {} is not supported.'.format(d_type))

        return tf_type

    def parse_fn(self, example):
        raise NotImplementedError('Dataset not implement parse_fn.')

    def preview(self):
        from tensorflow import data, InteractiveSession

        sess = InteractiveSession()
        samples = data.make_one_shot_iterator(self.input_fn()).get_next()

        data = []
        for i in range(1):
            data.append(sess.run(samples))

        return data

    def _get_dataset_fn(self):
        _type = self._dataset_conf['set_type']
        if _type == DatasetType.TF_RECORD:
            return tf.data.TFRecordDataset
        elif _type == DatasetType.CSV:
            return tf.data.TextLineDataset
        else:
            raise NotImplementedError(f'Invalid DataType {_type}.')

    def _assert(self):
        assert self._flags.batch_size > 0, 'batch size should be > 0.'
        assert self._data_files, 'data_files empty.'
        assert self._slots, 'slots not specified.'
        seen = set()
        dupes = [s for s in self._slots if s in seen or seen.add(s)]
        assert not dupes, f'duplicated slots: {dupes}'

    def _get_slots(self):
        return [int(f.strip()) for f in self._dataset_conf['slots'].strip().split(',')]

    def _get_files(self):
        path = self._get_path()
        self._logger.info(f'#path: {len(path)}, value: {path}\n')
        files = self._read(path)
        self._logger.info(f'#files: {len(files)}')
        return files

    def _read(self, directory):
        files = []
        if self._is_training:
            for i in range(len(directory)):
                training_files = []
                sub_dir = directory[i]
                last = i == len(directory) - 1
                training_files.extend([i for i in self._fs.ls(sub_dir) if '_SUCCESS' not in i])
                if last and 0 < self._flags.train_test_split < 1.0:
                    file_num = math.ceil(self._flags.train_test_split * len(training_files))
                    training_files = training_files[:file_num]
                files.extend(training_files)
        else:
            test_files = []
            sub_dir = directory[-1]
            test_files.extend([i for i in self._fs.ls(sub_dir) if '_SUCCESS' not in i])
            if 0 < self._flags.train_test_split < 1.0:
                file_num = math.ceil(self._flags.train_test_split * len(test_files))
                test_files = test_files[:file_num]
            files.extend(test_files)

        random.shuffle(files)
        return files

    def _get_label(self):
        label = self._dataset_conf.get('label')
        if not label:
            return None, None

        if ':' not in label:
            raise RuntimeError('Dataset label invalid: {}'.format(label))

        name, _type = label.split(':')
        if _type in ['int32', 'int64', 'float32']:
            return name, _type
        else:
            self._logger.warn(f'warning, label type: {_type} is invalid.')
            return name, _type

    def _get_path(self):
        if 'dataset' in self._dataset_conf:
            data_root = self._dataset_conf['dataset'].rstrip('/')
        else:
            raise ValueError('dataset not specified.')

        data_path = [i for i in self._fs.ls(data_root) if i.split('/')[-1].isdigit()]
        start, end = self._start, self._end if self._end else data_path[-1].split('/')[-1]
        dates_span = self._list_date(start, end)
        data_path = [i for i in data_path if i.split('/')[-1] in dates_span]

        return data_path

    def _file_system(self):
        path = self._dataset_conf['dataset']
        if path.startswith('/'):
            return local.Local()
        else:
            raise NotImplementedError('Dataset file system not supported.')

    @staticmethod
    def _list_date(start, end, pattern='%Y%m%d'):
        dates = []
        dt = datetime.strptime(start, pattern)
        date = start
        while date <= end:
            dates.append(date)
            dt += timedelta(days=1)
            date = dt.strftime(pattern)
        return dates
