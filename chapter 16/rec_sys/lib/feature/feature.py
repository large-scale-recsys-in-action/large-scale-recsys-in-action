# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import feature_column


class Feature:
    def __init__(self, conf):
        self._conf = conf
        self.slot = conf['slot']
        self.f_type = conf['f_type']
        self.name = conf['name']
        self.encoder = conf.get('encoder')
        self.args = conf.get('args')
        self.d_type = conf['d_type']
        self.len = int(conf.get('len', '0'))
        self.column = self._parse() if self.encoder else None

    @property
    def conf(self):
        return self._conf

    def _parse(self):
        raise NotImplementedError('Feature not implement _col.')

    def __str__(self):
        return str(self._conf)


class Categorical(Feature):
    def __init__(self, conf):
        super(Categorical, self).__init__(conf)

    def _parse(self):
        _column = None
        if self.encoder == 'hash':
            self.args = int(self.args)
            d_type = tf.string
            if self.d_type == 'int64':
                d_type = tf.int64
            if self.d_type == 'int32':
                d_type = tf.int32
            _column = feature_column.categorical_column_with_hash_bucket(
                self.name,
                hash_bucket_size=self.args,
                dtype=d_type
            )
        elif self.encoder == 'identity':
            self.args = self.args.split('|')
            num_buckets, default_value = self.args
            _column = feature_column.categorical_column_with_identity(
                self.name,
                num_buckets=num_buckets,
                default_value=default_value)
        elif self.encoder == 'matrix':
            pass
        else:
            raise NotImplementedError('Categorical not support'
                                      f' {self.encoder}: slot {self.slot}')

        return _column


class Continuous(Feature):
    def __init__(self, conf):
        super(Continuous, self).__init__(conf)

    def _parse(self):
        _column = None
        if self.encoder == 'bucketize':
            self.args = list(map(float, self.args.split('|')))
            if self.d_type == 'int32':
                d_type = tf.int32
            elif self.d_type == 'int64':
                d_type = tf.int64
            else:
                d_type = tf.float32

            shape = self.len or 1

            col = feature_column.numeric_column(self.name,
                                                shape=(shape,),
                                                default_value=0,
                                                dtype=d_type,
                                                normalizer_fn=None)

            _column = feature_column.bucketized_column(
                source_column=col,
                boundaries=self.args)
        else:
            raise NotImplementedError('Continuous not support'
                                      f' {self.encoder}: slot {self.slot}')
        return _column


class Sequential(Feature):
    def __init__(self, conf):
        super(Sequential, self).__init__(conf)

    def _parse(self):
        _column = None
        if self.encoder == 'hash':
            self.args = int(self.args)
            d_type = tf.string

            _column = (feature_column.sequence_categorical_column_with_hash_bucket(
                self.name,
                hash_bucket_size=self.args,
                dtype=d_type))
        elif self.encoder == 'identity':
            self.args = self.args.split('|')
            num_buckets, default_value = self.args
            _column = feature_column.sequence_categorical_column_with_identity(
                self.name,
                num_buckets=num_buckets,
                default_value=default_value)
        else:
            raise NotImplementedError('Sequence not support'
                                      f' {self.encoder}: slot {self.slot}')

        return _column
