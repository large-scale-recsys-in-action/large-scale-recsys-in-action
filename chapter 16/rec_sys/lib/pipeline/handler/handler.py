# -*- coding: utf-8 -*-
import importlib
from abc import ABC
from datetime import datetime
import tensorflow as tf


class Handler(ABC):
    def __init__(self):
        self._next = None

    def next(self):
        return self._next

    @staticmethod
    def _build_model_fn(model_name):
        model_class = importlib.import_module(f'model.{model_name}.estimator').Estimator

        def model_fn(features, labels, mode, params):
            return model_class(features, labels, mode, params).model_fn()

        return model_fn

    def handle(self, conf_factory):
        self._handle(conf_factory)
        _next = self.next()
        if _next:
            _next.handle(conf_factory)

    def _handle(self, conf_factory):
        raise NotImplementedError('Operator not implement handle.')

    def set_next(self, _next):
        self._next = _next
        return _next

    @staticmethod
    def _build_params(conf_factory):
        params = {
            'feature_conf': conf_factory.feature_conf,
            'model_conf': conf_factory.model_conf
        }
        return params

    @staticmethod
    def _tf_type(d_type):
        if d_type == 'string':
            tf_type = tf.string
        elif d_type == 'int64':
            tf_type = tf.int64
        elif d_type == 'int32':
            tf_type = tf.int32
        elif d_type == 'float32':
            tf_type = tf.float32
        else:
            raise NotImplementedError('data type {} is not supported.'.format(d_type))
        return tf_type
