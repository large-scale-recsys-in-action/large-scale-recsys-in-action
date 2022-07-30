# -*- coding: utf-8 -*-

from typing import Dict
import tensorflow as tf
from tensorflow import initializers
from tensorflow import train
from lib.feature.feature_factory import FeatureFactory
from tensorflow.estimator import ModeKeys
from tensorflow_core import Tensor


class MetaEstimator:
    TRAIN = ModeKeys.TRAIN
    EVAL = ModeKeys.EVAL
    PREDICT = ModeKeys.PREDICT

    def __init__(self, features: Dict[str, Tensor], labels: Dict[str, Tensor], mode: str, h_params: Dict):
        self._features = features
        self._labels = labels
        self._mode = mode
        self._h_params = h_params
        self._feature_factory = FeatureFactory(h_params['feature_conf'])

    def _exponential_decay(self, global_step):
        learning_rate = self._h_params['model_conf']['learning_rate']
        decay_steps = self._h_params['model_conf']['decay_steps']
        decay_rate = self._h_params['model_conf']['decay_rate']

        return train.exponential_decay(learning_rate=learning_rate,
                                       global_step=global_step,
                                       decay_steps=decay_steps,
                                       decay_rate=decay_rate,
                                       staircase=False)

    def model_fn(self):
        raise NotImplementedError('MetaBuilder not support model_fn.')

    def input_layer(self, slots,
                    embedding_size=None,
                    bucket_size=None) -> Tensor:
        if not slots:
            raise RuntimeError('Estimator input_layer slots params NONE.')

        is_cross = type(slots[0]) in [set, tuple, list]
        columns = []
        if not is_cross:
            for slot in slots:
                embedding_name = '{}_emb'.format(self._slot_name(slot))
                column = self.look_up(slot,
                                      name=embedding_name,
                                      embedding_size=embedding_size,
                                      hash_bucket_size=bucket_size)
                columns.append(column)
        else:
            for cross_slots in slots:
                column = self.cross(cross_slots, hash_bucket_size=bucket_size)
                columns.append(column)

        return tf.feature_column.input_layer(self._features, columns)

    def _slot_name(self, slot):
        return self._feature_factory.slot_feature_map[slot].name

    def cross(self, slots, hash_bucket_size, embedding_size=None):
        return self._feature_factory.cross(slots, hash_bucket_size, embedding_size)

    def look_up(self, slot, hash_bucket_size=None, embedding_size=None, name=None):
        return self._feature_factory.look_up(slot,
                                             self._features,
                                             hash_bucket_size,
                                             embedding_size,
                                             name)

    @staticmethod
    def attention(history_emb,
                  current_emb,
                  history_masks,
                  units,
                  name='attention'):
        time_steps = tf.shape(history_emb)[1]
        embedding_size = current_emb.get_shape().as_list()[-1]
        current_emb_tmp = tf.tile(current_emb, [1, time_steps])
        current_emb = tf.reshape(current_emb_tmp, shape=[-1, time_steps, embedding_size])
        net = tf.concat([history_emb,
                         history_emb - current_emb,
                         current_emb,
                         history_emb * current_emb], axis=-1)
        for units in units:
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.relu)
        attention_weights = tf.layers.dense(inputs=net, units=1, activation=None)
        scores = tf.transpose(attention_weights, [0, 2, 1])
        history_masks = tf.expand_dims(history_masks, axis=1)
        padding = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(history_masks, scores, padding)
        scores = tf.nn.softmax(scores)
        scores = tf.where(history_masks, scores, padding)
        outputs = tf.matmul(scores, history_emb)
        outputs = tf.reduce_sum(outputs, 1, name=name)
        return outputs

    @staticmethod
    def fully_connected_layers(mode,
                               net,
                               units,
                               dropout=0.0,
                               activation=None,
                               name='fc_layers'):
        layers = len(units)
        for i in range(layers - 1):
            num = units[i]
            net = tf.layers.dense(net,
                                  units=num,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.initializers.he_uniform(),
                                  name=f'{name}_units_{num}_{i}')
            net = tf.layers.dropout(inputs=net,
                                    rate=dropout,
                                    training=mode == tf.estimator.ModeKeys.TRAIN)
        num = units[-1]
        net = tf.layers.dense(net, units=num, activation=activation,
                              kernel_initializer=tf.initializers.glorot_uniform(),
                              name=f'{name}_units_{num}')
        return net

    @staticmethod
    def exponential_decay(global_step,
                          learning_rate=0.01,
                          decay_steps=10000,
                          decay_rate=0.9):
        return tf.train.exponential_decay(learning_rate=learning_rate,
                                          global_step=global_step,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          staircase=False)
