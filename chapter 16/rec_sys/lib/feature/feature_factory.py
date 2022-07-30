# -*- coding: utf-8 -*-
import math
import tensorflow as tf
from tensorflow import initializers
from tensorflow import get_variable
from tensorflow.feature_column import embedding_column, crossed_column
from lib.feature.feature import Categorical, Continuous, Sequential


def get_embedding_size(category_num):
    return int(2 ** math.ceil(math.log2(category_num ** 0.25)))


def get_bucket_size(feature):
    bucket_size = None
    if isinstance(feature, Categorical):
        bucket_size = feature.args
        if feature.encoder == 'hash':
            bucket_size = feature.args
    elif isinstance(feature, Continuous):
        bucket_size = len(feature.args) + 1

    if not bucket_size:
        raise RuntimeError(f'slot {feature.slot} buckets is None.')
    return bucket_size


class FeatureFactory:
    """
    Args:
        feature_conf: 模型特征配置, 解析后得到 slot_feature_map,
                      slot_feature_map 的 key 是 slot, value 是 lib.feature.feature.Feature,
                      slot_embedding_map 的 key 也是 slot, 但是 value 的类型有两种:
                        1) tf.feature_column, 也就是特征的处理函数, 注意这里存储的是特征如何处理的方式;
                        2) tensor, 具体的 tensor。有些特征需要共享 embedding matrix, 比如 item_id 与 history behaviors,
                           此时必须存储为 tensor, 如果存储为 tf.feature_column, 则使用起来非常的不方便。
    """

    def __init__(self, feature_conf):
        self._slot_feature_map = self.parse(feature_conf)
        self._slot_embedding_map = {}

    def look_up(self, slot,
                features=None,
                hash_bucket_size=None,
                embedding_size=None,
                name=None):
        """
        look up slot_embedding_map by slot
        return 1. tf.feature_column (how to process one feature.) or
               2. tensor (low-level tensor representation.)
        :param slot: feature ID
        :param features: input feature, {'name': 'tensor value'}
        :param hash_bucket_size: bucket size, could be none.
        :param embedding_size: embedding size, could be none.
        :param name: ops name
        :return: tf.feature_column or tensor
        """

        depend_slot = self._slot_feature_map[slot].conf.get('depend')
        if depend_slot and depend_slot not in self._slot_embedding_map:
            raise LookupError(f'slot: {slot}, depend slot is missing.')

        if depend_slot:
            fc_or_matrix = self._slot_embedding_map[depend_slot]
        else:
            if slot not in self._slot_embedding_map:
                self._slot_embedding_map[slot] = self._create_embedding(slot,
                                                                        hash_bucket_size,
                                                                        embedding_size,
                                                                        name)
            fc_or_matrix = self._slot_embedding_map[slot]
        col = self._slot_feature_map[slot]
        if depend_slot or col.conf.get('encoder') == 'matrix':  # tensor
            ids = features[col.name]
            ids = tf.strings.to_hash_bucket_fast(ids, fc_or_matrix.bucket_size)
            matrix = fc_or_matrix.embedding
            return tf.nn.embedding_lookup(matrix, ids, name=name)
        else:
            if self._slot_feature_map[slot].encoder:  # feature_column
                fc = fc_or_matrix.embedding
                return fc
            else:
                raise RuntimeError('feature without encoder.')

    def cross(self, slots, hash_bucket_size, embedding_size=None):
        names = [self.slot_feature_map[slot].name for slot in slots]
        col = crossed_column(names, hash_bucket_size)
        emb_size = get_embedding_size(hash_bucket_size)
        if embedding_size:
            emb_size = embedding_size
        return embedding_column(col, dimension=emb_size)

    @classmethod
    def parse(cls, feature_conf):
        slot_feature_map = {}
        for slot, slot_conf in feature_conf.items():
            f_type = slot_conf['f_type']
            if f_type == 'categorical':
                col = Categorical(slot_conf)
            elif f_type == 'continuous':
                col = Continuous(slot_conf)
            elif f_type == 'sequence':
                col = Sequential(slot_conf)
            else:
                raise NotImplementedError(f'slot {slot}, '
                                          f'feature type {f_type} not supported.')
            slot_feature_map[slot] = col

        return slot_feature_map

    @property
    def slot_feature_map(self):
        return self._slot_feature_map

    def _create_embedding(self, slot, hash_bucket_size=None, embedding_size=None, name=None):
        col = self._slot_feature_map[slot]
        return Embedding(col, hash_bucket_size, embedding_size, name)


class Embedding:
    def __init__(self, feature,
                 bucket_size=None,
                 embedding_size=None,
                 name=None,
                 initializer=initializers.glorot_uniform()):
        self._feature = feature
        self._initializer = initializer
        self.name = name if name else feature.name + '_emb'
        self.embedding_size = embedding_size
        self.bucket_size = bucket_size
        self.embedding = self._get_embedding()

    def _get_embedding(self):
        encoder = self._feature.conf.get('encoder')
        args = self._feature.args
        if not self.bucket_size:
            if encoder == 'matrix':
                bucket_size, _ = args.split('|')
                self.bucket_size = int(bucket_size)
            else:
                self.bucket_size = get_bucket_size(self._feature)

        if not self.embedding_size:
            if encoder == 'matrix':
                _, embedding_size = args.split('|')
                self.embedding_size = int(embedding_size)
            else:
                self.embedding_size = get_embedding_size(self.bucket_size)

        if encoder == 'matrix':
            return get_variable(name=self.name,
                                shape=(self.bucket_size, self.embedding_size),
                                dtype=tf.float32,
                                initializer=self._initializer)
        elif encoder:
            return embedding_column(self._feature.column, self.embedding_size)
        else:
            raise ValueError('feature without encoder and matrix')
