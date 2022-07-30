# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.compat.v1.feature_column as tfc
import math


class FeatureBuilder:
    @staticmethod
    def _get_embedding_size(bucket_size):
        return int(2 ** math.ceil(math.log2(bucket_size ** 0.25)))

    def user_features(self):
        user_embedding = self._hash_embedding(key='user_id',
                                              hash_bucket_size=1000,
                                              embedding_size=8)
        gender_embedding = self._hash_embedding(key='gender',
                                                hash_bucket_size=10,
                                                embedding_size=2)

        _boundaries = [0, 18, 25, 36, 45, 55, 65, 80]
        age_embedding = self._bucketized_embedding('age', _boundaries,
                                                   embedding_size=2)

        return [user_embedding, gender_embedding, age_embedding]

    def context_features(self):
        device_embedding = self._hash_embedding(key='device',
                                                hash_bucket_size=100,
                                                embedding_size=4)
        return [device_embedding]

    def item_and_histories_features(self, features, name='item'):
        _keys = ['item_id', 'clicks']
        item_tensor, clicks_tensors = self._share_embedding_v2(_keys,
                                                               features,
                                                               hash_bucket_size=100,
                                                               embedding_size=2,
                                                               name=name)
        return item_tensor, clicks_tensors

    def _hash_embedding(self, key, hash_bucket_size, embedding_size=None, dtype=tf.string):
        _hash = tfc.categorical_column_with_hash_bucket(
            key=key,
            hash_bucket_size=hash_bucket_size,
            dtype=dtype)
        _embedding_size = embedding_size or self._get_embedding_size(hash_bucket_size)
        _embedding_column = tfc.embedding_column(_hash, _embedding_size)

        return _embedding_column

    def _bucketized_embedding(self, key, boundaries, embedding_size=None, dtype=tf.int64):
        # 1. 读取原始数据
        raw = tfc.numeric_column(
            key=key,
            dtype=dtype)

        # 2. 根据 boundaries 得到桶号
        bucketized = tfc.bucketized_column(
            source_column=raw,
            boundaries=boundaries)
        _embedding_size = embedding_size or self._get_embedding_size(len(boundaries) + 1)

        # 3. 根据桶号, 得到 embedding
        _embedding_column = tfc.embedding_column(bucketized, _embedding_size)

        return _embedding_column

    def _share_embedding_v2(self, keys, features, hash_bucket_size, embedding_size=None, name=''):
        # 1. 手动计算各特征的 hash 值
        _hashes = [
            tf.string_to_hash_bucket_fast(
                features[key],
                num_buckets=hash_bucket_size) for key in keys
        ]

        # 2. 手动生成共享 embedding 矩阵
        _embedding_size = embedding_size or self._get_embedding_size(hash_bucket_size)
        embedding_matrix = tf.get_variable(
            name=f'{name}_embedding_matrix',
            shape=(hash_bucket_size, _embedding_size))

        # 3. 手动查询各 hash 对应的 embedding 向量
        _vectors = [
            tf.nn.embedding_lookup(embedding_matrix, _hash)
            for _hash in _hashes
        ]

        return _vectors
