# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import lookup_ops

"""
numpy: 1.19.3
tf: 1.15.0
"""

features = ...
item_ids = features['item_ids']

# 查询物品 embedding 方式一: 随机初始化
V = 10000
D = 128
embedding_matrix = tf.get_variable(name='embedding_matrix', dtype=tf.float32, shape=(V, D))
hash_ids = tf.strings.to_hash_bucket(item_ids, num_buckets=V)
item_embeddings = tf.nn.embedding_lookup(embedding_matrix, hash_ids, name='item_embeddings')
...

# 查询物品 embedding 方式二: 使用第三方 embedding, 文件名 item_embedding.npy, 存储 dict 数据, key 分别为 ids 和 embeddings
pretrain_embeddings = np.load('item_embedding.npy', allow_pickle=True).item()
ids = lookup_ops.index_table_from_tensor(pretrain_embeddings['ids'],
                                         num_oov_buckets=0,
                                         default_value=0,
                                         hasher_spec=lookup_ops.FastHashSpec,
                                         dtype=tf.string,
                                         name='ids')
embedding_matrix_v2 = tf.get_variable(name='embedding_matrix_v2',
                                      dtype=tf.float32,
                                      initializer=pretrain_embeddings['embeddings'])
indices = ids.lookup(item_ids)
item_embeddings_v2 = tf.nn.embedding_lookup(embedding_matrix_v2, indices, name='item_embeddings_v2')
...
