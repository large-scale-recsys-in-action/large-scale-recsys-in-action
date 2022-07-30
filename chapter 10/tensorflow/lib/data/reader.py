# -*- coding: utf-8 -*-

"""
文件名：reader.py
启动命令: python reader.py
"""

import os

import tensorflow as tf  # 1.15
from tensorflow.compat.v1 import data, InteractiveSession
from tensorflow.compat.v1.data import experimental


class Reader:
    def __init__(self, num_parallel_calls=None):
        self._num_parallel_calls = num_parallel_calls or os.cpu_count()

    # 1. 定义每个特征的格式和类型
    @staticmethod
    def get_example_fmt():
        example_fmt = dict()

        example_fmt['user_id'] = tf.FixedLenFeature([], tf.string)
        example_fmt['age'] = tf.FixedLenFeature([], tf.int64)
        example_fmt['gender'] = tf.FixedLenFeature([], tf.string)
        example_fmt['device'] = tf.FixedLenFeature([], tf.string)
        # 下列数据长度不固定
        example_fmt['clicks'] = tf.VarLenFeature(tf.string)
        example_fmt['item_id'] = tf.VarLenFeature(tf.string)
        example_fmt['relevance'] = tf.VarLenFeature(tf.int64)

        return example_fmt

    @staticmethod
    def _default_value(d_type):
        if d_type == 'string':
            return tf.constant('0')
        elif d_type == 'int64':
            return tf.constant(0, tf.int64)
        elif d_type == 'float32':
            return tf.constant(0.0)
        else:
            raise NotImplementedError('d_type {} error'.format(d_type))

    # 2. 定义解析函数
    def parse_fn(self, example):
        example_fmt = self.get_example_fmt()
        parsed = tf.parse_single_example(example, example_fmt)
        for name, fmt in example_fmt.items():
            if name == 'relevance':
                continue
            # VarLenFeature 解析的特征是 Sparse 的，需要转成 Dense 便于操作
            d_type = fmt.dtype
            default_value = self._default_value(d_type)
            if isinstance(fmt, tf.io.VarLenFeature):
                parsed[name] = tf.sparse.to_dense(parsed[name], default_value)

        parsed['relevance'] = tf.sparse.to_dense(parsed['relevance'], -2 ** 32)  # 1
        label = parsed.pop('relevance')
        features = parsed
        return features, label

    # pad 返回的数据格式与形状必须与 parse_fn 的返回值完全一致。
    def padded_shapes_and_padding_values(self):
        example_fmt = self.get_example_fmt()

        padded_shapes = {}
        padding_values = {}

        for f_name, f_fmt in example_fmt.items():
            if 'relevance' == f_name:
                continue
            if isinstance(f_fmt, tf.FixedLenFeature):
                padded_shapes[f_name] = []
            elif isinstance(f_fmt, tf.VarLenFeature):
                padded_shapes[f_name] = [None]
            else:
                raise NotImplementedError('feature {} feature type error.'.format(f_name))

            if f_fmt.dtype == tf.string:
                value = '0'
            elif f_fmt.dtype == tf.int64:
                value = 0
            elif f_fmt.dtype == tf.float32:
                value = 0.0
            else:
                raise NotImplementedError('feature {} data type error.'.format(f_name))

            padding_values[f_name] = tf.constant(value, dtype=f_fmt.dtype)

        # parse_fn 返回的是 tuple 结构，这里也必须是 tuple 结构
        padded_shapes = (padded_shapes, [None])
        padding_values = (padding_values, tf.constant(-2 ** 32, tf.int64))  # 2
        return padded_shapes, padding_values

    # 3. 定义读数据函数
    def input_fn(self, mode, flags):
        pattern, epochs, batch_size = flags.pattern, flags.num_epochs, flags.batch_size
        padded_shapes, padding_values = self.padded_shapes_and_padding_values()
        files = tf.data.Dataset.list_files(pattern)
        data_set = files.apply(
            experimental.parallel_interleave(
                tf.data.TFRecordDataset,
                cycle_length=8,
                sloppy=True
            )
        ) 
        data_set = data_set.apply(experimental.ignore_errors())
        data_set = data_set.map(map_func=self.parse_fn,
                                num_parallel_calls=self._num_parallel_calls)

        if mode == 'train':
            data_set = data_set.shuffle(buffer_size=10000)
            data_set = data_set.repeat(epochs)
        data_set = data_set.padded_batch(batch_size,
                                         padded_shapes=padded_shapes,
                                         padding_values=padding_values)

        data_set = data_set.prefetch(buffer_size=1)
        return data_set


def input_fn(flags, mode='train'):
    factory = Reader()
    return factory.input_fn(mode, flags)


if __name__ == '__main__':
    reader = Reader()
    from lib import flags as _flags

    dataset = input_fn(mode='train', flags=_flags)

    sess = tf.InteractiveSession()
    samples = tf.data.make_one_shot_iterator(dataset).get_next()

    records = []
    for i in range(1):
        records.append(sess.run(samples))

    print(records)
