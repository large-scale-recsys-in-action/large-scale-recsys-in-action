# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.dataset.datasets.dataset import Dataset
from lib.feature.feature import Categorical, Continuous, Sequential


class TFDataset(Dataset):
    def __init__(self, mode, conf_factory):
        super().__init__(mode, conf_factory)

    # {field_name: field_type}
    def _get_example_fmt(self):
        example_fmt = {}
        if self._label_type:
            label_type = self._tf_type(self._label_type)
            example_fmt[self._label_name] = tf.io.FixedLenFeature([], label_type)

        for slot in self._slots:
            col = self._slot_feature_map[slot]
            name = col.name
            d_type = self._tf_type(col.d_type)

            if isinstance(col, Categorical) or isinstance(col, Continuous):
                if col.len:
                    example_fmt[name] = tf.io.FixedLenFeature([col.len], d_type)
                else:
                    example_fmt[name] = tf.io.FixedLenFeature([], d_type)
            elif isinstance(col, Sequential):
                example_fmt[name] = tf.io.VarLenFeature(d_type)
            else:
                raise NotImplementedError('TFDataset slot {} configuration error.'.format(slot))

        return example_fmt

    def parse_fn(self, example):
        example_fmt = self._get_example_fmt()
        parsed = tf.io.parse_single_example(example, example_fmt)
        for slot in self._slots:
            col = self._slot_feature_map[slot]
            name = col.name
            default_value = self._default_value(col.d_type)
            if isinstance(example_fmt[name], tf.io.VarLenFeature):
                parsed[name] = tf.sparse.to_dense(parsed[name], default_value)

            if col.len > 0:
                parsed[name] = parsed[name][0:col.len]

        label = parsed.pop(self._label_name)
        return parsed, label

    def _padded_shapes_and_padding_values(self):
        padded_shapes = {self._slot_feature_map[s].name: [] for s in self._slots}
        padding_values = {}

        for slot in self._slots:
            col = self._slot_feature_map[slot]
            name = col.name
            padding_value = self._default_value(col.d_type)
            padding_values[name] = padding_value

            if isinstance(col, Sequential) or col.len > 0:
                padded_shapes[name] = [col.len if col.len > 0 else None]

        if self._label_type:
            padded_shapes = (padded_shapes, [])
            padding_values = (padding_values, self._default_value(self._label_type))

        return padded_shapes, padding_values
