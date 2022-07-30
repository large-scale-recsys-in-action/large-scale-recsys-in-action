# -*- coding: utf-8 -*-
import tensorflow as tf
from lib import model_fn
from lib import flags


def export_model(_flags):
    _flag = _flags[0]

    def serving_input_receiver_fn():
        receiver_tensors = \
            {
                'user_id': tf.placeholder(dtype=tf.string,
                                          shape=(None, None),
                                          name='user_id'),
                'age': tf.placeholder(dtype=tf.int64,
                                      shape=(None, None),
                                      name='age'),
                'gender': tf.placeholder(dtype=tf.string,
                                         shape=(None, None),
                                         name='gender'),
                'device': tf.placeholder(dtype=tf.string,
                                         shape=(None, None),
                                         name='device'),
                'item_id': tf.placeholder(dtype=tf.string,
                                          shape=(None, None),
                                          name='item_id'),
                'clicks': tf.placeholder(dtype=tf.string,
                                         shape=(None, None),
                                         name='clicks')
            }

        return tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_tensors)

    params = {}
    params.update(_flag.__dict__)
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=str(_flag.checkpoint_dir),
        params=params
    )

    model.export_savedmodel(str(_flag.model_dir), serving_input_receiver_fn())


def main(_flags):
    export_model(_flags)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.FATAL)
    tf.app.run(main=main, argv=[flags])
