# -*- coding: utf-8 -*-
import tensorflow as tf

# sess = tf.InteractiveSession()
#
#
# def get_lr(lr):
#     sess.run(tf.global_variables_initializer())
#     lr = sess.run(lr)
#     return lr
#
#
# learning_rate = 0.1
# decay_steps = 10000
# decay_rate = 0.9
# global_steps = [0, 10000, 10001, 20000, 20001]

# _lr = [get_lr(tf.train.inverse_time_decay(learning_rate,
#                                           global_step,
#                                           decay_steps,
#                                           decay_rate)) for global_step in global_steps]

# _lr = [get_lr(tf.train.exponential_decay(learning_rate,
#                                          global_step,
#                                          decay_steps,
#                                          decay_rate)) for global_step in global_steps]
# _lr = [get_lr(tf.train.polynomial_decay(learning_rate,
#                                         global_step,
#                                         decay_steps,
#                                         end_learning_rate=0.0001,
#                                         power=1.0,
#                                         cycle=True)) for global_step in global_steps]
import numpy as np


# def plateau_decay(_learning_rate,
#                   global_step,
#                   loss,
#                   factor=0.1,
#                   patient_steps=10,
#                   min_delta=1e-4,
#                   cooldown_steps=0,
#                   min_lr=0.0001):
#     if not isinstance(_learning_rate, tf.Tensor):
#         _learning_rate = tf.get_variable('learning_rate', initializer=tf.constant(learning_rate), trainable=False)
#
#     with tf.variable_scope('plateau_decay'):
#         step = tf.get_variable('step', trainable=False, initializer=global_step)
#         best = tf.get_variable('best', trainable=False, initializer=tf.constant(np.Inf, tf.float32))
#
#         def _update_best():
#             with tf.control_dependencies([
#                 tf.assign(best, loss),
#                 tf.assign(step, global_step),
#                 tf.print('Plateau Decay: Updated Best - Step:', global_step, 'Next Decay Step:',
#                          global_step + patient_steps, 'Loss:', loss)
#             ]):
#                 return tf.identity(learning_rate)
#
#         def _decay():
#             with tf.control_dependencies([
#                 tf.assign(best, loss),
#                 tf.assign(learning_rate, tf.maximum(tf.multiply(learning_rate, factor), min_lr)),
#                 tf.assign(step, global_step + cooldown_steps),
#                 tf.print('Plateau Decay: Decayed LR - Step:', global_step, 'Next Decay Step:',
#                          global_step + cooldown_steps + patient_steps, 'Learning Rate:', learning_rate)
#             ]):
#                 return tf.identity(learning_rate)
#
#         def _no_op(): return tf.identity(learning_rate)
#
#         met_threshold = tf.less(loss, best - min_delta)
#         should_decay = tf.greater_equal(global_step - step, patient_steps)
#
#         return tf.cond(met_threshold, _update_best, lambda: tf.cond(should_decay, _decay, _no_op))

def reduce_lr_on_plateau(learning_rate,
                         global_step,
                         decay_steps,
                         decay_rate,
                         auc,
                         patient_steps=10000,
                         cooldown_steps=5000,
                         min_delta=1e-4,
                         min_lr=0.0001):
    if not isinstance(learning_rate, tf.Tensor):
        learning_rate = tf.get_variable('learning_rate',
                                        initializer=tf.constant(learning_rate),
                                        trainable=False)

    def exponential_decay(lr):
        return tf.train.exponential_decay(lr, global_step, decay_steps, decay_rate)

    with tf.variable_scope('reduce_lr_on_plateau'):
        step = tf.get_variable('step',
                               trainable=False,
                               initializer=global_step)
        best = tf.get_variable('best',
                               trainable=False,
                               initializer=tf.constant(0.0, tf.float32))

        def _update_best():
            with tf.control_dependencies([tf.assign(best, auc),
                                          tf.assign(step, global_step)]):
                return tf.identity(learning_rate)

        def _decay():
            with tf.control_dependencies(
                    [tf.assign(best, auc),
                     tf.assign(learning_rate,
                               tf.maximum(exponential_decay(learning_rate), min_lr)),
                     tf.assign(step, global_step + cooldown_steps)]):
                return tf.identity(learning_rate)

        def _no_op(): return tf.identity(learning_rate)

        met_threshold = tf.greater(auc, best + min_delta)
        should_decay = tf.greater_equal(global_step - step, patient_steps)

        return tf.cond(met_threshold,
                       _update_best,
                       lambda: tf.cond(should_decay, _decay, _no_op))
