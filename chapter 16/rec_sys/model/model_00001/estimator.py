# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.estimator.estimator import MetaEstimator


class Estimator(MetaEstimator):
    def __init__(self, features, labels, mode, h_params):
        super().__init__(features, labels, mode, h_params)
        self._attention_units = [8, 4]
        self._fc_units = [8, 4, 1]

    def model_fn(self):
        with tf.name_scope('user'):
            u_id = self.input_layer([1], embedding_size=16)
            u_other = self.input_layer([2, 3], embedding_size=4)
            user = tf.concat([u_id, u_other], axis=-1)

        with tf.name_scope('context'):
            context = self.input_layer([4], embedding_size=4)

        with tf.name_scope('item'):
            self._features[self._slot_name(5)] = tf.reshape(self._features[self._slot_name(5)], [-1, 1])
            item_embedding = self.look_up(5)
            item_embedding = tf.squeeze(item_embedding, axis=1)

        with tf.name_scope('interactions'):
            clicks_embeddings = self.look_up(6)
            clicks_mask = tf.not_equal(self._features[self._slot_name(6)], b'0')

        with tf.name_scope('x_features'):
            x_features = self.input_layer([(2, 4),
                                           (3, 4),
                                           (2, 3, 4)], bucket_size=100)

        with tf.name_scope('attention'):
            attention = self.attention(history_emb=clicks_embeddings,
                                       current_emb=item_embedding,
                                       history_masks=clicks_mask,
                                       units=self._attention_units,
                                       name='attention')

        fc_inputs = [user, context, attention, x_features, item_embedding]
        fc_inputs = tf.concat(fc_inputs, axis=-1, name='fc_inputs')
        logits = self.fully_connected_layers(mode=self._mode,
                                             net=fc_inputs,
                                             units=self._fc_units,
                                             dropout=0.3,
                                             name='logits')
        predictions = tf.sigmoid(logits, name='predictions')
        if self._mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'predictions': tf.reshape(predictions, [-1, 1])}
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(self._mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)
        else:
            labels = tf.reshape(self._labels, [-1, 1])
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)

            if self._mode == tf.estimator.ModeKeys.EVAL:
                metrics = {'auc': tf.metrics.auc(labels=labels,
                                                 predictions=predictions,
                                                 num_thresholds=500)}
                return tf.estimator.EstimatorSpec(self._mode,
                                                  loss=loss,
                                                  eval_metric_ops=metrics)
            else:
                global_step = tf.train.get_global_step()
                learning_rate = self.exponential_decay(global_step)
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(self._mode,
                                                  loss=loss,
                                                  train_op=train_op)
