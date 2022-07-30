# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.feature.feature_builder import FeatureBuilder


class Estimator:
    def __init__(self, features, labels, mode, params):
        self._features = features
        self._labels = labels
        self._mode = mode
        self._params = params
        self._fb = FeatureBuilder()
        self._attention_units = [8, 4]
        self._fc_units = [8, 4, 1]

    def model_fn(self):
        with tf.name_scope('user'):
            user_fc = self._fb.user_features()
            user = tf.feature_column.input_layer(self._features, user_fc)

        with tf.name_scope('context'):
            context_fc = self._fb.context_features()
            context = tf.feature_column.input_layer(self._features, context_fc)

        with tf.name_scope('item'):
            self._features['item_id'] = tf.reshape(self._features['item_id'], [-1, 1])
            item_embedding, clicks_embedding = self._fb.item_and_histories_features(self._features)
            item_embedding = tf.squeeze(item_embedding, axis=1)
            clicks_mask = tf.not_equal(self._features['clicks'], b'0')  # pad 的是 b'0'

        if self._mode == tf.estimator.ModeKeys.PREDICT:  # 0 与输入有关
            batch_size = tf.shape(input=item_embedding)[0]
            user = tf.tile(user, [batch_size, 1])
            context = tf.tile(context, [batch_size, 1])
            clicks_embedding = tf.tile(clicks_embedding, [batch_size, 1, 1])
            clicks_mask = tf.tile(clicks_mask, [batch_size, 1])

        with tf.name_scope('user_behaviour_sequence'):
            attention = self.attention(history_emb=clicks_embedding,
                                       current_emb=item_embedding,
                                       history_masks=clicks_mask,
                                       units=self._attention_units,
                                       name='attention')

        fc_inputs = [user, context, attention, item_embedding]

        fc_inputs = tf.concat(fc_inputs, axis=-1, name='fc_inputs')

        logits = self.fully_connected_layers(mode=self._mode,
                                             net=fc_inputs,
                                             units=self._fc_units,
                                             dropout=0.3,
                                             name='logits')
        probability = tf.sigmoid(logits, name='probability')

        if self._mode == tf.estimator.ModeKeys.PREDICT:  # 1. 这个分支对应线上推理阶段
            predictions = {
                'predictions': tf.reshape(probability, [-1, 1])
            }
            # 推理阶段直接返回预测概率
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(self._mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)
        else:  # 这个分支对应训练和验证阶段
            labels = tf.reshape(self._labels, [-1, 1])
            loss = tf.losses.sigmoid_cross_entropy(labels, logits)
            if self._mode == tf.estimator.ModeKeys.EVAL:  # 2. 这个分支对应验证阶段
                # 验证阶段输出离线指标
                metrics = {
                    'auc': tf.metrics.auc(labels=labels,
                                          predictions=probability,
                                          num_thresholds=1000)
                }
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                return tf.estimator.EstimatorSpec(self._mode,
                                                  loss=loss,
                                                  eval_metric_ops=metrics)
            else:  # 3. 这个分支对应训练阶段
                global_step = tf.train.get_global_step()
                learning_rate = self.exponential_decay(global_step)
                # 训练阶段通过梯度下降实现参数更新
                optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
                tf.summary.scalar('learning_rate', learning_rate)
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
                return tf.estimator.EstimatorSpec(self._mode,
                                                  loss=loss,
                                                  train_op=train_op)

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
    def attention(history_emb,
                  current_emb,
                  history_masks,
                  units,
                  name='attention'):
        """
        param:history_emb: 历史行为 embedding, 形状: Batch Size * List Size * Embedding Size
        param:current_emb: 候选物品 embedding, 形状: Batch Size * Embedding Size
        param:history_masks: 历史行为 mask, pad 的信息不能投入计算, Batch Size * List Size
        param:units: list of hidden unit num
        param:name: output name
        param:weighted sum attention output
        """
        list_size = tf.shape(history_emb)[1]
        embedding_size = current_emb.get_shape().as_list()[-1]
        current_emb = tf.tile(current_emb, [1, list_size])
        current_emb = tf.reshape(current_emb, shape=[-1, list_size, embedding_size])
        net = tf.concat([history_emb,
                         history_emb - current_emb,
                         current_emb,
                         history_emb * current_emb,
                         history_emb + current_emb],
                        axis=-1)
        for unit in units:
            net = tf.layers.dense(net, units=unit, activation=tf.nn.relu)
        weights = tf.layers.dense(net, units=1, activation=None)
        weights = tf.transpose(weights, [0, 2, 1])
        history_masks = tf.expand_dims(history_masks, axis=1)
        padding = tf.zeros_like(weights)
        weights = tf.where(history_masks, weights, padding)
        outputs = tf.matmul(weights, history_emb)
        outputs = tf.reduce_sum(outputs, 1, name=name)
        return outputs

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
