# -*- coding: utf-8 -*-
import tensorflow as tf
from lib.feature.feature_builder import FeatureBuilder
from lib.common.ranking_metrics import metrics_impl


class Estimator:
    def __init__(self, features, labels, mode, params):
        self._features = features
        self._labels = labels
        self._mode = mode
        self._params = params
        self._fb = FeatureBuilder()
        self._attention_units = [8, 4]
        self._fc_units = [8, 4, 1]
        self._rank_discount_fn = lambda rank: tf.math.log(2.) / tf.math.log1p(rank)

    def model_fn(self):
        with tf.name_scope('user'):
            user_fc = self._fb.user_features()
            user = tf.feature_column.input_layer(self._features, user_fc)  # B * E_user

        with tf.name_scope('context'):
            context_fc = self._fb.context_features()
            context = tf.feature_column.input_layer(self._features, context_fc)  # B * E_contextual

        with tf.name_scope('item'):
            # item_embedding: B * L * E_item
            # clicks_embedding: B * S * E_item
            item_embedding, clicks_embedding = self._fb.item_and_histories_features(self._features)
            # clicks_mask: B * S
            clicks_mask = tf.not_equal(self._features['clicks'], b'0')  # pad 的是 b'0'

        # user 特征和 contextual 特征复制 L 份
        # L 等于物品特征的第二维
        # S 等于历史行为序列的第二维
        list_size = tf.shape(input=item_embedding)[1]
        time_steps = tf.shape(input=clicks_embedding)[1]
        item_embedding_size = clicks_embedding.get_shape().as_list()[-1]

        # user: B * E_user, 需要在第二维新增一维, 并在新增的维度上复制 L 份
        # contextual 特征同理
        user = tf.expand_dims(user, axis=1)
        user = tf.tile(user, [1, list_size, 1])

        context = tf.expand_dims(context, axis=1)
        context = tf.tile(context, [1, list_size, 1])

        # history sequence: B * S * E_item, 需要变为 B * L * S * E_item
        # history mask: B * S 同理需要变为 B * L * S
        # B * (L * S) * E_item
        clicks_embedding = tf.tile(clicks_embedding, [1, list_size, 1])
        # B * L * S * E_item
        clicks_embedding = tf.reshape(clicks_embedding, [-1,
                                                         list_size,
                                                         time_steps,
                                                         item_embedding_size])

        # B * (L * S)
        clicks_mask = tf.tile(clicks_mask, [1, list_size])
        # B * L * S
        clicks_mask = tf.reshape(clicks_mask, [-1, list_size, time_steps])

        # item: B * L * E_item, 需要变为 B * L * S * E_item
        # B * (L * S) * E_item
        item_embedding_temp = tf.tile(item_embedding, [1, time_steps, 1])
        # B * L * S * E_item
        item_embedding_temp = tf.reshape(item_embedding_temp, [-1,
                                                               list_size,
                                                               time_steps,
                                                               item_embedding_size])

        with tf.name_scope('user_behaviour_sequence'):
            # B * L * E_item
            attention = self.attention(history_emb=clicks_embedding,
                                       current_emb=item_embedding_temp,
                                       history_masks=clicks_mask,
                                       units=self._attention_units,
                                       name='attention')

        # B * L * (E_user + E_contextual + E_item + E_item)
        fc_inputs = [user, context, attention, item_embedding]

        fc_inputs = tf.concat(fc_inputs, axis=-1, name='fc_inputs')

        logits = self.fully_connected_layers(mode=self._mode,
                                             net=fc_inputs,
                                             units=self._fc_units,
                                             dropout=0.3,
                                             name='logits')
        # B * L
        logits = tf.squeeze(logits, axis=-1)

        if self._mode == tf.estimator.ModeKeys.PREDICT: 
            probability = tf.nn.softmax(logits, name='predictions')  # B * L
            predictions = {
                'predictions': tf.reshape(probability, [-1, 1])
            }
         
            export_outputs = {
                'predictions': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(self._mode,
                                              predictions=predictions,
                                              export_outputs=export_outputs)
        else: 
            relevance = tf.cast(self._labels, tf.float32)
            # 1. 这里使用 softmax 将 relevance 转化为概率
            # 因此 relevance 的默认值和 pad 值必须是一个很小的值
            soft_max = tf.nn.softmax(relevance, axis=-1)
            mask = tf.cast(relevance >= 0.0, tf.bool)

            """Softmax cross-entropy loss with masking."""
            # 2. 求 loss
            padding = tf.ones_like(logits) * -2 ** 32
            logits = tf.where(mask, logits, padding)
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=relevance))

            if self._mode == tf.estimator.ModeKeys.EVAL:
                # 3. 计算 ndcg 时需要剔除掉 pad 的数据
                gauc_labels = tf.cast(relevance > 0.0, tf.float32)
                weights = tf.cast(mask, tf.float32)

                metrics = {
                    'gauc': tf.metrics.auc(labels=gauc_labels,
                                           predictions=soft_max,
                                           weights=weights,
                                           num_thresholds=1000)
                }

                metrics.update(self.ndcg(relevance,
                                         logits,
                                         weights=weights,
                                         name='ndcg'))
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                return tf.estimator.EstimatorSpec(self._mode,
                                                  loss=loss,
                                                  eval_metric_ops=metrics)
            else: 
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
        param:history_emb: 历史行为 embedding, 形状: Batch Size * List Size * Time Steps * Embedding Size
        param:current_emb: 候选物品 embedding, 形状: Batch Size * List Size * Time Steps * Embedding Size
        param:history_masks: 历史行为 mask, pad 的信息不能投入计算, Batch Size * List Size * Time Steps
        param:units: list of hidden unit num
        param:name: output name
        param:weighted sum attention output
        """
        net = tf.concat([history_emb,
                         history_emb - current_emb,
                         current_emb,
                         history_emb * current_emb,
                         history_emb + current_emb],
                        axis=-1)
        for unit in units:
            net = tf.layers.dense(net, units=unit, activation=tf.nn.relu)
        # B * L * S * 1
        weights = tf.layers.dense(net, units=1, activation=None)
        # B * L * 1 * S
        weights = tf.transpose(weights, [0, 1, 3, 2])
        padding = tf.zeros_like(weights)
        # B * L * S --> B * L * 1 * S
        history_masks = tf.expand_dims(history_masks, axis=2)
        weights = tf.where(history_masks, weights, padding)
        # [B * L * 1 * S] * [B * L * S * E] --> [B * L * 1 * E]
        outputs = tf.matmul(weights, history_emb)
        # B * L * E
        outputs = tf.squeeze(outputs, axis=2, name=name)
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

    def ndcg(self, relevance, predictions, ks=(1, 4, 8, 20, None), weights=None, name='ndcg'):
        ndcgs = {}
        for k in ks:
            metric = metrics_impl.NDCGMetric('ndcg',
                                             topn=k,
                                             gain_fn=lambda label: tf.pow(2.0, label) - 1,
                                             rank_discount_fn=self._rank_discount_fn)

            with tf.name_scope(metric.name,
                               'normalized_discounted_cumulative_gain',
                               (relevance, predictions, weights)):
                per_list_ndcg, per_list_weights = metric.compute(relevance, predictions, weights)

            ndcgs.update({'{}_{}'.format(name, k): tf.metrics.mean(per_list_ndcg, per_list_weights)})

        return ndcgs
