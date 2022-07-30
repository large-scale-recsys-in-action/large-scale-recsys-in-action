# -*- coding: utf-8 -*-

import re
import math
import random

"""
python: 3.6
"""


class FM:
    def __init__(self, params):
        self._params = params
        assert 'k' in self._params, 'k has to be set.'
        self._k = self._params['k']
        self._w_0 = 0
        self._w = {}
        self._v = {}
        random.seed(123)

    def __str__(self):
        return f'w_0: {self._w_0}, w: {self._w}, v: {self._v}'

    def _input_fn(self, mode, data):
        assert mode in ('train', 'predict'), f'mode only support train or predict, but get {mode}'
        epochs = self._params.get('epochs', 1) if mode == 'train' else 1
        for line in data * epochs:
            label_features = re.split('\\s+', line)
            label = self._label_convert(float(label_features[0])) if mode == 'train' else None
            feature_values = label_features[1:] if mode == 'train' else label_features[0:]
            features = {}
            for feature_value in feature_values:
                feature, value = feature_value.split(':')
                features[feature] = float(value)

            yield label, features

    # sigmoid 函数
    @staticmethod
    def _sigmoid(x):
        x = min(max(x, -35), 35)
        return 1 / (1 + math.exp(-x))

    # 符号函数
    @staticmethod
    def _label_convert(x):
        return 1 if x > 0 else -1

    # 预测函数, 公式 (11)
    def _predict(self, features):
        wx_plus_b = self._w_0
        _sum = [0.0] * self._k
        square_sum = [0.0] * self._k

        for feature, value in features.items():
            w_i = self._w.get(feature, 0.0)
            wx_plus_b += w_i * value
            v = self._v.get(feature, [0.0] * self._k)
            for f in range(self._k):
                v_f_x = v[f] * value
                square_sum[f] += v_f_x ** 2
                _sum[f] += v_f_x

        # 预测值, 公式 (11)
        prediction = wx_plus_b + 0.5 * sum([_sum[i] ** 2 - square_sum[i] for i in range(self._k)])

        # 返回 _sum, 因为公式 (14) 需要求和项
        return prediction, _sum

    def predict(self, data):
        _predictions = []
        data_set = self._input_fn('predict', data)
        for _, features in data_set:
            prediction, _ = self._predict(features)
            _predictions.append(self._sigmoid(prediction))

        return _predictions

    # 交叉熵损失
    def _loss(self, prediction, label):
        return -math.log(self._sigmoid(prediction * label))

    def fit(self, data):
        # 1.输入数据
        data_set = self._input_fn('train', data)
        learning_rate = self._params.get('learning_rate', 0.01)
        mu = self._params.get('mu', 0.0)
        sigma = self._params.get('sigma', 0.1)

        steps = 0
        for label, features in data_set:
            steps += 1
            # 2.读取参数 和 3.数学运算
            prediction, _sum = self._predict(features)
            # 4.计算损失 和 5.求导更新参数
            # 公式 (13)
            g_constant = (self._sigmoid(prediction * label) - 1) * label

            # 更新 w_0, 公式 (14)
            self._w_0 = self._w_0 - learning_rate * g_constant
            for feature, value in features.items():
                w = self._w.get(feature, 0.0)
                # 更新 w, 公式 (14)
                self._w[feature] = w - learning_rate * g_constant * value

                # 更新 v, v 初始化为服从正态分布的随机变量, 公式 (14)
                v = (self._v.setdefault(feature,
                                        [random.normalvariate(mu, sigma) for _ in range(self._k)]))
                for f in range(self._k):
                    v_f = v[f]
                    self._v[feature][f] = (v_f - learning_rate * g_constant *
                                           (value * _sum[f] - v_f * value * value))

            if steps and not steps % 1000:
                loss = self._loss(prediction, label)
                print(f'loss = {loss}, step = {steps}')

        return self


if __name__ == '__main__':
    hyper_parameters = {'learning_rate': 0.05, 'k': 4, 'epochs': 200}
    # 格式: label feature1:value1 feature2:value2
    data_train = [
        "0 item_id2:1 user_id2:1",
        "1 item_id1:1 user_id1:1",
        "0 item_id2:1 user_id1:1",
        "0 item_id1:1 user_id2:1",
        "1 item_id1:1 user_id1:1"
    ]

    fm = FM(hyper_parameters)
    model = fm.fit(data_train)

    data_predict = [
        "item_id2:1 user_id2:1",
        "item_id1:1 user_id1:1",
        "item_id2:1 user_id1:1",
        "item_id1:1 user_id2:1",
        "item_id1:1 user_id1:1"
    ]
    predictions = model.predict(data_predict)

    # [0.014356894809136269,
    #  0.9897084711625975,
    #  0.021539235148611335,
    #  0.019228125772935124,
    #  0.9897084711625975]
    print(predictions)
    # w_0: -1.0629511974149788
    # w: {
    #   'item_id2': -1.999285268270898,
    #   'user_id2': -1.9196686596353496,
    #   'item_id1': 0.9363340708559196,
    #   'user_id1': 0.8567174622203717
    #   }
    # v: {...}
    print(model)
