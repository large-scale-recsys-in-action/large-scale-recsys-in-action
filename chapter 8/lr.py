# -*- coding: utf-8 -*-
import re
import math

"""
python: 3.6
"""


class LogisticRegression:
    def __init__(self, params):
        self._params = params
        self._b = 0.0
        self._w = {}

    def __str__(self):
        return f'w: {self._w}, b: {self._b}'

    def _input_fn(self, mode, data):
        assert mode in ('train', 'predict'), f'mode only support train or predict, but get {mode}'
        epochs = self._params.get('epochs', 1) if mode == 'train' else 1
        for line in data * epochs:
            label_features = re.split('\\s+', line)
            if not label_features or len(label_features) < 2:
                continue
            label = float(label_features[0]) if mode == 'train' else None
            feature_values = label_features[1:] if mode == 'train' else label_features
            features = {}
            for feature_value in feature_values:
                feature, value = feature_value.split(':')
                features[feature] = float(value)

            yield label, features

    # sigmoid 函数
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # 符号函数
    @staticmethod
    def _sign(x):
        if x > 1e-6:
            return 1
        elif x > -1e-6:
            return 0
        else:
            return -1

    # 预测函数
    def _predict(self, features):
        wx_plus_b = self._b
        for feature, value in features.items():
            weight = self._w.get(feature, 0.0)
            wx_plus_b += weight * value
        # 预测值
        prediction = self._sigmoid(wx_plus_b)

        return prediction

    def predict(self, data):
        _predictions = []
        data_set = self._input_fn('predict', data)
        for _, features in data_set:
            prediction = self._predict(features)
            _predictions.append(prediction)

        return _predictions

    # 交叉熵损失
    @staticmethod
    def _loss(prediction, label):
        return -math.log(prediction) if label > 0 else -math.log(1 - prediction)

    def fit(self, data):
        # 1.输入数据
        data_set = self._input_fn('train', data)
        learning_rate = self._params.get('learning_rate', 0.01)
        lambda1 = self._params.get('lambda1', 0.0)
        lambda2 = self._params.get('lambda2', 0.0)
        steps = 0
        for label, features in data_set:
            steps += 1
            # 2.读取参数 和 3.数学运算
            prediction = self._predict(features)
            # 4.计算损失 和 5.求导更新参数
            g_bias = (prediction - label) + lambda1 * self._sign(self._b) + lambda2 * self._b
            self._b = self._b - learning_rate * g_bias
            for feature, value in features.items():
                w = self._w.get(feature, 0.0)
                # 公式 (8)
                g_w = (prediction - label) * value + lambda1 * self._sign(w) + lambda2 * w
                # 更新参数, SGD
                self._w[feature] = w - learning_rate * g_w

            if steps == 1 or (steps and not steps % 1000):
                loss = self._loss(prediction, label)
                print(f'loss = {loss}, step = {steps}')

        return self


if __name__ == '__main__':
    hyper_parameters = {'learning_rate': 0.05, 'lambda1': 0.01, 'lambda2': 0.01, 'epochs': 200}
    # 格式: label feature1:value1 feature2:value2
    data_train = [
        "0 item_id2:1 user_id2:1",
        "1 item_id1:1 user_id1:1",
        "0 item_id2:1 user_id1:1",
        "0 item_id1:1 user_id2:1",
        "1 item_id1:1 user_id1:1"
    ]

    lr = LogisticRegression(hyper_parameters)
    model = lr.fit(data_train)

    data_predict = [
        "item_id2:1 user_id2:1",
        "item_id1:1 user_id1:1",
        "item_id2:1 user_id1:1",
        "item_id1:1 user_id2:1",
        "item_id1:1 user_id1:1"
    ]
    predictions = model.predict(data_predict)

    # [0.004406334558092134, 
    #  0.8786431612197027, 
    #  0.15131339987498774, 
    #  0.15234633822452381, 
    #  0.8786431612197027]
    print(predictions)

    # w: {
    #     'item_id2': -2.467453798710349,
    #     'user_id2': -2.462844191694607,
    #     'item_id1': 1.2365265665362821,
    #     'user_id1': 1.2331150420070116
    #    }
    # b: -0.4899980396316112
    print(model)
