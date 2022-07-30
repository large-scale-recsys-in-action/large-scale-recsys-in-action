# -*- coding: utf-8 -*-
import sklearn.metrics as sk_metrics


class AUC:
    def __init__(self, labels, predictions, threshold_num):
        """
        :param labels: list, 只有 0 和 1
        :param predictions: list, 形状与 labels 相同, 概率
        :param threshold_num: 阈值个数
        :return: AUC 对象
        """
        self._labels = labels
        self._predictions = predictions
        self._threshold_num = threshold_num
        assert len(labels) == len(predictions), \
            f'labels len: {len(labels)} != predictions len: {len(predictions)}'
        assert threshold_num > 0, 'threshold_num has to be positive'

    # 面积法
    def trapezoidal_auc(self):

        # 阈值从大到小排列
        thresholds = [(self._threshold_num - i) / self._threshold_num
                      for i in range(self._threshold_num + 1)]
        tpr_fpr = []

        # 正类个数
        p = sum(self._labels)
        # 负类个数
        n = len(self._labels) - p

        for threshold in thresholds:
            this_tp = 0
            this_fp = 0
            for label, prediction in zip(self._labels, self._predictions):
                if prediction >= threshold:
                    if label > 0:
                        this_tp += 1
                    else:
                        this_fp += 1
            tpr = this_tp / p
            fpr = this_fp / n
            # 添加 tpr, fpr 坐标点
            tpr_fpr.append((tpr, fpr))

        _auc = 0
        for i in range(1, len(tpr_fpr)):
            tpr_1, fpr_1 = tpr_fpr[i - 1]
            tpr_2, fpr_2 = tpr_fpr[i]
            # (上底 + 下底) * 高 / 2
            _auc += (tpr_1 + tpr_2) * (fpr_2 - fpr_1) / 2

        return _auc

    # 概率法
    def probabilistic_auc(self):
        # 正类的排序位置
        p_ranks = [i for i in range(len(self._labels)) if self._labels[i] == 1]
        # 负类的排序位置
        n_ranks = [i for i in range(len(self._labels)) if self._labels[i] == 0]
        # 正类个数
        m = len(p_ranks)
        # 负类个数
        n = len(n_ranks)

        # 正类概率大于等于负类概率的个数
        num_p_ge_n = 0.0
        for p_rank in p_ranks:
            for n_rank in n_ranks:
                p_p = self._predictions[p_rank]
                p_n = self._predictions[n_rank]
                if p_p > p_n:
                    num_p_ge_n += 1.0
                elif p_p == p_n:
                    num_p_ge_n += 0.5

        return num_p_ge_n / (m * n)

    def validate(self):
        _trapezoidal_auc = self.trapezoidal_auc()
        _prob_auc = self.probabilistic_auc()
        _sklearn_auc = self._sklearn_auc()

        assert _trapezoidal_auc == _sklearn_auc, \
            f'trapezoidal_auc: {_trapezoidal_auc} != sklearn_auc: {_sklearn_auc}'
        assert _prob_auc == _sklearn_auc, \
            f'probabilistic_auc: {_prob_auc} != sklearn_auc: {_sklearn_auc}'

    # sklearn 作为手动计算 auc 的校验工具
    def _sklearn_auc(self):
        # 调用 sklearn api, 获得 fpr 和 tpr, 两个返回值均为数组形式
        fpr, tpr, _ = sk_metrics.roc_curve(self._labels, self._predictions, pos_label=1)
        # 调用 sklearn api, 获得 auc
        _auc = sk_metrics.auc(fpr, tpr)
        return _auc


if __name__ == '__main__':
    _labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    _predictions = [0.3, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4, 0.1, 0.2, 0.0]

    auc = AUC(_labels, _predictions, threshold_num=100)
    # trapezoidal: 0.5714285714285714
    print('trapezoidal: {}'.format(auc.trapezoidal_auc()))
    # probabilistic: 0.5714285714285714
    print('probabilistic: {}'.format(auc.probabilistic_auc()))
    # pass validation
    auc.validate()
