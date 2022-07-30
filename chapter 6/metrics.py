# -*- coding: utf-8 -*-
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType, LongType

"""
spark: 2.4.0
python: 3.6
"""


class Metrics:
    def __init__(self, spark, ground_truth_table, i2i_table,
                 user_col="user", item_col="item",
                 relevance_col="relevance", timestamp_col="timestamp",
                 item1_col="item1", item2_col="item2", score_col="score"):
        self._spark = spark
        self._ground_truth_table = ground_truth_table
        self._i2i_table = i2i_table
        self._user_col = user_col
        self._item_col = item_col
        self._relevance_col = relevance_col
        self._timestamp_col = timestamp_col
        self._item1_col = item1_col
        self._item2_col = item2_col
        self._score_col = score_col

    def _get_ground_truth_dataset(self):
        return (self._spark.sql(
            f'select {self._user_col}, '
            f'{self._item_col}, '
            f'{self._relevance_col}, '
            f'{self._timestamp_col} from '
            f'{self._ground_truth_table}'))

    def _get_i2i_dataset(self):
        return (self._spark.sql(
            f'select {self._item1_col}, '
            f'{self._item2_col}, '
            f'{self._score_col}, from '
            f'{self._i2i_table}'))

    def _label_predictions(self):
        def _list_append(acc, element):
            acc.append(element)
            return acc

        def _list_merge(acc1, acc2):
            acc1.extend(acc2)
            return acc1

        user = col(self._user_col).alias('user')
        item = col(self._item_col).alias('item')
        relevance = col(self._relevance_col).cast(DoubleType()).alias('relevance')
        timestamp = col(self._timestamp_col).cast(LongType()).alias('timestamp')
        ground_truth = self._get_ground_truth_dataset()
        ground_truth = (ground_truth.select(user, item, relevance, timestamp)
                        .rdd
                        .map(lambda row: (row.item, (row.user, row.relevance, row.timestamp))))

        item1 = col(self._item1_col).alias('item1')
        item2 = col(self._item2_col).alias('item2')
        score = col(self._score_col).cast(DoubleType()).alias('score')
        # 获取算法生成的物品相似/相关表
        i2i = self._get_i2i_dataset()
        # 根据 item1 进行聚合, 得到所有与之相似/相关的物品, 一般是 top n 个
        i2i = (i2i.select(item1, item2, score)
               .rdd
               .map(lambda row: (row.item1, (row.item2, row.score)))
               .aggregateByKey([], _list_append, _list_merge))

        def _rearrange(record):
            _item1, ((_user, _relevance, _timestamp), _item2_and_rating) = record
            if not _item2_and_rating:
                _item2_and_rating = []
            return _user, ((_item1, _relevance, _timestamp), _item2_and_rating)

        def _single_user_metrics(user_recs):
            # user_recs 格式: (user_id, ((truth, relevance, timestamp), [(rec1, rating1), (rec2, rating2) ...]))
            _, recs = user_recs
            sort_by_timestamp = sorted(recs, key=lambda r: r[0][2])

            truths = []
            # 注意: 取后 n-1 个 truth
            for t in sort_by_timestamp[1:]:
                (truth, rel, _), _ = t
                truths.append((truth, rel))

            recs = []
            # 注意: 取前 n-1 个 recs
            for t in sort_by_timestamp[:-1]:
                _, rec_and_ratings = t
                # 根据得分降序排列
                rec_and_ratings = sorted(rec_and_ratings, key=lambda r: r[1], reverse=True)
                recs.extend([rec[0] for rec in rec_and_ratings])

            return zip(truths, recs)

        def _valid_records(user_recs):
            _, recs = user_recs
            return recs

        return (ground_truth.leftOuterJoin(i2i)
                .map(_rearrange)
                .aggregateByKey([], _list_append, _list_merge)
                .filter(_valid_records)
                .flatMap(_single_user_metrics))

    @staticmethod
    def _calc_metrics(ground_truth, relevance, recs):
        def log2(x):
            return math.log(x) / math.log(2)

        index = recs.index(ground_truth) if ground_truth in recs else -1

        if index >= 0:  # 说明命中了
            index += 1  # 计算 ndcg 时, index 从 1 开始
            dcg = (math.pow(2, relevance) - 1) / log2(index + 1)
            idcg = (math.pow(2, relevance) - 1) / log2(1 + 1)
            ndcg = dcg / idcg
            mrr = 1.0 / index
            tp = 1
            tpfn = 1
            tpfp = len(recs)
            return ndcg, mrr, tp, tpfn, tpfp
        else:
            return 0.0, 0.0, 0, 1, len(recs)

    @staticmethod
    def _merge_metrics(metrics1, metrics2):
        ndcg1, mrr1, tp1, tpfn1, tpfp1 = metrics1
        ndcg2, mrr2, tp2, tpfn2, tpfp2 = metrics2
        return ndcg1 + ndcg2, mrr1 + mrr2, tp1 + tp2, tpfn1 + tpfn2, tpfp1 + tpfp2

    @staticmethod
    def _final_metric(metric):
        ndcg, mrr, tp, tpfn, tpfp = metric
        precision = tp * 1.0 / tpfp
        recall = tp * 1.0 / tpfn
        f1 = 2 * precision * recall / (precision + recall + 1E-8)
        mrr = mrr / tpfn
        ndcg = ndcg / tpfn
        return precision, recall, f1, mrr, ndcg

    def metrics_at(self, ks):
        def k_metrics(record):
            (truth, relevance), recs = record
            result = []
            for k in ks:
                top_k = recs[:k]
                this_metric = Metrics._calc_metrics(truth, relevance, top_k)
                result.append((k, this_metric))
            return result

        label_predictions = self._label_predictions()
        # [(k, metrics), (k, metrics)]
        metrics_at_k = label_predictions.flatMap(k_metrics).reduceByKey(self._merge_metrics).collect()
        # 根据 k 值升序排列
        metrics_at_k = sorted(metrics_at_k, key=lambda m: m[0])
        metrics_at_k = [(k, self._final_metric(metric)) for (k, metric) in metrics_at_k]
        return metrics_at_k


if __name__ == '__main__':
    spark_session = (SparkSession.builder.appName('metrics').master('yarn')
                     .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
                     .config('spark.network.timeout', '600')
                     .config('spark.driver.maxResultSize', '5g')
                     .enableHiveSupport().getOrCreate())

    metrics = Metrics(spark=spark_session,
                      ground_truth_table="recsys.user_behaviour",
                      i2i_table="recsys.i2i_table")
    # [(k1, (precision1, recall1, f11, mrr1, ndcg1)), (k2, (precision2, recall2, f12, mrr2, ndcg2))]
    metrics = metrics.metrics_at([1, 2, 4, 8])
    spark_session.stop()
