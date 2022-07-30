# -*- coding: utf-8 -*-
import math
import itertools
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import countDistinct, rank, col

"""
spark: 2.4.0
python: 3.6
"""


class ItemCF:
    def __init__(self, spark, table,
                 user_col='user', item_col='item', rating_col='rating',
                 top_n=10, lower=2, upper=3000):
        self._spark = spark
        self._table = table
        self._user_col = user_col
        self._item_col = item_col
        self._rating_col = rating_col
        self._top_n = top_n
        self._lower = lower
        self._upper = upper

    def _data_set(self):
        return self._spark.sql(f'select {self._user_col}, {self._item_col}, {self._rating_col} from {self._table}')

    def _clean_data_set(self):
        # 过滤无效用户
        data_set = self._data_set()
        invalid_users = (data_set
                         .groupBy(self._user_col).agg(countDistinct(self._item_col).alias('item_count'))
                         .filter(f'item_count < {self._lower} and item_count > {self._upper}')
                         .select(self._user_col).collect())

        invalid_users = [record.user for record in invalid_users]
        invalid_users = self._spark.sparkContext.broadcast(invalid_users)
        data_set = data_set.filter(~data_set.user.isin(invalid_users.value))

        # 物品的流行度以消费人数衡量
        item_count = (data_set
                      .groupBy(self._item_col).agg(countDistinct(self._user_col).alias('user_count'))
                      .select(self._item_col, 'user_count')
                      .rdd
                      .collectAsMap())
        item_count = self._spark.sparkContext.broadcast(item_count)

        def _collection_append(collection, element):
            collection.append(element)
            return collection

        def _collection_merge(collection1, collection2):
            collection1.extend(collection2)
            return collection1

        data_set = (data_set
                    .rdd
                    .map(lambda record: (record.user, (record.item, record.rating)))
                    .aggregateByKey([],
                                    _collection_append,
                                    _collection_merge)
                    # record: [[user, items], [user, items]]
                    # 至少消费 2 个物品
                    .filter(lambda record: len(record[1]) >= 2))

        return item_count, data_set

    # RDD[(str, str, float)]
    @property
    def similarities(self):
        item_count, data_set = self._clean_data_set()

        def _sim_from_one_user(user_item_ratings):
            user, item_ratings = user_item_ratings
            size = len(item_ratings)
            for (item_rating1, item_rating2) in itertools.combinations(item_ratings, 2):
                item1, rating1 = item_rating1
                item2, rating2 = item_rating2

                # 惩罚热门用户
                local_similarity = 1 / (1 + math.fabs(rating1 - rating2)) / math.log1p(size)

                yield (item1, item2), local_similarity

        item_sim_from_one_user = data_set.flatMap(_sim_from_one_user)

        def _intra_partition(sim_sum_count, local_sim):
            sim_sum, count = sim_sum_count
            return sim_sum + local_sim, count + 1

        def _inter_partition(sim_sum_count1, sim_sum_count2):
            sim_sum1, count1 = sim_sum_count1
            sim_sum2, count2 = sim_sum_count2
            return sim_sum1 + sim_sum2, count1 + count2

        def _item_sim(pair_sim_sum_count):
            (item1, item2), (sim_sum, count) = pair_sim_sum_count
            item_count1 = item_count.value[item1]
            item_count2 = item_count.value[item2]

            # 惩罚热门物品
            sim = sim_sum / count / math.sqrt(item_count1 * item_count2)
            return (item1, item2, sim), (item2, item1, sim)

        item_similarity = (item_sim_from_one_user
                           .aggregateByKey([0.0, 0],
                                           _intra_partition,
                                           _inter_partition)
                           .flatMap(_item_sim)
                           .toDF(["item1", "item2", "sim"]))

        window = Window.partitionBy(item_similarity.item1).orderBy(item_similarity.sim.desc())
        item_top_n_similarity = (item_similarity
                                 .select('*', rank().over(window).alias('rank'))
                                 .filter(col('rank') <= self._top_n)
                                 .select("item1", "item2", "sim"))

        return item_top_n_similarity


if __name__ == '__main__':
    spark_session = (SparkSession.builder.appName('item_cf').master('yarn')
                     .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
                     .config('spark.network.timeout', '600')
                     .config('spark.driver.maxResultSize', '5g')
                     .enableHiveSupport().getOrCreate())

    item_cf = ItemCF(spark=spark_session, table='recsys.data_itemcf')
    similarities = item_cf.similarities
    similarities.write.mode("overwrite").saveAsTable("recsys.model_itemcf")
    spark_session.stop()
