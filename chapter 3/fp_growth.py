# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.window import Window
from pyspark.sql.functions import col, split, size, rank

"""
spark: 2.4.0
python: 3.6
"""


class FPG:
    def __init__(self,
                 spark,
                 table,
                 items_col='items',
                 min_support_count=3,
                 min_confidence=0.1,
                 top_n=10,
                 partitions=2000):
        self._spark = spark
        self._table = table
        self._items_col = items_col
        self._min_support_count = min_support_count
        self._min_confidence = min_confidence
        self._top_n = top_n
        self._partitions = partitions

    def _dataset(self):
        return (self._spark.sql(f'select {self._items_col} from {self._table}')
                .select(split(self._items_col, '\\s+').alias("items")))

    @property
    def rules(self):
        dataset = self._dataset()
        transactions_count = dataset.count()
        fp = FPGrowth(minSupport=self._min_support_count * 1.0 / transactions_count,
                      minConfidence=self._min_confidence,
                      itemsCol="items",
                      numPartitions=self._partitions)
        fpm = fp.fit(dataset)

        association_rules = (fpm.associationRules
                             # 只保留长度为 1 的结果
                             .filter((size("antecedent") == 1) & (size("consequent") == 1))
                             .withColumn('antecedent', col("antecedent")[0])
                             .withColumn('consequent', col('consequent')[0]))
        window = Window.partitionBy(association_rules.antecedent).orderBy(association_rules.lift.desc())
        association_rules = (association_rules.select('*', rank().over(window).alias('rank'))
                             .filter(col('rank') <= self._top_n)
                             .select("antecedent", "consequent", "lift"))
        return association_rules


if __name__ == '__main__':
    spark_session = (SparkSession.builder.appName('fpgrowth').master('yarn')
                     .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
                     .config('spark.network.timeout', '600')
                     .config('spark.driver.maxResultSize', '5g')
                     .enableHiveSupport().getOrCreate())

    fpg = FPG(spark_session, table='recsys.data_fpgrowth')
    rules = fpg.rules
    rules.write.mode("overwrite").saveAsTable("recsys.model_fpgrowth")
    spark_session.stop()
