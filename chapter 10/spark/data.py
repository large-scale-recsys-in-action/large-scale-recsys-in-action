# -*- coding: utf-8 -*-

"""
文件名：data.py
这里因为要将数据保存在本地，所以 master 指定为 local, 同时指定 jars.
启动命令: spark-submit --master local --jars spark-tensorflow-connector_2.11-1.15.0.jar data.py
"""

from pyspark.sql.types import *
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ltr_dataset').getOrCreate()

# 保存在本地，可以换成 HDFS、S3 等分布式存储路径
path = "file:///home/recsys/chapter10/ltr/dataset"


def data(records):
    """
    records: 同一个 pv_id 下的数据集合, 格式为二维数组
             二维数组中每一行的元素及其格式为:
             单值    单值    单值  单值    单值    数组    单值      单值
             pv_id, user_id, age, gender, device, clicks, item_id, relevance
    """

    # 先拿到 relevances
    relevances = [record[7] for record in records]

    # 如果此 pv_id 下全是曝光数据, 此 list 丢弃
    if not any(relevances):
        return []

    pv_id = records[0][0]

    # records 中的 user 信息是一样的, 格式为单值
    user_id = records[0][1]
    age = records[0][2]
    gender = records[0][3]

    # records 中的 context 信息是一样的, 格式为单值
    device = records[0][4]

    # records 中的 user behaviour 信息是一样的, 格式为数组
    clicks = records[0][5]

    # records 中的 item 信息是不一样的, 格式为数组
    items = [record[6] for record in records]

    row = [pv_id, user_id, age, gender, device, clicks, items, relevances]
    return row


# 指定各字段类型
feature_names = [
    # pv id: 单值(scalar)
    StructField("pv_id", StringType()),

    # user: 单值(scalar)
    StructField("user_id", StringType()),
    StructField("age", LongType()),
    StructField("gender", StringType()),

    # context: 单值(scalar)
    StructField("device", StringType()),

    # user behaviour: 数组(array)
    StructField("clicks", ArrayType(StringType())),

    # item: 数组(array)
    StructField("item_id", ArrayType(StringType())),

    # relevance: 数组(array)
    StructField("relevance", ArrayType(LongType())),
]

schema = StructType(feature_names)
rows = [
    # pv_id,  user_id, age, gender,    device,          clicks,     item_id, relevance
    ["pv123", "uid012", 18, "0", "huawei p40pro max", ["item011", "item012"], "item012", 1],
    ["pv123", "uid012", 18, "0", "huawei p40pro max", ["item011", "item012"], "item345", 0],
    ["pv456", "uid345", 25, "1", "iPhone 13", ["item345"], "item456", 2],
    ["pv456", "uid345", 25, "1", "iPhone 13", ["item345"], "item567", 1],
    ["pv456", "uid345", 25, "1", "iPhone 13", ["item345"], "item678", 0]
]
rdd = spark.sparkContext.parallelize(rows)
rdd = rdd.keyBy(lambda row: row[0]).groupByKey().mapValues(list)
rdd = rdd.map(lambda pv_id_and_records: data(pv_id_and_records[1]))

df = spark.createDataFrame(rdd, schema)

# 存储为 tfrecord 文件格式，文件内部的数据格式为 Example
df.write.format("tfrecords").option("recordType", "Example").save(path, mode="overwrite")
df = spark.read.format("tfrecords").option("recordType", "Example").load(path)

df.show()
# +------------------+--------------------+-----+-----------------+---+------+---------+-------+
# |            clicks|             item_id|pv_id|           device|age|gender|relevance|user_id|
# +------------------+--------------------+-----+-----------------+---+------+---------+-------+
# |[item011, item012]|  [item012, item345]|pv123|huawei p40pro max| 18|     0|   [1, 0]| uid012|
# |         [item345]|[item456, item567...|pv456|        iPhone 13| 25|     1|[2, 1, 0]| uid345|
# +------------------+--------------------+-----+-----------------+---+------+---------+-------+


# 打印 dataframe 结构
df.printSchema()
# 输出
# root
#  |-- clicks: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- item_id: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- pv_id: string (nullable = true)
#  |-- device: string (nullable = true)
#  |-- age: long (nullable = true)
#  |-- gender: string (nullable = true)
#  |-- relevance: array (nullable = true)
#  |    |-- element: long (containsNull = true)
#  |-- user_id: string (nullable = true)
