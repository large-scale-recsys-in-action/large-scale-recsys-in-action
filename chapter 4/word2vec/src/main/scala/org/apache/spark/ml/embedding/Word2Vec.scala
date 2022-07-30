/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.embedding

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg.{BLAS, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.embedding
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

/**
  * Params for [[Word2Vec]] and [[Word2VecModel]].
  */
private[embedding] trait Word2VecBase extends Params
  with HasInputCol with HasOutputCol with HasMaxIter with HasStepSize with HasSeed {

  /**
    * The dimension of the code that you want to transform from words.
    * Default: 100
    *
    * @group param
    */
  final val vectorSize = new IntParam(
    this, "vectorSize", "the dimension of codes after transforming from words (> 0)",
    ParamValidators.gt(0))
  setDefault(vectorSize -> 100)
  /**
    * The window size (context words from [-window, window]).
    * Default: 5
    *
    * @group expertParam
    */
  final val windowSize = new IntParam(
    this, "windowSize", "the window size (context words from [-window, window]) (> 0)",
    ParamValidators.gt(0))
  /**
    * Number of partitions for sentences of words.
    * Default: 1
    *
    * @group param
    */
  final val numPartitions = new IntParam(
    this, "numPartitions", "number of partitions for sentences of words (> 0)",
    ParamValidators.gt(0))
  setDefault(windowSize -> 5)
  /**
    * The minimum number of times a token must appear to be included in the word2vec model's
    * vocabulary.
    * Default: 5
    *
    * @group param
    */
  final val minCount = new IntParam(this, "minCount", "the minimum number of times a token must " +
    "appear to be included in the word2vec model's vocabulary (>= 0)", ParamValidators.gtEq(0))
  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size.
    * Default: 1000
    *
    * @group param
    */
  final val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Maximum length " +
    "(in words) of each sentence in the input data. Any sentence longer than this threshold will " +
    "be divided into chunks up to the size (> 0)", ParamValidators.gt(0))
  setDefault(numPartitions -> 1)
  /**
    * Use continues bag-of-words model.
    * Default: 0
    *
    * @group param
    */
  final val cbow = new IntParam(this, "cbow", "Use continues bag-of-words model",
    ParamValidators.inArray(Array(0, 1)))
  /**
    * Use hierarchical softmax method to train the model.
    * Default: 1
    *
    * @group param
    */
  final val hs = new IntParam(this, "hs", "Use hierarchical softmax method to train the model",
    ParamValidators.inArray(Array(0, 1)))
  setDefault(minCount -> 5)
  /**
    * Use negative sampling method to train the model.
    * Default: 0
    *
    * @group param
    */
  final val negative = new IntParam(this, "negative", "Use negative sampling method to train the model",
    ParamValidators.gtEq(0))
  /**
    * Use sub-sampling trick to improve the performance.
    * Default: 0
    *
    * @group param
    */
  final val sample = new DoubleParam(this, "sample", "Use sub-sampling trick to improve the performance",
    ParamValidators.inRange(0, 1, true, true))
  setDefault(maxSentenceLength -> 1000)

  /** @group getParam */
  def getVectorSize: Int = $(vectorSize)

  /** @group expertGetParam */
  def getWindowSize: Int = $(windowSize)
  setDefault(cbow -> 0)

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /** @group getParam */
  def getMinCount: Int = $(minCount)
  setDefault(hs -> 1)

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** @group getParam */
  def getCBOW: Int = $(cbow)
  setDefault(negative -> 0)

  /** @group getParam */
  def getHS: Int = $(hs)

  /** @group getParam */
  def getNegative: Int = $(negative)
  setDefault(sample -> 0)

  /** @group getParam */
  def getSample: Double = $(sample)

  setDefault(stepSize -> 0.025)
  setDefault(maxIter -> 1)

  /**
    * Validate and transform the input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val typeCandidates = List(new ArrayType(StringType, true), new ArrayType(StringType, false))
    SchemaUtils.checkColumnTypes(schema, $(inputCol), typeCandidates)
    SchemaUtils.appendColumn(schema, $(outputCol), new VectorUDT)
  }
}

/**
  * Word2Vec trains a model of `Map(String, Vector)`, i.e. transforms a word into a code for further
  * natural language processing or machine learning process.
  */
final class Word2Vec(override val uid: String)
  extends Estimator[Word2VecModel] with Word2VecBase with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("org/apache/spark/w2v"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setVectorSize(value: Int): this.type = set(vectorSize, value)

  /** @group expertSetParam */
  def setWindowSize(value: Int): this.type = set(windowSize, value)

  /** @group setParam */
  def setStepSize(value: Double): this.type = set(stepSize, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setMinCount(value: Int): this.type = set(minCount, value)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = set(maxSentenceLength, value)

  /** @group getParam */
  def setCBOW(value: Int): this.type = set(cbow, value)

  /** @group getParam */
  def setHS(value: Int): this.type = set(hs, value)

  /** @group getParam */
  def setNegative(value: Int): this.type = set(negative, value)

  /** @group getParam */
  def setSample(value: Double): this.type = set(sample, value)

  override def fit(dataset: Dataset[_]): Word2VecModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(inputCol)).rdd.map(_.getAs[Seq[String]](0))
    val wordVectors = new embedding.Word2Vec()
      .setLearningRate($(stepSize))
      .setMinCount($(minCount))
      .setNumIterations($(maxIter))
      .setNumPartitions($(numPartitions))
      .setSeed($(seed))
      .setVectorSize($(vectorSize))
      .setWindowSize($(windowSize))
      .setMaxSentenceLength($(maxSentenceLength))
      .setCBOW($(cbow))
      .setHS($(hs))
      .setNegative($(negative))
      .setSample($(sample))
      .fit(input)
    copyValues(new Word2VecModel(uid, wordVectors).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): Word2Vec = defaultCopy(extra)
}

object Word2Vec extends DefaultParamsReadable[Word2Vec] {

  override def load(path: String): Word2Vec = super.load(path)
}

/**
  * Model fitted by [[Word2Vec]].
  */
class Word2VecModel private[ml](
  override val uid: String,
  @transient private val wordVectors: embedding.Word2VecModel)
  extends Model[Word2VecModel] with Word2VecBase with MLWritable {

  import Word2VecModel._

  /**
    * Returns a dataframe with two fields, "word" and "vector", with "word" being a String and
    * and the vector the DenseVector that it is mapped to.
    */
  @transient lazy val getVectors: DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    val wordVec = wordVectors.getVectors.mapValues(vec => Vectors.dense(vec.map(_.toDouble)))
    spark.createDataFrame(wordVec.toSeq).toDF("word", "vector")
  }

  /**
    * Find "num" number of words closest in similarity to the given word, not
    * including the word itself. Returns a dataframe with the words and the
    * cosine similarities between the synonyms and the given word.
    */
  def findSynonyms(word: String, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(wordVectors.findSynonyms(word, num)).toDF("word", "similarity")
  }

  /**
    * Find "num" number of words whose vector representation most similar to the supplied vector.
    * If the supplied vector is the vector representation of a word in the model's vocabulary,
    * that word will be in the results.  Returns a dataframe with the words and the cosine
    * similarities between the synonyms and the given word vector.
    */
  def findSynonyms(vec: Vector, num: Int): DataFrame = {
    val spark = SparkSession.builder().getOrCreate()
    spark.createDataFrame(wordVectors.findSynonyms(vec, num)).toDF("word", "similarity")
  }

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
    * Transform a sentence column to a vector column to represent the whole sentence. The transform
    * is performed by averaging all word vectors it contains.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val vectors = wordVectors.getVectors
      .mapValues(vv => Vectors.dense(vv.map(_.toDouble)))
      .map(identity) // mapValues doesn't return a serializable map (SI-7005)
    val bVectors = dataset.sparkSession.sparkContext.broadcast(vectors)
    val d = $(vectorSize)
    val word2Vec = udf { sentence: Seq[String] =>
      if (sentence.isEmpty) {
        Vectors.sparse(d, Array.empty[Int], Array.empty[Double])
      } else {
        val sum = Vectors.zeros(d)
        sentence.foreach { word =>
          bVectors.value.get(word).foreach { v =>
            BLAS.axpy(1.0, v, sum)
          }
        }
        BLAS.scal(1.0 / sentence.size, sum)
        sum
      }
    }
    dataset.withColumn($(outputCol), word2Vec(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): Word2VecModel = {
    val copied = new Word2VecModel(uid, wordVectors)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new Word2VecModelWriter(this)
}

object Word2VecModel extends MLReadable[Word2VecModel] {

  override def read: MLReader[Word2VecModel] = new Word2VecModelReader

  override def load(path: String): Word2VecModel = super.load(path)

  private[Word2VecModel]
  class Word2VecModelWriter(instance: Word2VecModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.wordVectors.wordIndex, instance.wordVectors.wordVectors.toSeq)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }

    private case class Data(wordIndex: Map[String, Int], wordVectors: Seq[Float])
  }

  private class Word2VecModelReader extends MLReader[Word2VecModel] {

    private val className = classOf[Word2VecModel].getName

    override def load(path: String): Word2VecModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("wordIndex", "wordVectors")
        .head()
      val wordIndex = data.getAs[Map[String, Int]](0)
      val wordVectors = data.getAs[Seq[Float]](1).toArray
      val oldModel = new embedding.Word2VecModel(wordIndex, wordVectors)
      val model = new Word2VecModel(metadata.uid, oldModel)
      metadata.getAndSetParams(model)
      model
    }
  }
}
