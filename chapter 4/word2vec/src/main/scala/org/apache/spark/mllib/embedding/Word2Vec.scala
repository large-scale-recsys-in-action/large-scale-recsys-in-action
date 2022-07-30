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

package org.apache.spark.mllib.embedding

import java.lang.{Iterable => JavaIterable}

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
 *  Entry in vocabulary
 */
private case class VocabWord(
  var word: String,
  var cn: Int,
  var point: Array[Int],
  var code: Array[Int],
  var codeLen: Int
)

/**
 * Word2Vec creates vector representation of words in a text corpus.
 * The algorithm first constructs a vocabulary from the corpus
 * and then learns vector representation of words in the vocabulary.
 * The vector representation can be used as features in
 * natural language processing and machine learning algorithms.
 *
 * Two models cbow and skip-gram are used in our implementation.
 * Both hierarchical softmax and negative sampling methods are
 * supported to train the model. The variable names in the implementation
 * matches the original C implementation.
 *
 * For original C implementation, see https://code.google.com/p/word2vec/
 * For research papers, see
 * Efficient Estimation of Word Representations in Vector Space
 * and
 * Distributed Representations of Words and Phrases and their Compositionality.
 */
class Word2Vec extends Serializable with Logging {

  private var vectorSize = 100
  private var learningRate = 0.025
  private var numPartitions = 1
  private var numIterations = 1
  private var seed = Utils.random.nextLong()
  private var minCount = 5
  private var maxSentenceLength = 1000
  private var cbow = 0
  private var negative = 0
  private var sample = 0.0
  private var hs = 1
  private val tableSize = 1e8.toInt

  /**
   * Sets vector size (default: 100).
   */
  def setVectorSize(vectorSize: Int): this.type = {
    require(vectorSize > 0, s"vectorSize must be greater than 0 but got $vectorSize")
    this.vectorSize = vectorSize
    this
  }

  /**
   * Sets initial learning rate (default: 0.025).
   */
  def setLearningRate(learningRate: Double): this.type = {
    require(learningRate > 0 && learningRate <= 1, s"learningRate must be between 0 and 1 but got $learningRate")
    this.learningRate = learningRate
    this
  }

  /**
   * Sets number of partitions (default: 1). Use a small number for accuracy.
   */
  def setNumPartitions(numPartitions: Int): this.type = {
    require(numPartitions > 0, s"numPartitions must be greater than 0 but got $numPartitions")
    this.numPartitions = numPartitions
    this
  }

  /**
   * Sets number of iterations (default: 1), which should be smaller than or equal to number of
   * partitions.
   */
  def setNumIterations(numIterations: Int): this.type = {
    require(numIterations > 0, s"numIterations must be greater than 0 but got $numIterations")
    this.numIterations = numIterations
    this
  }

  /**
   * Sets random seed (default: a random long integer).
   */
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }

  /**
   * Sets the window of words (default: 5)
   */
  def setWindowSize(window: Int): this.type = {
    require(window > 0, s"window must be greater than 0 but got $window")
    this.window = window
    this
  }

  /**
   * Sets minCount, the minimum number of times a token must appear to be included in the word2vec
   * model's vocabulary (default: 5).
   */
  def setMinCount(minCount: Int): this.type = {
    require(minCount >= 0, s"minCount must be greater than or equal to 0 but got $minCount")
    this.minCount = minCount
    this
  }

  /**
    * Sets the maximum length (in words) of each sentence in the input data.
    * Any sentence longer than this threshold will be divided into chunks of
    * up to `maxSentenceLength` size (default: 1000)
    */
  def setMaxSentenceLength(maxSentenceLength: Int): this.type = {
    require(maxSentenceLength > 0, s"maxSentenceLength must be greater than 0 but got $maxSentenceLength")
    this.maxSentenceLength = maxSentenceLength
    this
  }

  /**
    * Sets cbow. Use continues bag-of-words model (default: 0).
    */
  def setCBOW(cbow: Int): this.type = {
    require(cbow == 0 || cbow == 1, s"cbow must be equal to 0 or 1 but got $cbow")
    this.cbow = cbow
    this
  }

  /**
    * Set sample. Use sub-sampling trick to improve the performance (default: 0).
    */
  def setSample(sample: Double): this.type = {
    require(sample >= 0 && sample <= 1, s"sample must be between 0 and 1 but got $sample")
    this.sample = sample
    this
  }

  /**
    * Set hs. Use hierarchical softmax method to train the model (default: 1).
    */
  def setHS(hs: Int): this.type = {
    require(hs == 0 || hs == 1, s"hs must be equal to 0 or 1 but got $hs")
    this.hs = hs
    this
  }

  /**
    * Set negative. Use negative sampling method to train the model (default: 0).
    */
  def setNegative(negative: Int): this.type = {
    require(negative >= 0, s"negative must be greater than or equal to 0 but got $negative")
    this.negative = negative
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val MAX_CODE_LENGTH = 40
  private val MAX_SENTENCE_LENGTH = 1000

  /** context words from [-window, window] */
  private var window = 5

  private var trainWordsCount = 0L
  private var vocabSize = 0
  @transient private var vocab: Array[VocabWord] = null
  @transient private var vocabHash = mutable.HashMap.empty[String, Int]
  @transient private var table: Array[Int] = null


  private def learnVocab[S <: Iterable[String]](dataset: RDD[S]): Unit = {
    val words = dataset.flatMap(x => x)

    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .filter(_._2 >= minCount)
      .map(x => VocabWord(
        x._1,
        x._2,
        new Array[Int](MAX_CODE_LENGTH),
        new Array[Int](MAX_CODE_LENGTH),
        0))
      .collect()
      .sortWith((a, b) => a.cn > b.cn)

    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    var a = 0
    while (a < vocabSize) {
      vocabHash += vocab(a).word -> a
      trainWordsCount += vocab(a).cn
      a += 1
    }
    logInfo(s"vocabSize = $vocabSize, trainWordsCount = $trainWordsCount")
  }

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }

  private def createBinaryTree(): Unit = {
    val count = new Array[Long](vocabSize * 2 - 1)
    val binary = new Array[Int](vocabSize * 2 - 1)
    val parentNode = new Array[Int](vocabSize * 2 - 1)
    val code = new Array[Int](MAX_CODE_LENGTH)
    val point = new Array[Int](MAX_CODE_LENGTH)
    var a = 0
    while (a < vocabSize) {
      count(a) = vocab(a).cn
      a += 1
    }
    while (a < 2 * vocabSize - 1) {
      count(a) = 1e9.toInt
      a += 1
    }
    var pos1 = vocabSize - 1
    var pos2 = vocabSize

    var min1i = 0
    var min2i = 0

    a = 0
    while (a < vocabSize - 1) {
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min1i = pos1
          pos1 -= 1
        } else {
          min1i = pos2
          pos2 += 1
        }
      } else {
        min1i = pos2
        pos2 += 1
      }
      if (pos1 >= 0) {
        if (count(pos1) < count(pos2)) {
          min2i = pos1
          pos1 -= 1
        } else {
          min2i = pos2
          pos2 += 1
        }
      } else {
        min2i = pos2
        pos2 += 1
      }
      count(vocabSize + a) = count(min1i) + count(min2i)
      parentNode(min1i) = vocabSize + a
      parentNode(min2i) = vocabSize + a
      binary(min2i) = 1
      a += 1
    }
    // Now assign binary code to each vocabulary word
    var i = 0
    a = 0
    while (a < vocabSize) {
      var b = a
      i = 0
      while (b != vocabSize * 2 - 2) {
        code(i) = binary(b)
        point(i) = b
        i += 1
        b = parentNode(b)
      }
      vocab(a).codeLen = i
      vocab(a).point(0) = vocabSize - 2
      b = 0
      while (b < i) {
        vocab(a).code(i - b - 1) = code(b)
        vocab(a).point(i - b) = point(b) - vocabSize
        b += 1
      }
      a += 1
    }
  }

  private def initUnigramTable(): Unit = {
    var a = 0
    val power = 0.75
    var trainWordsPow = 0.0
    table = new Array[Int](tableSize)

    while (a < vocabSize) {
      trainWordsPow += Math.pow(vocab(a).cn, power)
      a += 1
    }

    var i = 0
    var d1 = Math.pow(vocab(i).cn, power) / trainWordsPow
    a = 0
    /* d1 marks the accumulative power-law probability up to the current word */
    /* a / TABLE_SIZE marks the sampling probability up to the current word */
    /* 说计算出的负采样概率*1亿=单词在表中出现的次数 */
    while (a < tableSize) {
      table(a) = i
      if (a.toDouble / tableSize > d1) {
        i += 1
        d1 += Math.pow(vocab(i).cn, power) / trainWordsPow
      }
      if (i >= vocabSize) {
        i = vocabSize - 1
      }
      a += 1
    }
  }

  /**
   * Computes the vector representation of each word in vocabulary.
   * @param dataset an RDD of sentences, each sentence is expressed as an iterable collection of words
   * @return a Word2VecModel
   */
  def fit[S <: Iterable[String]](dataset: RDD[S]): Word2VecModel = {

    learnVocab(dataset)

    createBinaryTree()

    if (negative > 0) {
      initUnigramTable()
    } else if (hs == 0) {
      throw new RuntimeException(s"negative and hs can not both be equal to 0.")
    }

    val sc = dataset.context

    val expTable = sc.broadcast(createExpTable())
    val bcVocab = sc.broadcast(vocab)
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcTable = sc.broadcast(table)

    // each partition is a collection of sentences,
    // will be translated into arrays of Index integer
    val sentences: RDD[Array[Int]] = dataset.mapPartitions { sentenceIter =>
      // Each sentence will map to 0 or more Array[Int]
      sentenceIter.flatMap { sentence =>
        // Sentence of words, some of which map to a word index
        val wordIndexes = sentence.flatMap(bcVocabHash.value.get)
        // break wordIndexes into trunks of maxSentenceLength when has more
        wordIndexes.grouped(maxSentenceLength).map(_.toArray)
      }
    }

    val newSentences = sentences.repartition(numPartitions).cache()
    val initRandom = new XORShiftRandom(seed)

    if (vocabSize.toLong * vectorSize >= Int.MaxValue) {
      throw new RuntimeException("Please increase minCount or decrease vectorSize in Word2Vec" +
        " to avoid an OOM. You are highly recommended to make your vocabSize*vectorSize, " +
        "which is " + vocabSize + "*" + vectorSize + " for now, less than `Int.MaxValue`.")
    }

    val syn0Global =
      Array.fill[Float](vocabSize * vectorSize)((initRandom.nextFloat() - 0.5f) / vectorSize)
    val syn1Global = new Array[Float](vocabSize * vectorSize)
    val syn1NegGlobal = new Array[Float](vocabSize * vectorSize)
    var alpha = learningRate

    for (k <- 1 to numIterations) {
      val bcSyn0Global = sc.broadcast(syn0Global)
      val bcSyn1Global = sc.broadcast(syn1Global)
      val bcSyn1NegGlobal = sc.broadcast(syn1NegGlobal)

      val partial = newSentences.mapPartitionsWithIndex { case (idx, iter) =>
        val random = new XORShiftRandom(seed ^ ((idx + 1) << 16) ^ ((-k - 1) << 8))
        val syn0Modify = new Array[Int](vocabSize)
        val syn1Modify = new Array[Int](vocabSize)
        val syn1NegModify = new Array[Int](vocabSize)
        val sen = new Array[Int](MAX_SENTENCE_LENGTH)

        val model = iter.foldLeft((bcSyn0Global.value, bcSyn1Global.value, bcSyn1NegGlobal.value, 0L, 0L)) {
          case ((syn0, syn1, syn1Neg, lastWordCount, wordCount), sentence) =>
            var sentenceLength = 0

            // The sub-sampling trick randomly discards frequent words while keeping the ranking same
            var sentencePosition = 0
            while (sentencePosition < sentence.length && sentencePosition < MAX_SENTENCE_LENGTH) {
              val word = sentence(sentencePosition)
              if (sample > 0) {
                val ran = Math.sqrt(bcVocab.value(word).cn / (sample * trainWordsCount) + 1) *
                  (sample * trainWordsCount) / bcVocab.value(word).cn
                if (ran >= random.nextFloat()) {
                  sen(sentenceLength) = word
                  sentenceLength += 1
                }
              } else {
                sen(sentenceLength) = word
                sentenceLength += 1
              }
              sentencePosition += 1
            }

            var lwc = lastWordCount
            var wc = wordCount
            if (wordCount - lastWordCount > 10000) {
              lwc = wordCount
              alpha =
                learningRate * (1 - numPartitions * wordCount.toDouble / (trainWordsCount + 1))
              if (alpha < learningRate * 0.0001) alpha = learningRate * 0.0001
              logInfo("wordCount = " + wordCount + ", alpha = " + alpha)
            }
            wc += sentence.length
            sentencePosition = 0
            while (sentencePosition < sentenceLength) {
              val word = sen(sentencePosition)
              val b = random.nextInt(window)
              val neu1 = new Array[Float](vectorSize)

              if (cbow == 1) {
                // Train CBOW
                val neu1e = new Array[Float](vectorSize)
                var a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = sentencePosition - window + a
                    if (c >= 0 && c < sentence.length) {
                      val lastWord = sen(c)
                      val l1 = lastWord * vectorSize
                      blas.saxpy(vectorSize, 1, syn0, l1, 1, neu1, 0, 1)
                    }
                  }
                  a += 1
                }

                if (hs == 1) {
                  // Hierarchical softmax
                  var d = 0
                  while (d < bcVocab.value(word).codeLen) {
                    val inner = bcVocab.value(word).point(d)
                    val l2 = inner * vectorSize
                    // Propagate hidden -> output
                    var f = blas.sdot(vectorSize, neu1, 0, 1, syn1, l2, 1)
                    if (f > -MAX_EXP && f < MAX_EXP) {
                      val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                      f = expTable.value(ind)
                      val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                      blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                      blas.saxpy(vectorSize, g, neu1, 0, 1, syn1, l2, 1)
                      syn1Modify(inner) += 1
                    }
                    d += 1
                  }
                }

                if (negative > 0) {
                  // Negative sampling
                  var target = -1
                  var label = -1
                  var d = 0
                  while (d < negative + 1) {
                    var isContinued = false
                    if (d == 0) {
                      target = word
                      label = 1
                    } else {
                      target = bcTable.value(random.nextInt(tableSize))
                      if (target == word) {
                        isContinued = true
                      }
                      if (!isContinued.equals(true)) {
                        label = 0
                      }
                    }
                    if (!isContinued.equals(true)) {
                      val l2 = target * vectorSize
                      val f = blas.sdot(vectorSize, neu1, 0, 1, syn1Neg, l2, 1)
                      var g = 0.0
                      if (f > MAX_EXP) {
                        g = (label - 1) * alpha
                      } else if (f < -MAX_EXP) {
                        g = (label - 0) * alpha
                      } else {
                        val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                        g = (label - expTable.value(ind)) * alpha
                      }
                      blas.saxpy(vectorSize, g.toFloat, syn1Neg, l2, 1, neu1e, 0, 1)
                      blas.saxpy(vectorSize, g.toFloat, neu1, 0, 1, syn1Neg, l2, 1)
                      syn1NegModify(target) += 1
                    }
                    d += 1
                  }
                }

                // Hidden -> in
                a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = sentencePosition - window + a
                    if (c >= 0 && c < sentenceLength) {
                      val lastWord = sen(c)
                      val l1 = lastWord * vectorSize
                      blas.saxpy(vectorSize, 1, neu1e, 0, 1, syn0, l1, 1)
                      syn0Modify(lastWord) += 1
                    }
                  }
                  a += 1
                }
                sentencePosition += 1
              } else {
                // Train Skip-gram
                var a = b
                while (a < window * 2 + 1 - b) {
                  if (a != window) {
                    val c = sentencePosition - window + a
                    if (c >= 0 && c < sentenceLength) {
                      val lastWord = sen(c)
                      val l1 = lastWord * vectorSize
                      val neu1e = new Array[Float](vectorSize)

                      if (hs == 1) {
                        // Hierarchical softmax
                        var d = 0
                        while (d < bcVocab.value(word).codeLen) {
                          val inner = bcVocab.value(word).point(d)
                          val l2 = inner * vectorSize
                          // Propagate hidden -> output
                          var f = blas.sdot(vectorSize, syn0, l1, 1, syn1, l2, 1)
                          if (f > -MAX_EXP && f < MAX_EXP) {
                            val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                            f = expTable.value(ind)
                            val g = ((1 - bcVocab.value(word).code(d) - f) * alpha).toFloat
                            blas.saxpy(vectorSize, g, syn1, l2, 1, neu1e, 0, 1)
                            blas.saxpy(vectorSize, g, syn0, l1, 1, syn1, l2, 1)
                            syn1Modify(inner) += 1
                          }
                          d += 1
                        }
                      }

                      if (negative > 0) {
                        // Negative sampling
                        var target = -1
                        var label = -1
                        var d = 0
                        while (d < negative + 1) {
                          var isContinued = false
                          if (d == 0) {
                            target = word
                            label = 1
                          } else {
                            target = bcTable.value(random.nextInt(tableSize))
                            if (target == word) {
                              isContinued = true
                            }
                            if (!isContinued.equals(true)) {
                              label = 0
                            }
                          }
                          if (!isContinued.equals(true)) {
                            val l2 = target * vectorSize
                            val f = blas.sdot(vectorSize, syn0, l1, 1, syn1Neg, l2, 1)
                            var g = 0.0
                            if (f > MAX_EXP) {
                              g = (label - 1) * alpha
                            } else if (f < -MAX_EXP) {
                              g = (label - 0) * alpha
                            } else {
                              val ind = ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2.0)).toInt
                              g = (label - expTable.value(ind)) * alpha
                            }
                            blas.saxpy(vectorSize, g.toFloat, syn1Neg, l2, 1, neu1e, 0, 1)
                            blas.saxpy(vectorSize, g.toFloat, syn0, l1, 1, syn1Neg, l2, 1)
                            syn1NegModify(target) += 1
                          }
                          d += 1
                        }
                      }

                      syn0Modify(lastWord) += 1
                      blas.saxpy(vectorSize, 1.0f, neu1e, 0, 1, syn0, l1, 1)
                    }
                  }
                  a += 1
                }
                sentencePosition += 1
              }
            }
            (syn0, syn1, syn1Neg, lwc, wc)
        }
        val syn0Local = model._1
        val syn1Local = model._2
        val syn1NegLocal = model._3
        // Only output modified vectors.
        Iterator.tabulate(vocabSize) { index =>
          if (syn0Modify(index) > 0) {
            Some((index, syn0Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1Modify(index) > 0) {
            Some((index + vocabSize, syn1Local.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten ++ Iterator.tabulate(vocabSize) { index =>
          if (syn1NegModify(index) > 0) {
            Some((index + 2 * vocabSize, syn1NegLocal.slice(index * vectorSize, (index + 1) * vectorSize)))
          } else {
            None
          }
        }.flatten
      }

      // FIXED: embedding normalized
      val synAgg = partial.mapPartitions { iter =>
        iter.map { case (id, vec) =>
          (id, (vec, 1))
        }
      }.reduceByKey { case ((v1, count1), (v2, count2)) =>
        blas.saxpy(vectorSize, 1.0f, v2, 1, v1, 1)
        (v1, count1 + count2)
      }.map { case (id, (vec, count)) =>
        blas.sscal(vectorSize, 1.0f / count, vec, 1)
        (id, vec)
      }.collect()

      var i = 0
      while (i < synAgg.length) {
        val index = synAgg(i)._1
        if (index < vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn0Global, index * vectorSize, vectorSize)
        } else if (index >= vocabSize && index < 2 * vocabSize) {
          Array.copy(synAgg(i)._2, 0, syn1Global, (index - vocabSize) * vectorSize, vectorSize)
        } else {
          Array.copy(synAgg(i)._2, 0, syn1NegGlobal, (index - 2 * vocabSize) * vectorSize, vectorSize)
        }
        i += 1
      }
      bcSyn0Global.unpersist(false)
      bcSyn1Global.unpersist(false)
      bcSyn1NegGlobal.unpersist(false)
    }
    newSentences.unpersist()
    expTable.destroy(false)
    bcVocab.destroy(false)
    bcVocabHash.destroy(false)

    val wordArray = vocab.map(_.word)
    new Word2VecModel(wordArray.zipWithIndex.toMap, syn0Global)
  }

  /**
    * Computes the vector representation of each word in vocabulary (Java version).
    * @param dataset a JavaRDD of words
    * @return a Word2VecModel
    */
  def fit[S <: JavaIterable[String]](dataset: JavaRDD[S]): Word2VecModel = {
    fit(dataset.rdd.map(_.asScala))
  }
}

/**
 * Word2Vec model
 * @param wordIndex maps each word to an index, which can retrieve the corresponding
 *                  vector from wordVectors
 * @param wordVectors array of length numWords * vectorSize, vector corresponding
 *                    to the word mapped with index i can be retrieved by the slice
 *                    (i * vectorSize, i * vectorSize + vectorSize)
 */
class Word2VecModel private[spark] (
    private[spark] val wordIndex: Map[String, Int],
    private[spark] val wordVectors: Array[Float]) extends Serializable with Saveable {

  val numWords = wordIndex.size
  // vectorSize: Dimension of each word's vector.
  val vectorSize = wordVectors.length / numWords

  // wordList: Ordered list of words obtained from wordIndex.
  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  // wordVecNorms: Array of length numWords, each value being the Euclidean norm
  //               of the wordVector.
  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  def this(model: Map[String, Array[Float]]) = {
    this(Word2VecModel.buildWordIndex(model), Word2VecModel.buildWordVectors(model))
  }

  override protected def formatVersion = "1.0"

  def save(sc: SparkContext, path: String): Unit = {
    Word2VecModel.SaveLoadV1_0.save(sc, path, getVectors)
  }

/*  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }*/

  /**
   * Transforms a word to its vector representation
   * @param word a word
   * @return vector representation of word
   */
  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  /**
   * Find synonyms of a word
   * @param word a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  /**
    * Find synonyms of the vector representation of a word, possibly
    * including any words in the model vocabulary whose vector respresentation
    * is the supplied vector.
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @return array of (word, cosineSimilarity)
    */
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    findSynonyms(vector, num, None)
  }


  /**
    * Find synonyms of the vector representation of a word, rejecting
    * words identical to the value of wordOpt, if one is supplied.
    * @param vector vector representation of a word
    * @param num number of synonyms to find
    * @param wordOpt optionally, a word to reject from the results list
    * @return array of (word, cosineSimilarity)
    */
  private def findSynonyms(
      vector: Vector,
      num: Int,
      wordOpt: Option[String]): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0
    // Normalize input vector before blas.sgemv to avoid Inf value
    val vecNorm = blas.snrm2(vectorSize, fVector, 1)
    if (vecNorm != 0.0f) {
      blas.sscal(vectorSize, 1 / vecNorm, fVector, 0, 1)
    }
    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    val cosVec = cosineVec.map(_.toDouble)
    var ind = 0
    while (ind < numWords) {
      val norm = wordVecNorms(ind)
      if (norm == 0.0) {
        cosVec(ind) = 0.0
      } else {
        cosVec(ind) /= norm
      }
      ind += 1
    }

    val scored = wordList.zip(cosVec).toSeq.sortBy(-_._2)

    val filtered = wordOpt match {
      case Some(w) => scored.take(num + 1).filter(tup => w != tup._1)
      case None => scored
    }

    filtered.take(num).toArray
  }

  /**
   * Returns a map of words to their vector representations.
   */
  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }
}

object Word2VecModel extends Loader[Word2VecModel] {

  private def buildWordIndex(model: Map[String, Array[Float]]): Map[String, Int] = {
    model.keys.zipWithIndex.toMap
  }

  private def buildWordVectors(model: Map[String, Array[Float]]): Array[Float] = {
    require(model.nonEmpty, "Word2VecMap should be non-empty")
    val (vectorSize, numWords) = (model.head._2.size, model.size)
    val wordList = model.keys.toArray
    val wordVectors = new Array[Float](vectorSize * numWords)
    var i = 0
    while (i < numWords) {
      Array.copy(model(wordList(i)), 0, wordVectors, i * vectorSize, vectorSize)
      i += 1
    }
    wordVectors
  }

  private object SaveLoadV1_0 {

    val formatVersionV1_0 = "1.0"

    val classNameV1_0 = "org.apache.spark.mllib.feature.Word2VecModel"

    case class Data(word: String, vector: Array[Float])

    def load(sc: SparkContext, path: String): Word2VecModel = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
      val dataFrame = spark.read.parquet(Loader.dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      Loader.checkSchema[Data](dataFrame.schema)

      val dataArray = dataFrame.select("word", "vector").collect()
      val word2VecMap = dataArray.map(i => (i.getString(0), i.getSeq[Float](1).toArray)).toMap
      new Word2VecModel(word2VecMap)
    }

    def save(sc: SparkContext, path: String, model: Map[String, Array[Float]]): Unit = {
      val spark = SparkSession.builder().sparkContext(sc).getOrCreate()

      val vectorSize = model.values.head.length
      val numWords = model.size
      val metadata = compact(render(
        ("class" -> classNameV1_0) ~ ("version" -> formatVersionV1_0) ~
          ("vectorSize" -> vectorSize) ~ ("numWords" -> numWords)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

      // We want to partition the model in partitions smaller than
      // spark.kryoserializer.buffer.max
      val bufferSize = Utils.byteStringAsBytes(
        spark.conf.get("spark.kryoserializer.buffer.max", "64m"))
      // We calculate the approximate size of the model
      // We only calculate the array size, considering an
      // average string size of 15 bytes, the formula is:
      // (floatSize * vectorSize + 15) * numWords
      val approxSize = (4L * vectorSize + 15) * numWords
      val nPartitions = ((approxSize / bufferSize) + 1).toInt
      val dataArray = model.toSeq.map { case (w, v) => Data(w, v) }
      spark.createDataFrame(dataArray).repartition(nPartitions).write.parquet(Loader.dataPath(path))
    }
  }

  override def load(sc: SparkContext, path: String): Word2VecModel = {

    val (loadedClassName, loadedVersion, metadata) = Loader.loadMetadata(sc, path)
    implicit val formats = DefaultFormats
    val expectedVectorSize = (metadata \ "vectorSize").extract[Int]
    val expectedNumWords = (metadata \ "numWords").extract[Int]
    val classNameV1_0 = SaveLoadV1_0.classNameV1_0
    (loadedClassName, loadedVersion) match {
      case (classNameV1_0, "1.0") =>
        val model = SaveLoadV1_0.load(sc, path)
        val vectorSize = model.getVectors.values.head.length
        val numWords = model.getVectors.size
        require(expectedVectorSize == vectorSize,
          s"Word2VecModel requires each word to be mapped to a vector of size " +
            s"$expectedVectorSize, got vector of size $vectorSize")
        require(expectedNumWords == numWords,
          s"Word2VecModel requires $expectedNumWords words, but got $numWords")
        model
      case _ => throw new Exception(
        s"Word2VecModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $loadedVersion).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}
