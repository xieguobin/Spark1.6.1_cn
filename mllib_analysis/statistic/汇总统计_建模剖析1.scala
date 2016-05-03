//汇总统计_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.statistic

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

object Summary_statistics extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//获取数据，转换成rdd类型
val observations_path = sc.textFile("C:/my_install/spark/data/mllib2/sample_stat.txt")
val observations = sc.textFile(observations_path).map(_.split("\t")).map(f => f.map(f => f.toDouble))  
val observations1 = observations.map(f => Vectors.dense(f))  

//统计计算
val summary: MultivariateStatisticalSummary = Statistics.colStats(observations1)        //colStats 主方法
println(summary.max)         // a dense vector containing the mean value for each column  最大值
println(summary.min)         // a dense vector containing the mean value for each column  最小值
println(summary.mean)        // a dense vector containing the mean value for each column  均值
println(summary.variance)    // column-wise variance                                      方差
println(summary.numNonzeros) // number of nonzeros in each column                         非零值
println(summary.normL1)      //                                                           L1范数
println(summary.normL2)      //                                                           L2范数
}

//三、源码调用解析
//1、colStats方法
def colStats(X: RDD[Vector]): MultivariateStatisticalSummary = {
    new RowMatrix(X).computeColumnSummaryStatistics()
  }   //利用RDD创建RowMatrix对象，通过方法computeColumnSummaryStatistics统计指标。

def computeColumnSummaryStatistics(): MultivariateStatisticalSummary = {
    val summary = rows.treeAggregate(new MultivariateOnlineSummarizer)(
      (aggregator, data) => aggregator.add(data),
      (aggregator1, aggregator2) => aggregator1.merge(aggregator2))
    updateNumRows(summary.count)
    summary
  }  
//调用RDD的treeAggregate方法，treeAggregate是聚合方法，它迭代处理RDD中的数据。
//其中，(aggregator, data) => aggregator.add(data)处理每条数据，将其添加到MultivariateOnlineSummarizer。
//(aggregator1, aggregator2) => aggregator1.merge(aggregator2)将不同分区的MultivariateOnlineSummarizer对象汇总。
//实现的重点是add方法和merge方法。它们都定义在MultivariateOnlineSummarizer中。 下面描述这两种方法。
  
//2、add方法
//使用在线算法来计算均值和方差
@Since("1.1.0")
  def add(sample: Vector): this.type = add(sample, 1.0)
  private[spark] def add(instance: Vector, weight: Double): this.type = {
    if (weight == 0.0) return this
    if (n == 0) {
      n = instance.size
      currMean = Array.ofDim[Double](n)
      currM2n = Array.ofDim[Double](n)
      currM2 = Array.ofDim[Double](n)
      currL1 = Array.ofDim[Double](n)
      nnz = Array.ofDim[Double](n)
      currMax = Array.fill[Double](n)(Double.MinValue)
      currMin = Array.fill[Double](n)(Double.MaxValue)
    }
    val localCurrMean = currMean
    val localCurrM2n = currM2n
    val localCurrM2 = currM2
    val localCurrL1 = currL1
    val localNnz = nnz
    val localCurrMax = currMax
    val localCurrMin = currMin
    instance.foreachActive { (index, value) =>
      if (value != 0.0) {
        if (localCurrMax(index) < value) {
          localCurrMax(index) = value
        }
        if (localCurrMin(index) > value) {
          localCurrMin(index) = value
        }
        val prevMean = localCurrMean(index)
        val diff = value - prevMean
        localCurrMean(index) = prevMean + weight * diff / (localNnz(index) + weight)
        localCurrM2n(index) += weight * (value - localCurrMean(index)) * diff
        localCurrM2(index) += weight * value * value
        localCurrL1(index) += weight * math.abs(value)
        localNnz(index) += weight
      }
    }
    weightSum += weight
    weightSquareSum += weight * weight
    totalCnt += 1
    this
  }

//3、merge方法
//merge方法相对比较简单，它只是对两个MultivariateOnlineSummarizer对象的指标作合并操作。
 def merge(other: MultivariateOnlineSummarizer): this.type = {
    if (this.weightSum != 0.0 && other.weightSum != 0.0) {
      totalCnt += other.totalCnt
      weightSum += other.weightSum
      weightSquareSum += other.weightSquareSum
      var i = 0
      while (i < n) {
        val thisNnz = nnz(i)
        val otherNnz = other.nnz(i)
        val totalNnz = thisNnz + otherNnz
        if (totalNnz != 0.0) {
          val deltaMean = other.currMean(i) - currMean(i)
          // merge mean together
          currMean(i) += deltaMean * otherNnz / totalNnz
          // merge m2n together，不单纯是累加
          currM2n(i) += other.currM2n(i) + deltaMean * deltaMean * thisNnz * otherNnz / totalNnz
          // merge m2 together
          currM2(i) += other.currM2(i)
          // merge l1 together
          currL1(i) += other.currL1(i)
          // merge max and min
          currMax(i) = math.max(currMax(i), other.currMax(i))
          currMin(i) = math.min(currMin(i), other.currMin(i))
        }
        nnz(i) = totalNnz
        i += 1
      }
    } else if (weightSum == 0.0 && other.weightSum != 0.0) {
      this.n = other.n
      this.currMean = other.currMean.clone()
      this.currM2n = other.currM2n.clone()
      this.currM2 = other.currM2.clone()
      this.currL1 = other.currL1.clone()
      this.totalCnt = other.totalCnt
      this.weightSum = other.weightSum
      this.weightSquareSum = other.weightSquareSum
      this.nnz = other.nnz.clone()
      this.currMax = other.currMax.clone()
      this.currMin = other.currMin.clone()
    }
    this
  }

//4、真实的样本均值和样本方差
//在线算法的并行化实现是一种特殊情况，样本方差进行无偏估计。
//真实的样本均值和样本方差通过下面的代码实现。
override def mean: Vector = {
    val realMean = Array.ofDim[Double](n)
    var i = 0
    while (i < n) {
      realMean(i) = currMean(i) * (nnz(i) / weightSum)
      i += 1
    }
    Vectors.dense(realMean)
  }
 override def variance: Vector = {
    val realVariance = Array.ofDim[Double](n)
    val denominator = weightSum - (weightSquareSum / weightSum)
    // Sample variance is computed, if the denominator is less than 0, the variance is just 0.
    if (denominator > 0.0) {
      val deltaMean = currMean
      var i = 0
      val len = currM2n.length
      while (i < len) {
        realVariance(i) = (currM2n(i) + deltaMean(i) * deltaMean(i) * nnz(i) *
          (weightSum - nnz(i)) / weightSum) / denominator
        i += 1
      }
    }
    Vectors.dense(realVariance)
  }
  
