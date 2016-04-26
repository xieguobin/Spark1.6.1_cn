//相关系数_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.statistic

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics

object Summary_statistics extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//获取数据，转换成rdd类型
val observations_path = sc.textFile("C:/my_install/spark/data/mllib2/sample_stat.txt")
val observations = sc.textFile(observations_path).map(_.split("\t")).map(f => f.map(f => f.toDouble))  
val observations1 = observations.map(f => Vectors.dense(f))  

//统计计算
val summary: MultivariateStatisticalSummary = Statistics.colStats(observations1)
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
