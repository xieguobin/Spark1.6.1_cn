//假设检验_建模剖析1

//一、数学理论


//二、建模实例
//spark.mllib目前支持皮尔森卡方检测。输入属性的类型决定是作拟合优度(goodness of fit)检测还是作独立性检测。 拟合优度检测需要输入数据的类型是vector，独立性检测需要输入数据的类型是Matrix。
//spark.mllib也支持输入数据类型为RDD[LabeledPoint]，它用来通过卡方独立性检测作特征选择。Statistics提供方法用来作皮尔森卡方检测。

package org.apache.spark.mllib_analysis.statistic

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics._

object Hypothesis_testing extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
val vec: Vector = ... // a vector composed of the frequencies of events

// compute the goodness of fit. If a second vector to test against is not supplied as a parameter,   //皮尔森拟合优度检测 
// the test runs against a uniform distribution.  
val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
println(goodnessOfFitTestResult) // summary of the test including the p-value, degrees of freedom, 
                                 // test statistic, the method used, and the null hypothesis.

val mat: Matrix = ... // a contingency matrix

// conduct Pearson's independence test on the input contingency matrix                               //皮尔森独立性检测
val independenceTestResult = Statistics.chiSqTest(mat) 
println(independenceTestResult) // summary of the test including the p-value, degrees of freedom...

val obs: RDD[LabeledPoint] = ... // (feature, label) pairs.

// The contingency table is constructed from the raw (feature, label) pairs and used to conduct
// the independence test. Returns an array containing the ChiSquaredTestResult for every feature 
// against the label.                                                                               //独立性检测用于特征选择
val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
var i = 1
  featureTestResults.foreach { result =>
    println(s"Column $i:\n$result")
    i += 1
  } // summary of the test
}

//另外，spark.mllib提供了一个Kolmogorov-Smirnov (KS)检测的1-sample, 2-sided实现，用来检测概率分布的相等性。通过提供理论分布（现在仅仅支持正太分布）的名字以及它相应的参数， 或者提供一个计算累积分布(cumulative distribution)的函数，用户可以检测原假设或零假设(null hypothesis)：即样本是否来自于这个分布。用户检测正太分布，但是不提供分布参数，检测会默认该分布为标准正太分布。
//Statistics提供了一个运行1-sample, 2-sided KS检测的方法，下面就是一个应用的例子。

import org.apache.spark.mllib.stat.Statistics
val data: RDD[Double] = ... // an RDD of sample data
// run a KS test for the sample versus a standard normal distribution
val testResult = Statistics.kolmogorovSmirnovTest(data, "norm", 0, 1)
println(testResult) 
// perform a KS test using a cumulative distribution function of our making
val myCDF: Double => Double = ...
val testResult2 = Statistics.kolmogorovSmirnovTest(data, myCDF)


//三、源码调用解析
