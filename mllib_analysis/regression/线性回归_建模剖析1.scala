//线性回归_建模剖析1

//一、数学理论


//二、建模实例
package org.apache.spark.mllib_analysis.regression

import org.apache.spark.{SparkConf,SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.linalg.Vectors

object Lr01 extends App{
  val conf = new SparkConf().setAppName("Spark_Lr").setMaster("local")
  val sc = new SparkContext(conf)
  
//获取数据
val data = sc.textFile("C:/my_install/spark/data/mllib/ridge-data/lpsa.data")
val parsedData = data.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
}.cache()

//模型训练
val numIterations = 100
val stepSize = 0.00000001
val model = LinearRegressionWithSGD.train(parsedData, numIterations, stepSize)

//模型评价
val valuesAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
println("training Mean Squared Error = " + MSE)

//模型保存与加载
//model.save(sc, "myModelPath")
//val sameModel = LinearRegressionModel.load(sc, "myModelPath")

}

//三、源码调用解析
//1、LeastSquaresGradient
//    训练过程均使用GeneralizedLinearModel中的run训练，只是训练使用的Gradient和Updater不同。在一般的线性回归中，
//使用LeastSquaresGradient计算梯度，使用SimpleUpdater进行更新。 它的实现过程分为4步。普通线性回归的损失函数是最
//小二乘损失。
//2、SimpleUpdater
//    普通线性回归的不适用正则化方法，所以它用SimpleUpdater实现Updater。
