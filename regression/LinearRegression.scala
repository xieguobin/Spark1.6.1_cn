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

package org.apache.spark.mllib.regression

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.pmml.PMMLExportable
import org.apache.spark.mllib.regression.impl.GLMRegressionModel
import org.apache.spark.mllib.util.{Saveable, Loader}
import org.apache.spark.rdd.RDD

/**
 * Regression model trained using LinearRegression.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 *
 */
@Since("0.8.0")
class LinearRegressionModel @Since("1.1.0") (
    @Since("1.0.0") override val weights: Vector,
    @Since("0.8.0") override val intercept: Double)
  extends GeneralizedLinearModel(weights, intercept) with RegressionModel with Serializable
  with Saveable with PMMLExportable {

  override protected def predictPoint(
      dataMatrix: Vector,
      weightMatrix: Vector,
      intercept: Double): Double = {
    weightMatrix.toBreeze.dot(dataMatrix.toBreeze) + intercept
  }

  @Since("1.3.0")
  override def save(sc: SparkContext, path: String): Unit = {
    GLMRegressionModel.SaveLoadV1_0.save(sc, path, this.getClass.getName, weights, intercept)
  }

  override protected def formatVersion: String = "1.0"
}

@Since("1.3.0")
object LinearRegressionModel extends Loader[LinearRegressionModel] {

  @Since("1.3.0")
  override def load(sc: SparkContext, path: String): LinearRegressionModel = {
    val (loadedClassName, version, metadata) = Loader.loadMetadata(sc, path)
    // Hard-code class name string in case it changes in the future
    val classNameV1_0 = "org.apache.spark.mllib.regression.LinearRegressionModel"
    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val numFeatures = RegressionModel.getNumFeatures(metadata)
        val data = GLMRegressionModel.SaveLoadV1_0.loadData(sc, path, classNameV1_0, numFeatures)
        new LinearRegressionModel(data.weights, data.intercept)
      case _ => throw new Exception(
        s"LinearRegressionModel.load did not recognize model with (className, format version):" +
        s"($loadedClassName, $version).  Supported:\n" +
        s"  ($classNameV1_0, 1.0)")
    }
  }
}

/**
 * Train a linear regression model with no regularization using Stochastic Gradient Descent.
 * This solves the least squares regression formulation
 *              f(weights) = 1/n ||A weights-y||^2^
 * (which is the mean squared error).
 * Here the data matrix has n rows, and the input RDD holds the set of rows of A, each with
 * its corresponding right hand side label y.
 * See also the documentation for the precise formulation.
 */
@Since("0.8.0")
class LinearRegressionWithSGD private[mllib] (                //随机梯度下降，损失函数f(weights) = 1/n ||A weights-y||^2^
    private var stepSize: Double,                             //迭代步长
    private var numIterations: Int,                           //迭代次数
    private var miniBatchFraction: Double)                    //参与计算的样本比例
  extends GeneralizedLinearAlgorithm[LinearRegressionModel] with Serializable {

  private val gradient = new LeastSquaresGradient()           //最小平方损失函数的梯度下降，实例化最优化包中Gradient的类LeastSquaresGradient
  private val updater = new SimpleUpdater()                   //简单梯度，无正则化，实例化最优化包中Updater的类SimpleUpdater
  @Since("0.8.0")
  override val optimizer = new GradientDescent(gradient, updater)    //实例化最优化包中GradientDescent的类GradientDescent
    .setStepSize(stepSize)                                           //setStepSize源于GradientDescent
    .setNumIterations(numIterations)
    .setMiniBatchFraction(miniBatchFraction)

  /**
   * Construct a LinearRegression object with default parameters: {stepSize: 1.0,
   * numIterations: 100, miniBatchFraction: 1.0}.
   */
  @Since("0.8.0")
  def this() = this(1.0, 100, 1.0)                              //默认参数

  override protected[mllib] def createModel(weights: Vector, intercept: Double) = {
    new LinearRegressionModel(weights, intercept)
  }
}

/**
 * Top-level methods for calling LinearRegression.
 *
 */
@Since("0.8.0")
object LinearRegressionWithSGD {                //伴生对象，train静态方法

  /**
   * Train a Linear Regression model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate a stochastic gradient. The weights used
   * in gradient descent are initialized using the initial weights provided.
   *
   * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
   *              matrix A as well as the corresponding right hand side label y
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   * @param initialWeights Initial set of weights to be used. Array should be equal in size to
   *        the number of features in the data.
   *
   */
  @Since("1.0.0")
  def train(                                               //train方法。Jlbas矩阵、Breeze数值计算、BLAS线性代数是基础
      input: RDD[LabeledPoint],                            //训练样本
      numIterations: Int,                                  //迭代次数
      stepSize: Double,                                    //迭代步长，默认为1
      miniBatchFraction: Double,                           //参与计算样本比例，默认为1
      initialWeights: Vector): LinearRegressionModel = {   //初始权重
    new LinearRegressionWithSGD(stepSize, numIterations, miniBatchFraction)      //调用伴生类
      .run(input, initialWeights)                          //调用run方法
  }

  /**
   * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. Each iteration uses
   * `miniBatchFraction` fraction of the data to calculate a stochastic gradient.
   *
   * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
   *              matrix A as well as the corresponding right hand side label y
   * @param numIterations Number of iterations of gradient descent to run.
   * @param stepSize Step size to be used for each iteration of gradient descent.
   * @param miniBatchFraction Fraction of data to be used per iteration.
   *
   */
  @Since("0.8.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double,
      miniBatchFraction: Double): LinearRegressionModel = {
    new LinearRegressionWithSGD(stepSize, numIterations, miniBatchFraction).run(input)
  }

  /**
   * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using the specified step size. We use the entire data set to
   * compute the true gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
   *              matrix A as well as the corresponding right hand side label y
   * @param stepSize Step size to be used for each iteration of Gradient Descent.
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a LinearRegressionModel which has the weights and offset from training.
   *
   */
  @Since("0.8.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int,
      stepSize: Double): LinearRegressionModel = {
    train(input, numIterations, stepSize, 1.0)
  }

  /**
   * Train a LinearRegression model given an RDD of (label, features) pairs. We run a fixed number
   * of iterations of gradient descent using a step size of 1.0. We use the entire data set to
   * compute the true gradient in each iteration.
   *
   * @param input RDD of (label, array of features) pairs. Each pair describes a row of the data
   *              matrix A as well as the corresponding right hand side label y
   * @param numIterations Number of iterations of gradient descent to run.
   * @return a LinearRegressionModel which has the weights and offset from training.
   *
   */
  @Since("0.8.0")
  def train(
      input: RDD[LabeledPoint],
      numIterations: Int): LinearRegressionModel = {
    train(input, numIterations, 1.0, 1.0)
  }
}
