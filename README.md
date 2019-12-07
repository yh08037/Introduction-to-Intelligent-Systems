# Introduction to Intelligent Systems
* 2019년 경북대학교 여름계절학기 지능시스템개론 강의를 듣고 정리하였습니다.
* 기계학습의 기초적인 수학 이론을 공부하고, 이를 python으로 구현해봅니다.
* `tensorflow`, `pytorch` 등의 딥러닝 라이브러리를 사용하지 않고,<br> `numpy`, `matplotlib.pyplot` 모듈만을 사용해 모든 것을 구현하는 것을 목표로 합니다.

## Lab1 : Non-Regularized Linear Regression
### TODO
* Batch Gradient Descent
* Stochastic Gradient Descent
* Closed-form solution (Ordinary Least Square)
### Linear Regression
#### Hypothesis Function
![h](images/image1_1.PNG)
#### Definition of Problem : Cost Minimization
![def](images/image1_2.PNG)
#### Batch Gradient Descent
![bgd](images/image1_3.PNG)
#### Stochastic Gradient Descent
![sgd](images/image1_4.PNG)
#### Closed-form Solution (Ordinary Least Square)
![ols](images/image1_5.PNG)


<br>

## Lab2 : Regularized Regression
### TODO
* Compute and compare solutions for
  1. unregularized linear
  2. unregularized parabolic
  3. unregularized 5th-order polynomial
  4. regularized 5th-order polynomial (RIDGE)
### RIDGE and LASSO
![ridge and lasso](images/image2.PNG)
#### Problem definition of Regularized Regression
![def](images/image2_1.PNG)
#### Unconstrained version of Problem
![laplace](images/image2_2.PNG)
#### Closed-form Solutoin of RIDGE Problem
![ols](images/image2_3.PNG)

<br>

## Lab 3 : Feed Forward Neural Network
### TODO
* Implementing FFNN for classification problem
* Back Propagation with Gradient Descent
### About Training
![training](images/image1.PNG)
### Model of 2-Layered FFNN
![ffnn_model](images/image3_1.PNG)
### Gradient Descent of 2-Layered FFNN
#### Update Rule of FFNN
![update rule of ffnn](images/image3_2.PNG)
#### Gradient of W
![Gradient of W](images/image3_3.PNG)
#### Gradient of V
![Gradient of V](images/image3_4.PNG)

<br>

## Lab 4 : Feed Back Neural Network (Recurrent Neural Network)
### TODO
* Back Propagation
* Resilient Propagation
* Gradient Clipping
### Elman Model of RNN
![elman](images/image4_1.PNG)
### Gradient Descent of RNN
#### Update Rule of RNN
![rule](images/image4_2.PNG)
#### Gradient of Vx
![Vx](images/image4_3.PNG)
#### Gradient of Vf
![Vf](images/image4_4.PNG)
### Issue : Gradient Vanishing / Explosion
![issue](images/image4_5.PNG)
#### Resilient Propagation : accelerate / slow down steps
![rp](images/image4_6.PNG)
#### Gradient Clipping : Prevent Explosion
![gc](images/image4_7.PNG)


<br>

## Lab5 : Unsupervised Learning : K-means & PCA
### TODO
* K-means
* PCA
### K-means
#### Difference of Classification and Clustering
![kmeans](images/image5_1.PNG)
#### Algorithm for K-means
![alg_kmeans](images/image5_2.PNG)
### PCA
#### What is 'Principal Component'
![pca](images/image5_3.PNG)
#### Algorithm for PCA
![alg_pca](images/image5_4.PNG)

<br>

## Lab7 : Generative Model : Naive Bayes
### TODO
* Spam Mail Detector with Naive Bayes Classifier
### Discriminitive model and Generative model
* Discriminative model 
  * learns the conditional probability distribution `p(y|x)`
  * learns `p(y|x)` directly from the data and then try to classify data
  * generally give better performance in classification tasks
* Generative model 
  * learns the joint probability distribution `p(x, y)`
  * learns `p(x, y)` which can be transformed into p(y|x) later to classify the data
  * we can use `p(x, y)` to generate new data similar to existing data

### Naive Bayes Classifier
#### Prediction Criterion
![predict](images/predict.PNG)
#### Model Parameters
![params](images/image7_1.PNG)
#### Principle of the Maximum Likelihood Estimation (MLE)
![mle](images/image7_2.PNG)
#### Issue : divide by zero
![issue](images/issue.PNG)
#### Laplace Smoothing : kind of regularization
![smooth](images/smoothing.PNG)
  
