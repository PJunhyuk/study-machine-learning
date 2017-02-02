# 머신 러닝

## 등장 배경

이미지를 주고, 이걸 high-level에서 예측하기 위해.

## 정의

```
어떤 데이터를 넣었을 때 그에 대한 결과를 출력하는 시스템을,
컴퓨터에게 많은 양의 샘플 데이터를 학습시키는 방법으로 디자인하고,
이를 활용하여 문제에 대한 답을 얻어내는 방법.
```

Feature -> [ black box ] -> Target value

Target value(목적값): 우리가 관심을 갖고 살펴보는 data의 속성. black box의 output.
Feature/Attribute: Target value를 결정하는데 사용되는 data의 속성.
Training data: black box를 디자인하기 위한 학습에 사용되는 data.

ex.
data: 야옹이 사진.
Feature/Attribute: 사진의 RGB 값.
Target value: 'Cat'
Training data: 야옹이 사진 잔뜩.

## 과정

### 요약

Training(학습 단계)
data 수집 -> hypothesis(가설) 정의 -> cost(비용) 함수 설정 -> Optimizer를 이용한 학습

Prediction(예측 단계)
학습된 모델을 활용한 예측

### Cost Function(코스트 함수, 비용 함수)

설정한 모델과 Training data 사이의 차이를 계산하는 함수.

ex.
Linear Regression: sum(f(x)-y)

### Optimizer(옵티마이져)

Cost Function의 최솟값을 찾는 알고리즘.

ex.
Gradient Descent

## 분류

### Supervised Learning(지도 학습)

Training data에 Target value가 있는 경우.

Labeling(라벨링): Training data에 Target value가 있는 것.

Regression problem과 Classification problem으로 나누어짐.

ex.
[ 남자, 22살, 인천 ] -> [ A형 ]
[ 여자, 23살, 서울 ] -> [ O형 ]

#### Regression problem(회귀 문제)

Target value가 continuous한 경우.

Linear regression, Locally weighted linear regression

ex.
[ 키 174cm ] -> [ 몸무게 65kg ]

#### Classification problem

Target value가 continuous하지 않고, 몇 개의 분류로 딱 떨어지는 경우.

kNN, Naive Bayes, SVM, Decision tree 등

ex.
[ 남자, 22살 ] -> [ 집이 없다 ]
[ 여자, 49살 ] -> [ 집이 있다 ]

##### 이항 분류 모델

분류 결과가 두 개만 있는 경우.

ex.
참, 거짓.

##### 다항 분류 모델

분류 결과가 두 개 이상인 경우.

### Unsupervised Learning(비지도 학습)

Training data에 Target value가 없는 경우.

Clustering, K-means

ex.
[ 96cm, 32kg ]
[ 168cm, 55kg ]
[ 178cm, 94kg ]

#### Cluster(군집)

Unsupervised Learning에서 비슷한 data끼리 묶은 그룹.

## 알고리즘

### Linear Regression(선형 회귀 문제)

하나의 Feature와 하나의 Target value가 있는 data에서, 이들 간의 관계를 가장 잘 대변할 수 있는 1차 함수를 찾는 방법.

모든 Training data에 대해 f(x)-y의 값이 가장 작은 f.

#### Gradient Descent(경사 하강법)

Linear regression에서 사용하는 Optimizer의 한 종류.

샘플 값을 설정하고, 그 값에서의 Cost function을 미분했을 때 기울기가 음수인 방향으로 내려가면서 최소값을 찾아나가는 방법.

### Logistics Regression(로지스틱스 회귀 분석)

이항 분류 모델에서 주로 사용하는 분석 방법.
Linear regression을 적용하면 문제가 생기기 때문에 고안.
Sigmoid 함수를 사용.

#### Sigmoid(시그모이드)

-inf일 때 0, 0일 때 0.5, +inf일 때 1인 함수.

#### Cost function in Logistics regression

Logistics regression의 경우에 일반적으로 Cost function을 계산하면 극솟값이 너무 많아 Optimizer을 적용했을 경우 최솟값으로 수렴하지 않고 극솟값에서 멈춘다.

이를 방지하기 위해 Sigmoid 함수값에 -log를 취한 값을 새로운 Cost function으로 설정한다.

### kNN: k Nearest Neighbors

Training data 중 가장 유사한 k개의 data를 이용하여 값을 예측하는 방법.

### Naive Bayesian Classification(나이브 베이즈 분류)

#### Laplace smoothing

## 기타 용어

### 언더 피팅

Training data가 모자라거나 학습이 제대로 되지 않아 Training data에 가깝게 가지 못한 경우.

### Overfitting(오버 피팅)

Training data에 너무 정확하게 학습이 되어 Training data에는 아주 높은 정확도를 보이지만 다른 data에는 적용되지 못하는 문제이다.

#### 해결

1. 충분히 많은 Training data를 넣는다.
1. Regularization(정규화)를 이용한다.
1. 피쳐의 수를 줄인다.

## Reference

조대협의 블로그 http://bcho.tistory.com