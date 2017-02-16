# Deep Learning(딥 러닝)

## 정의

multiple layers를 사용한 머신 러닝.
이미지를 low-level features, high-level features, classifier 순으로 거쳐 분류하는 방식.

깊게 신경망을 구축한 뉴럴 네트워크를 말함.

## 과정

데이터셋 로드 -> 네트워크 정의 -> loss function 정의 -> Training dataset 으로 학습 진행 -> Testing dataset으로 학습 평가


## 용어

### learningRate(학습률)

각 epoch이 끝날 때마다 가중치 데이터(theta)를 손실함수가 최소화되는 방향으로의 값(gradTheta)만큼 업데이트하는데, 이 때 업데이트하는 비율을 조정하기 위한 변수.

## 등장 배경과 역사

### Artificial neural network(인공 지능망, 뉴럴 네트워크)

사람의 신경망이 여러 개의 뉴런으로 이루어진 것처럼, 머신 러닝의 학습 모델을 여러 개의 계산 노드와 층으로 연결해서 만들어낸 모델.

원래도 있었는데, 계층이 깊어지면 연산이 불가능해 관심을 받지 못하다가 추가적인 연구와 하드웨어의 발달로 재주목.

#### 뉴런의 구조

뉴런은 돌기를 통해서 여러 신경 자극을 입력 받고, 이를 세포체가 인지하여 신호로 변환해주는 구조.

#### Perceptron

뉴런의 구조를 응용한 계산 유닛.

외부에서 입력값 X1, ... Xn 을 입력 받고, 내부의 어떤 함수 f(x)를 거쳐 그 값의 범위에 따라 0 또는 1을 출력하는 모델이다.

입력 X를 받아, W*X+b인 선을 기준으로 Y에 0 또는 1을 출력하는 계산 유닛이다.

##### Activation function

Sigmoid function과 같이 Y를 0과 1로 분류하는데 사용되는 함수.

ex.
Activation function in Logistics regression : Sigmoid function

#### Perceptron의 한계

입력값 X1, X2을 각각 x축, y축으로 한 그래프에서 Perceptron은 하나의 선을 그어 나누는 방식이다.
그러나 이 방식으로는 AND와 OR은 해결 가능하지만, XOR 문제는 해결할 수 없다.

#### MLP(Multi Layer Perceptron: 다중 계층 퍼셉트론)

Perceptron을 다중으로 겹치면 XOR 문제를 해결할 수 있다.

그러나 레이어가 복잡해질수록 각각의 Perceptron을 디자인하기가 너무 어렵다.

#### Back Propagation

뉴럴 네트워크를 순방향(X->Y)으로 한 번 연산을 한 후, 그 결과 값을 가지고 다시 역방향으로 계산하며 값을 구한다는 개념이다.

##### Vanishing Gradient

Back Propagation에서 뉴럴 네트워크가 깊어질수록 값의 전달이 되지 않는 문제이다.

#### ReLu

Activation function의 한 종류로, Back Propagation을 해결할 수 있다.

#### 뉴럴 네트워크의 초기값

뉴럴 네트워크가 학습을 잘 하기 위해서는 초기값을 잘 설정해야 한다.

##### RBM(Restricted Boltzmann Machine)

초기값을 계산할 수 있는 알고리즘.

## 모델

### CNN(Convolutional Neural Network)

이미지 인식에 자주 사용됨.

#### 구조

이미지 -> Convolutional Layer -> Feature(특징) -> Neural Network -> 분류

#### Convolutional Layer

특징을 추출하는 Filter(필터)와 Filter의 값을 비선형 값으로 바꾸어주는 Activation function으로 이루어진다.

##### Filter(필터)

Filter는 그 특징이 data에 있는지 없는지를 검출해주는 함수이다.

행렬 형태로, 특성을 가지고 있는 input과 곱한다면 큰 결과 값(Multiplication and Summation, 행렬곱의 합)이 나온다.

###### Stride

size가 큰 이미지에 size가 작은 Filter를 적용시킬 때, Filter를 이동하는 간격.

필터를 일정 간격 씩 이동시키며 결과값을 계산한 행렬을 저장한다.

###### Feature map(Activation map)

Filter를 Stride 간격으로 적용시켜 얻어낸 결과값을 저장한 행렬.

ex.
Image size: 6x6
Filter size: 3x3
Stride: 1
Convolved Feature: 4x4

###### Padding

Filter에 의해 특징이 유실되는 것을 방지하기 위해 Convolved Feature 주위로 0 값을 넣어 size를 인위적으로 키우는 기법.

Filter를 적용하여 얻은 Convolved Feature의 size는 원본 image보다 작기 마련이라, 처음에 비해 특징이 유실되게 됨.
Padding은 이를 방지하기 위한 기법으로, 원본 image의 size가 되도록 Convolved Feature 주위를 0으로 둘러싼다.

원래의 특징을 희석시켜 Overfitting 현상을 방지하는 역할도 한다.

##### Activation function

Filter로 얻어진 값을 0과 1 사이의 값으로 리턴하는 함수.

ex.
Sigmoid function, ReLU function

###### ReLU

R(z) = max(0, z)

Sigmoid function이 Gradient vanishing을 야기하는 것을 방지하기 위해 고안됨.

#### Pooling(Sub sampling, 풀링, 서브 샘플링)

추출된 Activation map을 인위로 줄여서 요약하는 작업.

1. 전체 data의 size가 줄어들어 연산에 들어가는 컴퓨팅 리소스가 적어진다.
1. data의 크기를 줄이면서 소실이 발생하기 때문에 Overfitting을 방지할 수 있다.

##### Max pooling(맥스 풀링)

Activation map을 일정 크기로 잘라낸 후, 그 안에서 가장 큰 값을 뽑아내는 방법.

ex.
Image size: 4x4
Max pooling filter size: 2x2
Stride: 2
Result size: 2x2

#### 구조도

Convolutional filter, Activation function, 그리고 Pooling을 반복적으로 저합하여 특징을 추출한다.

ex.
Image -> CONV RELU CONV RELU POOL CONV RELU CONV RELU POOL CONV RELU CONV RELU POOL -> Feature

#### Neural Network

한 image에 여러 Convolutional filter를 적용하여 다양한 Feature들을 추출하고, 이를 Neural Network에 넣어서 분류한다.

##### 구조

여러 Feature -> Fully-connected Layer -> Dropout Layer -> Softmax Function -> Result

##### Fully-connected Layer

일반적인 Neural Network.

##### Softmax function

Activation fuction의 일종으로, 여러 개의 분류를 가지는 함수.

ex.
A형일 확률 0.7, B형일 확률 0.2, C형일 확률 0.1..

##### Dropout Layer

Overfitting을 방지하기 위해 Neural Network가 학습 중일 때 랜덤하게 뉴런을 꺼서 학습을 방해해, 모델이 Training data에 치중하는 것을 방지하는 것.

## Frameworks

### Caffe

비전 데이터에 특화되어 있는 머신-비전 라이브러리.

1. C++(Python, Matlab)
1. CUDA 가속 지원
1. CPU와 GPU 연산 전환 가능

### Google TensorFlow

복수의 노드에서 확장될 수 있도록 개발된 기계학습 프레임워크.

1. C++, Python

### Theano/Pylearn2

1. Python
1. Theano: 다차원 배열을 다루는 라이브러리.
1. Pylearn2: 머신러닝 라이브러리.

### Torch

1. Lua
1. 머신러닝 프레임워크.
1. Google과 Facebook에서 사용.

## Reference

조대협의 블로그 http://bcho.tistory.com
Deep Learning for Vision by Adam Coates http://videolectures.net/bmvc2013_coates_machine_vision/
A Beginner's Guide To Understanding Convolutional Neural Networks https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/