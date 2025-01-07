---
title:  "(WIL) 딥러닝 - 07. 딥 러닝"
excerpt: "퍼셉트론: 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘   
x: 입력값 
w: 가중치(Weight) 
y: 출력값"

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-02
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 07. 딥러닝

# 퍼셉트론
퍼셉트론: 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘
![](https://wikidocs.net/images/page/24958/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A01.PNG)   
x: 입력값 
w: 가중치(Weight)
y: 출력값

![](/images/../../AI-Study-2024/images/2025-01-02-17-29-05.png)   
계단 함수(Step function): 가중치의 곱의 전체 합이 임계치(threshold, Θ)를 넘으면 종착지에 있는 인공 뉴런은 출력 신호로서 1을 출력하고, 그렇지 않을 경우에는 0을 출력하는 함수

![](/images/../../AI-Study-2024/images/2025-01-02-17-29-16.png)   
임계치를 좌변으로 넘기고 **편향(bias)**으로 표현할 수도 있다.

![](https://wikidocs.net/images/page/24958/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A02.PNG)

## 다층 퍼셉트론
**단층 퍼셉트론의 한계:**   
단층 퍼셉트론은 AND 게이트, NAND 게이트, OR 게이트를 구현할 수 있으나 지금부터 설명할 XOR 게이트는 구현할 수 없다. (XOR 게이트: 입력값 두 개가 서로 다른 값을 갖고 있을때에만 출력값이 1이 되고, 입력값 두 개가 서로 같은 값을 가지면 출력값이 0이 되는 게이트)   
-> 다층 퍼셉트론(MultiLayer Perceptron, MLP)으로 해결 가능!

![](https://wikidocs.net/images/page/24958/perceptron_4image.jpg)

단층 퍼셉트론은 입력층과 출력층만 존재하지만, 다층 퍼셉트론은 중간에 층을 더 추가하였다. 이렇게 입력층과 출력층 사이에 존재하는 층을 은닉층(hidden layer)이라고 한다.

# 인공 신경망
피드 포워드 신경망(Feed-Forward Neural Network, FFNN): 층 퍼셉트론(MLP)과 같이 오직 입력층에서 출력층 방향으로 연산이 전개되는 신경망
순환 신경망(RNN): 은닉층의 출력값을 출력층으로 값을 보내면서도, 동시에 은닉층의 출력값이 다시 은닉층의 입력으로 사용되는 신경망

전결합층(Fully-connected layer, FC, Dense layer): 다층 퍼셉트론과 같이, 어떤 층의 모든 뉴런이 이전 층의 모든 뉴런과 연결돼 있는 층

## 활성화 함수
활성화 함수(Activation function): 퍼셉트론의 계단 함수와 같이, 은닉층과 출력층의 뉴런에서 출력값을 결정하는 함수

활성화 함수는 비선형 함수여야 한다.

### Sigmoid
![](https://wikidocs.net/images/page/60683/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%ED%95%A8%EC%88%982.PNG)

시그모이드 함수를 미분한 값은 최대 0.25로 작은 편이다. 이런 시그모이드 함수를 활성화 함수로하는 인공 신경망의 층을 쌓는다면, 가중치와 편향을 업데이트 하는 과정인 역전파 과정에서 0에 가까운 값이 누적해서 곱해지게 되면서, 앞단에는 기울기(미분값)가 잘 전달되지 않는 **기울기 소실(Vanishing Gradient)** 문제가 발생한다. -> 매개변수 w가 업데이트 되지 않아 학습이 되지 않을 수 있다. (뒤에서 더 설명)

### 하이퍼볼릭탄젠트
![](https://wikidocs.net/images/page/60683/%ED%95%98%EC%9D%B4%ED%8D%BC%EB%B3%BC%EB%A6%AD%ED%83%84%EC%A0%A0%ED%8A%B8.PNG)

하이퍼볼릭탄젠트 함수의 경우에는 시그모이드 함수와는 달리 0을 중심으로 하고있는 형태의 함수이다. 하이퍼볼릭탄젠트 함수를 미분한 값은 최대 1로, 시그모이드 함수의 최대값인 0.25보다 크다. 따라서 시그모이드 함수보다는 기울기 소실 증상이 적은 편이며 은닉층에서 시그모이드 함수보다는 선호된다.

### ReLU
![](https://wikidocs.net/images/page/60683/%EB%A0%90%EB%A3%A8%ED%95%A8%EC%88%98.PNG)

입력값이 음수면 기울기(미분값)가 0이 되어버리는 죽은 렐루(**dying ReLU**) 문제 발생   

### Leaky ReLU
![](https://wikidocs.net/images/page/60683/%EB%A6%AC%ED%82%A4%EB%A0%90%EB%A3%A8.PNG)      
![](/images/../../AI-Study-2024/images/2025-01-02-17-55-28.png)
dying ReLU를 보완한 함수 중 하나가 Leaky ReLU이다. Leaky ReLU는 입력값이 음수일 경우에 0이 아니라 0.001과 같은 매우 작은 수를 반환한다. 

### Softmax
- 은닉층) 일반적으로 ReLU(또는 ReLU 변형) 함수들을 사용한다. 
- 출력층) 소프트맥스 함수와 시그모이드 함수가  주로 사용
  - 시그모이드 함수가 두 가지 선택지 중 하나를 고르는 이진 분류 (Binary Classification) 문제에 사용된다면 소프트맥스 함수는 세 가지 이상의 (상호 배타적인) 선택지 중 하나를 고르는 다중 클래스 분류(MultiClass Classification) 문제에 주로 사용됨.
  - 즉, 이진 분류를 할 때는 출력층에 앞서 배운 로지스틱 회귀를 사용하고, 다중 클래스 분류 문제를 풀 때는 출력층에 소프트맥스 회귀를 사용한다.

# 행렬곱으로 이해하는 신경망

# 딥러닝의 학습 방법

# 역전파

# 과적합(Overfitting)을 막는 방법들
## 1. 데이터의 양을 늘리기
## 2. 모델의 복잡도 줄이기
## 3. 가중치 규제(Regularization) 적용하기
- L1 규제 : 가중치 w들의 **절대값 합계**를 비용 함수에 추가
- L2 규제 : 모든 가중치 w들의 **제곱합**을 비용 함수에 추가
## 4. 드롭아웃(Dropout)
드롭아웃: 학습 과정에서 신경망의 일부를 사용하지 않는 방법

# 기울기 소실(Gradient Vanishing)과 폭주(Exploding)
## 기울기 소실(Gradient Vanishing)
기울기 소실: 역전파 과정에서 입력층으로 갈 수록 기울기(Gradient)가 점차적으로 작아지는 현상. 이렇게 입력층에 가까운 층들에서 가중치들이 업데이트가 제대로 되지 않으면 결국 최적의 모델을 찾을 수 없게 된다.

## 기울기 폭주(Gradient Exploding) 
기울기 폭주: 기울기가 점차 커지더니 가중치들이 비정상적으로 큰 값이 되면서 결국 발산되는 현상. 순환 신경망(Recurrent Neural Network, RNN)에서 쉽게 발생(챕터 08에서 설명!)

## 기울기 소실 또는 기울기 폭주를 막는 방법들
### 1. ReLU와 ReLU의 변형들
- 은닉층에서는 시그모이드 함수를 사용 X. ReLU나 Leaky ReLU와 같은 ReLU 함수의 변형들을 사용
- 모든 입력값에 대해서 기울기가 0에 수렴하지 않도록 Leaky ReLU를 사용함으로써 죽은 ReLU 문제를 해결

### 2. 그래디언트 클리핑(Gradient Clipping)
: 그대로 기울기 값을 자르는 것.   
기울기 폭주를 막기 위해 임계값을 넘지 않도록 값을 자름

### 3. 가중치 초기화(Weight initialization)
### 4. 배치 정규화(Batch Normalization)
### 5. 층 정규화(Layer Normalization)

# 이주의 회고
자연어 처리 스터디가 슬슬 어렵네요.   
저의 태도는 저번주보다는 나았어요. 그렇지만 강의들이 밀린 것은 스노우볼이 더 커지기 전에 꾸준히 상쇄해나가야겠습니다. 주말에 할 생각보다, 주어진 시간에 집중합시다.