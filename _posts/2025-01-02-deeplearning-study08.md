---
title:  "(TIL) 딥러닝 - 08. 순환 신경망(Recurrent Neural Network)"
excerpt: ""

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-02
thumbnail: ../images/2024-12-04-11-03-02.png
---
![](/images/../images/2024-12-04-11-03-02.png)

# 08. 순환 신경망(Recurrent Neural Network)
피드 포워드 신경망은 입력의 길이가 고정되어 있어 자연어 처리를 위한 신경망으로는 한계가 있었다. 다양한 길이의 입력 시퀀스를 처리할 수 있는 순환 신경망 - 바닐라 RNN, 이를 개선한 LSTM, GRU -를 공부해보자.

# 순환 신경망(Recurrent Neural Network, RNN)
입력과 출력을 시퀀스 단위로 처리하는 시퀀스(Sequence) 모델 중 가장 기본적인 인공 신경망 시퀀스 모델

![](https://wikidocs.net/images/page/22886/rnn_image1_ver2.PNG)      
(편향 b도 입력으로 존재할 수 있지만 당분간 그림에서는 생략)
셀: RNN에서 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드. 메모리 셀 또는 RNN 셀이라고도 함.

![](/images/../images/2025-01-02-18-32-52.png)

![](/images/../images/2025-01-02-18-38-41.png)

RNN은 시점마다 파라미터가 같다고 가정함 => Parameter Sharing!   
그래서 파라미터가 많아보이더라도 사실상 세 개다.

중간은 반복(Recurrent)되므로 간단한 그림으로 표현할 수 있다.

## 순환 신경망 구조 다양성
Many to many 구조:
- 순차적인 x로 순차적인 y를 예측하는 문제
- 예시: 영어 문장 -> 한글 문장 번역

### Sequence to Sequence (seq2seq)
![](/images/../images/2025-01-02-18-41-57.png)
4번째 그림: 입력이 다 끝나고 출력이 이루어지는 구조

- L_t = y_t와 yhat_t의 차이
- 1 to many, many to many의 구조에서는 **각 시점별 loss의 평균**을 전체 loss로 활용
![](/images/../images/2025-01-02-18-52-32.png)

## 양방향 순환 신경망(Bidirectional Recurrent Neural Network)
시점 t에서의 출력값을 예측할 때 이전 시점의 입력뿐만 아니라, 이후 시점의 입력 또한 예측에 기여할 수 있는 순환 신경망

```
운동을 열심히 하는 것은 [        ]을 늘리는데 효과적이다.

1) 근육
2) 지방
3) 스트레스
```
![](https://wikidocs.net/images/page/22886/rnn_image5_ver2.PNG)

주황색 메모리 셀: **앞** 시점의 은닉 상태(Forward States)를 전달받아 현재의 은닉 상태를 계산
연두색 메모리 셀: **뒤** 시점의 은닉 상태(Backward States) 를 전달 받아 현재의 은닉 상태를 계산

![](https://wikidocs.net/images/page/22886/rnn_image6_ver3.PNG) 
양방향 RNN도 다수의 은닉층을 가질 수 있다. 은닉층을 추가하면 학습할 수 있는 양이 많아지지만 반대로 훈련 데이터 또한 많은 양이 필요하다.

# 장단기 메모리(Long Short-Term Memory, LSTM)
## 바닐라 RNN의 한계
바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있다. 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생! => **장기 의존성 문제(the problem of Long-Term Dependencies)**      

어쩌면 가장 중요한 정보가 시점의 앞 쪽에 위치할 수도 있다.

## LSTM
LSTM은 은닉층의 메모리 셀에 입력 게이트, 망각 게이트, 출력 게이트를 추가하여 불필요한 기억을 지우고, 기억해야할 것들을 정함. 
=> 은닉 상태(hidden state)를 계산하는 식이 전통적인 RNN보다 조금 더 복잡해졌으며 **셀 상태(cell state)**라는 값이 추가됨.  
## LSTM 구조
![](/images/../images/2025-01-02-19-25-59.png)

세 개의 Gate, Cell state 개념이 사용됨.

## 1. 세 개의 Gate 계산
### Forget gate 
![](/images/../images/2025-01-02-19-28-50.png)

### Input gate
![](/images/../images/2025-01-02-19-35-35.png)

### Output gate  
![](/images/../images/2025-01-02-19-37-08.png)

=> 세 개의 게이트 각각이 가중치 역할을 한다. (위의 LSTM 구조 다시 참조)

## 2. cell state 업데이트
시점 t에 대한 cell state 업데이트 식:

![](/images/../images/2025-01-02-19-45-24.png)


![](/images/../images/2025-01-02-19-42-46.png)

즉, Forget gate가 이전 Cell state의 중요도를 결정해줌.

![](/images/../images/2025-01-02-19-44-46.png)

즉, Input 데이터가 현재 정보의 중요도 결정

*=> 불필요한 정보는 잊고, 추가할 정보는 추가해서 **cell state**를 구성*

## 3. hidden state 업데이트
최종적으로 단기적 정보 hidden state를 업데이트

![](/images/../images/2025-01-02-19-49-48.png)

즉, Output gate: 어떤 정보를 output으로 내보낼지 결정!

## LSTM 전체 구조 정리
![](/images/../images/2025-01-02-19-52-50.png)

# 게이트 순환 유닛(Gated Recurrent Unit, GRU)
LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄여 LSTM의 구조를 간단화

LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재한 반면, GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재. But GRU는 LSTM보다 학습 속도가 빠르면서도 LSTM과 비슷한 성능을 보인다.
 
경험적으로 데이터 양이 적을 때는 매개 변수의 양이 적은 GRU가 조금 더 낫고, 데이터 양이 더 많으면 LSTM이 더 낫다고도 함.

# 오늘의 회고
GPT2 영어 모델에 넣을 데이터 전처리와, 기본 틀을 정리한 날이다. 오후에는 자연어 처리 스터디 준비를 했다. 다음주부터 본격적으로 쿠버네티스 스터디가 시작되니, 그전에 도커 개념을 잘 이해하고 혼자 연습해봐야겠다. 아래 도커 강의의 도움을 받고 있다.   
[추석특집 도커 (Docker) 기초 강의 몰아보기](https://www.youtube.com/watch?v=IqnAiM1A0d8&list=PLlTylS8uB2fDLJRJCXqUowsOViG-ZKnWy&index=16)

[v] 개인과제 제출
[v] 월, 화 TIL 일단 초안으로 제출
[v] GPT2 모델 구현 -> 내일 마저
[v] 2시 회의
[v] 3시 쿠버네티스 스터디
[] FastAPI, 장고 공부 -> 내일
[v] 자연어 처리 자료 정리!!
[v] 7시 반 자연어 처리 스터디
