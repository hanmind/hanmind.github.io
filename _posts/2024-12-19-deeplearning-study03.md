---
title:  "딥러닝 - 03. 언어 모델"
excerpt: "언어 모델(Language Model, LM): 언어라는 현상을 모델링하고자 **단어 시퀀스(문장)에 확률을 할당(assign)하는 모델**"

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2024-12-19
thumbnail: ../images/TIL.png
---

# 03. 언어 모델
언어 모델(Language Model, LM): 언어라는 현상을 모델링하고자 **단어 시퀀스(문장)에 확률을 할당(assign)하는 모델**

## 모델링 방법
- 통계를 이용한 방법
- 인공 신경망을 이용한 방법 -> 성능 good!

단어 시퀀스에 확률을 할당하게 하기 위해서 가장 보편적으로 사용되는 방법은 언어 모델이 **이전 단어들이 주어졌을 때 다음 단어를 예측**하도록 하는 방식.

## 확률 할당의 의미
언어 모델은 확률을 통해 **보다 적절한 문장을 판단**한다.
```
P(나는 버스를 탔다) > P(나는 버스를 태운다)
선생님이 교실로 부리나케  
P(달려갔다) > P(잘려갔다)
P(머신러닝) > P(머닝러신)
P(나는 메롱을 먹는다) < P(나는 메론을 먹는다)
```
이렇게 기계 번역, 오타 교정, 음성인식 등에서 유용하다.

# 통계적 언어 모델

## 조건부 확률
각 단어는 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어입니다. 그리고 모든 단어로부터 하나의 문장이 완성된다. 
=> 따라서 조건부 확률의 연쇄 법칙을 이용해 문장의 확률을 구할 수 있다.

- 다섯번째 단어의 확률    
![](/images/../images/2024-12-19-19-14-20.png)

- 전체 단어 시퀀스 W의 확률   
![](/images/../images/2024-12-19-19-15-04.png)

문장의 확률 = 각 단어들이 이전 단어가 주어졌을 때 다음 단어로 등장할 확률의 **곱**

예시:  'An adorable little boy is spreading smiles'의 확률    
![](/images/../images/2024-12-19-19-17-07.png)

## n-gram 언어 모델
카운트에 기반한 통계적 접근을 사용하고 있으므로 SLM의 일종이지만, 앞서 배운 언어 모델과는 달리 이전에 등장한 모든 단어를 고려하는 것이 아니라 **일부 단어만 고려**하는 접근 방법.

SLM의 한계: 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다.
-> 앞 몇 개의 단어만 고려해서 근사하자!         
![](/images/../images/2024-12-19-19-19-54.png)

- n-gram: n개의 연속적인 단어 나열
    - n이 1일 때는 유니그램(unigram), 2일 때는 바이그램(bigram), 3일 때는 트라이그램(trigram)이라고 명명하고 n이 4 이상일 때는 gram 앞에 그대로 숫자를 붙여서 명명

Q. 그러면 unigram은 어떻게 되는 건지.?

### n-gram의 한계
![](/images/../images/2024-12-19-19-24-31.png)

n=4일 때, 앞 세 개의 단어만 고려시 boy is spreading 뒤에 등장할 확률이 높은 단어는 insulting이다. 그런데 그 앞에 있었던 표현을 포함해서 다시 문장을 읽어보자:   
```(An adorable little) boy is spreading insulting   ```   
❗ 앞의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다.
- (1) 희소 문제(Sparsity Problem)
- (2) n을 선택하는 것은 trade-off 문제