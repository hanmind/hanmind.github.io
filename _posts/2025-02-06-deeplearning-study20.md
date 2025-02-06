---
title:  "(TIL) 자연어 처리 - 20. 사전 훈련된 인코더-디코더 모델"
excerpt: ""

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-02-06
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

본 게시글은 아래 위키독스를 바탕으로 자연어 처리를 공부하며 정리한 내용입니다.   
※ [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155)

# 20. 사전 훈련된 인코더-디코더 모델 

인코더 모델 BERT와 디코더 모델 GPT와는 달리 인코더-디코더 구조를 유지한 채 사전 학습된 모델인 BART와 T5에 대해서 알아보자.

## 인코더 BERT와 디코더 GPT
트랜스포머 구조를 제안한 논문 Attention is All you need에서 제안되었던 초기 트랜스포머는 인코더-디코더 구조를 가지고 있었다.
- 인코더는 일반적으로 자연어 이해(Natural Language Understanding, NLU) 능력이 뛰어남
- 디코더는 자연어 생성(Natural Language Generation, NLG) 능력을 보유

이후 인코더와 디코더는 서로 분리되어 각자의 장점을 극대화시켜 별도의 모델들로 발전하게 되었다.

## BART(Bidirectional Auto-Regressive Transformers)
BART(Bidirectional Auto-Regressive Transformers)는 2020년 페이스북 AI에서 발표한 모델로, BERT, GPT와 달리 인코더-디코더 구조로 **자연어 이해**와 **자연어 생성** 능력을 모두 적절히 가지고 있다.

### 학습 방식
![](https://wikidocs.net/images/page/256572/bert_gpt_bart.PNG)

- BERT: 양방향의 문맥을 반영하여 가려진 단어를 맞추는 식으로 학습
- GPT: 이전 단어들로부터 다음 단어를 예측하는 방식으로 학습
- BART: 인코더에서 훼손된 문장을 디코더에서 복원하는 방식으로 학습

BART는 사전 학습 시에 총 5개의 방법을 사용한다. 이 5가지 방법은 모두 인코더에 고의적으로 원본 문장을 훼손된 문장을 넣고 디코더에서 원본 문장을 다시 재복원하도록 하여 언어에 대한 이해를 높이는 데에 집중한다!

### 1) Token Masking
: 입력 문장에서 임의의 단어 **하나**를 마스킹하여 입력 문장을 훼손시키고, 디코더가 원래 문장을 정확하게 예측, 복원하도록 학습하는 방식

**인코더 입력과 디코더의 레이블**   
- 인코더 입력(변형된 문장): "The children played `<mask>` outside until it got dark."
- 디코더의 레이블(원본 문장): "The children played soccer outside until it got dark."

=> 변형된 문장으로부터 원본 문장을 복원하도록 학습

=> 전체 문장 구조와 흐름을 이해하고 재구성하는 능력을 개발

### 2) Text Infilling
: 입력 문장에서 하나 이상의 **연속된 단어**를 마스킹하여 입력 문장을 훼손시키고, 디코더가 원래 문장을 정확하게 예측, 복원하도록 학습하는 방식

**인코더 입력과 디코더의 레이블**     
- 인코더 입력(변형된 문장): "He goes `<mask>` to the gym."
- 디코더의 레이블(원본 문장): "He goes **to school and then** to the gym."

=> Token Masking보다 더 어려운 문제이기 때문에 모델의 문맥 이해 능력과 연속적인 텍스트 생성 능력을 더욱 향상

### 3) Sentence Permutation; 문장 순서 바꾸기
: 여러 문장으로 구성된 텍스트의 문장 순서를 무작위로 섞고, 디코더가 원래 순서의 문장들을 정확하게 예측, 복원하도록 학습하는 방식

**인코더 입력과 디코더의 레이블**   
- 인코더 입력(순서가 섞인 문장들): "They went to the cinema. He met his friend. He left the house."
- 디코더의 레이블(원래 순서의 문장들): "He left the house. He met his friend. They went to the cinema."

=> Token Masking이나 Text Infilling보다 더 큰 범위의 문맥을 다루기 때문에, 모델의 문서 수준 이해 능력을 크게 향상

### 4) Document Rotation; 문서 회전
: 전체 문서의 일부를 잘라내어 그 부분을 문서의 시작으로 설정하는 방식

문서의 일부를 잘라 시작 부분으로 옮긴다면 BART는 인코더에 재배열된 문서가 들어가고, 디코더의 레이블은 원래 순서의 문서가 들어가서 재배열된 문서로부터 원래 순서의 문서를 복원하도록 학습된다. 

**인코더 입력과 디코더의 레이블**   
- 인코더 입력(재배열된 문서): "She stayed dry. It was raining. She opened her umbrella."
- 디코더의 레이블(원래 순서의 문서): "It was raining. She opened her umbrella. She stayed dry."

=> BART의 문서 이해 능력을 종합적으로 향상

### 5) Token Deletion
: 입력 문장에서 무작위로 토큰을 삭제하는 방식

**인코더 입력과 디코더의 레이블**   
- 인코더 입력(토큰이 삭제된 문장): "The cat on the mat."
- 디코더의 레이블(원래 문장): "The cat sat on the mat."

=> 더 강력한 문맥 추론 능력과 누락된 정보 복원 능력을 개발