---
title:  "(TIL) 자연어 처리 - 19. GPT(Generative Pre-trained Transformer)"
excerpt: " "

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-31
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 19. GPT(Generative Pre-trained Transformer)

BERT가 트랜스포머의 인코더로 설계된 모델
GPT는 트랜스포머의 디코더로 설계된 모델이다.

![](https://wikidocs.net/images/page/184363/gpt0.PNG)

![](/images/../images/2025-01-31-19-06-40.png)

### 복습
언어 모델(Language Model): 인공 지능 분야에서 컴퓨터가 사람의 언어를 이해하고 생성할 수 있도록 하는 기술

언어 모델은 이전 단어들로부터 다음 단어를 예측하는 생성 모델이고, GPT는 수많은 언어 모델 중 하나.

LLM: 대규모 데이터 학습 및 다양한 자연어 처리 작업 수행

### GPT 구조

![](https://wikidocs.net/images/page/184363/gpt2.png)       
이전 단어들로부터 다음 단어를 예측하는 모델이기 때문에, 다음 단어를 지속적으로 생성할 수 있어 기본적으로 글쓰기가 가능한 생성 모델

![](https://wikidocs.net/images/page/184363/gpt3.PNG)       
트랜스포머 디코더 층이 16 챕터에서 배웠던 초기 트랜스포머 디코더 층과 크게 다르지 않음 