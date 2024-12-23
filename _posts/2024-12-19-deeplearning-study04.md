---
title:  "딥러닝 - 04. 카운트 기반의 단어 표현"
excerpt: "**Bag of Words**: 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법"

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP, WIL]

toc: true

last_modified_at: 2024-12-19
thumbnail: ../images/2024-12-04-11-03-02.png
---

# 04. 카운트 기반의 단어 표현

**Bag of Words**: 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법
-> 단어의 빈도수를 카운트(Count)하여 단어를 수치화한다.

- Bag of Words 생성 과정    
```
(1) 각 단어에 고유한 정수 인덱스를 부여합니다.  # 단어 집합 생성.
(2) 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만듭니다.  
```

```py
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])
print('bag of words vector :',vect.fit_transform(text).toarray())
print('vocabulary :',vect.vocabulary_)
```

```
bag of words vector : [[1 1 1 1 1]]
vocabulary : {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
```

# 이주의 회고
흐어. 책 네 챕터를 정리하느라 진을 다 뺐다. 그래도 이렇게 공부하면서 차곡차곡 쌓여가는 기록을 보니 많이 뿌듯하다. ^-^ 한편 백엔드 프로젝트가 본격화되었다. 챗봇 서비스에 대한 플로우차트는 팀원 분들과 논의하며 좀더 고도화하자. 

플로우차트 작성법: 사용자의 입장에서 필요한 화면이 합리적인 순서대로 노출되고 있는지 불필요한 프로세스는 없는지 지속적으로 점검한다.  
[UIUX : 앱 서비스 설계 - 플로우차트 작성법](https://yeon-design.tistory.com/21#google_vignette)