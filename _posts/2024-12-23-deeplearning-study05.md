---
title:  "딥러닝 - 05. 벡터의 유사도"
excerpt: "벡터의 유사도
문장이나 문서의 유사도를 구하는 작업

## 코사인 유사도
코사인 유사도: 두 벡터 간의 **코사인 각도**를 이용하여 구할 수 있는 두 벡터의 유사도"

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2024-12-23
thumbnail: ../images/TIL.png
---

# 05. 벡터의 유사도
문장이나 문서의 유사도를 구하는 작업

## 코사인 유사도
코사인 유사도: 두 벡터 간의 **코사인 각도**를 이용하여 구할 수 있는 두 벡터의 유사도
- 1: 두 벡터의 방향이 완전히 동일한 경우
- 0: 두 벡터의 방향이 90°의 각을 이루는 경우
- -1: 두 벡터의 방향이 180°로 반대일 경우

문서 단어 행렬(Document-Term Matrix, DTM)이나 TF-IDF 행렬을 통해서 문서의 유사도를 구하는 경우에는 문서 단어 행렬이나 TF-IDF 행렬이 각각의 특징 벡터 A, B가 된다.
예시:
- 문서1: 나는 딸기 좋아
- 문서2: 나는 바나나 좋아
- 문서3: 나는 바나나 좋아 나는 바나나 좋아

 . | 바나나 | 딸기 | 나는 | 좋아
----|----|----|----|----
문서1 | 0 | 1 | 1 | 1
문서2 | 1 | 0 | 1 | 1
문서3 | 2 | 0 | 2 | 2

이렇게 해서 각각의 문서가 벡터의 형태로 표현되었다. 이제 벡터 간 코사인 유사도를 계산해보자.
```py
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

print('문서 1과 문서2의 유사도 :',cos_sim(doc1, doc2))
print('문서 1과 문서3의 유사도 :',cos_sim(doc1, doc3))
print('문서 2와 문서3의 유사도 :',cos_sim(doc2, doc3))
```
결과:
```
문서 1과 문서2의 유사도 : 0.67
문서 1과 문서3의 유사도 : 0.67
문서 2과 문서3의 유사도 : 1.00
```
### 코사인 유사도의 장점
예를 들어 어떤 문서 A, B, C가 있다. A, B는 동일한 주제이고 C는 다른 주제의 문서이다. 그런데 A, C는 문서 길이가 비슷한 반면 B는 A, C 두 배의 길이를 가진다. 이 경우 문서의 유사도를 유클리드 거리로 연산하면 문서 A가 문서 B보다 문서 C와 유사도가 더 높게 나오는 상황이 발생할 수 있다! 이는 문서의 길이가 유사도 연산에 영향을 주었기 때문이다.  

💡 코사인 유사도는 유사도를 구할 때 벡터의 방향(패턴)에 초점을 두므로 코사인 유사도는 **문서의 길이가 다른 상황에서 비교적 공정한 비교**를 할 수 있도록 도와준다.

+a. 유클리드 거리란?  
![](https://wikidocs.net/images/page/24654/2%EC%B0%A8%EC%9B%90_%ED%8F%89%EB%A9%B4.png)

유클리드 거리는 우리가 일반적으로 피타고라스 정리를 통해 구하는 두 점 사이의 거리를 의미한다. 문서의 유사도를 구할 때 자카드 유사도나 코사인 유사도만큼 유용하지는 않지만, 거리 연산의 기본적인 방법 중 하나이다.

## 자카드 유사도(Jaccard similarity)
자카드 유사도: A와 B 두개의 집합이 있을 때, 합집합 대비 교집합의 비율
- 0과 1 사이의 범위
- 두 집합이 동일하다면 1의 값을 가지고, 두 집합의 공통 원소가 없다면 0의 값을 가짐
- 두 문서 사이의 자카드 유사도 = 두 집합의 **교집합** 크기 / 두 집합의 **합집합** 크기

예시:
```py
doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"
# 토큰화
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()
print('문서1 :',tokenized_doc1)
print('문서2 :',tokenized_doc2)
```
```
문서1 : ['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
문서2 : ['apple', 'banana', 'coupon', 'passport', 'love', 'you']
```
문서1과 2의 합집합 및 교집합:
```py
union = set(tokenized_doc1).union(set(tokenized_doc2))
print('문서1과 문서2의 합집합 :',union)
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print('문서1과 문서2의 교집합 :',intersection)
```
합집합:
```
문서1과 문서2의 합집합 : {'you', 'passport', 'watch', 'card', 'love', 'everyone', 'apple', 'likey', 'like', 'banana', 'holder', 'coupon'}
```
교집합:
```
문서1과 문서2의 교집합 : {'apple', 'banana'}
```
이제 교집합의 크기 2를 합집합의 크기 12로 나누어주면 자카드 유사도를 구할 수 있다.
```py
print(len(intersection)/len(union)) # 0.16666666666666666
```
