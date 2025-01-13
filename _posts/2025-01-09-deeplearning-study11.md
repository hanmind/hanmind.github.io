---
title:  "(TIL) 딥러닝 - 11. NLP를 위한 합성곱 신경망(Convolution Neural Network)"
excerpt: "11. NLP를 위한 합성곱 신경망(Convolution Neural Network)
비전 분야에서 사용되는 알고리즘이지만 이를 응용해서 자연어 처리에 사용하기 위한 시도가 있었다."

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-09
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 11. NLP를 위한 합성곱 신경망(Convolution Neural Network)
비전 분야에서 사용되는 알고리즘이지만 이를 응용해서 자연어 처리에 사용하기 위한 시도가 있었다.

자연어 처리를 위한 1D 합성곱 신경망을 이해해보자.

## 합성곱 신경망의 구성
- 합성곱층과(Convolution layer)
- 풀링층(Pooling layer)

![](https://wikidocs.net/images/page/64066/convpooling.PNG)         
CONV는 합성곱 연산을 의미하고, 합성곱 연산의 결과가 활성화 함수 ReLU를 지납니다. 이 두 과정을 합성곱층이라고 합니다. 그 후에 POOL이라는 구간을 지나는데 이는 풀링 연산을 의미하며 풀링층

1차원으로 변환된 결과는 이게 원래 어떤 이미지였는지 알아보기가 어렵다.      
-> 결국 이미지의 공간적인 구조 정보를 보존하면서 학습할 수 있는 방법이 필요해졌고, 이를 위해 합성곱 신경망을 사용하게 됨

- 이미지 = (높이, 너비, 채널)이라는 3차원 텐서
- 채널: 색 성분

## 합성곱 연산      
![](https://wikidocs.net/images/page/64066/conv5.png)       
- (1×1) + (2×0) + (3×1) + (2×1) + (1×0) + (0×1) + (3×0) + (0×1) + (1×0) = 6
- (2×1) + (3×0) + (4×1) + (1×1) + (0×0) + (1×1) + (0×0) + (1×1) + (1×0) = 9

위 연산을 9번의 스텝까지 마친 결과 
![](https://wikidocs.net/images/page/64066/conv8.png)       
: 위와 같이 입력으로부터 커널을 사용하여 합성곱 연산을 통해 나온 결과 **특성맵**이라고 부름       
-  커널의 크기, 스트라이드(이동 범위)는 사용자가 직접 정할 수 있다.

### 패딩
![](https://wikidocs.net/images/page/64066/conv10.png)
합성곱 연산 이후에도 특성 맵의 크기가 입력의 크기와 동일하게 유지되도록 하기 위해 가장자리에 지정된 개수의 폭만큼 행, 열 추가

...(생략)

## 1D 합성곱(1D Convolutions)
![](https://wikidocs.net/images/page/80437/%EB%84%A4%EB%B2%88%EC%A7%B8%EC%8A%A4%ED%85%9D.PNG)

1D 합성곱 연산과 자연어 처리 관점에서는 커널의 크기에 따라서 참고하는 단어의 묶음의 크기가 달라진다 = 이는 참고하는 n-gram이 달라지는 것을 의미!

## 맥스 풀링
이미지 처리에서의 CNN에서 그랬듯이, 일반적으로 1D 합성곱 연산을 사용하는 1D CNN에서도 합성곱 층(합성곱 연산 + 활성화 함수) 다음에는 풀링 층을 추가한다. 
- 그 중 대표적으로 사용되는 것이 **맥스 풀링(Max-pooling)**

