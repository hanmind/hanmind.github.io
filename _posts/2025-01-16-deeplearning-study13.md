---
title:  "(TIL) 자연어 처리 - 13. 서브워드 토크나이저(Subword Tokenizer)"
excerpt: " "

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-16
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 13. 서브워드 토크나이저(Subword Tokenizer)

## OOV
- 기계가 모르는 단어가 등장하면 그 단어를 단어 집합에 없는 단어란 의미에서 OOV(Out-Of-Vocabulary) 또는 UNK(Unknown Token)라고 표현
- OOV 문제: 모르는 단어로 인해 문제를 푸는 것이 까다로워지는 상황

## 서브워드 분리(Subword segmenation) 
: 하나의 단어는 더 작은 단위의 의미있는 여러 서브워드들(Ex) birthplace = birth + place)의 조합으로 구성된 경우가 많기 때문에, 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩하겠다는 의도를 가진 전처리 작업        
*이 책에서는 이런 작업을 하는 토크나이저를 서브워드 토크나이저라고 명명한다.*

## BPE(Byte Pair Encoding)
: OOV(Out-Of-Vocabulary) 문제를 완화하는 대표적인 서브워드 분리 알고리즘

어떤 훈련 데이터로부터 각 단어들의 빈도수를 카운트했다고 해보자.        
```
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
low : 5, lower : 2, newest : 6, widest : 3
```     
*+a. 여기서는 각 단어와 각 단어의 빈도수가 기록되어져 있는 결과를 임의로 딕셔너리라고 부르기로 한다.*

그렇다면 이 훈련 데이터의 단어 집합(vocabulary)은:        
```
# vocabulary
low, lower, newest, widest
```

이 경우 테스트 과정에서 'lowest'란 단어가 등장한다면 기계는 이 단어를 학습한 적이 없으므로 해당 단어에 대해서 제대로 대응하지 못하는 OOV 문제가 발생

-> BPE 알고리즘을 사용하자

1. 딕셔너리의 모든 단어들을 글자(chracter) 단위로 분리      
```
# dictionary
l o w : 5,  l o w e r : 2,  n e w e s t : 6,  w i d e s t : 3
```

2. 위 딕셔너리를 참고로 한 단어 집합(vocabulary)은:
```
# vocabulary
l, o, w, e, r, n, s, t, i, d
```

BPE에서는 알고리즘의 동작을 몇 회 반복(iteration)할 것인지를 사용자가 정한다. 여기서 알고리즘 동작이란, **가장 빈도수가 높은 유니그램의 쌍을 하나의 유니그램으로 통합**하는 과정이다.

1회) 빈도수가 9로 가장 높은 (e, s)의 쌍을 es로 통합        
```
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3
```

2회) 빈도수가 9로 가장 높은 (es, t)의 쌍을 est로 통합       
```
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3
```

3회) 빈도수가 7로 가장 높은 (l, o)의 쌍을 lo로 통합     
```
# dictionary update!
lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3
```

...

이런 식으로 10회를 반복하면 아뢔 같은 딕셔너리와 vocabulary 집합을 얻는다.      
```
# dictionary update!
low : 5,
low e r : 2,
newest : 6,
widest : 3
```

```
# vocabulary update!
l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
```

![](https://wikidocs.net/images/page/22592/%EA%B7%B8%EB%A6%BC.png)

이 경우 테스트 과정에서 'lowest'란 단어가 등장한다면, 
- 기계는 우선 'lowest'를 전부 글자 단위로 분할
    - 즉, 'l, o, w, e, s, t'가 됨
- 그리고 기계는 위의 단어 집합을 참고로 하여 'low'와 'est'를 찾아낸다. 
    - 즉, 'lowest'를 기계는 'low'와 'est' 두 단어로 인코딩

13-1 코드로 이해하기

BPE 외에도 BPE를 참고하여 만들어진 Wordpiece Tokenizer나 Unigram Language Model Tokenizer와 같은 서브워드 분리 알고리즘이 있다.

## 센텐스피스(SentencePiece)
- BPE를 포함하여 기타 서브워드 토크나이징 알고리즘들을 내장한 구글의 패키지
- 내부 단어 분리에 유용
- 사전 토큰화 작업없이 단어 분리 토큰화를 수행하므로 어떤 언어에도 적용 가능

## SubwordTextEncoder
: 텐서플로우를 통해 사용할 수 있는 서브워드 토크나이저      
- BPE와 유사한 알고리즘인 Wordpiece Model을 채택

## 허깅페이스 토크나이저
허깅페이스가 개발한 패키지 tokenizers는 자주 등장하는 서브워드들을 하나의 토큰으로 취급하는 다양한 서브워드 토크나이저를 제공한다.

### BERT의 워드피스 토크나이저(BertWordPieceTokenizer)
구글이 공개한 딥 러닝 모델 BERT에는 WordPiece Tokenizer가 사용되었다. 허깅페이스는 해당 토크나이저를 직접 구현하여 **tokenizers**라는 패키지를 통해 버트 워드피스토크나이저(BertWordPieceTokenizer)를 제공한다.

이외에도 ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer 등이 있으므로 자유롭게 선택해 사용할 수 있다.

- BertWordPieceTokenizer : BERT에서 사용된 워드피스 토크나이저(WordPiece Tokenizer)
- CharBPETokenizer : 오리지널 BPE
- ByteLevelBPETokenizer : BPE의 바이트 레벨 버전
- SentencePieceBPETokenizer : 앞서 본 패키지 센텐스피스(SentencePiece)와 호환되는 BPE 구현체