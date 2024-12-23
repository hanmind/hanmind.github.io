---
title:  "딥러닝 - 02. 텍스트 전처리"
excerpt: "토큰화
- 코퍼스를 토큰 단위로 나누는 작업
- 토큰의 단위는 단어, 문장 등 다양하며, **의미 있는 단위**로 정의됨."

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP, TIL]

toc: true

last_modified_at: 2024-12-19
thumbnail: ../images/2024-12-04-11-03-02.png
---

# 02. 텍스트 전처리
## 토큰화
- 코퍼스를 토큰 단위로 나누는 작업
- 토큰의 단위는 단어, 문장 등 다양하며, **의미 있는 단위**로 정의됨.

## NLTK와 기타 도구를 이용한 토큰화
- **`word_tokenize`**:
  - `Don't` → `Do`와 `n't`로 분리.
  - `Jone's` → `Jone`과 `'s`로 분리.
- **`WordPunctTokenizer`**:
  - 구두점을 별도로 분리.
  - `Don't` → `Don`, `'`, `t`.
  - `Jone's` → `Jone`, `'`, `s`.
- **`text_to_word_sequence`** (Keras):
  - 모든 문자를 소문자로 변환.
  - 단순한 기준으로 토큰화 진행.

## 한국어에서의 토큰화
- 한국어는 수많은 코퍼스에서 띄어쓰기가 무시되는 경우가 많아 자연어 처리가 어려워졌다.
- 한국어는 띄어쓰기로만 단어를 분리하기 어려움 → 형태소 분석기를 활용하여 더 정교한 토큰화 필요

## 품사 태깅(Part-of-speech tagging)
fly   
(동사) 날다   
(명사) 파리

못   
(부사) 동작을 할 수 없다는 의미   
(명사) 망치를 사용해서 목재 따위를 고정하는 물건
→ 각 단어가 어떤 품사로 쓰였는지를 구분해놓는 작업, 품사 태깅을 진행할 수 있다.

### 영어 품사 태깅
```py
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence = word_tokenize(text)

print('단어 토큰화 :',tokenized_sentence)
print('품사 태깅 :',pos_tag(tokenized_sentence))
```
```
단어 토큰화 : ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
품사 태깅 : [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), ('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), ('student', 'NN'), ('.', '.')]
```

### 한국어 품사 태깅
한국어 자연어 처리를 위해서는 KoNLPy(코엔엘파이)라는 파이썬 패키지를 사용할 수 있다. KoNLPy로 사용할 수 있는 형태소 분석기로는 Okt(Open Korea Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)가 있다.

한국어 NLP에서 형태소 분석기를 사용하여 단어 토큰화, 더 정확히는 형태소 토큰화(morpheme tokenization)를 수행해보자.   
**1. Okt**
```py
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요")) 
```
```
OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
```
**2. 꼬꼬마**
```py
print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
```

```
꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']
```

## 정제 및 정규화
- 정제(cleaning) : 갖고 있는 코퍼스로부터 노이즈 데이터를 제거한다.
- 정규화(normalization) : 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어준다.
    - 예시: USA와 US, uh-huh와 uhhuh
    - 소문자 변환을 언제 사용할지 결정하는 머신 러닝 시퀀스 모델 사용 가능
    - 대문자를 써야하는 표현에서도 소문자를 쓴 데이터를 가지고 있다면, 모든 코퍼스를 소문자로 바꾸는 것이 더 실용적인 해결책이 될 수 있음

## 불필요한 단어 제거
- 등장 빈도가 적은 단어
    - 스팸 메일 분류기를 설계 중 100,000개의 메일 데이터에서 총 합 5번 밖에 등장하지 않은 단어 -> 분류에 거의 도움 X
- 길이가 짧은 단어
    - 영어권 언어에서는 길이가 짧은 단어를 삭제하는 것만으로도 자연어 처리에서 크게 의미가 없는 단어들을 제거하는 효과 有

## 어간 추출(Stemming) and 표제어 추출(Lemmatization)

### 표제어 추출
: 기본 사전형 단어 추출   
```
am, are, is → be   
```
단어의 빈도수를 기반으로 문제를 풀고자 하는 상황에서 주로 사용됨(추후 학습 예정).
### 표제어 추출 방법: 단어의 형태학적 파싱
형태학적 파싱: 어간과 접사를 분리하는 작업
예시: cats → cat, -s
1) 어간(stem)
: 단어의 의미를 담고 있는 단어의 핵심 부분
2) 접사(affix)
: 단어에 추가적인 의미를 주는 부분

### 어간 추출(stemming) 
: 어간(Stem)을 추출하는 작업   
```
formalize → formal
allowance → allow
electricical → electric
```
규칙 기반의 접근을 하고 있으므로 어간 추출 후의 결과에는 사전에 없는 단어들이 포함되어 있기도 하다.
포터 알고리즘의 어간 추출 예시:
```
ALIZE → AL
ANCE → 제거
ICAL → IC
```
포터 어간 추출기는 정밀하게 설계되어 정확도가 높은 편. NLTK에서는 포터 알고리즘 외에도 랭커스터 스태머(Lancaster Stemmer) 알고리즘을 지원함.

### 표제어 추출 vs 어간 추출 비교
Lemmatization
```
am → be
the going → the going
having → have
```

Stemming
```
am → am
the going → the go
having → hav
```

<!-- ### 한국어에서의 어간 추출
활용이란 용언의 어간(stem)이 어미(ending)를 가지는 일
- 규칙 활용: 잡(어간) + 다(어미)
- 불규칙 활용: 오르+ 아/어→올라, 하+아/어→하여, 이르+아/어→이르러, 푸르+아/어→푸르러 -->

## 불용어
유의미한 단어 토큰만을 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거하는 작업이 필요하다.
예: I, my, me, over, 조사, 접미사

### 영어 불용어 제거
```py
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt
```
`stopwords.words("english")`: NLTK가 정의한 영어 불용어 리스트

예시:
```
불용어 제거 전 : ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
불용어 제거 후 : ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```

### 한국어에서 불용어 제거
사용자가 직접 불용어 사전을 만들게 되는 경우가 많음.   
불용어가 많은 경우에는 코드 내에서 직접 정의하지 않고 txt 파일이나 csv 파일로 정리해놓고 이를 불러와서 사용하기도 함.

## 정규 표현식(Regular Expression)
모듈 `re`: 특정 규칙이 있는 텍스트 데이터를 빠르게 정제할 수 있게 함.   
![](/images/../images/2024-12-19-18-32-30.png)
![](/images/../images/2024-12-19-18-33-49.png)

개인적으로, 이걸 꼭 외우기보다는 필요할 때 문법을 찾아보고 그에 맞춰 표현식을 작성하면 된다고 생각한다. 지피티 역시 어느 정도 틀을 잘 잡아준다.

## 정수 인코딩(Integer Encoding)
텍스트보다는 숫자를 더 잘 처리하는 컴퓨터의 특성에 기반해, 자연어 처리에서는 텍스트를 숫자로 바꾸는 여러가지 기법들이 있다.

### 정수 인코딩 과정 - 딕셔너리 사용하기 (귀찮음, 비추천)
텍스트를 수치화하는 단계라는 것은 본격적으로 자연어 처리 작업에 들어간다는 의미이므로, 단어가 **텍스트**일 때만 할 수 있는 **최대한의 전처리를 끝내놓아야* 한다.    
```py
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
```     

```
raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
```

- 문장 토큰화
```
['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 'a barber kept his word.', 'His barber kept his secret.', 'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 'the barber went up a huge mountain.']
```

- 단어 토큰화
정제 작업 및 정규화 작업을 병행해주었다.  
예시: 단어 소문자화, 불용어 및 단어 길이 2 이하의 단어 일부 제외 등 
```
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
```

- 단어 빈도수 높은 순서대로 정렬    
```
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
``` 

- 인덱스 부여
높은 빈도수를 가진 단어일수록 낮은 정수를 부여한다. 정수는 1부터 부여한다.  
```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
```

- 상위 빈도수 단어만 남기기
자연어 처리를 하다보면, 텍스트 데이터에 있는 단어를 모두 사용하기 보다는 빈도수가 가장 높은 n개의 단어만 사용하고 싶은 경우가 있다. 본 예시에서는 상위 5개의 단어만 사용하도록 설정해주었다.
```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

- 단어 집합에 없는 단어에 인덱스 부여
첫번째 문장은 ['barber', 'person'] -> [1, 5]로 인코딩이 성공적으로 완료되었다. 그런데 두번째 문장을 살펴보면 ['barber', 'good', 'person']에는 더 이상 word_to_index에는 존재하지 않는 단어인 'good'이 있다.
-> Out-Of-Vocabulary(단어 집합에 없는 단어) 문제!
-> 'OOV'란 단어를 새롭게 추가하고, 단어 집합에 없는 단어들은 'OOV'의 인덱스로 인코딩해줄 수 있다.   
```
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'OOV': 6}
```

- 최종 인코딩 결과  
```
[[1, 5], [1, 6, 5], [1, 3, 5], [6, 2], [2, 4, 3, 2], [3, 2], [1, 4, 6], [1, 4, 6], [1, 4, 2], [6, 6, 3, 2, 6, 1, 6], [1, 6, 3, 6]]
```

그러나 리스트로 직접 일일이 정수 인코딩 과정을 하는 것은 귀찮고 시간이 오래 걸린다. Counter, FreqDist, enumerate / 케라스 토크나이저 등을 사용하는 것을 추천!

## 패딩
서로 길이가 다른 문장(또는 문서)가 있다. 이때 문장 길이를 동일하게 맞추어 자연어 처리에서 병렬 연산을 가능하게 하는 작업. 

### Numpy로 패딩하기
```py
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

# 단어 집합을 만들고, 정수 인코딩을 수행
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences) # 모든 단어가 고유한 정수로 변환

max_len = max(len(item) for item in encoded)
print('최대 길이 :',max_len) # 7

for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0) # 길이가 7보다 짧은 문장에는 숫자 0을 채워서 길이 7로 맞춰줌

padded_np = np.array(encoded)
```

### 케라스 전처리 도구로 패딩하기
```py
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences) # 모든 단어가 고유한 정수로 변환

padded = pad_sequences(encoded) # 알아서 패딩 작업 수행
```
넘파이보다 훨씬 간편하게 패딩 작업이 완료되었다. 다만, 문서의 뒤에 0을 채우던 넘파이와 달리 pad_sequences는 앞에서부터 0으로 채운다. 뒤에 0을 채우고 싶다면 `padding='post'`로 설정이 가능하다.     
예시: `padded = pad_sequences(encoded, padding='post')`

그런데, 모든 문서의 평균 길이가 20인데 문서 1개의 길이가 5,000이라고 해서 굳이 모든 문서의 길이를 5,000으로 패딩할 필요는 없을 것이다. 이런 경우에 사용하는 인자가 `maxlen`이다. `maxlen`을 설정하면 해당 길이보다 짧은 문서들은 기존처럼 0으로 패딩하되, maxlen보다 길이가 긴 경우에는 데이터가 손실된다.  
예시: 
```py
padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
```
`padding='post'`: 뒷자리들을 빈자리로 패딩
`truncating='post'`: 뒤의 단어부터 삭제
`maxlen=5`: 길이가 5가 넘어가는 문장은 5까지만 남겨두고 삭제

## 원-핫 인코딩
```
[2, 5, 1, 6, 3, 7]
```

### 원-핫 인코딩 수행   
케라스는 정수 인코딩 된 결과로부터 원-핫 인코딩을 수행하는 to_categorical()를 지원한다.    
```py
one_hot = to_categorical(encoded) # 원-핫 인코딩을 수행
print(one_hot)
```
```
[[0. 0. 1. 0. 0. 0. 0. 0.] # 인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] # 인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] # 인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] # 인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] # 인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] # 인덱스 7의 원-핫 벡터
```
단점:
- 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 벡터의 차원이 늘어난다. -> 매우 비효율적인 표현 방법
- 단어의 **유사도**를 표현하지 못한다.

# 오늘의 회고
특강이 많아!! 정리할 게 많아!! 점점 밀려가고 있어!! 떠밀려가면 안돼!!!