---
title:  "(WIL) 딥러닝 - 12. 태깅 작업(Tagging Task)"
excerpt: "케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)
여기서는 자연어 처리에서 중요한 태깅 작업과 이를 위한 양방향 LSTM 활용에 대해 다룬다."

categories:
  - Deep Learning
tags:
  - [AI, 딥러닝, 파이썬, 자연어 처리, NLP]

toc: true

last_modified_at: 2025-01-10
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 12. 태깅 작업(Tagging Task)

## 12-01 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)
여기서는 자연어 처리에서 중요한 태깅 작업과 이를 위한 양방향 LSTM 활용에 대해 다룬다.

### 태깅 작업
: 텍스트 내 각 단어의 유형을 식별하는 작업. 지도 학습에 속함

태깅을 해야하는 단어 데이터를 X, 레이블에 해당되는 태깅 정보 데이터는 y라고 이름을 붙였다고 하자. 데이터는 다음과 같은 구조를 가질 수 있다.   
![](../images/2025-01-13-21-27-28.png)

### 시퀀스 레이블링 작업(Sequence Labeling Task)
위와 같이 입력 시퀀스 X = [x1, x2, x3, ..., xn]에 대하여 레이블 시퀀스 y = [y1, y2, y3, ..., yn]를 각각 부여하는 작업. 태깅 작업은 대표적인 시퀀스 레이블링 작업이다.

### 양방향 LSTM
```py
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
```     
양방향 LSTM은 이전 시점의 단어 정보 뿐만 아니라, 다음 시점의 단어 정보도 참고한다.

### RNN 다대다 문제
![](https://wikidocs.net/images/page/33805/forwardrnn_ver2.PNG)   
RNN의 은닉층은 모든 시점에 대해서 은닉 상태의 값을 출력할 수도, 마지막 시점에 대해서만 은닉 상태의 값을 출력할 수도 있다. 인자로 `return_sequences=True`를 넣을 것인지, 넣지 않을 것인지로(기본값은 return_sequences=False) 설정할 수 있는데, 태깅 작업의 경우에는 다 대 다(many-to-many) 문제로 return_sequences=True를 설정하여** 출력층에 모든 은닉 상태의 값을 보낸다**.

## 12-02 양방향 LSTM를 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)

양방향 LSTM을 이용해서 **품사 태깅** 모델을 만들 수 있다.

```py
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 토큰화에 품사 태깅이 된 데이터 받아오기
tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences)) # 품사 태깅이 된 문장 개수:  3914

print(tagged_sentences[0])
# [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
```   
훈련을 시키려면 훈련 데이터에서 단어에 해당되는 부분과 품사 태깅 정보에 해당되는 부분을 분리시켜야 한다. -> [('Pierre', 'NNP'), ('Vinken', 'NNP')]와 같은 문장 샘플이 있다면 Pierre과 Vinken을 같이 저장하고, NNP와 NNP를 같이 저장하자.

파이썬의 **`zip()` 함수**는 동일한 개수를 가지는 시퀀스 자료형에서 동일한 순서에 등장하는 원소들끼리 **묶어주는** 역할    
```py
sentences, pos_tags = [], [] 
for tagged_sentence in tagged_sentences: # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 품사 태깅 정보들은 tag_info에 저장한다.
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(list(tag_info)) # 각 샘플에서 품사 태깅 정보만 저장한다.
```
각 문장 샘플에 대해서 **단어**는 `sentences`에, **태깅 정보**는 `pos_tags`에 저장하였다.

이제 케라스 토크나이저를 이용해 정수 인코딩을 하자.   
```py
def tokenize(samples):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(samples)
  return tokenizer

src_tokenizer = tokenize(sentences) # 문장 데이터 전용 토크나이저
tar_tokenizer = tokenize(pos_tags) # 품사 태깅 정보 전용 토크나이저

# 정수 인코딩 수행
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(pos_tags)
```

다음으로 패딩으로 샘플들의 모든 길이를 통일해준다.    
```py
max_len = 150
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

print('훈련 샘플 문장의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
print('테스트 샘플 문장의 크기 : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))
```

결과:     
```
훈련 샘플 문장의 크기 : (3131, 150)
훈련 샘플 레이블의 크기 : (3131, 150)
테스트 샘플 문장의 크기 : (783, 150)
테스트 샘플 레이블의 크기 : (783, 150)
```

### 양방향 LSTM으로 POS Tagger 만들기
- 임베딩 벡터의 차원과 LSTM의 은닉 상태의 차원은 128로 지정
- 다대다 문제이므로 LSTM의 return_sequences의 값은 True로 지정
- 방향 사용을 위해 LSTM을 Bidirectional()로 감쌈
- validation_data로는 테스트 데이터를 기재하여 학습 중간에 테스트 데이터의 정확도 확인함 (테스트 데이터를 검증에 쓰는 경우는 못봤는데, 이 교재에서는 단순 실습이라 그런지 이렇게 했더라.)
- 레이블에 대해서 원-핫 인코딩을 하고 손실 함수를 categorical_crossentropy를 사용할 수도 있겠지만, 만약 레이블에 원-핫 인코딩을 하지 않고 학습을 진행하고자 한다면 categorical_crossentropy 대신 sparse_categorical_crossentropy를 선택 -> 여기선 후자

```py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

embedding_dim = 128
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data=(X_test, y_test))

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
```

결과:   
```
25/25 [==============================] - 0s 6ms/step - loss: 0.0720 - accuracy: 0.9016
테스트 정확도: 0.9016
```

실제로 맞추고 있는지를 특정 테스트 샘플(10번 인덱스)을 통해 확인해보자. 

```py
index_to_word = src_tokenizer.index_word
index_to_tag = tar_tokenizer.index_word

i = 10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측값 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 확률 벡터를 정수 레이블로 변환.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], y_test[i], y_predicted[0]):
    if word != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_tag[tag].upper(), index_to_tag[pred].upper()))
```

결과:   
```
단어             |실제값  |예측값
-----------------------------------
in               : IN      IN
addition         : NN      NN
,                : ,       ,
buick            : NNP     NNP
is               : VBZ     VBZ
a                : DT      DT
relatively       : RB      RB
respected        : VBN     VBN
nameplate        : NN      NN
among            : IN      IN
american         : NNP     NNP
express          : NNP     NNP
card             : NN      NN
holders          : NNS     NNS
,                : ,       ,
says             : VBZ     VBZ
0                : -NONE-  -NONE-
*t*-1            : -NONE-  -NONE-
an               : DT      DT
american         : NNP     NNP
express          : NNP     NNP
spokeswoman      : NN      NN
.                : .       .
```

과하리만치 정확하다. 내 생각엔 validation_data에 (X_test, y_test)를 써서 더 그럴 수 있을 것 같다.