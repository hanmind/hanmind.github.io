---
title:  "Redis 입문 - Redis와 FastAPI로 간단 챗봇 만들기 2"
excerpt: "유사도 기반 질문 매칭 -
챗봇에서 중요한 부분은 사용자가 입력한 질문에 대해 가장 적합한 답변을 찾아주는 것이다. 이 과정에서 코사인 유사도를 사용하여 벡터 간의 유사도를 계산했다."

categories:
  - Backend
tags:
  - [파이썬, 리눅스, 우분투, NoSQL, Redis, FastAPI]

toc: true

last_modified_at: 2024-12-18
thumbnail: ../images/2024-12-04-11-03-02.png
---

# Redis와 FastAPI로 간단 챗봇 만들기 2
- 지난 실습 요약
    - 질문-답변 데이터 저장
    - 대화 기록 관리
    - 대화 기록 조회
    - `rpush`로 대화기록 저장

## 유사도 기반 질문 매칭
챗봇에서 중요한 부분은 사용자가 입력한 질문에 대해 가장 적합한 답변을 찾아주는 것이다. 이 과정에서 코사인 유사도를 사용하여 벡터 간의 유사도를 계산했다.

### 코사인 유사도
코사인 유사도는 두 벡터 간의 각도를 이용하여 유사도를 측정하는 방법으로, 값이 1에 가까울수록 두 벡터는 유사하다고 판단한다. 본 실습에서는 유사도 측정을 위해 `sentence-transformers` 라이브러리를 활용하였다. `sentence-transformers`에서 제공하는 모델을 사용하여 질문을 벡터화하고, Redis에서 저장된 질문들의 벡터와 비교하여 가장 유사한 질문을 찾아 답변을 제공하고자 했다.

```py
# 유사도 기반 답변 찾기
@app.post("/chat")
def chat(user_id: str, message: str):
    user_embedding = model.encode(message)  # 사용자의 질문 벡터화
    max_similarity = -1
    best_match = None

    # Redis에서 기존 질문 가져와서 유사도 계산
    for key in r.scan_iter("embedding:*"):
        question = key.split("embedding:")[1]
        stored_embedding = eval(r.get(key))  # Redis에서 벡터 가져오기
        similarity = util.cos_sim(user_embedding, stored_embedding)[0].item()  # 코사인 유사도 계산

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = question

    # 가장 유사한 질문에 대한 답변 가져오기
    if best_match and max_similarity > 0.5:  # 임계값 설정 (0.5 이상만 반환)
        answer = r.get(f"question:{best_match}")
    else:
        r.rpush("unanswered_questions", message)  # 답변 없는 질문 기록
        answer = "죄송해요, 그 질문에 대한 답변은 아직 없어요."
    
    # 대화 기록 저장
    r.rpush(f"chat:{user_id}:history", f"사용자: {message}")
    r.rpush(f"chat:{user_id}:history", f"챗봇: {answer}")
    
    return {"message": answer}
```

## 질문-답변 데이터 확장
### KorQuAD(Korean Question Answering Dataset)
챗봇이 더 많은 질문과 답변을 학습할 수 있도록 하는 방법을 찾다가 **KorQuAD**라는 데이터셋을 알게 되었다. KorQuAD는 LG CNS에서 운영하는 한국어 질의응답 데이터셋으로, 자연어 처리(NLP)에서 많이 사용되는 데이터셋이다. 모든 질의에 대한 답변은 해당 Wikipedia article을 기반으로 만들어졌다고 한다.   
[KorQuAD 1.0](https://korquad.github.io/category/1.0_KOR.html)

### 데이터를 Redis에 저장하기
KorQuAD 데이터를 Redis에 저장하는 함수는 load_korquad_to_redis로, 이 함수는 JSON 파일을 읽어 질문-답변 데이터를 Redis에 저장한다. 또한, 질문에 대한 벡터도 Redis에 저장하여, 이후 유사도 검색 시 사용할 수 있도록 한다.   
```py
# KorQuAD 데이터를 Redis에 저장
def load_korquad_to_redis(korquad_file):
    print("load_korquad_to_redis 함수 실행")
    with open(korquad_file, 'r', encoding='utf-8') as f:
        korquad_data = json.load(f)
    print("korquad_data 열기")
    # 질문-답변 데이터 추출 및 Redis에 저장
    for article in korquad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']  # 첫 번째 정답만 사용

                # Redis에 질문-답변 저장
                print(f"질문: {question}, 답변: {answer}")
                
                r.set(f"question:{question}", answer)
                embedding = model.encode(question).tolist()
                r.set(f"embedding:{question}", str(embedding))
    
    print("KorQuAD 데이터 저장 완료!")

# KorQuAD 파일 경로를 지정해 주세요.
korquad_file_path = "data/KorQuAD_v1.0_train.json"
# load_korquad_to_redis(korquad_file_path) # 한 번 저장 후에는 재실행되지 않도록 주석 처리
```

+a. 참고

![](/images/../images/2024-12-19-01-10-28.png)

URL에 접속했는데 연결이 되지 않아 코드나 설정에 오류가 있는 건가 많이 혼란스러웠다. 알고보니 데이터가 많아서 Redis에 저장하는 데 시간이 오래 걸리는 것이었다. "Application startup complete." 메시지가 출력되고 난 후에야 서버가 접속되니 처음 한 번은 인내심을 갖고 기다려주어야 한다. 또한, 한 번 데이터를 저장한 후에는 불필요하게 재실행되는 일을 막도록 `load_korquad_to_redis(korquad_file_path)`을 주석 처리해준다. 아니면 아예 `load_korquad_to_redis` 함수 정의 및 실행 코드를 별도 파일로 관리해주는 것도 좋은 방법!

### 미답변 질문 저장 및 조회
미답변 질문을 처리하는 로직은 간단하다. 사용자가 입력한 질문에 대해 유사한 답변이 없으면 unanswered_questions 리스트에 질문을 저장하고, "죄송해요, 그 질문에 대한 답변은 아직 없어요."라는 메시지를 반환한다.

**미답변 질문 저장**    
```py
# (생략)
else:
    r.rpush("unanswered_questions", message)  # 답변 없는 질문 기록
    answer = "죄송해요, 그 질문에 대한 답변은 아직 없어요."
```

**미답변 질문 조회**    
```py
@app.get("/unanswered")
def get_unanswered():
    unanswered = r.lrange("unanswered_questions", 0, -1)
    return {"history": unanswered}
```
이렇게 redis에 저장한 unanswered_questions 데이터는 추후 한 번에 조회하여 챗봇을 업데이트하는 데 활용할 수 있다.

## 통신 방법
기본적으로 HTTP를 통해 통신한다. RESTful API로 사용자와 챗봇 간의 메시지를 주고받을 수 있다. 다른 멋지고 빠른 실시간 통신 방식도 많겠지만, 초보자로서 일단 http 통신을 유지했다.

# 추후 고도화 아이디어
## 오타 교정
질문에 오타가 있을 경우, 맞춤법 교정 라이브러리를 사용해서 오타를 자동으로 수정할 수 있다. 이를 위해 맞춤법 교정 라이브러리 hanspell을 사용할 수 있다.
```py
pip install hanspell
```

## 유사도 검색 방식 개선
현재는 유사도 검색시 간단한 코사인 유사도를 사용했지만, 데이터가 늘어나면 더 고도화된 방법이 필요하다.
- 빠른 검색을 위해 **FAISS** 사용을 고려할 수 있다. FAISS는 대규모 벡터 데이터를 빠르게 검색하는 Facebook AI 라이브러리로, 유사도 검색 속도를 크게 향상시킬 수 있다.
- **KoBERT, GPT 기반 모델**을 Fine-tuning하면 훨씬 자연스러운 답변을 제공할 수 있을 것이다.