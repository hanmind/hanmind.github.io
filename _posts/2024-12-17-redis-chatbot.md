---
title:  "Redis 입문 - Redis와 FastAPI로 간단 챗봇 만들기 1"
excerpt: "리스트는 순차적인 데이터를 저장하는 구조로, 데이터를 선입선출(FIFO) 방식으로 처리할 수 있다. 또한 **리스트**는 **키 하나에 여러 개의 값**을 추가하는 형태로 이루어져 있다. "

categories:
  - Backend
tags:
  - [파이썬, 리눅스, 우분투, NoSQL, Redis, FastAPI]

toc: true

last_modified_at: 2024-12-17
thumbnail: ../images/TIL.png
---

# Redis와 FastAPI로 간단 챗봇 만들기 1
## Redis 리스트 자료구조
리스트는 순차적인 데이터를 저장하는 구조로, 데이터를 선입선출(FIFO) 방식으로 처리할 수 있다. 또한 **리스트**는 **키 하나에 여러 개의 값**을 추가하는 형태로 이루어져 있다. 키-값 쌍으로 **단일 데이터**를 저장하는 구조인 **String**과 달리 여러 개의 값을 저장할 수 있다.

## 질문-답변 데이터 저장하기
```py
# Redis에 질문-답변 데이터 저장
def initialize_qna():
    qna_data = {
        "안녕하세요": "안녕하세요! 무엇을 도와드릴까요?",
        "날씨가 어때?": "오늘은 맑고 화창한 날씨입니다!",
        "너는 누구야?": "저는 간단한 Redis 챗봇이에요!",
        "고마워": "천만에요! 또 질문이 있으면 물어보세요."
    }
    for question, answer in qna_data.items():
        r.set(f"question:{question}", answer)

# 초기화
initialize_qna()
```
먼저 사용자의 질문을 키(key), 챗봇의 답변을 값(value)으로 하는 키-값 쌍을 만들어주자. 미리 파이썬 딕셔너리로 질문-답변 쌍을 여러 개 만들어두었다. 이후 for문으로 딕셔너리를 순회하며 `r.set` 명령어로 **Redis**에 데이터 키-값 쌍을 저장한다. 여기서 키는 `question:안녕하세요`와 같은 **계층적** 형태로 구성하여, 데이터를 관리하기 쉽게 만들었다.   
```
키: question:안녕하세요?
값: 안녕하세요! 무엇을 도와드릴까요?

키: question:날씨가 어때?
값: 오늘은 맑고 화창한 날씨입니다!
```

## 대화 기록 관리하기
```py
# 사용자와의 대화 기록 관리
@app.post("/chat")
def chat(user_id: str, message: str):
    # Redis에서 질문에 대한 답변 찾기
    answer = r.get(f"question:{message}")
    if not answer:
        answer = "죄송해요, 그 질문에 대한 답변은 없어요."

    # 대화 기록 저장
    r.rpush(f"chat:{user_id}:history", f"사용자: {message}")
    r.rpush(f"chat:{user_id}:history", f"챗봇: {answer}")

    return {"message": answer}
```
`def chat(user_id: str, message: str)`은 두 개의 파라미터를 입력받는다. 첫번째는 `user_id`(사용자의 ID), 두번째는 `message`(사용자가 보낸 질문)이다.   
1. 답변 조회: 그중 첫번째 파라미터는 `r.get(f"question:{message}")`의 {message} 위치에 들어가서 아까 `r.set`으로 저장해뒀던 키-값 쌍에서 답변을 찾는다. 만약 해당 키가 없으면 "죄송해요, 그 질문에 대한 답변은 없어요"라는 기본값을 반환한다.   
2. 대화 기록 저장: 
그리고 이때, 대화 기록은 rpush를 통해 redis에 리스트 형태로 저장한다. Redis  리스트는 키 하나에 **순서가 있는 여러 개의 값**을 가질 수 있기 때문에 각각의 사용자에 대한 대화 기록을 관리하기에 적합하다. 키는 `chat:{user_id}:history` 형식으로 만들고, 값은 `사용자: {message}` 형식과 `챗봇: {answer} 형식` 두 가지가 되도록 구현했다.   
![](/images/../images/레디스%20대화%20기록%20저장.png)

```
키: chat:user123:history
값: 사용자: 안녕하세요?
값: 챗봇: 안녕하세요! 무엇을 도와드릴까요?
값: 사용자: 너 MBTI가 뭐니  
값: 챗봇: 죄송해요, 그 질문에 대한 답변은 없어요.  
```
그리고 대화를 이어나갈수록 Redis의 리스트에 값들이 차곡차곡 쌓이게 된다.

## 대화 기록 조회하기
```py
# 대화 기록 조회
@app.get("/history/{user_id}")
def get_history(user_id: str):
    history = r.lrange(f"chat:{user_id}:history", 0, -1)
    return {"history": history}
```
이 함수는 `user_id`를 입력받아 해당 사용자의 대화 기록을 조회한다.   
1. `lrange` 명령어: `Redis` 리스트에서 값을 가져온다. `0`부터 `-1`까지의 범위를 지정하면 리스트의 모든 값을 반환한다.
2. 결과 반환: 대화 기록을 리스트 형태로 반환한다.

![](/images/../images/레디스%20챗봇%20조회.png)

## 오늘의 회고
이제 Redis의 리스트 자료구조를 활용해 대화 기록을 저장하고 조회할 수 있게 되었다. 다음 편에서는 유사도 기반으로 비슷한 질문을 찾아 답변하는 기능 등을 추가해보겠다. 🚀