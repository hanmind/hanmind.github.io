---
title:  "Redis 입문 - Redis와 FastAPI로 간단 투두리스트 만들기"
excerpt: "Redis의 리스트 자료구조 관련 명령어
Redis는 **키-값 형태뿐만 아니라 리스트 자료구조**도 지원한다. 이를 통해 여러 값을 순서대로 저장하고, 추가하거나 조회하는 등의 작업이 가능하다."

categories:
  - Backend
tags:
  - [파이썬, 리눅스, 우분투, NoSQL, Redis, FastAPI]

toc: true

last_modified_at: 2024-12-16
thumbnail: ../images/2024-12-04-11-03-02.png
---

# Redis와 FastAPI로 간단 투두리스트 만들기

## Redis의 리스트 자료구조 관련 명령어
Redis는 **키-값 형태뿐만 아니라 리스트 자료구조**도 지원한다. 이를 통해 여러 값을 순서대로 저장하고, 추가하거나 조회하는 등의 작업이 가능하다.

명령어	| 설명
--------|-----
lpush	|리스트의 **왼쪽(head)**에 데이터 삽입
lrange	|리스트의 특정 범위 데이터 조회
lrem	|리스트에서 특정 값 제거
expire	|키에 TTL을 설정하여 데이터의 만료 시간 지정

(1) `lpush`
```py
r.lpush("list_name", task)
```

- `list_name`: 리스트의 이름. 여기서는 "tasks"를 지정

(1) `lrange`
```py
r.lrange("list_name", start, end)
```
- 리스트의 특정 범위에 있는 데이터를 조회할 수 있는 명령어
- `list_name`: 리스트의 이름. 여기서는 "tasks"를 지정
- `start`: 조회 시작 인덱스입니다. 0은 리스트의 첫 번째 요소를 의미
- `end`: 조회 종료 인덱스. -1은 리스트의 마지막 요소를 의미

(2) `lrem`
```py
r.lrem("list_name", count, value)
```
- 리스트에서 특정 값을 삭제하는 명령어
- `list_name`: 리스트의 이름. 여기서는 "tasks"를 지정
- `count`: 삭제할 값의 **개수**
    - count = 1: 첫 번째로 일치하는 값을 삭제
    - count = -1: 일치하는 모든 값을 삭제
    - count = 0: 리스트에서 일치하는 값을 모두 삭제(하지만 삭제한 갯수를 반환하지 않음).
- `value`: 삭제할 값. 여기서는 task를 지정
- 예시: `r.lrem("tasks", 1, "설거지하기")`
    - `tasks` 리스트에서 "설거지하기"라는 값을 하나만 찾아 삭제

(3) `expire`
```py
r.expire("key_name", time_in_seconds)
```
- `key_name`: TTL을 설정할 키의 이름. 여기서는 리스트 이름인 "tasks"을 지정하여 리스트 전체에 대해 TTL이 설정되었음.
    - ※ B*리스트*에서는 **개별 항목**에 대해 TTL을 설정할 수 없다는 것 같다. 따라서 일정 시간이 지나면 리스트 전체가 삭제된다.
- `time_in_seconds`: 데이터가 살아있는 시간을 초 단위로 지정

## 전체 코드(`main.py`)
```py
from fastapi import FastAPI
import redis

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to the To-Do app!"}

# 할 일 추가
@app.post("/add")
def add_task(task: str):
    r.lpush("tasks", task) # 리스트에 추가
    r.expire("tasks", 86400)  # "tasks" 리스트의 TTL 설정. 하루 = 60*60*24초
    
    r.set(f"task:{task}:status", "pending〰") # 상태를 "pending"으로 설정
    
    # r.set(f"task:{task}:status", 10)  # 각 할 일 상태의 TTL 설정
    return {"message": f"Task '{task}' added successfully!"}

# 할 일 완료 시 상태를 "done"으로 변경
@app.post("/complete")
def complete_task(task: str):
    r.set(f"task:{task}:status", "done✔")  # 상태 변경: "done"
    return {"message": f"Task '{task}' marked as done!"}

# 할 일 목록 조회 (상태 포함)
@app.get("/tasks")
def get_tasks():
    tasks = r.lrange("tasks", 0, -1)
    tasks_with_status = []
    
    for task in tasks:
        status = r.get(f"task:{task}:status")  # 상태 조회
        tasks_with_status.append({"Task": task, "Status": status})
    
    return {"tasks": tasks_with_status}

# 할 일 삭제
@app.delete("/delete")
def delete_task(task: str):
    # 리스트에서 삭제
    r.lrem("tasks", 1, task) # 작업복잡도 고려
    
    # 해당 상태도 삭제
    r.delete(f"task:{task}:status")
    
    return {"message": f"Task '{task}' deleted successfully!"}
```