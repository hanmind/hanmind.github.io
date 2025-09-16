---
title:  "Django 시작하기"
excerpt: "Django 시작하기"

categories:
  - TIL
tags:
  - [TIL, Django, 시작하기]

toc: true

last_modified_at: 2025-07-25
---

# 순서
## 0. 가상환경 생성 및 활성화 
## 1. 프로젝트 생성

```bash
django-admin startproject myproject
```


## 2. App 생성

```bash
python manage.py startapp articles
```

## 3. App 등록
`settings.py` 파일의 INSTALLED_APPS 리스트에 추가

## 4. 서버 실행해서 확인
```bash
python manage.py runserver
```

# render 함수
```
render(request, template_name, context)
```

주어진 template을 context와 결합해서 렌더링을 거친 후 → 완성된 html을 HttpResponse로 돌려주는 함수