---
title:  "(TIL) 판다스를 이용한 데이터 변형: 정렬과 병합"
excerpt: "`sort_values()`: 특정 열의 값을 기준으로 데이터를 오름차순 또는 내림차순으로 정렬"

categories:
  - TIL
tags:
  - [TIL, 내일배움캠프, 스파르타내일배움캠프, AI, 파이썬, 라이브러리, 판다스, 데이터프레임, 정렬, 병합]

toc: true

last_modified_at: 2024-12-03
thumbnail: ../assets/TIL.jpg
---

# 데이터 변형: 정렬과 병합
## 정렬
### `sort_values()`
- `sort_values()`: 특정 열의 값을 기준으로 데이터를 오름차순 또는 내림차순으로 정렬
```py
sorted_df = df.sort_values(by='나이')
```
기본 정렬은 오름차순으로 설정되어있다. 내림차순으로 정렬하려면 `ascending=False`를 사용

또한 여러 열을 기준으로도 정렬이 가능하다. 우선순위에 따라 첫 번째 열부터 정렬됨!
```py
# '직업'을 기준으로, 같은 직업 내에서 '나이' 오름차순 정렬
sorted_df_multi = df.sort_values(by=['직업', '나이'])
print(sorted_df_multi)
```

### `sort_index()`
- `sort_index()`: **인덱스**를 기준으로 데이터를 정렬
```py
# 예시: 인덱스 기준, 내림차순으로 정렬
sorted_index_df_desc = df.sort_index(ascending=False)
```
## 병합
### `merge`
- `merge`
  다양한 merge() 방식
  - inner (기본값): 공통된 데이터만 병합.
  - outer: 공통되지 않은 데이터도 포함하여 병합, 없는 값은 NaN으로 채움.
  - left: 왼쪽 데이터프레임 기준으로 병합.
  - right: 오른쪽 데이터프레임 기준으로 병합.
```py
# outer join을 사용한 병합
merged_df_outer = pd.merge(df1, df2, on='이름', how='outer')
print(merged_df_outer)
```

# 오늘의 회고
기계음을 계속 들어서 그런지 두통이 온다. 두통과 집중력 저하로 오후에 너무 공부가 미흡했어요
오전의 페이스를 침착하게 유지하자! 폰은 던져버려!

```
💭 #2. 아직까진 정확하지 않은 데이터 과학자의 역할(?)
데이터 과학자 일 공고를 보면 같은 타이틀 아래에서도 다양한 일을 하고 있는 것을 볼 수 있다. 데이터 과학자가 워낙 새로운 직종이다 보니, 하는 일이 회사마다 다르게 설정되는 것 같다. 내가 봤을 때 데이터 과학자의 역할은 크게 두 그룹으로 나뉘는 듯하다.

제품 중심 (product-focused) 데이터 과학자: 머신러닝 엔지니어와 유사한 역할을 수행하며, 머신러닝 모델 개발 및 구현이 주 업무. DevOps, Git과 같은 컴퓨터 과학 및 프로그래밍 기술을 활용하는 경우가 많아 보임.
비즈니스 중심 (business-focused) 데이터 과학자: 데이터 시각화, SQL, 비즈니스 분석 능력이 중요하며, KPI에 맞춘 모델링 및 분석을 수행함. 비즈니스 이해도가 높아야 하는 듯! 데이터 분석가에 더 가까운 느낌이다.

출처: https://benn.tistory.com/61 [Bee's 데이터 과학:티스토리]
```
나는 비즈니스 이해보다는 product-focused가 맞을 것 같다.