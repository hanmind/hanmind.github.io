---
title:  "(TIL) 결측값 처리의 3가지 방법, 다항 회귀"
excerpt: "결측값을 처리하는 방법에는 크게 세 가지가 있다.
- **제거**: 결측값이 포함된 행/열 제거"

categories:
  - TIL
tags:
  - [TIL, 내일배움캠프, 스파르타내일배움캠프, AI, 파이썬, 결측값 처리, 다항 회귀]

toc: true

last_modified_at: 2024-12-09
thumbnail: ../images/2024-12-04-11-03-02.png
---
![](/images/../images/2024-12-04-11-03-02.png)

# 데이터 전처리
🔎 궁금한 점: 데이터 불균형 vs 데이터 부족   
예를 들어, '강아지 50장, 고양이 50장' 데이터셋과 '강아지 500장, 고양이 50장' 데이터셋이 있다고 하자. 어떤 데이터셋이 더 높은 성능을 보일까? 
ChatGPT 답변:
```
Kubat & Matwin (1997): 불균형 데이터와 적은 데이터량을 모두 다룬 연구.
데이터 불균형이 극단적일 경우 소수 클래스의 재현율(Recall)이 크게 감소하는 반면, 데이터 부족은 모델의 전반적인 성능 저하를 초래.
```
이 두 가지 문제 다 결국은 기피해야 하는 상황이기 때문에, 둘 중 하나를 택할 것 없이 부족한 소수 클래스에 **데이터 증강**을 사용하는 게 좋겠다. 데이터가 너무 부족한 경우에는 모델 복잡도를 줄이거나 하고 말이다.


## 결측값 처리
결측값을 처리하는 방법에는 크게 세 가지가 있다.
- **제거**: 결측값이 포함된 행/열 제거
- **대체**: 결측값을 특정 값으로 채움
- **예측**: **머신러닝 모델**로 결측값 예측

### (1) 결측값 제거
그런데 첫번째 방법은 결측값이 하나라도 있는 행(또는 열)이면 그 데이터들을 다 사용하지 못하게 된다. 데이터 손실이 크기 때문에 실제로 잘 사용하지 않으며, 특히 데이터가 적은 경우엔 피하는 게 좋다. 그래서 제거보다는, 두번째 방법인 **대체**를 주로 사용한다.

### (2) 결측값 대체
평균, 중간값, 최빈값 등을 사용하면 빠르고 간단히 결측값을 채울 수 있다.

```py
# 결측값을 0으로 대체
df_filled = df.fillna(0)

# 결측값을 각 열의 평균값으로 대체
df_filled_mean = df.fillna(df.mean())

# 결측값을 각 열의 중간값으로 대체
df_filled_median = df.fillna(df.median())

# 결측값을 각 열의 최빈값으로 대체
df_filled_mode = df.fillna(df.mode().iloc[0])
```

❗ 주의: 대체 방식 선택 시 데이터의 분포를 잘 살펴봐야 함. 예를 들어, 평균값으로 채울 경우 왜곡될 수 있음.

### (3) 결측값 예측
다른 열들을 통해서 누락된 열의 값을 선형회귀(LinearRegression)를 통해 예측한다.
```py
from sklearn.linear_model import LinearRegression

# 결측값이 있는 행과 없는 행 분리
df_with_na = df[df['column_with_na'].isnull()] # 결측값 있는 행
df_without_na = df[df['column_with_na'].notnull()] # 결측값 없는 행

# 회귀 모델 학습
model = LinearRegression()
# 누락되지 않은 열의 값을 X, 누락이 있던 열의 값을 y로 설정
model.fit(df_without_na[['feature1', 'feature2']], df_without_na['column_with_na'])

# 결측값 예측
predicted_values = model.predict(df_with_na[['feature1', 'feature2']])

# 예측된 값으로 결측값 대체
df.loc[df['column_with_na'].isnull(), 'column_with_na'] = predicted_values

```
- `df_with_na`: 'column_with_na' 열에 결측값이 있는 행들만 선택한 데이터프레임
- `df_without_na`: 'column_with_na' 열에 결측값이 없는 행들만 선택한 데이터프레임   

두 개로 분리한 이유: 결측값이 없는 데이터(`df_without_na`)를 선형회귀 **학습 데이터**로 사용하고, 이렇게 피팅한 모델로 결측값이 있는 데이터(`df_with_na`)에 대해 값을 **예측**

- 누락되지 않은 열:'feature1', 'feature2' -> 독립 변수 X
- 데이터가 일부 누락된 열: 'column_with_na' -> 종속 변수 y

🔎 궁금한 점: 결국 데이터 전처리를 하는 목적은 머신러닝/딥러닝을 하기 위함인데, 결측치를 머신러닝으로 채워넣어도 되는가? 머신러닝 모델로 결측값을 예측해 처리하면 성능이 향상될까, 아니면 왜곡될 가능성이 있을까?
ChatGPT 답변:
```
- 논문/연구 사례:
몇몇 연구에 따르면, 머신러닝 기반 대체가 단순 대체보다 성능을 향상시키는 경우가 많았음. 특히, 결측값이 적절히 예측된 경우 모델의 예측 정확도가 개선됨.
하지만, 결측값의 양이 매우 많거나 독립 변수 간의 상관관계가 약하면, 성능 저하로 이어질 수 있음.

- 결론:
머신러닝으로 결측값을 예측해 대체하는 것은 강력한 방법이지만, 무조건 성능 향상을 보장하지는 않아.
데이터의 특성과 목표에 따라 적합한 방식(단순 대체, 예측 기반 대체, 제거 등)을 선택해야 함.
최선의 방법: 여러 가지 결측값 처리 방법을 비교하면서 최적의 방식을 찾는 것!
🔍 참고 자료: 이 주제에 관심이 많으면, "Imputation Techniques in Machine Learning" 관련 논문이나 블로그 포스트를 찾아보면 더 깊이 있는 사례를 확인할 수 있을 거야!
```
내 생각에도, **독립 변수 간의 상관관계가 강하면** 머신러닝 기반 결측값 예측 방법이 단순 제거, 대체보다 유리할 것 같다. 

## 중복 데이터 처리
- `.duplicated()`: 중복된 행 확인
중복된 행 중 나중에 등장한 행을 True로 출력한다.
중복된 행 제거할 때는 바로 `.drop_duplicates()`을 이용하면 된다. 

```py
# 중복된 행 확인
print(df.duplicated().sum())

# 중복된 행 제거
df_no_duplicates = df.drop_duplicates()
```

# ML 모델
## 다항 회귀(Polynomial Regression)
- 다항 회귀: 종속 변수와 독립 변수 간의 **비선형 관계**를 모델링하는 방법. 독립 변수의 제곱, 세제곱 등 고차항을 포함

- 기본 수식: 

    $y=β0​+β1​x+β2​x^2+⋯+βn​x^n+ϵ$

$\scriptsize\textsf{y: 종속변수, x1,x2,…,xn: 독립변수, b0: 절편, b1,b2,…,bn: 회귀계수, e: 오차}$

`PolynomialFeatures()`: 다항 특징을 생성해줌. 괄호 안에 차수를 입력한다.

```py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 4, 9, 16, 25, 36])

# 다항 특징 생성 (차수 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 다항 회귀 모델 생성 및 학습
model = LinearRegression() # 모델 생성할 때는 LinearRegression()을 이용해줌
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# +a. 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
```

+a. 다중선형회귀(Multi Linear Regression)와 다르니 헷갈리지 않기!
- 다중선형회귀: 단순선형회귀와 같이 독립변수 X의 변화에 따른 종속변수 y의 변화를 선으로서 예측하되, 독립변수 X가 여러개인 분석기법.

# 오늘의 회고
오늘 공부는,,, 네.. 많이 하지 않았지만 그래도 궁금한 점들이 꽤 쌓였다. 데이터 불균형 vs 데이터 부족, 결측값 예측의 한계에 대한 궁금증을 내일 튜터님과 이야기하고 싶다! 첫번째 궁금증은 먼저 직접 실험해도 좋을듯

💡 노션 단축키 꿀팁
- 모든 토글 열고 닫기
    - Windows : `Ctrl + alt + t `
    - Mac : `⌘ + ⌥ + t`