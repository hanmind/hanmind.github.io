---
title:  "모듈, 패키지, 라이브러리 개념 정리"
excerpt: "모듈: 파이썬 코드를 재사용하기 위해 작성된 파일"

categories:
  - TIL
tags:
  - [TIL, 내일배움캠프, 스파르타내일배움캠프, AI, 파이썬, 모듈, 패키지, 라이브러리]

toc: true

last_modified_at: 2024-11-28
thumbnail: ../assets/TIL.jpg
---
![](/images/../images/TIL.png)

# 1. 모듈(Module)
- 모듈: 파이썬 코드를 재사용하기 위해 작성된 **파일**
- 변수, 함수, 클래스 등을 정의한 `.py` 파일 하나가 모듈

```py
# 1. import A 형태로 가져오기
import math as m

result = m.sqrt(16)
print(result) # 4.0

# 2. from A import B 형태로 특정 기능만 가져오기
from math import sqrt, pow

## from A import B 형태로 가져온 경우, A 이름을 적을 필요없이 B 이름만으로 바로 사용 가능
print(sqrt(16)) # 4.0
print(pow(2, 3))  #  8.0
```

script: 예를 들어, `math`라는 모듈은 ~함수들을 정의한 모듈이다. 따라서 `math` 모듈을 import하면 제곱근, 지수 계산 등의 다양한 수학 연산을 `math` 모듈에 정의된 함수로 해낼 수 있다.
또한, 특정 기능만 필요하다면 `from math import sqrt`처럼 가져와 효율적으로 사용할 수 있다.
위 예시에서는 `math` 모듈에서 `sqrt`를 import한 경우, math.를 별도로 작성하지 않고 바로 sqrt를 써줄 수 있다.

---
# 2. 패키지(Package)
- 패키지: **모듈의 모음**. 관련된 여러 모듈을 논리적인 그룹으로 묶은 **디렉토리**
- 구조: 하나의 디렉토리에 여러 모듈 파일과 `__init__.py`가 포함되어 있음

- 예시:
```py
# 예: 머신러닝 패키지 scikit-learn
from sklearn.linear_model import LinearRegression

model = LinearRegression()
print(model)
```

script:   
- 우리는 머신러닝이나 딥러닝 모델 구현시, 모든 모델을 직접 일일이 모델 단위로 구현할 필요가 없다. 이미 머신러닝용 패키지, 딥러닝용 패키지라는 것이 있기 때문이다. 패키지란 **여러 개의 모듈을 논리적인 그룹으로 묶은 디렉토리**를 의미한다. `scikit-learn`, `pytorch`는 대표적인 머신러닝 관련 패키지로, 머신러닝의 주요 모델들이 내장(구현)되어있다. 따라서 필요한 패키지를 불러와 그 패키지가 가진 모델을 손쉽게 사용할 수 있다.

- 이 예시를 살펴봅시다. 여기 from에 적혀있는 `sklearn`은 '사이킷런'이라는 머신러닝을 위한 패키지로, 다양한 모듈들을 가지고 있습니다.
그렇다면 sklearn. 뒤에 있는 linear_model은 뭘까요? 바로 사이킷런이 가진 모듈 중 하나겠죠. 여기서는 linear_model이라는 모듈을 가져왔습니다. 그리고 이 모듈은 선형 모델 기반 알고리즘들을 함수, 클래스 형태로 가지고 있을 것입니다. 
LinearRegression은 linear_model이라는 모듈에 구현되어 있으며, 이를 불러와 간단히 사용할 수 있습니다. 

그러면 사이킷런 패키지를 자세히 살펴볼까요?
파이썬이 디렉토리를 패키지로 인식하기 위해서는 해당 디렉토리 내에 `__init__.py` 파일을 만들어주어야 한다.

---
# 3. 라이브러리(Library)
- 라이브러리: 특정 기능을 수행하는 **모듈이나 함수들의 집합**
개발자가 필요한 기능을 선택적으로 호출해서 사용할 수 있다. 