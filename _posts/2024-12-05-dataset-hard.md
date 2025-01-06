---
title:  "(TIL) 코딩 꿀팁: 단순명료한 init함수, return과 break 비교, VSCode 단축키"
excerpt: "__init__ 함수의 역할 분리
구조적 관점에서 `__init__` 함수는 **생성자의 역할**만 담당하도록 하는 것이 좋다!
즉, 객체의 초기 속성을 설정하고 기본 상태를 정의하는 데 집중하고, 제어 로직은 별도 함수로 분리해야 한다."

categories:
  - TIL
tags:
  - [TIL, 내일배움캠프, 스파르타내일배움캠프, AI, 파이썬, init함수, return, break, VSCode, 단축키]

toc: true

last_modified_at: 2024-12-05
thumbnail: ../images/TIL.png
---
![](/images/../images/TIL.png)

# 코드 개선

## `__init__` 함수의 역할 분리
구조적 관점에서 `__init__` 함수는 **생성자의 역할**만 담당하도록 하는 것이 좋다!
즉, 객체의 초기 속성을 설정하고 기본 상태를 정의하는 데 집중하고, 제어 로직은 별도 함수로 분리해야 한다.

예를 들어, `get_name`, `get_gender`, `get_age와` 같은 메서드를 분리하여 제어 로직을 명확히 나눈다. 이렇게 하면 코드의 가독성과 유지 보수성이 크게 향상된다.

- 코드 구조:
    - 정의부 (`__init__`): 속성 초기화
    - 선언부: 함수와 메서드 정의
    - 제어부: 실제 로직 호출 및 실행
    => 세부 역할이 명확히 구분되어야 한다.



## 중첩된 반복문에서 `break`와 `return` 비교
과제 해설 강의 중, 함수에서 중첩 while문을 완전히 끝내려 할 때 동일한 `if-break`를 두 번 작성하는 게 좋은지, `return`으로 한번에 끝내는 게 좋은지 궁금해 질문드렸다. (아래의 방법 1, 2 참고)

```py
import random as rd
# 방식1. `if-break`문 두 번 사용
def homework_1():
    # 게임 시작                                                                                                               
    while True :                                                               
        Q_num = random.randint(1,10)                                           
        print("1과 10 사이의 숫자를 하나 정했습니다.")
        print("이 숫자는 무엇일까요?")
        while True:                                                            
            try :
                A_num = int(input('예상숫자: '))
                if not (1 <= A_num <= 10): 
                    print("1과 10 사이의 숫자를 입력해주세요!")
                    continue
                elif Q_num < A_num :                                
                    print('너무 큽니다. 다시 입력하세요.')  
                elif Q_num > A_num :                                    
                    print('너무 작습니다. 다시 입력하세요.')
                else:
                    print('정답입니다!')                             
                    break
            except ValueError:
                print('잘못된 입력입니다. 숫자를 입력하세요')

        while True:                                                         
            re_game = input('한 판 더 ? (y/n) >> ').lower()             
            if re_game == 'y' :                                           
                break
            elif re_game == 'n' : # re_game == 'n'으로 게임을 종료할 시, 여기서 한번 break를 걸어주고
                print('게임을 종료합니다. 즐거우셨나요? 또 만나요!')
                break
            else :                                                          
                print('잘못 입력하셨습니다. y나 n을 입력해주세요')
                continue
        if re_game == 'n' : # 동일한 조건문을 바깥에 한번 더 감싸 바깥의 반복문에 또 break를 걸어줌
            break
homework_1()

# 2. `return` 사용
def game():
    # 게임 시작
    while True:    
        number = rd.randrange(1,11)
        print("1과 10 사이의 숫자를 하나 정했습니다.")
        print("이 숫자는 무엇일까요?")
        while True:
            try:
                answer = int(input("예상 숫자: "))
                if 1 <= answer <= 10:
                    if answer > number:
                        print("너무 큽니다. 다시 입력하세요.")
                    elif answer < number:
                        print("너무 작습니다. 다시 입력하세요.")
                    elif answer == number:
                        print("정답입니다!")
                        break
                else:
                    print("1과 10 사이의 숫자를 입력하여야 합니다.")
            except ValueError:
               print("1과 10 사이의 숫자를 입력하여야 합니다.")
    
        # 재시작 여부 확인
        while True:
            retry = input("게임을 다시 하시겠습니까? (y/n): ")
            if retry.lower() == "y":
                break # 게임 재시작
            elif retry.lower() == "n":
                print("게임을 종료합니다. 즐거우셨나요? 또 만나요!")
                return # return: 실행 중인 함수를 즉시 종료. 함수 호출 지점으로 되돌아감
            else:
                print("잘못된 입력입니다. y/n으로 대답해주세요.")
# 게임 실행
game()

```
나는 내심 return으로 끝내는 코드가 깔끔하고 마음에 들어 튜터님도 그렇게 답변해주실 것이라 기대했다. 그런데 튜터님께서는 return할 것이 없는 함수에서 '함수를 종료하기 위해' return을 이용하는는 것은 일종의 꼼수로 볼 수 있다고 하셨다! 생각지 못한 지점이라 굉장히 흥미로웠다. 그런데 나는 반환값이 없더라도 return으로 끝내는 코드를 자주 본 것 같아서, 조금 더 검색을 해보았다.

- [break 랑 return 이랑 차이가 무엇인가요? - 인프런](https://www.inflearn.com/community/questions/747573/break-%EB%9E%91-return-%EC%9D%B4%EB%9E%91-%EC%B0%A8%EC%9D%B4%EA%B0%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94?srsltid=AfmBOopLNuPzLUWxpiD0GskhZCL0r_jRIq2G9GRoFVMj-HH_Sqfq8kxp)
- [break, continue, return 차이 - SOOBIN](https://velog.io/@subin0214/break-continue-return-%EC%B0%A8%EC%9D%B4)

위 게시물들에서는 return을 **값을 반환하는 명령어**이기도 하지만, **함수를 종료하는 명령어**이기도 하다고 설명하고 있다.

챗지피티에게도 물어보았다.
```md
1. 반복문 이후의 로직이 필요하다면: break
- 반복문 이후에 추가적인 동작이 필요하거나, 조건에 따라 반복문을 단계적으로 종료해야 한다면 break를 사용하는 것이 적합합니다.
- 다만, 외부 반복문까지 종료하려면 조건을 다시 작성해야 하므로 코드의 가독성을 유지하도록 주석이나 구조적 설계를 신경 써야 합니다.

2. 함수 전체를 종료하려면: return
- 반복문을 포함한 함수 실행을 완전히 종료하고 싶다면 return이 간결하고 직관적입니다.
- 반환값이 없더라도 return으로 의도를 명확히 드러낼 수 있습니다.
- 추천: 중첩된 반복문에서 모든 흐름을 중단하려는 경우에는 return을 사용하는 것이 더 깔끔합니다.
```
얘는 답변이 애매해서 신빙성이 조금 떨어진다.
흠, 아무래도 개인별 선호 방식이나 회사별 컨벤션이 있을 것 같다. 두 방식 다 기억하고 적절한 상황에서 잘 활용해야겠다! 내일 심심하면 영어로 구글링 더 해봐야지

## 유용한 VSCode 단축키
- Ctrl + 좌우 방향키: 단어 단위로 커서 이동
- Ctrl + Tab: 열린 파일 간 빠르게 전환

# 오늘의 회고
딴짓을 많이 했는데 그래도 깃허브 테마 찾는 일처럼 언젠가 하긴 해야할 일도 있었다.. 하지만 웬만한 건 공부 시간 끝나고 하자! 아니면 쉬는 시간에 빠르게 끝내자 😁