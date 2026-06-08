# numpy 알아보기 (행렬 연산)
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]
print("========================")
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
print([[1,2,3],[4,5,6]])  # 이건 단순히 파이썬 리스트 두개를 묶어놓은것.
print("========================")
zeros = np.zeros((2, 3))
print(zeros)
# [[0. 0. 0.]
#  [0. 0. 0.]]
print("========================")
ones = np.ones((2, 3))
print(ones)
# [[1. 1. 1.]
#  [1. 1. 1.]]
print("========================")
full = np.full((2, 3), 7)     #7로 채우겠다 라는 의미.
print(full)
# [[7 7 7]
#  [7 7 7]]
print("========================")
arange_array = np.arange(0, 10, 2)   
print(arange_array)  # [0 2 4 6 8]  arange는 간격 지정.
print("========================")
linspace_array = np.linspace(0, 1, 5)
print(linspace_array)  # [0.   0.25 0.5  0.75 1. ]  #구간 지정.
print("========================")
# 배열 조작
array = np.array([1, 2, 3, 4, 5, 6])
reshaped_array = array.reshape((2, 3))
print(reshaped_array)
# [[1 2 3]
#  [4 5 6]]
print("========================")
# 평탄화
flattened = reshaped_array.flatten()
print(flattened)  # [1 2 3 4 5 6]
print("========================")
raveled = reshaped_array.ravel()
print(raveled)  # [1 2 3 4 5 6]  
print("========================")
# 배열 연산
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print("========================")
print(x + y)  # [5 7 9]
print(x - y)  # [-3 -3 -3]
print(x * y)  # [4 10 18]
print(x / y)  # [0.25 0.4  0.5 ]
print("========================")
# 통계 함수
print(np.sum(x))  # 6  총합
print(np.mean(x))  # 2.0 평균
print(np.std(x))  # 0.816496580927726   #표준편차
print(np.min(x))  # 1  최소값
print(np.max(x))  # 3 최대값
print("========================")
# 배열 인덱싱 및 슬라이싱
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])  # 1
print(arr[-1])  # 5
print("========================")
arr = np.array([1, 2, 3, 4, 5])
print(arr[1:3])  #[2 3]
print(arr[:2])   #[1 2]
print(arr[2:])   #[3 4 5]
print("========================")
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3by3 행렬
print(arr2d[1, 2])  # 1번 행과 2번 열 좌표의 값 : 6
print(arr2d[:2, 1:])  # 1번 행까지와, 1번 열부터 끝까지의 겹치는 행렬 출력
# [[2 3]
#  [5 6]]
print("========================")
# 브로드캐스팅 : 어떤 조건만 만족한다면 모양이 다른 배열끼리의 연산도 가능
array1 = np.array([1, 2, 3])
array2 = np.array([4])
print(array1 + array2)  # [5 6 7]
print("========================")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix + array1)  
# [[2 4 6]
#  [5 7 9]]
print("========================")
# 난수 생성
random_array = np.random.rand(3, 2)   #기본적으로 0~1 까지의 난수 생성
print(random_array)
# [[0.37454012 0.95071431]
#  [0.73199394 0.59865848]
#  [0.15601864 0.15599452]]
print("========================")
random_int_array = np.random.randint(1, 10, (2, 3))    #1~10사이에서 난수 생성.
print(random_int_array)
# [[7 4 8]
#  [3 5 2]]
print("========================")