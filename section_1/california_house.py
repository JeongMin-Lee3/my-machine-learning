import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#ssl 오류 방지
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 데이터셋 로드
data = fetch_california_housing(as_frame=True)
df = data.frame

# print(df.head())
# 결측치 확인 -> 데이터 전처리
print(df.isnull().sum()) # 각 컬럼에 결측치가 있는지 확인 : 없음

# 특성과 타겟 분리
X = df.drop(columns=["MedHouseVal"])  # 특징 (특성) : 집값 컬럼을 제외한 모든 컬럼
y = df["MedHouseVal"]  # 목표 변수 (집값)

print(X.head()) # 특성 컬럼 5개 조회
print(y.head()) # 집값 컬럼 5개 조회

# 데이터 개수
print(len(X)) # 전체 데이터 개수 : 20640개
print("==========================================================")
# 학습 및 테스트 데이터 분리 train:공부, test:시험
# test_size=0.2 : 20%를 테스트 데이터로 사용, random_state=42 : 랜덤 시드 고정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
print("==========================================================")

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 평가지표
y_pred = model.predict(X_test) # 테스트 데이터로 예측. 근데 여기서 y_pred와 y_test는 다른 값이 나옴.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE:', mse) # 평균 제곱 오차 : 예측값과 실제값의 차이를 제곱한 후 평균을 낸 값입니다. = 절대크기
print('R2:', r2) # 결정 계수 : 예측값과 실제값의 차이를 제곱한 후 평균을 낸 값입니다. = 절대크기
# 해석 : 컬럼 갯수에 비해 데이터 갯수가 많아서 r2가 0.57로 낮음. 오버피팅(과적합)의 위험이 있음.
# XGBRegressor : 트리 기반 모델, 오버피팅 방지 가능.

print("==========================================================")

# 새로운 데이터 예측 (DataFrame으로 넣어 컬럼명·순서 일치)
new_data = X.iloc[[0]]  # 데이터셋 0번째 행
predicted_price = model.predict(new_data)
print('예측 가격:', predicted_price[0])  # 단위: $100,000 (예: 4.526 → 약 $452,600)
print('실제 가격:', y.iloc[0])

print("==========================================================")

# 모델 저장 _ 실무 활용
import joblib
joblib.dump(model, 'models/california_house_model.pkl')
# 여기서 pk1은 파이썬 객체를 저장하는 파일 확장자입니다.