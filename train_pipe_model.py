# 모델 학습 용도
import os
import certifi
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('csv/pipe_cleaned.csv')
# print(df.info())

# 재질 컬럼을 원핫인코딩 처리
df = pd.get_dummies(df, columns=['재질'], drop_first=True)

# 피처 컬럼 설정
feature_cols = ['외경_mm', '두께_mm', '길이_m'] + [col for col in df.columns if col.startswith('재질_')]

# 데이터 분리 
X = df[feature_cols]
y = df['생산단가_원']

# 재질 컬럼의 종류 확인
# print(df['재질'].unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
pipe_model = LinearRegression()
pipe_model.fit(X_train, y_train)

# 학습한 모델로 예측을 해보자
y_pred = pipe_model.predict(X_test)

# 예측한 값을 평가해보자
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 결과 출력
print("Mean Squared Error (MSE):", mse) 
print("R-squared (R²):", r2.round(4)) # 소수점 4자리까지 출력
print("Mean Absolute Error (MAE):", mae.round(2)) # 소수점 2자리까지 출력
print("Root Mean Squared Error (RMSE):", rmse.round(2)) # 소수점 2자리까지 출력

