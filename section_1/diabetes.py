from sklearn.model_selection import train_test_split # 데이터 분리
from sklearn.linear_model import LinearRegression # 모델
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # 평가지표

# 자가학습
# 1. 질병 진행 정도 예측
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes(as_frame=True)  # 당뇨 환자 442명, 특성 10개 (혈청 측정값 등), target : 1년 후 질병 진행 정도 (25~346)
df = data.frame
# target컬럼명을 degree로 변경
df.rename(columns={'target': 'degree'}, inplace=True)
df.to_csv("csv/diabetes.csv", index=False)

# 타깃과 상관이 큰 순 (대략)
# bmi, s5, bp, s4, s3, s6 ...
X = df[['bmi', 'bp', 's5', 's4']]  # 예시
y = df['target']

print(df.head())
print(data.feature_names)  # age, sex, bmi, bp, s1, ...

print("==========================================")
print(df.describe())
print(df.info())
print(df.corr())


# print(df.isnull().sum()) # 결측치 확인 - 없음

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 학습한 모델로 예측하기
y_pred = model.predict(X_test)

# 예측한 모델 평가하기
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.2f}")   # 평균적으로 target에서 약 43 정도 벗어남
print(f"RMSE: {rmse:.2f}")