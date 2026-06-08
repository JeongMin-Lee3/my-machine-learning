# 데이터 전처리, 시각화 용도

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel('csv/pipe_dummy_data.xlsx')

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# 데이터 전처리
df = df.dropna()
print(df.isnull().sum())

# 데이터 저장
# df.to_csv('csv/pipe_cleaned.csv', index=False)

# 전체 데이터 개수 확인
print(df.shape)
print("==========================================================")

X = df[['외경_mm','두께_mm','길이_m','무게_kg','생산시간_분']] # 피처 컬럼 설정
y = df['생산단가_원'] # 타깃 변수 설정

# 데이터 시각화
# 한글깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재질 컬럼을 제외한 피처간의 상관관계 확인
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')

# 이미지로 저장
# plt.savefig('result_img/pipe_correlation.png')

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
pipe_model = LinearRegression()
pipe_model.fit(X_train, y_train)

# 모델 예측
y_pred = pipe_model.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

# 결과 출력
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
