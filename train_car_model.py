import os

import certifi
import numpy as np # 데이터 처리
import pandas as pd # 데이터 처리

# Windows에서 seaborn/sklearn 데이터 다운로드 시 SSL 인증서 오류 방지
os.environ['SSL_CERT_FILE'] = certifi.where()

import seaborn as sns # 데이터 시각화
from sklearn.model_selection import train_test_split # 데이터 분리
from sklearn.linear_model import LinearRegression # 모델
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # 평가지표

# # 1. 데이터 로드 및 전처리
# df = sns.load_dataset("mpg")

# # df를 csv로 저장
# df.to_csv('csv/car_data.csv', index=False)

# 엑셀 파일로 불러와서 실습해보자
df = pd.read_csv('csv/car_data_cleaned.csv')
print(df['연비'].max()) # 연비 컬럼의 최대값 :46.6
# print(df.head())
# print("==============info==============")
# print(df.info())
# print("============describe============")
# print(df.describe())

# # 결측치 확인
# print(df.isnull().sum())

# # 있다면 제거
# df = df.dropna()
# print(df.isnull().sum())

# # 깨끗한 csv파일 저장
# df.to_csv('csv/car_data_cleaned.csv', index=False)


# 피처랑 타깃 변수 설정하기 / 연비 컬럼을 예측해보자. 가장 영향력 있는 독립변수 3개를 AI한테 물어서 진행하자.
X = df[["배기량", "중량", "기통수"]]  # drop메서드로 특징변수를 제외하고 나머지 피처들을 선택하는 것보다 이렇게 직접 지정.
y = df["연비"]  # 타깃 변수 (연비)

# print(X.head())
# print(y.head())

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 학습한 모델로 예측을 해보자
y_pred = model.predict(X_test)

# 예측한 값을 평가해보자
mse = mean_squared_error(y_test, y_pred) # 평균 제곱 오차 : 예측값과 실제값의 차이를 제곱한 후 평균을 낸 값입니다. = 절대크기
r2 = r2_score(y_test, y_pred) # 결정 계수 : 예측값과 실제값의 차이를 제곱한 후 평균을 낸 값입니다. = 절대크기
mae = mean_absolute_error(y_test, y_pred) # 평균 절대 오차 : 예측값과 실제값의 차이의 절대값을 평균을 낸 값입니다. = 절대크기
rmse = np.sqrt(mse) # 평균 제곱 오차의 제곱근 : 예측값과 실제값의 차이를 제곱한 후 평균을 낸 값의 제곱근을 낸 값입니다. = 절대크기

# 결과 출력
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# 모델 저장
import joblib
VERSION = '1.0.0'
joblib.dump(model, f'models/car_model_ver_{VERSION}.pkl')

# 모델 로드
loaded_model = joblib.load(f'models/car_model_ver_{VERSION}.pkl')

# 임의의 값으로 예측 (학습 시와 동일하게 컬럼명있는 DataFrame으로 넣어 컬럼명·순서 일치)
temp_X = pd.DataFrame([[307.0, 3504.0, 8]], columns=['배기량', '중량', '기통수'])
temp_y_pred = loaded_model.predict(temp_X)
print('예측 값:', temp_y_pred[0])

print("==========================================================")

# 데이터 시각화
# (1) 특성과 목표 변수의 관계 시각화
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# sns.pairplot(df, x_vars=["배기량", "중량", "기통수"], y_vars="연비", height=4)
# plt.suptitle("특성과 연비 간 관계", y=1.02)
# plt.show()

# (2) 예측값 vs 실제값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("실제 연비")
plt.ylabel("예측 연비")
plt.title("실제 연비 VS 예측 연비")
plt.legend()
plt.grid(True)
plt.show()

# (3) 잔차(Residual) 시각화
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors="k")
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("예측한 연비")
plt.ylabel("잔차")
plt.title("잔차 그림")
plt.grid(True)
plt.show()

# 피처간의 상관관계 확인
sns.heatmap(df[['배기량', '중량', '기통수', '연비']].corr(), annot=True, cmap='coolwarm')
plt.title('피처간의 상관관계')
plt.xlabel('피처')
plt.ylabel('피처')
plt.show()
