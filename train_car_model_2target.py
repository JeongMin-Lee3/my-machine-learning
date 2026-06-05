import os

import certifi
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Windows에서 seaborn/sklearn 데이터 다운로드 시 SSL 인증서 오류 방지
os.environ['SSL_CERT_FILE'] = certifi.where()

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('csv/car_data_cleaned.csv')

TARGETS = ['연비', '제로백']  # 타깃 변수 2개
FEATURES = ['배기량', '중량', '기통수']

X = df[FEATURES]
y = df[TARGETS]

print('타깃별 최댓값:')
print(y.max())

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 모델 학습 (다중 출력 회귀)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 평가 (타깃별 + 전체 평균)
mse = mean_squared_error(y_test, y_pred, multioutput='uniform_average')
r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
mae = mean_absolute_error(y_test, y_pred, multioutput='uniform_average')
rmse = np.sqrt(mse)

print('=== 전체 평균 ===')
print('Mean Squared Error (MSE):', mse)
print('R-squared (R²):', r2)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)

print('=== 타깃별 R² ===')
for i, target in enumerate(TARGETS):
    print(f'{target}:', r2_score(y_test[target], y_pred[:, i]))

# 모델 저장
VERSION = '1.0.0'
joblib.dump(model, f'models/car_model_2target_ver_{VERSION}.pkl')

loaded_model = joblib.load(f'models/car_model_2target_ver_{VERSION}.pkl')

# 새 데이터 예측
temp_X = pd.DataFrame([[307.0, 3504.0, 8]], columns=FEATURES)
temp_y_pred = loaded_model.predict(temp_X)
print('예측 연비:', temp_y_pred[0][0])
print('예측 제로백:', temp_y_pred[0][1])
print('실제 연비:', y.iloc[0]['연비'])
print('실제 제로백:', y.iloc[0]['제로백'])

print('==========================================================')

# (1) 특성과 타깃 변수 관계 시각화
sns.pairplot(df, x_vars=FEATURES, y_vars=TARGETS, height=3)
plt.suptitle('특성과 타깃 변수(연비, 제로백) 간 관계', y=1.02)
plt.show()

# (2) 타깃별 예측값 vs 실제값
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, target in enumerate(TARGETS):
    axes[i].scatter(y_test[target], y_pred[:, i], alpha=0.7, edgecolors='k')
    min_val = y[target].min()
    max_val = y[target].max()
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[i].set_xlabel(f'실제 {target}')
    axes[i].set_ylabel(f'예측 {target}')
    axes[i].set_title(f'실제 {target} VS 예측 {target}')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# (3) 타깃별 잔차 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, target in enumerate(TARGETS):
    residuals = y_test[target] - y_pred[:, i]
    axes[i].scatter(y_pred[:, i], residuals, alpha=0.7, edgecolors='k')
    axes[i].axhline(0, color='red', linestyle='--', lw=2)
    axes[i].set_xlabel(f'예측 {target}')
    axes[i].set_ylabel('잔차')
    axes[i].set_title(f'{target} 잔차 그림')
    axes[i].grid(True)

plt.tight_layout()
plt.show()
