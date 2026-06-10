# 선형회귀 모델을 KFold로 학습하고 성능을 측정하는 코드

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 예시 데이터
X = np.linspace(0, 10, 100).reshape(100, 1) # 0~10 범위의 값을 100개 생성
y = 3 * X.squeeze() + np.random.randn(100)  # y = 3x + noise ; 랜덤 노이즈 추가

# KFold 객체 생성 (5개 폴드로 나눔)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LinearRegression()

mse_list = [] # 성능 측정 결과를 저장할 리스트. 여러 번 반복하여 평균을 내기 위함.

# K번 반복
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)                  # 학습
    y_pred = model.predict(X_test)              # 예측
    mse = mean_squared_error(y_test, y_pred)    # 성능 측정
    mse_list.append(mse)
    print(f"개별 MSE: {mse}")

print("평균 MSE:", np.mean(mse_list))
