# 랜덤 좌표값의 모임을 선형 회귀로 분석

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100개의 랜덤 X값 생성
y = 4 + 3 * X + np.random.randn(100, 1)  # 선형 방정식 y = 4 + 3x + noise / 여기서 np.random.randn()을 한 것은 기본적인 3x+4 직선에서 위아래(+/-)로 랜덤성을 더한것임.

# 데이터 시각화
plt.scatter(X, y, color='blue', alpha=0.5)
plt.title("Generated Data")
plt.xlabel("X")
plt.ylabel("y")
# plt.show()

# 2. 선형회귀 모델 생성 및 학습
model = LinearRegression() # scikit-learn에 있는 LinearRegression 클래스를 가져와 model이라는 이름으로 객체 만듦.
model.fit(X, y)            # 객체가 가진 메서드 fit으로 랜덤 생성한 데이터에 맞게 회귀식 학습

# 학습된 모델의 절편과 기울기 확인
print("절편 (Intercept):", model.intercept_)
print("기울기 (Slope):", model.coef_)

# 3. 예측 및 결과 시각화
X_new = np.array([[0], [2]])  # X=0과 X=2 두 가지 좌표일 때
y_pred = model.predict(X_new) # 모델 객체가 가진 predict 메서드로 회귀선을 바탕으로 예측

print('x=0일 때 예측값: ', y_pred[0])
print('x=2일 때 예측값:', y_pred[1])

plt.scatter(X, y, color='blue', alpha=0.5, label='Data')
plt.plot(X_new, y_pred, color='red', label='Prediction Line')  # 예측된 직선
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 모델 성능 평가
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X)

print('=======================================================')
print('y:', y) # 실제값
print('=======================================================')
print('y_pred:', y_pred) # 예측값

# MSE : 평균 제곱 오차(Mean Squared Error) 예측값과 실제값의 차이(오차)를 제곱한 후 평균을 낸 값입니다. = 절대크기
# mse가 작을수록 예측값과 실제값의 차이가 작은 것을 의미합니다. 근데 작다고 다 좋은건 아니고, 모델의 복잡도에 따라 다름.
mse = mean_squared_error(y, y_pred)
print('MSE:', mse)

# r2 출력 / r2는 결정 계수(R-squared)를 의미합니다. / “그냥 평균만 맞추는 것”보다 얼마나 더 잘 맞췄나?
# r2가 1에 가까울수록 예측값과 실제값의 차이가 작은 것을 의미합니다. 근데 1에 가까운건 좋지만, 1이 되면 오버피팅(과적합)의 위험이 있습니다.
r2 = r2_score(y, y_pred)
print('R2:', r2)

# 4. 특정 값 예측
test_X = np.array([[1.5], [3.0]])  # X=1.5, X=3.0
test_y_pred = model.predict(test_X)

print("X=1.5일 때 예측 값:", test_y_pred[0])
print("X=3.0일 때 예측 값:", test_y_pred[1])

# 모델 저장 _ 실무 활용
import joblib
joblib.dump(model, 'models/linear_regression_model.pkl')

# 모델 로드 _ 실무 활용
loaded_model = joblib.load('models/linear_regression_model.pkl')

# 임의의 값으로 예측
new_X = np.array([[1.5]])
new_y_pred = loaded_model.predict(new_X)
print('x=1.5일 때 예측 값:', new_y_pred[0])