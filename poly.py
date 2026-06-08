# 다항 회귀 모델 학습

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 데이터 준비
# 예제 데이터 생성 (X는 독립 변수, y는 종속 변수)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 0~10 범위의 값
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1) * 10  # 2차 방정식에 노이즈 추가

# 2. 다항 특징 생성
degree = 2  # 다항식 차수
poly_features = PolynomialFeatures(degree=degree, include_bias=False) # include_bias=False : 절편 항을 제외하고 특징 생성
X_poly = poly_features.fit_transform(X) # X를 다항식 특징으로 변환

# 3. 모델 학습
model = LinearRegression()
model.fit(X_poly, y)

# 4. 예측
X_new = np.linspace(0, 10, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_pred = model.predict(X_new_poly)

# 5. 결과 시각화
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_new, y_pred, color="red", label="Predicted Model")
plt.title(f"Polynomial Regression (degree={degree})")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 6. 모델 평가
mse = mean_squared_error(y, model.predict(X_poly))
print(f"Mean Squared Error: {mse:.2f}")

print("==========================================================")

# Lasso 모델을 사용하여 조금 더 과적합을 방지해보자
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# 1. 데이터 준비
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 0~10 범위의 값
y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1) * 10  # 2차 방정식에 노이즈 추가

# 2. 다항 특징 생성
degree = 5  # 다항식 차수 ; 아까는 2차 방정식이었지만, 5차 방정식으로 변경
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

#### 3. Lasso 모델 학습 ###
alpha = 0.1  # 정규화 강도 (0.1은 강한 정규화, 1.0은 약한 정규화)
lasso_model = Lasso(alpha=alpha, max_iter=10000)
lasso_model.fit(X_poly, y.ravel())

# 4. 예측
X_new = np.linspace(0, 10, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_pred = lasso_model.predict(X_new_poly)

# 5. 결과 시각화
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_new, y_pred, color="red", label="Lasso Predicted Model")
plt.title(f"Lasso Regression (degree={degree}, alpha={alpha})")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# 6. 모델 평가
mse = mean_squared_error(y, lasso_model.predict(X_poly))
print(f"Mean Squared Error: {mse:.2f}")
print(f"Lasso Coefficients: {lasso_model.coef_}")
