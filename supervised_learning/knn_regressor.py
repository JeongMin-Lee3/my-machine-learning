# KNN 회귀모델 학습

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 준비
# 거대한 하나의 직선을 중심으로 무작위 오차(noise=15)를 흩뿌리는 방식으로 데이터를 생성 ; 1개의 특성을 가진 데이터를 100개 생성
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 정규화를 할것 같으면 여기서 해주면 됨.

# KNN 회귀기 생성
knn_reg = KNeighborsRegressor(n_neighbors=5)  # K=5
knn_reg.fit(X_train, y_train)

# 예측
y_pred = knn_reg.predict(X_test)

# 성능 평가
print("KNN 회귀모델 MSE:", mean_squared_error(y_test, y_pred))

#r2 점수
print("KNN 회귀모델 R2 Score:", r2_score(y_test, y_pred))

print("==========================================================")
# 번외 : 랜덤포레스트 회귀모델 학습
# 랜덤포레스트 회귀모델 생성
from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor(
    random_state=42, 
    n_estimators=100) # 100개의 결정트리를 만듦
random_forest_reg.fit(X_train, y_train)

# 예측
y_pred = random_forest_reg.predict(X_test)

# 성능 평가
print("랜덤포레스트 회귀모델 MSE:", mean_squared_error(y_test, y_pred))
print("랜덤포레스트 회귀모델 R2 Score:", r2_score(y_test, y_pred))

print("==========================================================")

# 번외 : 선형회귀모델 학습
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# 예측
y_pred = linear_reg.predict(X_test)

# 성능 평가
print("선형회귀모델 MSE:", mean_squared_error(y_test, y_pred))
print("선형회귀모델 R2 Score:", r2_score(y_test, y_pred))

# 선형회귀모델 시각화
import matplotlib.pyplot as plt
#선형회귀 모델 시각화
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, linear_reg.predict(X), color='red')
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
