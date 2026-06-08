import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = iris.target

# print(iris.feature_names)
# print(iris.target_names)

# print(X[0:3])
# print(y)

# 이진 분류 문제를 위해 클래스 0과 1만 선택
binary_mask = y < 2  # 클래스 0과 1만 선택

# print(binary_mask) # True, False 로 나옴 -> 이걸 이용해서 X와 y를 분리

X_binary = X[binary_mask]
y_binary = y[binary_mask]

print(X_binary[0:3])
print(y_binary)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 예측 및 성능 평가 (분류)
y_pred = model.predict(X_test) # 여기서는 결과가 0 또는 1로 나옴

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# 모델 저장
import joblib
joblib.dump(model, 'models/logistic_model.pkl')

# 모델 로드
loaded_model = joblib.load('models/logistic_model.pkl')

# 임의의 값으로 예측
new_X = np.array([[5.1, 3.5, 1.4, 0.2]])
new_y_pred = loaded_model.predict(new_X)
print(iris.target_names)
print('예측값:', new_y_pred[0])