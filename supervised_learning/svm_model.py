# SVM 모델 학습

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. 데이터 생성
X, y = make_classification(   # 데이터 생성 함수
    n_samples=200,      # 데이터 샘플 수
    n_features=2,       # 특징 수
    n_informative=2,    # 유효한 특징 수
    n_redundant=0,      # 중복 특징 수
    n_classes=2,        # 클래스 수
    random_state=42
)

# 2. 데이터 시각화
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 3. 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. SVM 모델 생성
svm_model = SVC(kernel='linear')  # 선형 커널 사용
svm_model.fit(X_train, y_train)  # 모델 학습

# 5. 예측
y_pred = svm_model.predict(X_test)

# 6. 평가
print("SVM 정확도:", accuracy_score(y_test, y_pred))

# 7. 결정 경계 시각화
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X, y, svm_model)

# 랜덤포레스트 모델
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# 모델성능평가
y_pred = random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("랜덤포레스트 정확도:", accuracy)

#그레디언트 부스팅 모델
from sklearn.ensemble import GradientBoostingClassifier
gradient_boosting_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
gradient_boosting_model.fit(X_train, y_train)

#모델 성능 평가
y_pred = gradient_boosting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('그레디언트 부스팅 모델 정확도:', accuracy)

# 로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# 모델성능평가
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("로지스틱 회귀모델 정확도:", accuracy)