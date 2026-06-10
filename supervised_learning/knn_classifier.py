# KNN 모델 학습 - Iris 데이터셋을 이용한 모델 학습

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN 분류기 생성
knn = KNeighborsClassifier(n_neighbors=3)  # K(훈련 데이터와 가장 가까운 이웃의 수)=3
knn.fit(X_train, y_train)

# 예측
y_pred = knn.predict(X_test)

# 정확도 평가 accuracy_score, f1_score, 혼동행렬(confusion_matrix)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro')) # iris 데이터셋은 3개의 클래스가 있으므로 macro 평균을 사용
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

# 모델 성능 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 모델 저장
import joblib
joblib.dump(knn, 'models/knn_model.pkl')

# 모델 로드
loaded_model = joblib.load('models/knn_model.pkl')

# 임의의 값으로 예측
import numpy as np
temp_X = np.array([[5.1, 3.5, 1.4, 0.2]])
temp_y_pred = loaded_model.predict(temp_X)
print('예측 종:', data.target_names[temp_y_pred[0]])