# Iris 데이터셋을 이용한 Random Forest 모델 학습
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH = 'models/iris_randomforest.pkl'
SCALER_PATH = 'models/iris_scaler.pkl'

# 데이터 로드 및 전처리
iris = load_iris()
X = iris.data
y = iris.target

# 이진 분류에서는 두 클래스만 선택했었는데, 이번에는 세 클래스 모두 사용

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 훈련 데이터 정규화
X_test = scaler.transform(X_test) # 테스트 데이터 정규화

# Random Forest 모델 학습
model = RandomForestClassifier(random_state=42)

# model = xgb.XGBClassifier(
#     random_state=42,
#     learning_rate=0.1,
#     max_depth=3,
#     n_estimators=100,
#     objective='multi:softmax', # 다중 클래스 분류 모델
#     num_class=3,
#     use_label_encoder=False # 라벨 인코더 사용 안함
# )

model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 평가 결과 출력
print("Accuracy:", accuracy_score(y_test, y_pred))  # 정확도 : 1.0 ; 테스트 세트들을 전부 맞춘 것
# 각 행(0, 1, 2) = Iris의 세 종(setosa, versicolor, virginica에 대응하는 클래스 번호). 각 열(precision, recall, f1-score) = 각 클래스에 대한 정밀도, 재현율, f1-score.
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
# 행 = 실제 정답(y_test), 열 = 모델 예측(y_pred) 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) # 대각선 (10, 9, 11): 맞춘 개수


# 모델·스케일러 저장 (예측 시 둘 다 필요)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# 모델·스케일러 로드
loaded_model = joblib.load(MODEL_PATH)
loaded_scaler = joblib.load(SCALER_PATH)

# 임의의 값으로 예측
new_X = np.array([[5.1, 3.5, 1.4, 0.2]])
new_X_scaled = loaded_scaler.transform(new_X)
new_y_pred = loaded_model.predict(new_X_scaled)
print('예측 종:', iris.target_names[new_y_pred[0]])

