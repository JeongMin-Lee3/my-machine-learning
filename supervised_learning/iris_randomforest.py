# Iris 데이터셋을 이용한 Random Forest 모델 학습
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

#pip install xgboost

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 2. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. XGBoost 모델 생성
model = xgb.XGBClassifier(
    objective='multi:softmax',  # 다중 클래스 분류
    num_class=3,                # 클래스 개수
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    use_label_encoder=False,    # 경고 방지용
    eval_metric='mlogloss'      # 다중 클래스 로스
)

# 4. 학습
model.fit(X_train, y_train)

# 5. 예측
y_pred = model.predict(X_test)

# 6. 평가
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


#모델 저장
import joblib
joblib.dump(model, '../models/iris_model_0610.pkl')


#모델 로드
loaded_model = joblib.load('../models/iris_model_0610.pkl')


#임의의 값으로 예측
temp_X = np.array([[5.1, 3.5, 1.4, 0.2]])
temp_y_pred = loaded_model.predict(temp_X)
print(temp_y_pred)

