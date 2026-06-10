# from sklearn.datasets import load_iris
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# # 1. 데이터 로딩
# iris = load_iris()
# X = iris.data
# y = iris.target

# # 2. 학습/테스트 데이터 분리
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. 모델 생성 ; 생성할 결정트리의 수 (n_estimators) : 100개, 학습률 (learning_rate) : 0.1, 트리의 깊이 (max_depth) : 3, 랜덤 시드 (random_state) : 42
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# # 4. 학습
# model.fit(X_train, y_train)

# # 5. 예측
# y_pred = model.predict(X_test)

# # 6. 평가
# print("정확도 (Accuracy):", accuracy_score(y_test, y_pred))
# print("\n분류 리포트:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# print("==========================================")

# 성능 개선 버전

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

