# 결정 트리 모델 학습
# 학생들의 공부 시간과 시험 점수를 입력하면 합격 여부를 예측하는 모델을 만들어보자.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import tree
import matplotlib.pyplot as plt

# 데이터 준비
data = {
    'Study Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 공부 시간
    'Exam Score': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],  # 시험 점수
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 합격 여부 : 1은 합격, 0은 불합격
} # 딕셔너리 형태로 데이터 준비

df = pd.DataFrame(data) # 데이터프레임 형태로 변환

# 독립 변수(X)와 종속 변수(y) 분리
X = df[['Study Hours', 'Exam Score']]
y = df['Pass']

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 결정 트리 모델 생성
model = DecisionTreeClassifier(max_depth=3, random_state=42) # max_depth=3 : 트리의 깊이를 3으로 제한
model.fit(X_train, y_train) # 모델 학습

# 모델 평가
accuracy = model.score(X_test, y_test)
print(f"모델 정확도: {accuracy:.2f}")

# 임의의 피처 값으로 예측
sample_features = [[7, 75]]  # 예: 공부 시간 7시간, 시험 점수 75점
prediction = model.predict(sample_features)
prediction_proba = model.predict_proba(sample_features)

# 예측 결과 출력
print(f"입력 값: {sample_features}")
print(f"예측 결과: {'Pass' if prediction[0] == 1 else 'Fail'}")
print(f"예측 확률: {prediction_proba}")

# 결정 트리 시각화
plt.figure(figsize=(10, 8))
tree.plot_tree(model, feature_names=['Study Hours', 'Exam Score'], class_names=['Fail', 'Pass'], filled=True)
plt.show()

# 텍스트 형태로 트리 규칙 출력
tree_rules = export_text(model, feature_names=['Study Hours', 'Exam Score'])
print(tree_rules)
