# 반도체 제조 공정 데이터 세트 (UCI SECOM) : 159개의 센서 데이터를 사용하여 공정 이상 여부를 예측하는 모델을 만들거임.
# 반도체 제조품 불량, 정상 판별 로직

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. CSV 파일 불러오기
df = pd.read_csv('csv/uci-secom.csv')  # 경로는 실제 위치에 맞게 수정
# print(df.info()) # 592개 컬럼. 
print(df.isnull().sum()) # 결측치 확인
print("==========================================")

# 2. 피처와 라벨 분리 (Time 컬럼 제거)
X = df.drop(['Pass/Fail', 'Time'], axis=1)  # axis=1 : 열 방향 기준, Time 컬럼도 제거
y = df['Pass/Fail']

# 3. 결측치 처리 (평균 대체)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X) 

# 4. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, stratify=y, random_state=42  # stratify=y : y의 비율을 유지하면서 데이터를 분할
)

# 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42) # 랜덤포레스트 분류 모델 사용
model.fit(X_train, y_train)

# 6. 예측 및 평가
y_pred = model.predict(X_test)
print("🔍 Classification Report:")
print(classification_report(y_test, y_pred))

print("==========================================")
#정확도
print(f"정확도: {accuracy_score(y_test, y_pred)}")

# 7. 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['정상', '불량'], yticklabels=['정상', '불량'])
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("혼동 행렬")
plt.show()

# 8. 중요 센서 TOP 20 시각화
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

top_features.plot(kind='barh')
plt.title("중요 센서 TOP 20")
plt.gca().invert_yaxis()
plt.show()

