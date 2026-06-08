# 와인 데이터셋을 이용한 Random Forest 모델 학습

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 데이터셋 로드
wine = load_wine()

# 와인 데이터셋 확인
# print(wine.feature_names) # 13개의 특성 (화학적 성분)
# print(wine.target_names) # 3개의 클래스 (0: 레드와인, 1: 화이트와인, 2: 스파클링와인)
# print(wine.data.shape) # 178개의 데이터, 13개의 특성
# print(wine.target.shape) # 178개의 데이터, 1개의 타깃

# 특성과 타깃 분리
X = wine.data
y = wine.target

# 데이터셋 나누기 (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # train 데이터의 평균과 분산을 계산하고, 이를 통해 데이터를 정규화 : 규칙 만들기 + 적용
X_test = scaler.transform(X_test) # test 데이터의 평균과 분산을 계산하고, 이를 통해 데이터를 정규화 : 만들어진 규칙 적용

# 랜덤 포레스트 분류 모델 생성
# - 13가지 화학 성분(알코올, 산도, proline 등)으로 와인 3종(클래스 0·1·2)을 구분하는 여러 결정나무를 만듦
# - random_state=42 : 나무를 만들 때 무작위로 뽑는 과정을 고정 → 실행할 때마다 같은 모델·같은 결과
model = RandomForestClassifier(random_state=42)

# train 와인 데이터(성분 X, 종류 y)로 학습 → "이 성분 조합이면 몇 번 와인인지" 패턴을 배움
model.fit(X_train, y_train)

# 모델 예측
y_pred = model.predict(X_test)

# 평가 결과 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
