# 그리드 서치 모델 예제 _ 비지도 학습 (군집화)
# 결측치 처리, 데이터 스케일링, KNN 분류기 하이퍼파라미터 튜닝
# 복수의 분류기에 개별로 여러 파라메터로 학습하여 하나의 모델 산출, 최적의 파라메터 산출

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# 파이프라인 구성
pipeline = Pipeline([ # 스케일링 → KNN 순서로 처리하는 파이프라인
    # 데이터 결측치 처리
    ('imputer', SimpleImputer(strategy='mean')), # 결측치 평균값으로 처리
    ('scaler', StandardScaler()),  # 데이터 스케일링
    ('knn', KNeighborsClassifier(n_neighbors=3))  # KNN 분류기
])

param_grid = { # 바꿔볼 하이퍼파라미터 조합
    'knn__n_neighbors': [3, 5, 7],
    'knn__weights': ['uniform', 'distance']
} 

grid_search = GridSearchCV(pipeline, param_grid, cv=5) # 각 조합마다 5-Fold 교차검증
grid_search.fit(X_train, y_train)
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
