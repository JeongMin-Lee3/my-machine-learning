# 철강산업 에너지 소비 예측 - 전력 사용량(Usage_kWh)을 예측하는 모델을 만들거임

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 데이터 불러오기
df = pd.read_csv("csv/Steel_industry_data.csv")

print(df.info()) # 11개 컬럼, 35040개 데이터
print("==========================================")
# print(df.isnull().sum()) # 결측치 확인 - 없음

# 결측치 제거
df = df.dropna()

# 수치형, 범주형, 타겟 분리 (date는 시각 정보라 피처에서 제외)
numeric_features = [
    'Lagging_Current_Reactive.Power_kVarh',  # 지연 전류 무효 전력 연속 kVarh   
    'Leading_Current_Reactive_Power_kVarh',  # 선행 전류 무효 전력 연속 kVarh
    'CO2(tCO2)',                             # 탄소 배출량 (tCO2)
    'Lagging_Current_Power_Factor',          # 지연 전류 역률 연속 %
    'Leading_Current_Power_Factor',          # 선행 전류 역률 연속 %
    'NSM',                                   # 비상대책 중단량 (kWh)
]
categorical_features = [
    'WeekStatus',    # 주중/주말
    'Day_of_week',   # 요일
    'Load_Type',     # 부하 유형
]
target_feature = 'Usage_kWh'  # 전력 사용량 (예측 타겟) / 수치형 컬럼을 예측할 것이므로 회귀 모델 사용

X = df[numeric_features + categorical_features]
y = df[target_feature]

# 피처 상관관계 시각화
# plt.rcParams['font.family'] = 'Malgun Gothic'
# sns.heatmap(df[numeric_features + [target_feature]].corr(), annot=True, cmap='coolwarm')
# # plt.savefig('result_img/steel_correlation.png')
# # plt.close()

# 데이터 분할 (전처리 fit은 train 데이터에서만 해야 함)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 원핫 + 스케일링 한번에 (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        # 데이터 결측치 처리
        ('imputer', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features), # 범주형 컬럼을 원핫 인코딩 처리
        ('num', StandardScaler(), numeric_features), # 수치형 컬럼을 정규화 처리
    ]
)

# 전처리 + 모델을 하나의 Pipeline으로 묶기
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor()), # 랜덤 포레스트 회귀 모델 사용
])

# 모델 학습
pipeline.fit(X_train, y_train)

# 모델 성능 평가
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print('R2:', r2)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)  

# 모델 저장
import joblib
joblib.dump(pipeline, 'models/steel_model.pkl')

# 모델 로드
loaded_model = joblib.load('models/steel_model.pkl')

# 임의의 값으로 예측
test_data = pd.DataFrame({
    'Lagging_Current_Reactive.Power_kVarh': [2.95],
    'Leading_Current_Reactive_Power_kVarh': [0],
    'CO2(tCO2)': [0],
    'Lagging_Current_Power_Factor': [73.21],
    'Leading_Current_Power_Factor': [100],
    'NSM': [900],
    'WeekStatus': ['Weekday'],
    'Day_of_week': ['Monday'],
    'Load_Type': ['Light_Load']
})

predicted_usage = loaded_model.predict(test_data)
print(f"예측 전력 사용량: {predicted_usage[0]}")