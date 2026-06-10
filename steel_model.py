# 철강제조 전력 소비량 예측, ML

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

# =========================================================
# 1) 데이터 로딩 (엑셀)
# =========================================================
df = pd.read_csv("csv/Steel_industry_data.csv")
print(df.head())
print(df.shape)

print("=== columns ===")
print(df.columns)

# =========================================================
# 2) 타겟 / 입력 분리
# =========================================================
TARGET_COL = "Usage_kWh"  # 타겟(예측할 값)

# 혹시 컬럼명이 다르면 여기서 TARGET_COL을 맞춰주세요.
X = df.drop(columns=[TARGET_COL, 'date'])
y = df[TARGET_COL]

# =========================================================
# 3) 범주형/수치형 컬럼 정의
# =========================================================
# Steel 데이터에서 주로 쓰는 범주형 컬럼(요일/주중주말/부하타입)
categorical_cols = ["Day_of_week", "WeekStatus", "Load_Type"]

# 실제 존재하는 범주형 컬럼만 사용
categorical_cols = [c for c in categorical_cols if c in X.columns]

# 실제 파일에 위 컬럼이 없거나 이름이 다르면 아래처럼 확인 후 수정
missing = [c for c in ["Day_of_week", "WeekStatus", "Load_Type"] if c not in X.columns]
if missing:
    print(f"[경고] 아래 범주형 컬럼이 데이터에 없습니다: {missing}")

numerical_cols = [c for c in X.columns if c not in categorical_cols]

print("\n=== categorical_cols ===")
print(categorical_cols)
print("\n=== numerical_cols ===")
print(numerical_cols)

# =========================================================
# 4) 전처리: OneHotEncoder + 수치 passthrough
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols),
    ],
    remainder="drop"
)

# =========================================================
# 5) 모델: RandomForestRegressor
# =========================================================
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# =========================================================
# 6) 파이프라인: 전처리 + 모델
# =========================================================
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)

# =========================================================
# 7) Train/Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 8) 학습
# =========================================================
pipeline.fit(X_train, y_train)

# =========================================================
# 9) 예측 및 평가
# =========================================================
pred = pipeline.predict(X_test)

# RMSE 계산 방법 (sklearn 버전 호환)
try:
    rmse = root_mean_squared_error(y_test, pred)
except:
    # root_mean_squared_error가 없으면 직접 계산
    rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)

print("\n=== Evaluation ===")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# =========================================================
# 10) Feature Importance (원-핫 포함 전체 중요도)
# =========================================================
# OneHotEncoder로 생성된 컬럼명 가져오기
ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

all_feature_names = np.concatenate([ohe_feature_names, np.array(numerical_cols)])
importances = pipeline.named_steps["model"].feature_importances_

fi = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n=== Top 20 Feature Importances ===")
print(fi.head(20))

# 필요하면 csv로 저장
fi.to_csv("csv/feature_importance.csv", index=False)
print("\n[저장] feature_importance.csv 생성 완료")

# 모델 저장
import joblib
joblib.dump(pipeline, 'steel_model.pkl')

# 모델 로드
steel_model = joblib.load('steel_model.pkl')

# 임의의 값 예측
# DataFrame 형태로 제공해야 함 (X와 동일한 컬럼 순서)
test_data = pd.DataFrame({
    'Lagging_Current_Reactive.Power_kVarh': [2.95],
    'Leading_Current_Reactive_Power_kVarh': [0],
    'CO2(tCO2)': [0],
    'Lagging_Current_Power_Factor': [73.21],
    'Leading_Current_Power_Factor': [100],
    'NSM': [900],
    'WeekStatus': ['Weekday'],  # 범주형: 실제 문자열 값
    'Day_of_week': ['Monday'],  # 범주형: 실제 문자열 값
    'Load_Type': ['Light_Load']  # 범주형: 실제 문자열 값
})

# 파이프라인이 자동으로 전처리(one-hot encoding 포함) 수행
predicted_usage = steel_model.predict(test_data)
print(f"Predicted Usage: {predicted_usage[0]}")