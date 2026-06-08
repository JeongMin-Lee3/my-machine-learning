import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

df = pd.read_csv('csv/pipe_cleaned.csv')

print(df.head())

numeric_features = ['외경_mm', '두께_mm', '길이_m', '무게_kg', '생산시간_분']
categorical_features = ['재질']
target_feature = ['생산단가_원']


X = df[categorical_features + numeric_features]
y = df[target_feature]

# 피처간의 상관관계 (수치형만)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 한글 폰트
sns.heatmap(df[numeric_features + ['생산단가_원']].corr(), annot=True, cmap='coolwarm')
plt.savefig('result_img/pipe_correlation.png')
plt.close()

#데이터 분석 시각화 
sns.pairplot(df[categorical_features + numeric_features + target_feature], corner=True)
plt.savefig('result_img/pipe_pairplot.png')
plt.close()



# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 재질(원핫) + 수치형(스케일링)을 한번에 처리
preprocessor = ColumnTransformer(  # 여러 개의 변환기를 하나로 묶어주는 역할
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features), # 원핫인코딩 처리
        ('num', StandardScaler(), numeric_features), # 수치형 데이터 정규화 처리
    ]
)

# 전처리 + 모델을 하나의 Pipeline으로 묶기
pipeline = Pipeline([
    ('preprocessor', preprocessor),    # 전처리 단계
    ('model', LinearRegression()),      # 모델 단계
])

pipeline.fit(X_train, y_train)         # 모델 학습

print('변환 후 특성:', pipeline.named_steps['preprocessor'].get_feature_names_out())

# 모델 성능 평가 r2, mse, rms, mae
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
print('R2:', r2)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAE:', mae)


#모델 저장
import joblib
joblib.dump(pipeline, f'models/pipe_model.pkl')


#모델 로드
loaded_model = joblib.load('models/pipe_model.pkl')


#임의의 값으로 예측
temp_X = pd.DataFrame([['Carbon Steel', 250.0, 4.78, 5.95, 176.144, 43.13]], columns=X.columns)
temp_y_pred = loaded_model.predict(temp_X)
print('예측값:', temp_y_pred[0])