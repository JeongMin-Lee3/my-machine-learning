# 대구광역시 도서관 도여 대여 예측 모델 만들기
# 구별로 할 수도 있음
# 공공데이터포털 or 한국부동산원-정보공개-공공데이터개방-공공데이터자료실

# 자가학습
# 1. 질병 진행 정도 예측
from sklearn.datasets import load_diabetes
import pandas as pd

data = load_diabetes(as_frame=True)  # 당뇨 환자 442명, 특성 10개
df = data.frame

X = df.drop(columns=['target'])
y = df['target']

print(df.head())
print(data.feature_names)  # age, sex, bmi, bp, s1, ...