import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('csv/employee_data.csv')

print(df.head())
print(df.info())
print(df.describe())

# 특정 열 선택 최신 5개 조회
names = df['이름']
print(names.head())

# 특정 행 선택 (인덱스로) 0이 첫번째 행.
first_row = df.iloc[0]

# 조건을 이용한 필터링
older_than_30 = df[df['나이'] > 30]
print(older_than_30.head())

# 나이 기준으로 정렬 (오름차순이 디폴트)
sorted_df = df.sort_values(by='나이')
print(sorted_df.head())

# 등록날짜 기준으로 내림차순 정렬
sorted_df_desc = df.sort_values(by='등록날짜', ascending=False)
print(sorted_df_desc.head())

# 부서별로 그룹화하여 나이의 평균 계산
grouped_df = df.groupby('부서')['나이'].mean()
print(grouped_df)

# 결측치(데이터 중 누락이 되었거나 필요없는(관계성이 없는) 데이터) 처리

# 결측치 확인
print(df.isnull().sum()) # 값이 없는 결측치의 합(갯수)

# 결측치 채우기 fillna 함수 이용(예: 나이의 결측치를 평균 나이로 채우기)
df['나이'].fillna(df['나이'].mean(), inplace=True)


# 결측치가 있는 행 제거
df_dropped = df.dropna()

# 데이터 추가 및 삭제
# 새로운 열 추가
df['연령대'] = df['나이'].apply(lambda x: '30대' if 30 <= x < 40 else '30대 이하' if x < 30 else '40대 이상')
print(df.head())

# 열 삭제
df.drop(columns=['연령대'], inplace=True)
print(df.head())

# 행 삭제
df.drop(index=[0, 1], inplace=True)  # 첫 두 행 삭제
print(df.head())

# 데이터 저장
# 결측치가 처리된 클린한 데이터를 다시 CSV 파일로 저장
df.to_csv('csv/employee_data_cleaned.csv', index=False)


