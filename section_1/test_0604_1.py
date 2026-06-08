# 데이터 시각화 (엑셀파일을 파이썬으로)
import pandas as pd   # 이름 충돌 방지를 위해 별칭 부여.
import matplotlib.pyplot as plt

df = pd.read_csv('csv/data.csv') # 함수를 통해 엑셀 파일 불러오기 (경로 확실히!)

top5 = df.head() # 불러온 데이터프레임의 상위 5개만 출력
print(top5)
print('===================')
print(df.info())
print('===================')
print(df.describe())

df['date'] = pd.to_datetime(df['date'])  # 'date' 열의 값을 datetime 형식으로 변환합니다. 이를 통해 날짜 관련 연산을 수행할 수 있게 됩니다.
df.set_index('date', inplace=True)  # 'date' 열을 데이터프레임의 인덱스로 설정합니다. 이렇게 하면 날짜를 기준으로 데이터를 쉽게 조회하고 조작할 수 있습니다.
monthly_data = df.resample('ME').sum()  # 데이터프레임을 월별('ME')로 리샘플링(resample)하고, 각 월의 합계를 계산하여 새로운 데이터프레임인 'monthly_data'에 저장합니다.

# print(monthly_data)

# 3. 데이터 시각화
# Matplotlib을 사용하여 시각화합니다.
plt.figure(figsize=(10, 6))  # 새로운 그림을 생성하고, 그림의 크기를 10x6 인치로 설정합니다.
plt.plot(monthly_data.index, monthly_data['value'], marker='o')  # 월별 데이터의 인덱스(날짜)를 x축으로, 값(value)을 y축으로 하여 점(marker='o')으로 그래프를 그립니다.
plt.title('Monthly Data')  # 그래프의 제목을 'Monthly Data'로 설정합니다.
plt.xlabel('Date')  # x축의 레이블을 'Date'로 설정합니다.
plt.ylabel('Value')  # y축의 레이블을 'Value'로 설정합니다.
plt.grid(True)  # 그래프에 그리드를 추가하여 가독성을 높입니다.

# plt.show()  # 그래프를 화면에 표시합니다.

# plt 사진으로 저장
plt.savefig('result_img/monthly_data.png')  # 경로 확실히!