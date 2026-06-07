# 데이터 시각화 (엑셀파일을 파이썬으로)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('csv/data.csv')
df['date'] = pd.to_datetime(df['date'])  # date 컬럼을 datetime 형식으로 변환
df.set_index('date', inplace=True)       # date 컬럼을 인덱스로 설정

monthly_data = df.resample('ME').sum()   # 월별 데이터를 합산하여 새로운 데이터프레임인 monthly_data에 저장
print(monthly_data)

plt.figure(figsize=(10, 6))              # 그래프 크기 설정
plt.scatter(monthly_data.index, monthly_data['value'], marker='o')  # 월별 데이터의 인덱스(날짜)를 x축으로, 값(value)을 y축으로 하여 점(marker='o')으로 그래프를 그립니다.
plt.title('Monthly Data')                                      # 그래프의 제목을 'Monthly Data'로 설정합니다.
plt.xlabel('Date')                                             # x축의 레이블을 'Date'로 설정합니다.
plt.ylabel('Value')                                            # y축의 레이블을 'Value'로 설정합니다.
plt.grid(True)                                                 # 그래프에 그리드를 추가하여 가독성을 높입니다.
plt.tight_layout()                                             # 그래프의 여백을 조정하여 그래프가 깔끔하게 보이도록 합니다.
plt.savefig('result_img/monthly_data.png')                                # 그래프를 'monthly_data.png' 파일로 저장합니다.
plt.show()                                                     # 그래프를 화면에 표시합니다.
