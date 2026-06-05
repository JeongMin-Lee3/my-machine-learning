import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

monthly_data = df.resample('ME').sum()

plt.figure(figsize=(10, 6))
plt.scatter(monthly_data.index, monthly_data['value'], marker='o')
plt.title('Monthly Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_data.png')
plt.show()
