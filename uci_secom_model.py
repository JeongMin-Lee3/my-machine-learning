# 반도체 제조 공정 데이터 세트 (UCI SECOM) : 159개의 센서 데이터를 사용하여 공정 이상 여부를 예측하는 모델을 만들거임.
# 반도체 제조품 불량, 정상 판별 로직

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. CSV 파일 불러오기
df = pd.read_csv('uci-secom.csv')  # 경로는 실제 위치에 맞게 수정