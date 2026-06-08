# logistic_model.py 모델을 사용하여 꽃 이름 예측

import joblib
import numpy as np
from sklearn.datasets import load_iris

sepal_length = input('꽃받침 길이를 입력하세요: ')
sepal_width = input('꽃받침 너비를 입력하세요: ')
petal_length = input('꽃잎 길이를 입력하세요: ')
petal_width = input('꽃잎 너비를 입력하세요: ')


def predict_iris(sepal_length : float, sepal_width : float, petal_length : float, petal_width : float):
    # 모델 로드
    loaded_model = joblib.load('models/logistic_model.pkl')
    
    # 임의의 값으로 예측
    new_X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    new_y_pred = loaded_model.predict(new_X)

    # 신뢰도
    confidence = loaded_model.predict_proba(new_X)
    print(confidence)
    # 꽃 이름 반환
    iris = load_iris()

    return iris.target_names[new_y_pred[0]]


result = predict_iris(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))
print('해당 꽃은', result, '입니다.')