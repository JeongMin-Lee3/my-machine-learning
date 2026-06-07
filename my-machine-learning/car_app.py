import joblib
import pandas as pd

VERSION = '1.0.0'

# 모델 로드
loaded_model = joblib.load(f'models/car_model_ver_{VERSION}.pkl')

# 임의의 값으로 예측 (학습 시와 동일하게 컬럼명있는 DataFrame으로 넣어 컬럼명·순서 일치)
temp_X = pd.DataFrame([[307.0, 3504.0, 8]], columns=['배기량', '중량', '기통수'])
temp_y_pred = loaded_model.predict(temp_X)
print('예측 값:', temp_y_pred[0])

print("======================================")
# 함수로 만들기
def predict_car_efficiency(version: str = '1.0.0', displacement: float = 0, weight: int = 0, cylinders: int = 0) -> float:
    # 모델 로드
    VERSION = '1.0.0'
    loaded_model = joblib.load(f'models/car_model_ver_{VERSION}.pkl')

    # 임의의 값으로 예측 (학습 시와 동일하게 컬럼명있는 DataFrame으로 넣어 컬럼명·순서 일치)
    temp_X = pd.DataFrame([[displacement, weight, cylinders]], columns=['배기량', '중량', '기통수'])
    temp_y_pred = loaded_model.predict(temp_X)
    print('예측 값:', temp_y_pred[0])

    return temp_y_pred[0]


# print(predict_car_efficiency(version='1.0.0', displacement=307.0, weight=3504.0, cylinders=8))

input_displacement = float(input("배기량을 입력하세요: "))
input_weight = int(input("중량을 입력하세요: "))
input_cylinders = int(input("기통수를 입력하세요: "))

result = predict_car_efficiency(
    version='1.0.0', 
    displacement=float(input_displacement), 
    weight=int(input_weight), 
    cylinders=int(input_cylinders))
print(f'예측 값: {result}')