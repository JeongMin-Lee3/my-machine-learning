import joblib
import numpy as np

# 모델 로드 _ 실무 활용
loaded_model = joblib.load('models/linear_regression_model.pkl')

# 임의의 값으로 예측
new_X = np.array([[1.5]])
new_y_pred = loaded_model.predict(new_X)
print('x=1.5일 때 예측 값:', new_y_pred[0])

print("==========================================================")

# 위 로직을 함수로 만들어서 재사용 가능하게 하기
def predict_data(value : float) -> float:
    # 모델 로드 _ 실무 활용
    loaded_model = joblib.load('models/linear_regression_model.pkl')
    
    # 임의의 값으로 예측
    new_X = np.array([[value]])
    new_y_pred = loaded_model.predict(new_X)

    return new_y_pred[0]

result1 = predict_data(1.2)
result2 = predict_data(1.3)
result3 = predict_data(1.4)

print(result1)
# 이 위 결과는 리스트 형태인데 요소가 하나밖에 없으므로 [0]을 써서 첫번째 요소를 출력   
print(result1[0])
print(result2)
# 이 위 결과는 리스트 형태인데 요소가 하나밖에 없으므로 [0]을 써서 첫번째 요소를 출력
print(result2[0])
print(result3)
# 이 위 결과는 리스트 형태인데 요소가 하나밖에 없으므로 [0]을 써서 첫번째 요소를 출력
print(result3[0])

# 아니면 애초에 함수에서 리턴값에 [0]을 써서 첫번째 요소를 출력하게 하기
# def predict_data(value : float) -> float:
#     loaded_model = joblib.load('models/linear_regression_model.pkl')
#     new_X = np.array([[value]])
#     new_y_pred = loaded_model.predict(new_X)
#     return new_y_pred[0][0] # 리스트 형태인데 요소가 하나밖에 없으므로 [0]을 써서 첫번째 요소를 출력

print("==========================================================")

# 사용자 입력 받아서 예측값 출력하기
input_value = float(input("값을 입력하세요: "))
result = predict_data(input_value)
print(f'x={input_value}일 때 예측 값: {result}')