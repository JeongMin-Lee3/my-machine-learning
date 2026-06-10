# 가우시안 혼합 모델 예제 _ 비지도 학습 (군집화)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs # 연습용 데이터셋 생성  

# 연습용 데이터셋 생성 (4덩어리 모양으로 흩뿌려진 300개의 데이터)
data, _ = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)
# _ : 라벨 데이터 (사용하지 않음)

# GMM 모델 생성 및 학습
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(data) # 모델 학습

# 클러스터 라벨 추출
labels = gmm.predict(data) # 모델 학습 및 클러스터 라벨 추출

# 4. 결과 시각화
plt.figure(figsize=(8, 6))

# 클러스터별 데이터 시각화
for i in range(4):
    cluster_data = data[labels == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {i}")

# GMM의 각 가우시안 중심 위치
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')

plt.title("Gaussian Mixture Model Clustering")
plt.legend()
plt.show()
