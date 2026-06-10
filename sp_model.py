# 스펙트럴 군집화 모델 예제 _ 비지도 학습 (군집화)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# 1. 데이터 생성
np.random.seed(0)
X, y = make_moons(n_samples=300, noise=0.02)  # 노이즈 감소

# 2. Spectral Clustering 모델 생성 및 학습
spectral_model = SpectralClustering(
    n_clusters=2,
    affinity='nearest_neighbors',  # 최근접 이웃 기반 유사도 계산
    n_neighbors=10,                # 이웃의 수
    random_state=0
)
labels = spectral_model.fit_predict(X)

# 3. 결과 시각화
plt.figure(figsize=(8, 6))

# 클러스터별 데이터 시각화
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], label='Cluster 0', s=50)
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], label='Cluster 1', s=50)

plt.title("Improved Spectral Clustering on Non-Linear Data")
plt.legend()
plt.show()