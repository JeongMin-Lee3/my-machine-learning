# 계층적 군집화 모델 예제 _ 비지도 학습 (군집화)

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering # 병합 방식(응집형, Bottom-up)으로 가까운 데이터끼리 병합하여 군집화

# 연습용 데이터셋 생성 (3덩어리 모양으로 흩뿌려진 100개의 데이터)
X, _ = make_blobs(n_samples=100, centers=3, random_state=42, cluster_std=1.5)
# _ : 라벨 데이터 (사용하지 않음)

# 덴드로그램 그리기 (계층적 군집화 모델 시각화)
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# 계층적 군집화 모델
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hc = hc.fit_predict(X)

# 결과 시각화
plt.scatter(X[:, 0], X[:, 1], c=y_hc, cmap='viridis', s=50)
plt.title("Hierarchical Clustering")
plt.show()
