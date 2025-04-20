import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist



df = pd.read_csv("customer_segmentation.csv")
df.head()


df = df.dropna()
X = df.select_dtypes(include=[np.number])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_euc = KMeans(n_clusters=3, random_state=42)
labels_euc = kmeans_euc.fit_predict(X_scaled)
df['Cluster_Euclidean'] = labels_euc

def kmeans_manhattan(X, k, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):
        labels = np.argmin(cdist(X, centroids, metric='cityblock'), axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

labels_man, _ = kmeans_manhattan(X_scaled, k=3)
df['Cluster_Manhattan'] = labels_man

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,5))




plt.subplot(1,2,1)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_Euclidean'], cmap='viridis')
plt.title('KMeans (Euclidean Distance)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')



plt.subplot(1,2,2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_Manhattan'], cmap='plasma')
plt.title('KMeans (Manhattan Distance)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.tight_layout()
plt.show()




pca_3d = PCA(n_components=3)
X_3d = pca_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=df['Cluster_Euclidean'], cmap='coolwarm')
ax.set_title('3D Clustering (Euclidean)')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.show()







