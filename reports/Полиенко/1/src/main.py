import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Wholesale customers data.csv')

X = data.drop([ 'Channel', 'Region'], axis=1)

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

cov_matrix = np.cov(X_normalized.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

pc2_manual = (X_normalized @ eigenvectors[:, :2])
pc3_manual = (X_normalized @ eigenvectors[:, :3])

pca_2 = PCA(n_components=2)
pc2_sklearn = pca_2.fit_transform(X_normalized)

pca_3 = PCA(n_components=3)
pc3_sklearn = pca_3.fit_transform(X_normalized)

colors = {1: 'red', 2: 'blue', 3: 'green'}
regions = data['Region']
color_values = regions.map(colors)

plt.subplot(1, 2, 1)
plt.scatter(-pc2_manual[:, 0], -pc2_manual[:, 1], c=color_values)
plt.title('PCA (2 Components) - Manual Calculation')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 2, 2)
plt.scatter(pc2_sklearn[:, 0], pc2_sklearn[:, 1], c=color_values)
plt.title('PCA (2 Components) - Sklearn')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(-pc3_manual[:, 0], -pc3_manual[:, 1], -pc3_manual[:, 2], c=color_values)
ax1.set_title('PCA (3 Components) - Manual Calculation')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pc3_sklearn[:, 0], pc3_sklearn[:, 1], pc3_sklearn[:, 2], c=color_values)
ax2.set_title('PCA (3 Components) - Sklearn')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')

plt.tight_layout()
plt.show()

total_variance = np.sum(eigenvalues)

explained_variance_2_manual = np.sum(eigenvalues[:2]) / total_variance
loss_2_manual = 1 - explained_variance_2_manual

explained_variance_3_manual = np.sum(eigenvalues[:3]) / total_variance
loss_3_manual = 1 - explained_variance_3_manual

explained_variance_2_sklearn = np.sum(pca_2.explained_variance_ratio_)
loss_2_sklearn = 1 - explained_variance_2_sklearn

explained_variance_3_sklearn = np.sum(pca_3.explained_variance_ratio_)
loss_3_sklearn = 1 - explained_variance_3_sklearn

print(f'Потери для 2 компонент (Manual): {loss_2_manual:.5f}')
print(f'Потери для 3 компонент (Manual): {loss_3_manual:.5f}')
print(f'Потери для 2 компонент (Sklearn): {loss_2_sklearn:.5f}')
print(f'Потери для 3 компонент (Sklearn): {loss_3_sklearn:.5f}')

