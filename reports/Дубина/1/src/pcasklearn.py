import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path = 'D:/PyCharm project/ИАД/1/heart_failure_clinical_records_dataset.csv'
data = pd.read_csv(file_path)

print(data.head())

# Разделим данные на признаки (X) и целевую переменную (y)
X = data.drop(['DEATH_EVENT'], axis=1)
y = data['DEATH_EVENT']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2[y == 0, 0], X_pca_2[y == 0, 1], c='blue', label='No Death Event', alpha=0.5)
plt.scatter(X_pca_2[y == 1, 0], X_pca_2[y == 1, 1], c='red', label='Death Event', alpha=0.5)
plt.title('PCA: Projection on First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Визуализация данных в 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3[y == 0, 0], X_pca_3[y == 0, 1], X_pca_3[y == 0, 2], c='blue', label='No Death Event', alpha=0.5)
ax.scatter(X_pca_3[y == 1, 0], X_pca_3[y == 1, 1], X_pca_3[y == 1, 2], c='red', label='Death Event', alpha=0.5)
ax.set_title('PCA: Projection on First Three Principal Components')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()

explained_variance = pca_2.explained_variance_ratio_
print(f"Explained variance for first 2 components: {explained_variance}")

loss = 1 - np.sum(explained_variance)
print(f"Loss in variance when projecting to 2 components: {loss}")

explained_variance_3 = pca_3.explained_variance_ratio_
print(f"Explained variance for first 3 components: {explained_variance_3}")

loss_3 = 1 - np.sum(explained_variance_3)
print(f"Loss in variance when projecting to 3 components: {loss_3}")
