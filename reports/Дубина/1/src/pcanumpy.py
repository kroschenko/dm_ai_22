import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
features = data.drop(columns=['DEATH_EVENT'])
target = data['DEATH_EVENT']

features_normalized = (features - features.mean()) / features.std()

cov_matrix = np.cov(features_normalized, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

projection_2d = features_normalized.dot(eigenvectors[:, :2])
projection_3d = features_normalized.dot(eigenvectors[:, :3])

# Визуализация первых двух главных компонент
plt.figure(figsize=(10, 7))
colors = ['blue' if label == 0 else 'red' for label in target]
plt.scatter(projection_2d.iloc[:, 0], projection_2d.iloc[:, 1], c=colors, alpha=0.5)
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.title("Проекция на первые две главные компоненты")
plt.show()

# Визуализация первых трех главных компонент в 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projection_3d.iloc[:, 0], projection_3d.iloc[:, 1], projection_3d.iloc[:, 2], c=colors, alpha=0.5)
ax.set_xlabel("Первая главная компонента")
ax.set_ylabel("Вторая главная компонента")
ax.set_zlabel("Третья главная компонента")
plt.title("Проекция на первые три главные компоненты")
plt.show()

# Вычисление потерь информации
explained_variance_ratio_2d = np.sum(eigenvalues[:2]) / np.sum(eigenvalues)
explained_variance_ratio_3d = np.sum(eigenvalues[:3]) / np.sum(eigenvalues)
loss_2d = 1 - explained_variance_ratio_2d
loss_3d = 1 - explained_variance_ratio_3d

print(f"Потери информации при проецировании на 2D: {loss_2d * 100: .2f}%")
print(f"Потери информации при проецировании на 3D: {loss_3d * 100: .2f}%")
