import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('C:\\Users\\vadim\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Python 3.12\\Python Files\\Programs\\ИАД\\heart_failure_clinical_records_dataset.csv')
X = data.drop(columns='DEATH_EVENT').values
y = data['DEATH_EVENT'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

cov_matrix = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

pc2_eigenvectors = eigenvectors[:, :2]
pc3_eigenvectors = eigenvectors[:, :3]

X_pca_2_manual = X @ pc2_eigenvectors
X_pca_3_manual = X @ pc3_eigenvectors

pca_2 = PCA(n_components=2)
X_pca_2_sklearn = pca_2.fit_transform(X)

pca_3 = PCA(n_components=3)
X_pca_3_sklearn = pca_3.fit_transform(X)

def plot_2d_projection(X_pca, y, title):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], 
                    label=f"Класс {label}", alpha=0.6)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_3d_projection(X_pca, y, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(y):
        ax.scatter(X_pca[y == label, 0], X_pca[y == label, 1], X_pca[y == label, 2], 
                   label=f"Класс {label}", alpha=0.6)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.set_title(title)
    plt.legend()
    plt.show()

plot_2d_projection(X_pca_2_manual.real, y, "Проекция на 2 главные компоненты (вручную)")
plot_2d_projection(X_pca_2_sklearn, y, "Проекция на 2 главные компоненты (sklearn)")    

plot_3d_projection(X_pca_3_manual.real, y, "Проекция на 3 главные компоненты (вручную)")
plot_3d_projection(X_pca_3_sklearn, y, "Проекция на 3 главные компоненты (sklearn)")

explained_variance_2_manual = np.sum(eigenvalues[:2]) / np.sum(eigenvalues)
explained_variance_2_sklearn = np.sum(pca_2.explained_variance_ratio_)
print(f"Сохранённая информация (2 главные компоненты, вручную): {explained_variance_2_manual * 100:.2f}%")
print(f"Сохранённая информация (2 главные компоненты, sklearn): {explained_variance_2_sklearn * 100:.2f}%")

explained_variance_3_manual = np.sum(eigenvalues[:3]) / np.sum(eigenvalues)
explained_variance_3_sklearn = np.sum(pca_3.explained_variance_ratio_)
print(f"Сохранённая информация (3 главные компоненты, вручную): {explained_variance_3_manual * 100:.2f}%")
print(f"Сохранённая информация (3 главные компоненты, sklearn): {explained_variance_3_sklearn * 100:.2f}%")
