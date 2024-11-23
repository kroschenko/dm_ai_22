import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Функция загрузки и подготовки данных
def load_data(file_path):
    
    dataset = pd.read_csv(file_path, sep=',').dropna()
    
    numerical_features = dataset.select_dtypes(include=['float64', 'int64'])
    
    labels = dataset['Category']
    
    return numerical_features.values, labels


# Функция для применения PCA
def apply_pca(features, components):
    
    pca = PCA(n_components=components)
    
    pca_transformed = pca.fit_transform(features)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    retained_variance = explained_variance_ratio.sum() * 100
    lost_info = 100 - retained_variance
    print(f"Потери: {lost_info:.2f}%")
    
    return pca_transformed


# Функция визуализации
def visualize_pca(pca_data, labels, components):
    classes = labels.unique()
    
    # 2D визуализация
    if components == 2:
        plt.figure(figsize=(8, 6))
        for cls in classes:
            cls_idx = labels == cls
            plt.scatter(pca_data[cls_idx, 0], pca_data[cls_idx, 1], label=cls)
        
        plt.title('PCA: 2 компоненты')
        plt.xlabel('Главная компонента 1')
        plt.ylabel('Главная компонента 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    # 3D визуализация
    elif components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cls in classes:
            cls_idx = labels == cls
            ax.scatter(pca_data[cls_idx, 0], pca_data[cls_idx, 1], pca_data[cls_idx, 2], label=cls)
        
        ax.set_title('PCA: 3 компоненты')
        ax.set_xlabel('Главная компонента 1')
        ax.set_ylabel('Главная компонента 2')
        ax.set_zlabel('Главная компонента 3')
        ax.legend()
        plt.show()


file_path = 'D:/7 семестр/ИАД лабы/ИАД лаба №1/hcvdat0.csv'
features, labels = load_data(file_path)

# Число главных компонент для PCA
components = 2

transformed_data = apply_pca(features, components)

visualize_pca(transformed_data, labels, components)