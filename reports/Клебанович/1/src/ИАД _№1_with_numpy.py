import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Функция загрузки и подготовки данных
def load_data(file_path):
    # Загружаем данные и удаляем строки с пропущенными значениями
    dataset = pd.read_csv(file_path, sep=',').dropna()
    
    # Извлекаем числовые столбцы для анализа
    numerical_features = dataset.select_dtypes(include=[np.number])
    
    # Получаем метки классов
    labels = dataset['Category']
    
    return numerical_features.values, labels


# Функция для вычисления PCA
def apply_pca(features, components):
    # Вычисляем ковариационную матрицу
    covariance_matrix = np.cov(features.T)
    
    # Находим собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Сортируем по убыванию собственных значений
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Преобразуем данные
    pca_transformed = np.dot(eigenvectors.T[:components], features.T).T
    
    # Информация о потерях
    total_variance = eigenvalues.sum()
    retained_variance = eigenvalues[sorted_indices[:components]].sum()
    lost_info = 100 - (retained_variance / total_variance * 100)
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


# Основная часть программы
file_path = 'D:/7 семестр/ИАД лабы/ИАД лаба №1/hcvdat0.csv'
features, labels = load_data(file_path)

# Число главных компонент для PCA
components = 3

# Применение PCA
transformed_data = apply_pca(features, components)

# Визуализация
visualize_pca(transformed_data, labels, components)