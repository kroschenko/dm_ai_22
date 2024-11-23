import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

# Путь к файлу CSV
data_path = r"C:\Universitet\Exasens.csv"

# Чтение данных из CSV
data = pd.read_csv(data_path, sep=",", engine='python')

# Обработка пропусков
data.fillna(np.nan, inplace=True)  # Замена пропусков на NaN

# Преобразование данных в нужные типы
numeric_cols = ['Imaginary Part', 'Real Part', 'Age', 'Smoking', 'Gender']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')  # Преобразование, замена ошибок на NaN

# Удаление строк с NaN
data.dropna(subset=numeric_cols, inplace=True)

# Фильтрация данных
filtered_data = data[data['Diagnosis'].isin(['COPD', 'HC', 'Asthma', 'Infected'])]

# Заполнение пропусков
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int']).columns
filtered_data[numeric_cols] = filtered_data[numeric_cols].fillna(filtered_data[numeric_cols].mean())

# Проверка данных
print("Первые 5 строк данных:")
print(filtered_data.head())

# Замена категориальных признаков на числовые
diagnosis_mapping = {
    'COPD': 1,
    'HC': 2,
    'Asthma': 3,
    'Infected': 4
}
filtered_data['Diagnosis'] = filtered_data['Diagnosis'].map(diagnosis_mapping)

# Отбор признаков и целевой переменной
X = filtered_data[['Imaginary Part', 'Real Part', 'Gender', 'Age', 'Smoking']]
y = filtered_data['Diagnosis']

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def run_pca_and_train(n_components):
    try:
        print(f"Запуск PCA с {n_components} компонентами...")
        
        # Использование IncrementalPCA
        ipca = IncrementalPCA(n_components=n_components, batch_size=10)  # Параметр batch_size можно настроить
        X_pca = ipca.fit_transform(X_scaled)
        
        print(f"Размеры X_pca: {X_pca.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
        
        param_grid = {
            'n_estimators': [500, 1000],
            'learning_rate': [0.01, 0.1],
            'max_depth': [4, 5, 6]
        }
        
        grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)  # Используйте все ядра
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        error_percent = (1 - accuracy) * 100
        
        print(f"\nРезультаты для {n_components} компонент:")
        print(f"Точность: {accuracy:.2f}")
        print(f"Процент ошибки: {error_percent:.2f}%")
        
        return X_pca, accuracy, mse, rmse, error_percent
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return None, None, None, None, None  # Возвращаем None в случае ошибки

# Запуск PCA и обучения для 2 и 3 компонент
results = {}
for n_components in [2, 3]:
    X_pca, accuracy, mse, rmse, error_percent = run_pca_and_train(n_components)
    results[n_components] = (X_pca, accuracy, mse, rmse, error_percent)

# Визуализация результатов
for n_components, (X_pca, accuracy, mse, rmse, error_percent) in results.items():
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
        plt.title(f'PCA с {n_components} компонентами (2D)')
        plt.xlabel('Первая компонента')
        plt.ylabel('Вторая компонента')
        plt.colorbar(label='Diagnosis')
        plt.show()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k', s=50)
        ax.set_title(f'PCA с {n_components} компонентами (3D)')
        ax.set_xlabel('Первая компонента')
        ax.set_ylabel('Вторая компонента')
        ax.set_zlabel('Третья компонента')
        plt.colorbar(ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k', s=50), label='Diagnosis')
        plt.show()
