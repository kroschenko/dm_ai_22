import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


class PCAM:
    input_size: int
    components: np.array
    dataset: np.array
    eig_val: np.array
    target: np.array
    sort: np.ndarray

    def __init__(self):
        self.load_dataset()
        self.get_main_components()

    def load_dataset(self):
        # Загрузка данных heart failure record
        data = pd.read_csv('heart_failure_record.csv')

        # Проверим загруженные данные
        print(data.head())  # Выведем первые 5 строк

        # Замена целевого класса на 'death_event' (убедитесь, что столбец правильно назван)
        self.target = data['DEATH_EVENT'].values  # Используем название столбца, а не срез
        self.dataset = data.drop(columns=['DEATH_EVENT']).values  # Убираем 'death_event' из признаков
        self.input_size = self.dataset.shape[1]

    def get_main_components(self):
        math_expectation = np.sum(self.dataset, axis=0) / self.dataset.shape[0]
        dataset = self.dataset - math_expectation
        cov_matrix = np.cov(dataset.T)

        eig_val, eig_vect = np.linalg.eig(cov_matrix)
        sort = np.argsort(-1 * eig_val)
        self.eig_val = eig_val[sort]
        self.components = eig_vect[:, sort]

    def encode(self, data: np.array, compr_size: int):
        full = np.sum(self.eig_val)
        compressed = np.sum(self.eig_val[0:compr_size])
        print(f'Потери - {100 * (1 - compressed / full)} %')
        return np.dot(data, self.components[:, :compr_size])


def main():
    pca = PCAM()
    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=3)

    # Самописный PCA, 2 компоненты
    compressed_data = pca.encode(pca.dataset, 2)
    plt.figure()
    for i in range(2):  # предполагаем, что 'death_event' имеет два класса: 0 и 1
        mask = (pca.target == i)
        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o', label=f"Class {i}")
    plt.title('Self-written PCA (2 components)')  # Заголовок
    plt.legend()
    plt.show()

    # Самописный PCA, 3 компоненты
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    compressed_data = pca.encode(pca.dataset, 3)
    for i in range(2):
        mask = (pca.target == i)
        points = compressed_data[mask]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Class {i}")
    ax.set_title('Self-written PCA (3 components)')  # Заголовок
    plt.legend()
    plt.show()

    # sklearn PCA, 2 компоненты
    compressed_data = pca2.fit_transform(pca.dataset)
    plt.figure()
    for i in range(2):
        mask = (pca.target == i)
        points = compressed_data[mask]
        plt.plot(points[:, 0], points[:, 1], 'o', label=f"Class {i}")
    plt.title('sklearn PCA (2 components)')  # Заголовок
    plt.legend()
    plt.show()

    # sklearn PCA, 3 компоненты
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    compressed_data = pca3.fit_transform(pca.dataset)
    for i in range(2):
        mask = (pca.target == i)
        points = compressed_data[mask]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f"Class {i}")
    ax.set_title('sklearn PCA (3 components)')  # Заголовок
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
