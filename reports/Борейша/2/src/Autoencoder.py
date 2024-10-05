import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.manifold import TSNE

FILENAME = 'seeds_dataset.txt'
INPUT_DIM = 7
NUM_EPOCHS = 200


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_data(filename):
    dataset = np.loadtxt(filename, usecols=range(7))
    classes = np.loadtxt(filename, usecols=[7])
    X = torch.tensor(dataset, dtype=torch.float32)
    y = torch.tensor(classes, dtype=torch.float32)
    return X, y


def main():
    X, y = load_data(FILENAME)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Autoencoder для 2-х компонент
    model_2 = Autoencoder(INPUT_DIM, 2)
    optimizer_2 = optim.Adam(model_2.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        for data, _ in dataloader:
            output = model_2(data)
            loss = criterion(output, data)
            optimizer_2.zero_grad()
            loss.backward()
            optimizer_2.step()
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        encoded_data_2 = model_2.encoder(X)
        print(encoded_data_2)
    encoded_data_2 = encoded_data_2.detach().numpy()

    labels = dataset.tensors[1].numpy()

    # Отображение сжатых данных в 2D
    plt.scatter(encoded_data_2[:, 0], encoded_data_2[:, 1], c=labels, cmap='plasma')
    plt.xlabel('1-я компонента')
    plt.ylabel('2-я компонента')
    plt.title('Сжатые данные (Autoencoder 2D)')
    plt.show()

    # Autoencoder для 3-х компонент
    model_3 = Autoencoder(INPUT_DIM, 3)
    optimizer_3 = optim.Adam(model_3.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCHS):
        for data, _ in dataloader:
            output = model_3(data)
            loss = criterion(output, data)
            optimizer_3.zero_grad()
            loss.backward()
            optimizer_3.step()
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss (3D): {loss.item():.4f}')

    with torch.no_grad():
        encoded_data_3 = model_3.encoder(X)
        print(encoded_data_3)
    encoded_data_3 = encoded_data_3.detach().numpy()

    # Отображение сжатых данных в 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(encoded_data_3[:, 0], encoded_data_3[:, 1], encoded_data_3[:, 2], c=labels, cmap='plasma')
    ax.set_xlabel('1-я компонента')
    ax.set_ylabel('2-я компонента')
    ax.set_zlabel('3-я компонента')
    plt.title('Сжатые данные (Autoencoder 3D)')
    plt.show()

    X_np = X.numpy()
    labels = y.numpy()

    # t-SNE для 2D
    tsne_2d = TSNE(n_components=2, init='pca')
    X_tsne_2d = tsne_2d.fit_transform(X_np)

    plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=labels, cmap='plasma')
    plt.xlabel('1-я компонента')
    plt.ylabel('2-я компонента')
    plt.title('t-SNE 2D визуализация')
    plt.show()

    # t-SNE для 3D
    tsne_3d = TSNE(n_components=3, init='pca')
    X_tsne_3d = tsne_3d.fit_transform(X_np)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=labels, cmap='plasma')
    ax.set_xlabel('1-я компонента')
    ax.set_ylabel('2-я компонента')
    ax.set_zlabel('3-я компонента')
    plt.title('t-SNE 3D визуализация')
    plt.show()


if __name__ == '__main__':
    main()
