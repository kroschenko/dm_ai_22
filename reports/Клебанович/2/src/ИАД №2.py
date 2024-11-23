import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

def load_data(file_path):
    dataset = pd.read_csv(file_path).dropna()
    numerical_features = dataset.select_dtypes(include=['float64', 'int64'])
    
    # Нормализация данных
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(numerical_features)
    
    labels = dataset['Category']
    return normalized_features, labels


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(autoencoder, data, epochs=50, lr=0.001):
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        data_tensor = torch.FloatTensor(data)
        encoded, decoded = autoencoder(data_tensor)
        loss = criterion(decoded, data_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return autoencoder


def visualize(data, labels, components, title):
    classes = labels.unique()
    
    if components == 2:
        plt.figure(figsize=(8, 6))
        for cls in classes:
            cls_idx = labels == cls
            plt.scatter(data[cls_idx, 0], data[cls_idx, 1], label=cls)
        
        plt.title(title)
        plt.xlabel('Главная компонента 1')
        plt.ylabel('Главная компонента 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    elif components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cls in classes:
            cls_idx = labels == cls
            ax.scatter(data[cls_idx, 0], data[cls_idx, 1], data[cls_idx, 2], label=cls)
        
        ax.set_title(title)
        ax.set_xlabel('Главная компонента 1')
        ax.set_ylabel('Главная компонента 2')
        ax.set_zlabel('Главная компонента 3')
        ax.legend()
        plt.show()


def apply_tsne(data, labels, components, perplexity=100):
    tsne = TSNE(n_components=components, perplexity=perplexity, init='random', random_state=0)
    tsne_transformed = tsne.fit_transform(data)
    visualize(tsne_transformed, labels, components, title=f't-SNE: {components} компоненты')

def apply_kernel_pca(data, labels, components, kernel='linear'):
    kpca = KernelPCA(n_components=components, kernel=kernel)
    kpca_transformed = kpca.fit_transform(data)
    visualize(kpca_transformed, labels, components, title=f'KernelPCA: {components} компоненты, kernel={kernel}')

file_path = 'D:/7 семестр/ИАД лабы/ИАД лаба №1/hcvdat0.csv'
features, labels = load_data(file_path)
input_dim = features.shape[1]

autoencoder_2d = Autoencoder(input_dim=input_dim, hidden_dim=2)
trained_autoencoder_2d = train_autoencoder(autoencoder_2d, features)
encoded_2d, _ = trained_autoencoder_2d(torch.FloatTensor(features))
encoded_2d = encoded_2d.detach().numpy()
visualize(encoded_2d, labels, 2, title='Autoencoder: 2 компоненты')

autoencoder_3d = Autoencoder(input_dim=input_dim, hidden_dim=3)
trained_autoencoder_3d = train_autoencoder(autoencoder_3d, features)
encoded_3d, _ = trained_autoencoder_3d(torch.FloatTensor(features))
encoded_3d = encoded_3d.detach().numpy()
visualize(encoded_3d, labels, 3, title='Autoencoder: 3 компоненты')

apply_tsne(features, labels, components=2, perplexity=40)
apply_tsne(features, labels, components=3, perplexity=40)

# Применение KernelPCA и визуализация для 2-х и 3-х компонент
apply_kernel_pca(features, labels, components=2, kernel='linear')
apply_kernel_pca(features, labels, components=3, kernel='linear')