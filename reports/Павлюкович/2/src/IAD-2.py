import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Загрузка и подготовка данных
data = pd.read_csv('heart_failure_record.csv')
X = data.drop(columns=['DEATH_EVENT']).values
y = data['DEATH_EVENT'].values

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Преобразование данных в тензоры
X_scaled_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Определение автоэнкодера с PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # Кодировщик
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        # Декодировщик
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Функция для обучения автоэнкодера
def train_autoencoder(model, data, epochs=50, batch_size=16, learning_rate=0.001):
    criterion = nn.MSELoss()  # Используем MSE для автокодировщика
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(data.size(0))
        for i in range(0, data.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = data[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Обучение автоэнкодера и визуализация
def train_and_visualize_autoencoder(latent_dim, X_scaled, y):
    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Обучаем модель
    train_autoencoder(model, X_scaled_tensor, epochs=50, batch_size=16)

    # Получение проекций данных
    model.eval()
    with torch.no_grad():
        X_encoded = model.encoder(X_scaled_tensor).numpy()

    # Визуализация
    plt.figure(figsize=(8, 6))
    if latent_dim == 2:
        scatter = plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap='coolwarm', s=30)
        plt.colorbar(scatter, label='DEATH_EVENT')
        plt.title(f'2D Autoencoder Projection')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif latent_dim == 3:
        ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection='3d')
        scatter = ax.scatter(X_encoded[:, 0], X_encoded[:, 1], X_encoded[:, 2], c=y, cmap='coolwarm', s=30)
        plt.colorbar(scatter, label='DEATH_EVENT')
        ax.set_title(f'3D Autoencoder Projection')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    plt.show()


# Обучение и визуализация автоэнкодера
train_and_visualize_autoencoder(latent_dim=2, X_scaled=X_scaled, y=y)
train_and_visualize_autoencoder(latent_dim=3, X_scaled=X_scaled, y=y)


# t-SNE для визуализации данных
def tsne_visualization(X, y, n_components, perplexity):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    if n_components == 2:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', s=30)
        plt.colorbar(scatter, label='DEATH_EVENT')
        plt.title(f'2D t-SNE with Perplexity {perplexity}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif n_components == 3:
        ax = plt.figure(figsize=(8, 6)).add_subplot(111, projection='3d')
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='coolwarm', s=30)
        plt.colorbar(scatter, label='DEATH_EVENT')
        ax.set_title(f'3D t-SNE with Perplexity {perplexity}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    plt.show()


# Визуализация t-SNE
for perplexity in [20, 40, 60]:
    tsne_visualization(X_scaled, y, n_components=2, perplexity=perplexity)
    tsne_visualization(X_scaled, y, n_components=3, perplexity=perplexity)
