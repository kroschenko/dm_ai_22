import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Загрузка и подготовка данных
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X = data.drop(columns=['DEATH_EVENT']).values
y = data['DEATH_EVENT'].values

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Функция для построения автоэнкодера
def build_autoencoder(latent_dim):
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    # Кодировщик
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(latent_dim, activation='linear')(encoder)  # Средний слой
    # Декодировщик
    decoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(decoder)
    output_layer = Dense(input_dim, activation='linear')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder_model


# Обучение автоэнкодера и визуализация
def train_and_visualize_autoencoder(latent_dim, X_scaled, y):
    autoencoder, encoder_model = build_autoencoder(latent_dim)
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=1, shuffle=True)

    # Получение проекций данных
    X_encoded = encoder_model.predict(X_scaled)

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