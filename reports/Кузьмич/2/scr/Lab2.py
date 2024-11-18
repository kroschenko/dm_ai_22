import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

Model = keras.Model
Input = keras.Input
Dense = keras.layers.Dense

os.environ["LOKY_MAX_CPU_COUNT"] = "6" 

data = pd.read_csv('C:\\Users\\vadim\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Python 3.12\\Python Files\\Programs\\ИАД\\heart_failure_clinical_records_dataset.csv')
X = data.drop(columns='DEATH_EVENT').values  
y = data['DEATH_EVENT'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


def build_autoencoder(input_dim, encoding_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

autoencoder_2d, encoder_2d = build_autoencoder(X.shape[1], 2)
autoencoder_2d.fit(X, X, epochs=50, batch_size=4, verbose=1)
X_autoencoded_2d = encoder_2d.predict(X)

autoencoder_3d, encoder_3d = build_autoencoder(X.shape[1], 3)
autoencoder_3d.fit(X, X, epochs=50, batch_size=4, verbose=1)
X_autoencoded_3d = encoder_3d.predict(X)

def plot_2d_projection(X, y, title):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Класс {label}", alpha=0.6)
    plt.xlabel("Компонента 1")
    plt.ylabel("Компонента 2")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_3d_projection(X, y, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(y):
        ax.scatter(X[y == label, 0], X[y == label, 1], X[y == label, 2], label=f"Класс {label}", alpha=0.6)
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.set_zlabel("Компонента 3")
    ax.set_title(title)
    plt.legend()
    plt.show()

plot_2d_projection(X_autoencoded_2d, y, "2D проекция (автоэнкодер)")
plot_3d_projection(X_autoencoded_3d, y, "3D проекция (автоэнкодер)")

tsne_2 = TSNE(n_components=2, perplexity=2, random_state=42)
X_tsne_2 = tsne_2.fit_transform(X)

tsne_3 = TSNE(n_components=3, perplexity=2, random_state=42)
X_tsne_3 = tsne_3.fit_transform(X)

plot_2d_projection(X_tsne_2, y, "2D проекция (t-SNE)")
plot_3d_projection(X_tsne_3, y, "3D проекция (t-SNE)")

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X)

pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X)

plot_2d_projection(X_pca_2, y, "2D проекция (PCA)")
plot_3d_projection(X_pca_3, y, "3D проекция (PCA)")

explained_variance_2 = np.sum(pca_2.explained_variance_ratio_) * 100
explained_variance_3 = np.sum(pca_3.explained_variance_ratio_) * 100
print(f"Сохранённая информация (2 главные компоненты, PCA): {explained_variance_2:.2f}%")
print(f"Сохранённая информация (3 главные компоненты, PCA): {explained_variance_3:.2f}%")
