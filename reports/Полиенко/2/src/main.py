import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.src.layers import Dense
from keras import Model, Input


data = pd.read_csv("Wholesale customers data.csv")


X = data.drop(columns=['Channel', 'Region'])
y = data['Region']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]

encoding_dim_2 = 2
input_layer = Input(shape=(input_dim,))
encoder_2 = Dense(encoding_dim_2, activation="relu")(input_layer)
decoder_2 = Dense(input_dim, activation="linear")(encoder_2)

autoencoder_2 = Model(inputs=input_layer, outputs=decoder_2)
autoencoder_2.compile(optimizer="adam", loss="mse")

autoencoder_2.fit(X_scaled, X_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2, verbose=0)

encoder_model_2 = Model(inputs=input_layer, outputs=encoder_2)
X_encoded_2 = encoder_model_2.predict(X_scaled)

encoding_dim_3 = 3
encoder_3 = Dense(encoding_dim_3, activation="relu")(input_layer)
decoder_3 = Dense(input_dim, activation="linear")(encoder_3)

autoencoder_3 = Model(inputs=input_layer, outputs=decoder_3)
autoencoder_3.compile(optimizer="adam", loss="mse")

autoencoder_3.fit(X_scaled, X_scaled, epochs=50, batch_size=16, shuffle=True, validation_split=0.2, verbose=0)

encoder_model_3 = Model(inputs=input_layer, outputs=encoder_3)
X_encoded_3 = encoder_model_3.predict(X_scaled)


plt.figure(figsize=(8, 6))
plt.scatter(X_encoded_2[:, 0], X_encoded_2[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.colorbar(label="Region")
plt.title("2D Projection using Autoencoder")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_encoded_3[:, 0], X_encoded_3[:, 1], X_encoded_3[:, 2], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.colorbar(sc, label="Region")
ax.set_title("3D Projection using Autoencoder")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")
plt.show()


tsne_2d = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.colorbar(label="Region")
plt.title("2D Projection using t-SNE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()

tsne_3d = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y, cmap='viridis', marker='o', edgecolor='k')
plt.colorbar(sc, label="Region")
ax.set_title("3D Projection using t-SNE")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_zlabel("t-SNE Component 3")
plt.show()
