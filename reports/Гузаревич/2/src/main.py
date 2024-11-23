import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def prepare_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df.dropna(inplace=True)
    data = df.drop(columns=['Diagnosis'])
    diagnosis = df['Diagnosis']
    return data.values, diagnosis


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 21),
            nn.ReLU(),
            nn.Linear(21, 14),
            nn.ReLU(),
            nn.Linear(14, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 14),
            nn.ReLU(),
            nn.Linear(14, 21),
            nn.ReLU(),
            nn.Linear(21, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(data, encoding_dim, epochs):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    data_tensor = torch.FloatTensor(data_scaled)

    input_dim = data.shape[1]
    model = Autoencoder(input_dim, encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data_tensor)
        loss = criterion(outputs, data_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data_tensor).numpy()

    return encoded_data


def tsne_method(data, n_components):
    tsne = TSNE(n_components=n_components, perplexity=30, init='pca', random_state=42)
    transformed_data = tsne.fit_transform(data)
    return transformed_data

def plot_data(reduced_data, diagnosis, n_components):
    unique_classes = diagnosis.unique()

    if n_components == 2:
        for label in unique_classes:
            class_indices = diagnosis == label
            x_values = reduced_data[class_indices, 0]
            y_values = reduced_data[class_indices, 1]

            plt.scatter(x_values, y_values, label=label)

        plt.title('Autoencoder Visualization (2 Components)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for label in unique_classes:
            class_indices = diagnosis == label
            x_values = reduced_data[class_indices, 0]
            y_values = reduced_data[class_indices, 1]
            z_values = reduced_data[class_indices, 2]

            ax.scatter(x_values, y_values, z_values, label=label)

        ax.set_title('Autoencoder Visualization (3 Components)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
        plt.show()


file_path = 'Exasens.csv'

data, diagnosis = prepare_data(file_path)

n_components = 3
epochs = 10000
reduced_data = train_autoencoder(data, n_components, epochs)

plot_data(reduced_data, diagnosis, n_components)

reduced_data_tsne = tsne_method(reduced_data, n_components)
plot_data(reduced_data_tsne, diagnosis, n_components)
