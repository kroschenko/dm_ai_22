import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

class AE(nn.Module):

    def __init__(self, p: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(7, 21),
            nn.ReLU(),
            nn.Linear(21, p),
        )

        self.decoder = nn.Sequential(
            nn.Linear(p, 21),
            nn.ReLU(),
            nn.Linear(21, 7),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def ae_1(dataset, target):
    ae = AE(2)

    loss_f = nn.MSELoss()

    optimizer = optim.Adam(
        ae.parameters(),
        lr=1.e-2,
        weight_decay=1.e-8
    )

    dataset = torch.from_numpy(dataset)
    dataset = dataset.type(torch.float32)

    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    epochs = 2000

    for epoch in range(epochs):
        reconstructed = ae(dataset)

        loss = loss_f(reconstructed, dataset)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())

    enc = ae.encoder(dataset)
    res: np.ndarray = enc.detach().numpy()

    mask = (target == 1)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='r')

    mask = (target == 2)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='g')

    mask = (target == 3)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='b')

    plt.show()


def ae_2(dataset, target):
    ae = AE(3)

    loss_f = nn.MSELoss()

    optimizer = optim.Adam(
        ae.parameters(),
        lr=1.e-2,
        weight_decay=1.e-8
    )

    dataset = torch.from_numpy(dataset)
    dataset = dataset.type(torch.float32)

    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    epochs = 2000

    for epoch in range(epochs):
        reconstructed = ae(dataset)
        loss = loss_f(reconstructed, dataset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item())

    enc = ae.encoder(dataset)
    res: np.ndarray = enc.detach().numpy()

    ax = plt.axes(projection='3d')

    mask = (target == 1)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='r')

    mask = (target == 2)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='g')

    mask = (target == 3)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b')

    plt.show()


def tsne_1(dataset, target):
    tsne = TSNE(2, perplexity=100,)
    res = tsne.fit_transform(dataset)

    mask = (target  == 1)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='r')

    mask = (target == 2)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='g')

    mask = (target == 3)
    mask = np.concatenate((mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 2, 2))
    plt.scatter(data[:, 0], data[:, 1], color='b')

    plt.show()


def tsne_2(dataset, target):
    tsne = TSNE(3, perplexity=100,)
    res = tsne.fit_transform(dataset)

    ax = plt.axes(projection='3d')
    mask = (target  == 1)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='r')

    mask = (target == 2)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='g')

    mask = (target == 3)
    mask = np.concatenate((mask, mask, mask), axis=1)
    data = res[mask]
    data = np.reshape(data, (data.size // 3, 3))
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b')

    plt.show()


def main():

    data = np.genfromtxt('Wholesale customers data.csv', skip_header=True, delimiter=',', dtype=None)
    target = data[:, 1:2]
    dataset = np.concatenate((data[:, 0:1], data[:, 2:]), axis=1)

    ae_1(dataset, target)
    ae_2(dataset, target)

    tsne_1(dataset, target)
    tsne_2(dataset, target)


if __name__ == "__main__":
    main()
