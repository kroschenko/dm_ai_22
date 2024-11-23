import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from os.path import exists
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

is_test = True
epochs = 1000
pretrain_epochs = 30
learning_rate = 0.01
pretrain_learning_rate = 0.0001
weights_path = "thyroid_net_weights_rbm_pretrained.pth"


def prepare_data():
    train_data = pd.read_csv("ann-train.data", header=None, sep=r'\s+')
    test_data = pd.read_csv("ann-test.data", header=None, sep=r'\s+')
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1] - 1
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1] - 1

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long), \
           torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long)


class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def forward(self, v):
        h = torch.relu(torch.matmul(v, self.W) + self.h_bias)
        return h

    def reconstruct(self, h):
        v = torch.relu(torch.matmul(h, self.W.t()) + self.v_bias)
        return v

    def contrastive_divergence(self, v, k=1):
        for _ in range(k):
            h = self.forward(v)
            v = self.reconstruct(h)
        return v


def train_rbm(rbm, data_loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate)
    all_outputs = []

    for epoch in range(epochs):
        epoch_loss = 0
        outputs = []
        for v in data_loader:
            v = v[0]
            v_reconstructed = rbm.contrastive_divergence(v)
            loss = ((v - v_reconstructed) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            outputs.append(rbm.forward(v).detach())
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(data_loader):.4f}")
        all_outputs = torch.cat(outputs, dim=0)
    return all_outputs


class ThyroidNet(nn.Module):
    def __init__(self):
        super(ThyroidNet, self).__init__()
        self.fc1 = nn.Linear(21, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if exists(weights_path) and is_test:
    loaded_model = ThyroidNet()
    loaded_model.load_state_dict(torch.load(weights_path, weights_only=True))
    loaded_model.eval()
    print("Веса модели загружены.")

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data()

    if torch.cuda.is_available():
        print("Cuda is available!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.to(device)
        X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted = predicted.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    precision = precision_score(y_test, predicted, average='weighted')
    recall = recall_score(y_test, predicted, average='weighted')
    f1 = f1_score(y_test, predicted, average='weighted')
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    conf_matrix = confusion_matrix(y_test, predicted)
    print("Confusion Matrix:")
    print(conf_matrix)
else:
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data()
    model = ThyroidNet()

    if torch.cuda.is_available():
        print("Cuda is available!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rbm_1 = RBM(21, 64).to(device)
    rbm_2 = RBM(64, 32).to(device)
    rbm_3 = RBM(32, 16).to(device)

    print(f"Training 1-st RBM")
    output_1 = train_rbm(rbm_1, DataLoader(TensorDataset(X_train_tensor)), pretrain_epochs, pretrain_learning_rate)
    print(f"Training 2-nd RBM")
    output_2 = train_rbm(rbm_2, DataLoader(TensorDataset(output_1)), pretrain_epochs, pretrain_learning_rate)
    print(f"Training 3-rd RBM")
    output_3 = train_rbm(rbm_3, DataLoader(TensorDataset(output_2)), pretrain_epochs, pretrain_learning_rate)

    model.fc1.weight.data = rbm_1.W.t()
    model.fc1.bias.data = rbm_1.h_bias
    model.fc2.weight.data = rbm_2.W.t()
    model.fc2.bias.data = rbm_2.h_bias
    model.fc3.weight.data = rbm_3.W.t()
    model.fc3.bias.data = rbm_3.h_bias

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}")


    torch.save(model.state_dict(), weights_path)
    print(f"Веса модели сохранены в {weights_path}.")



    plt.plot(range(1, epochs + 1), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
