import random

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(123456)

# Данные из UCI
from ucimlrepo import fetch_ucirepo
maternal_health_risk = fetch_ucirepo(id=863)
X = maternal_health_risk.data.features
y = maternal_health_risk.data.targets

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=43)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    loss_list = []
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
        avg_loss = epoch_loss / len(data_loader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        all_outputs = torch.cat(outputs, dim=0)
    return loss_list, all_outputs


architecture = [X_train.shape[1], 64, 32, 3]

rbm_stack = []
data = X_train_tensor
pretrain_loss_list = []

for i in range(len(architecture) - 1):
    print(f"Training RBM: Layer {i + 1} ({architecture[i]} -> {architecture[i + 1]})")
    rbm = RBM(architecture[i], architecture[i + 1])
    loss_list, data = train_rbm(rbm, DataLoader(TensorDataset(data), batch_size=64), epochs=50, learning_rate=0.001)
    rbm_stack.append(rbm)
    pretrain_loss_list.extend(loss_list)

plt.plot(pretrain_loss_list, label='Pretraining Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

class DeepNetwork(nn.Module):
    def __init__(self, rbm_stack, architecture):
        super(DeepNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(rbm_stack)):
            input_units, output_units = architecture[i], architecture[i + 1]
            layer = nn.Linear(input_units, output_units)
            layer.weight.data = rbm_stack[i].W.t()
            layer.bias.data = rbm_stack[i].h_bias
            self.layers.append(layer)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out(x)

model = DeepNetwork(rbm_stack, architecture)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
epochs = 1000
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_loss_list.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss_list[-1]:.4f}")

# Построение графика ошибки
plt.plot(train_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Оценка модели
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(predicted.numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
