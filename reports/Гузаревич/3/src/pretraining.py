import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded


pretrained_weights = []
pretrained_bias = []
input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))
layer_dims = [128, 64, 32]

X_pretrain = X_train_tensor
for hidden_dim in layer_dims:
    autoencoder = Autoencoder(input_dim, hidden_dim)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    current_dataset = TensorDataset(X_pretrain)
    current_loader = DataLoader(current_dataset, batch_size=64, shuffle=True)

    train_losses = []
    for epoch in range(50):
        autoencoder.train()
        epoch_loss = 0
        for X_batch, in current_loader:
            optimizer.zero_grad()
            encoded, decoded = autoencoder(X_batch)
            loss = criterion(decoded, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(current_loader))
        print(f"Слой с {hidden_dim} нейронами, Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    plt.plot(range(1, 51), train_losses, label=f"Layer {hidden_dim} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss for Layer {hidden_dim}")
    plt.legend()
    plt.show()

    pretrained_weights.append(autoencoder.encoder.weight.detach().clone())
    pretrained_bias.append(autoencoder.encoder.bias.detach().clone())

    autoencoder.eval()
    with torch.no_grad():
        X_pretrain = autoencoder.encoder(X_pretrain)  # Кодируем данные
    input_dim = hidden_dim  # Обновляем размер входных данных

class PretrainedDeepNN(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim, pretrained_weights, pretrained_bias):
        super(PretrainedDeepNN, self).__init__()
        layers = []
        for i, hidden_dim in enumerate(layer_dims):
            layer = nn.Linear(input_dim, hidden_dim)
            layer.weight.data = pretrained_weights[i]  # Инициализация предобученными весами
            layer.bias.data = pretrained_bias[i]
            layers.append(layer)
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


pretrained_model = PretrainedDeepNN(X_train.shape[1], layer_dims, output_dim, pretrained_weights, pretrained_bias)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.parameters(), lr=0.001)

epochs = 1000
train_losses = []

for epoch in range(epochs):
    pretrained_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = pretrained_model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch+1)%100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Pretraining")
plt.show()

pretrained_model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = pretrained_model(X_batch).argmax(dim=1)
        y_pred.extend(preds.numpy())

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
