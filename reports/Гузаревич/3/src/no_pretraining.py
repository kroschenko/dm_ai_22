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

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

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

class DeepNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 18),
            nn.ReLU(),
            nn.Linear(18, 54),
            nn.ReLU(),
            nn.Linear(54, 12),
            nn.ReLU(),
            nn.Linear(12, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


input_dim = X_train.shape[1]
output_dim = len(np.unique(y_train))
deep_nn = DeepNN(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deep_nn.parameters(), lr=0.001)

epochs = 1000
train_losses = []

for epoch in range(epochs):
    deep_nn.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = deep_nn(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("No pretraining")
plt.show()

deep_nn.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = deep_nn(X_batch).argmax(dim=1)
        y_pred.extend(preds.numpy())

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
