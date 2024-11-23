import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("D:/7 семестр/ИАД лабы/ИАД лаба №3/HCV-Egy-Data.csv")

print(data.head())

features = data.drop(columns=['Baselinehistological staging'])
target = data['Baselinehistological staging']

encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

class DeepNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 18),
            nn.ReLU(),
            nn.Linear(18, 54),
            nn.ReLU(),
            nn.Linear(54, 12),
            nn.ReLU(),
            nn.Linear(12, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
num_classes = len(np.unique(y_train))

model = DeepNN(input_size, num_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_history.append(total_loss / len(train_loader))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

plt.plot(range(1, num_epochs + 1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Without pretraining")
plt.legend()
plt.show()

model.eval()
predictions = []
with torch.no_grad():
    for batch_features, _ in test_loader:
        batch_predictions = model(batch_features).argmax(dim=1)
        predictions.extend(batch_predictions.numpy())

precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

print(f"\nPrecision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-Score:  {f1:.6f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))