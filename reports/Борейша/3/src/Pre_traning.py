from os.path import exists
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt


pretrain = False
is_test = False
epochs = 1000
pretrain_epoch = 80
learning_rate = 0.01
pretrain_learning_rate = 0.00001
weights_path = "thyroid_net_weights.pth"
pretrained_weights_path = "thyroid_net_weights_pretrained.pth"

def prepare_data():
    train_data = pd.read_csv("ann-train.data", header=None, sep=r'\s+')
    test_data = pd.read_csv("ann-test.data", header=None, sep=r'\s+')

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    y_train = y_train - 1
    y_test = y_test - 1

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


class AutoencoderLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoencoderLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def pretrain_autoencoder(autoencoder, data, epochs=pretrain_epoch, retain_graph=False):
    optimizer = optim.Adam(autoencoder.parameters(), lr=pretrain_learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, data)
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Autoencoder Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


class ThyroidNet(nn.Module):
    def __init__(self):
        super(ThyroidNet, self).__init__()
        self.fc1 = nn.Linear(21, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if exists(weights_path) and is_test:
    loaded_model = ThyroidNet()
    if pretrain:
        loaded_model.load_state_dict(torch.load(pretrained_weights_path, weights_only=True))
    else:
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
    model = ThyroidNet()

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = prepare_data()

    if torch.cuda.is_available():
        print("Cuda is available!")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)

    if pretrain:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        autoencoder1 = AutoencoderLayer(21, 32).to(device)
        autoencoder2 = AutoencoderLayer(32, 16).to(device)
        autoencoder3 = AutoencoderLayer(16, 8).to(device)

        print("Предобучение первого слоя...")
        pretrain_autoencoder(autoencoder1, X_train_tensor.to(device))
        model.fc1.weight.data = autoencoder1.encoder[0].weight.data
        model.fc1.bias.data = autoencoder1.encoder[0].bias.data
        print("Предобучение второго слоя...")
        pretrain_autoencoder(autoencoder2, autoencoder1.encoder(X_train_tensor.to(device)), retain_graph=True)
        model.fc2.weight.data = autoencoder2.encoder[0].weight.data
        model.fc2.bias.data = autoencoder2.encoder[0].bias.data
        print("Предобучение третьего слоя...")
        pretrain_autoencoder(autoencoder3, autoencoder2.encoder(autoencoder1.encoder(X_train_tensor.to(device))),
                             retain_graph=True)
        model.fc3.weight.data = autoencoder3.encoder[0].weight.data
        model.fc3.bias.data = autoencoder3.encoder[0].bias.data
        print("Предобучение окончено!")

        print("Веса энкодеров записаны в модель.")

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

    if pretrain:
        torch.save(model.state_dict(), weights_path)
        print(f"Веса модели сохранены в {pretrained_weights_path}.")
    else:
        torch.save(model.state_dict(), pretrained_weights_path)
        print(f"Веса модели сохранены в {weights_path}.")



    plt.plot(range(1, epochs + 1), losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()
