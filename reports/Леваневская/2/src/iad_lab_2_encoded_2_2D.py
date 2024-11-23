import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def prepare_data():
    df = pd.read_csv("seeds.csv")
    feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
    target_col = 'V8'
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

X, y = prepare_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

autoencoder = MLPRegressor(hidden_layer_sizes=(6, 4), activation='relu', solver='adam', max_iter=1000)

autoencoder.fit(X_scaled, X_scaled)

X_encoded_2D = autoencoder.predict(X_scaled)

plt.figure(figsize=(8, 6))
colors = {1: 'yellow', 2: 'purple', 3: '#10C999'}
class_names = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
for class_value in set(y):
    plt.scatter(X_encoded_2D[y == class_value, 0], X_encoded_2D[y == class_value, 1],
                c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)
plt.title('Autoencoder - 2D Projection (seeds dataset)')
plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.legend()
plt.show()
