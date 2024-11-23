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

autoencoder_3d = MLPRegressor(hidden_layer_sizes=(6, 4), activation='relu', solver='adam', max_iter=1000)

autoencoder_3d.fit(X_scaled, X_scaled)

X_encoded_3D = autoencoder_3d.predict(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = {1: 'yellow', 2: 'purple', 3: '#10C999'}
class_names = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}

for class_value in set(y):
    ax.scatter(X_encoded_3D[y == class_value, 0], X_encoded_3D[y == class_value, 1], X_encoded_3D[y == class_value, 2],
               c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)
    
ax.set_title('Autoencoder - 3D Projection (seeds dataset)')
ax.set_xlabel('Encoded Dimension 1')
ax.set_ylabel('Encoded Dimension 2')
ax.set_zlabel('Encoded Dimension 3')
plt.legend()
plt.show()
