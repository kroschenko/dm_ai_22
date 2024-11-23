from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

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

def visualize_tsne_2d(X_scaled, y, perplexity):
    plt.figure(figsize=(10, 8))
    
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    X_tsne_2D = tsne.fit_transform(X_scaled)
    
    colors = {1: 'yellow', 2: 'purple', 3: '#10C999'}
    class_names = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
    for class_value in set(y):
        plt.scatter(X_tsne_2D[y == class_value, 0], X_tsne_2D[y == class_value, 1],
                    c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)
    
    plt.title(f't-SNE - 2D Projection (perplexity={perplexity})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()

perplexity = 40
visualize_tsne_2d(X_scaled, y, perplexity)

def visualize_tsne_3d(X_scaled, y, perplexity):
    fig = plt.figure(figsize=(10, 8))
    
    tsne = TSNE(n_components=3, perplexity=perplexity, init='pca', random_state=42)
    X_tsne_3D = tsne.fit_transform(X_scaled)
    
    ax = fig.add_subplot(111, projection='3d')
    colors = {1: 'yellow', 2: 'purple', 3: '#10C999'}
    class_names = {1: 'Class 1', 2: 'Class 2', 3: 'Class 3'}
    for class_value in set(y):
        ax.scatter(X_tsne_3D[y == class_value, 0], X_tsne_3D[y == class_value, 1], X_tsne_3D[y == class_value, 2],
                   c=colors[class_value], label=class_names[class_value], edgecolor='k', s=100)

    ax.set_title(f't-SNE - 3D Projection (perplexity={perplexity})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.legend()
    plt.show()

visualize_tsne_3d(X_scaled, y, perplexity)
