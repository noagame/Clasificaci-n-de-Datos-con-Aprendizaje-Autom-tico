import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Carga de Datos
data = load_iris()
X = data.data
y = data.target

# Aplicar PCA para reducir a 2  dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

#Graficar
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolors='k', s=100)
plt.title('PCA del conjunto de datos Iris')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()