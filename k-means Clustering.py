import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Crear Datos de ejemplo
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0,6], [9, 11], [8, 2], [10, 2],
    [9, 3], [4, 7], [5, 5], [6, 6]
])

# Aplicar K-Means Clustering
kmeans = KMeans(n_clusters=3) # Numeros de clusters
kmeans.fit(X)

# Obtener los centroides y etiquetas
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(f"Centroides: {centroids}")
print(f'Etiquetas: {labels}')

# Visualizar los resultados 
colors = ['r','g','b']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], c=colors[labels[i]], marker='o')

plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('K-Means Clustering')
plt.show()