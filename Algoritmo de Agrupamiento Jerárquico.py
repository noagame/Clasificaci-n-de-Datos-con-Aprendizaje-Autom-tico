import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# Crear Datos de ejemplo
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8], [8, 8],
    [1, 0.6], [9, 11], [8, 2], [10, 2],
    [9, 3], [4, 7], [5, 5], [6, 6]
])

# Aplicar agrupamiento de jerárquico aglomerativo
Z = linkage(X, method='ward')

# Gráficar el dendrograma
plt.figure(figsize=(10,7))
dendrogram(Z)
plt.title('Dendograma de Agrupamiento Jerárquico')
plt.xlabel('Indice de muestras')
plt.ylabel('Distancia')
plt.show()

# Cortar en 3 el dendrograma para formar 3 clústeres
max_d = 7 # Valor de distancia para cortar
clusters= fcluster(Z, max_d, criterion='distance')

print(f'Clusters Asignados: {clusters}')

# Visualizar los cluster en un gráfico  de disperción
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')
plt.title('Agrupamiento Jerárquico Agrupamiento - Clusters Resultantes')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
