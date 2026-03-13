import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generar datos random
np.random.seed(0)
n_sample = 2000
time= np.linspace(0, 8, n_sample)

s1 = np.sin(2 * time) # Fuente 1: señal senoidal
s2 = np.sign(np.sin(3 * time)) # Fuente 2: señal cuadrada
s3 = np.random.normal(size=time.shape) # Fuente 3: Ruido Gaussiano

# Combinar las fuentes
S = np.c_[s1, s2, s3]
S /= S.std(axis = 0) # Estandar de la mezcla
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]) # Mezcla de Matriz
X = np.dot(S, A.T) # Mezcla las señales observadas

# Aplicar ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X) # Estimacion de las fuentes
A_ = ica.mixing_ # Estimación de la matriz de mezcla

# Graficar los resultados
plt.figure(figsize=(10, 8))
models = [X, S, S_]
names = ['Señales Mezcladas',
         'Señales Originales',
         'Señales Separadas por ICA']

colors = ['red','steelblue','orange']
for i, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot (3, 1, i)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()




