import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Configuración estética global
sns.set_theme(style="whitegrid") # Fondo limpio con cuadrícula tenue
plt.rcParams['figure.dpi'] = 100  # Mayor resolución

# Cargar el Conjunto de Datos Iris
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def plot_modern_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Usamos Seaborn para un heatmap elegante
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    
    plt.title('Análisis de Clasificación: Predicciones vs Realidad', fontsize=16, pad=20)
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_decision_boundaries(X, y, model, title):
    # Creamos una malla densa para que se vea suave
    h = .02 
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predecir sobre la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    # 'contourf' crea las áreas de color
    plt.contourf(xx, yy, Z, cmap='Pastel1', alpha=0.6)
    
    # Dibujamos los puntos reales con bordes blancos para que resalten
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', 
                         edgecolors='white', s=60, linewidth=1)
    
    # construir leyenda manualmente usando los colores y nombres
    # Set1 index 0=rojo,1=naranja,2=gris claro aprox
    class_colors = ['red', 'orange', 'grey']
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=class_colors[i],
                          markersize=8, label=iris.target_names[i])
               for i in range(len(class_colors))]
    plt.legend(handles=handles, title='Clases', loc='upper right')
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Característica 1 (Normalizada)', fontsize=12)
    plt.ylabel('Característica 2 (Normalizada)', fontsize=12)
    sns.despine() # Quita los bordes superior y derecho del gráfico
    plt.show()

if __name__ == '__main__':
    # Escalar los datos
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Inicializar y entrenar el modelo SVM
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)

    # Predecir las etiquetas de clase para los datos de prueba
    y_pred = svm_classifier.predict(X_test_scaled)

    # Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("Precisión del Modelo SVM: {:.2f}%".format(accuracy * 100))

    # Visualizaciones opcionales
    plot_modern_confusion_matrix(y_test, y_pred, iris.target_names)
    plot_decision_boundaries(np.vstack((X_train_scaled, X_test_scaled)),
                             np.hstack((y_train, y_test)),
                             svm_classifier,
                             title="Fronteras de decisión del SVM (kernel lineal)")