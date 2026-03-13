"""Demostración de árbol de decisión y bosque aleatorio con visualizaciones.

Requisitos:
  - Cree y active un entorno virtual (venv) y luego instale dependencias:
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt

Este script entrena un árbol de decisión y un RandomForest sobre Iris, imprime
las precisiones y muestra gráficas de los árboles y las importancias de
característica.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np


def plot_single_tree(model, feature_names, class_names, max_depth=None):
    """Visualiza un único árbol de decisión dentro del modelo.
    Si se pasa un RandomForest, dibuja su primer estimador interno.
    """
    plt.figure(figsize=(20, 10))
    if hasattr(model, "tree_"):
        estimator = model
    elif hasattr(model, "estimators_") and model.estimators_:
        estimator = model.estimators_[0]
    else:
        raise ValueError("Modelo no compatible para plot_single_tree")

    # usar parámetros extras para que haya más espacio entre cuadros
    tree.plot_tree(estimator,
                   feature_names=feature_names,
                   class_names=[str(c).capitalize() for c in class_names],
                   filled=True,
                   rounded=True,
                   max_depth=max_depth,
                   fontsize=10)
    # ajustar los márgenes/espacio para que los cuadros no se solapen
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.title("Árbol de decisión", fontsize=16)
    plt.ylabel('Características')
    plt.xlabel('Nodos')
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.title("Importancias de características (RandomForest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Cargar conjunto de datos Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Divir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Árbol de decisión simple
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    y_pred_tree = decision_tree.predict(X_test)
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print("La Precisión Del Árbol de Decisión es: {:.2f}%".format(accuracy_tree * 100))

    # Bosque aleatorio
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_forest = random_forest.predict(X_test)
    accuracy_forest = accuracy_score(y_test, y_pred_forest)
    print("La Precisión Del Bosque Aleatorio es: {:.2f}%".format(accuracy_forest * 100))

    # Visualizaciones
    feature_names = iris.feature_names
    class_names = iris.target_names
    plot_single_tree(decision_tree, feature_names, class_names, max_depth=3)
    plot_single_tree(random_forest, feature_names, class_names, max_depth=3)
    plot_feature_importances(random_forest, feature_names)


if __name__ == '__main__':
    main()

