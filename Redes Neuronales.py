import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constantes
EPOCHS = 10
BATCH_SIZE = 32
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10
LEARNING_RATE = 0.001

def load_and_preprocess_data():
    """
    Carga y preprocesa el conjunto de datos MNIST.

    Returns:
        tuple: Datos de entrenamiento y prueba preprocesados.
    """
    try:
        logging.info("Cargando datos MNIST...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Normalizar píxeles
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Convertir etiquetas a one-hot encoding
        y_train = to_categorical(y_train, NUM_CLASSES)
        y_test = to_categorical(y_test, NUM_CLASSES)

        logging.info(f"Datos cargados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")
        return (X_train, y_train), (X_test, y_test)

    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
        raise

def build_model():
    """
    Construye el modelo de red neuronal.

    Returns:
        tf.keras.Model: Modelo compilado.
    """
    try:
        model = Sequential([
            Flatten(input_shape=INPUT_SHAPE, name='flatten_layer'),
            Dense(128, activation='relu', name='hidden_layer'),
            Dense(NUM_CLASSES, activation='softmax', name='output_layer')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logging.info("Modelo construido y compilado exitosamente")
        model.summary()
        return model

    except Exception as e:
        logging.error(f"Error al construir modelo: {e}")
        raise

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Entrena el modelo y devuelve el historial.

    Args:
        model: Modelo a entrenar.
        X_train, y_train: Datos de entrenamiento.
        X_test, y_test: Datos de validación.

    Returns:
        tf.keras.callbacks.History: Historial del entrenamiento.
    """
    try:
        logging.info("Iniciando entrenamiento...")
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            verbose=1
        )
        logging.info("Entrenamiento completado")
        return history

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")
        raise

def plot_training_history(history):
    """
    Grafica el historial de entrenamiento.

    Args:
        history: Historial del entrenamiento.
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')  # Estilo profesional

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pérdida
        ax1.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validación', linewidth=2)
        ax1.set_title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precisión
        ax2.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
        ax2.set_title('Evolución de la Precisión', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Precisión')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        logging.info("Gráficas generadas exitosamente")

    except Exception as e:
        logging.error(f"Error al generar gráficas: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo en el conjunto de prueba.

    Args:
        model: Modelo entrenado.
        X_test, y_test: Datos de prueba.

    Returns:
        tuple: Pérdida y precisión en prueba.
    """
    try:
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logging.info(".4f")
        return test_loss, test_accuracy

    except Exception as e:
        logging.error(f"Error en evaluación: {e}")
        raise

def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    try:
        # Cargar y preprocesar datos
        (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()

        # Construir modelo
        model = build_model()

        # Entrenar modelo
        history = train_model(model, X_train, y_train, X_test, y_test)

        # Graficar resultados
        plot_training_history(history)

        # Evaluar modelo
        test_loss, test_accuracy = evaluate_model(model, X_test, y_test)

        # Resultados finales
        print("\n" + "="*50)
        print("RESULTADOS FINALES")
        print("="*50)
        print(".4f")
        print(".4f")
        print("="*50)

    except Exception as e:
        logging.error(f"Error en la ejecución principal: {e}")
        raise

if __name__ == "__main__":
    main()
