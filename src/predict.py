# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import ast
import numpy as np
import tensorflow as tf

from dataset import load_dataset
from preprocess import prepare_data, normalize_data


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "crop_classifier.keras"
)


# ============================================================
# CARGA DEL MODELO
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo:\n{MODEL_PATH}\n"
        "Ejecuta primero train.py."
    )

print("Cargando modelo entrenado...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✓ Modelo cargado correctamente.")


# ============================================================
# CARGA DEL DATASET
# ============================================================

dataset = load_dataset()


# ============================================================
# PREPROCESAMIENTO
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    class_names,
    num_classes,
    encoder,
) = prepare_data(dataset)

(
    X_train,
    X_test,
    X_min,
    feature_range,
) = normalize_data(
    X_train,
    X_test
)

# ============================================================
# PROCESAMIENTO DE LA ENTRADA
# ============================================================

def parse_input(text):

    text = text.strip()

    # ========================================================
    # FORMATO MATRIZ
    # ========================================================

    if text.startswith("["):

        values = ast.literal_eval(text)

        array = np.array(
            values,
            dtype=np.float32
        )

        if array.ndim == 1:
            array = array.reshape(1, -1)

        return array

    # ========================================================
    # VARIAS MUESTRAS
    # ========================================================

    if ";" in text:

        samples = [
            sample.strip()
            for sample in text.split(";")
            if sample.strip()
        ]

        matrix = [
            [float(value) for value in sample.split(",")]
            for sample in samples
        ]

        return np.array(
            matrix,
            dtype=np.float32
        )

    # ========================================================
    # UNA SOLA MUESTRA
    # ========================================================

    values = [
        float(value)
        for value in text.split(",")
    ]

    return np.array(
        [values],
        dtype=np.float32
    )


# ============================================================
# PREDICCIÓN
# ============================================================

def predict_crops(samples):

    # ========================================================
    # NORMALIZACIÓN
    # ========================================================

    samples = (
        samples - X_min
    ) / feature_range

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    probabilities = model.predict(
        samples,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return (
        probabilities,
        predictions
    )
    
    # ============================================================
# INTERFAZ EN CONSOLA
# ============================================================

print("=" * 80)
print("PREDICTOR DE CULTIVOS")
print("=" * 80)

print("Orden de entrada:")
print("N, P, K, Temperatura, Humedad, pH, Lluvia")

print("\nFormatos permitidos:\n")

print("Una muestra:")
print("90,42,43,20.8,82.0,6.5,202.9\n")

print("Varias muestras:")
print("90,42,43,20.8,82.0,6.5,202.9;20,60,20,25.5,60.0,7.0,100.0\n")

print("Formato matriz:")
print("[[90,42,43,20.8,82.0,6.5,202.9]]\n")

print("Escribe 'salir' para terminar.\n")


# ============================================================
# BUCLE PRINCIPAL
# ============================================================

while True:

    user_input = input("Introduce los valores: ").strip()

    if user_input.lower() in (
        "salir",
        "exit",
        "quit",
        "q",
        ""
    ):
        print("\nHasta luego.")
        break

    # ========================================================
    # LEER ENTRADA
    # ========================================================

    try:

        samples = parse_input(user_input)

    except Exception as error:

        print(f"\nError al leer la entrada:\n{error}")

        print("\nEjemplo válido:")

        print(
            "90,42,43,20.8,82.0,6.5,202.9\n"
        )

        continue

    # ========================================================
    # VALIDAR ENTRADA
    # ========================================================

    if samples.ndim != 2 or samples.shape[1] != 7:

        print("\nCada muestra debe contener exactamente 7 valores.\n")

        continue

    # ========================================================
    # REALIZAR PREDICCIÓN
    # ========================================================

    try:

        probabilities, predictions = predict_crops(samples)

    except Exception as error:

        print(f"\nError durante la predicción:\n{error}\n")

        continue

    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print("\n" + "=" * 120)

    print(
        f"{'#':>3}"
        f"{'Cultivo':>20}"
        f"{'Confianza':>18}"
    )

    print("-" * 120)

    for i in range(len(samples)):

        predicted_class = predictions[i]

        crop = class_names[predicted_class]

        confidence = (
            probabilities[i][predicted_class] * 100
        )

        print(
            f"{i+1:>3}"
            f"{crop:>20}"
            f"{confidence:>17.2f}%"
        )

    print("=" * 120)

    # ========================================================
    # TOP 3 CULTIVOS
    # ========================================================

    print("\nTOP 3 CULTIVOS SUGERIDOS\n")

    for i in range(len(samples)):

        print(f"Muestra {i+1}")

        top3 = np.argsort(
            probabilities[i]
        )[::-1][:3]

        for position, index in enumerate(top3, start=1):

            print(
                f"{position}. "
                f"{class_names[index]:<15}"
                f"{probabilities[i][index]*100:6.2f}%"
            )

        print()
