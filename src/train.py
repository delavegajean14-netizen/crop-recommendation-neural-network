# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks

from dataset import load_dataset
from preprocess import prepare_data, normalize_data
from model import build_model


# ============================================================
# CONFIGURACIÓN
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# CARGA DEL DATASET
# ============================================================

dataset = load_dataset()

print("Primeras filas del dataset:")
print(dataset.head())


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

print(f"\nNúmero de clases: {num_classes}")
print("Clases:", list(class_names))


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

model = build_model(
    input_shape=X_train.shape[1],
    num_classes=num_classes
)

print("\nResumen del modelo:")
model.summary()

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, "crop_classifier.keras"),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# ENTRENAMIENTO
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# ============================================================
# CURVAS DE APRENDIZAJE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Curva de pérdida
axes[0].plot(history.history["loss"], label="Entrenamiento", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validación", linewidth=2)
axes[0].set_title("Curva de Pérdida")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Pérdida")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Curva de accuracy
axes[1].plot(history.history["accuracy"], label="Entrenamiento", linewidth=2)
axes[1].plot(history.history["val_accuracy"], label="Validación", linewidth=2)
axes[1].set_title("Curva de Accuracy")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "learning_curves.png"),
    dpi=120,
    bbox_inches="tight"
)

plt.show()

print("\n✓ Modelo guardado en:")
print(os.path.join(MODELS_DIR, "crop_classifier.keras"))

print("\n✓ Curvas de aprendizaje guardadas en:")
print(os.path.join(RESULTS_DIR, "learning_curves.png"))