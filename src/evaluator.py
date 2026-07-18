# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

from sklearn.preprocessing import label_binarize

from dataset import load_dataset
from preprocess import prepare_data, normalize_data


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "crop_classifier.keras"
)

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
# CARGAR MODELO
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo:\n{MODEL_PATH}\n"
        "Ejecuta primero train.py."
    )

print("\nCargando modelo entrenado...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✓ Modelo cargado correctamente.")


# ============================================================
# PREDICCIONES
# ============================================================

y_prob = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_prob,
    axis=1
)

# ============================================================
# MÉTRICAS GENERALES
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

print("\n" + "=" * 60)
print("MÉTRICAS GENERALES")
print("=" * 60)
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")


# ============================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================

print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# AUC - ROC POR CLASE
# ============================================================

y_test_bin = label_binarize(
    y_test,
    classes=range(num_classes)
)

print("\n" + "=" * 60)
print("AUC - ROC POR CLASE")
print("=" * 60)

print(f"{'Cultivo':<20}{'AUC-ROC'}")
print("-" * 35)

auc_scores = []

for i, class_name in enumerate(class_names):

    auc = roc_auc_score(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    auc_scores.append(auc)

    print(f"{class_name:<20}{auc:.4f}")

print("-" * 35)
print(f"AUC promedio: {np.mean(auc_scores):.4f}")

# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(figsize=(10, 10))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False,
    cmap="Blues"
)

plt.title("Matriz de Confusión")
plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=120,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CURVAS ROC
# ============================================================

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        y_prob[:, i]
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_name} (AUC = {auc_scores[i]:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curvas ROC")

plt.grid(True, alpha=0.3)

plt.legend(
    fontsize=8,
    loc="lower right"
)

plt.tight_layout()

roc_path = os.path.join(
    RESULTS_DIR,
    "roc_curves.png"
)

plt.savefig(
    roc_path,
    dpi=120,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# RESUMEN DE LA EVALUACIÓN
# ============================================================

print("\n" + "=" * 60)
print("EVALUACIÓN FINALIZADA")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")
print(f"AUC Prom.: {np.mean(auc_scores):.4f}")

print("\nArchivos generados:")

print(f"• {confusion_path}")
print(f"• {roc_path}")

print("\n✓ Evaluación completada correctamente.")