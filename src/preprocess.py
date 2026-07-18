# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PREPARACIÓN DEL DATASET
# ============================================================

def prepare_data(dataset):

    # ========================================================
    # VARIABLES DE ENTRADA Y SALIDA
    # ========================================================

    X = dataset.drop("label", axis=1).values.astype(np.float32)
    y_text = dataset["label"].values

    # ========================================================
    # CODIFICACIÓN DE LAS ETIQUETAS
    # ========================================================

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    class_names = encoder.classes_
    num_classes = len(class_names)

    # ========================================================
    # DIVISIÓN DEL DATASET
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # DEVOLVER DATOS PREPARADOS
    # ========================================================

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        class_names,
        num_classes,
        encoder,
    )


# ============================================================
# NORMALIZACIÓN DEL DATASET
# ============================================================

def normalize_data(X_train, X_test):

    # ========================================================
    # CÁLCULO DE LOS PARÁMETROS DE NORMALIZACIÓN
    # ========================================================

    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)

    data_range = X_max - X_min
    data_range[data_range == 0] = 1

    # ========================================================
    # NORMALIZACIÓN DE LOS DATOS
    # ========================================================

    X_train = (X_train - X_min) / data_range
    X_test = (X_test - X_min) / data_range

    # ========================================================
    # DEVOLVER DATOS NORMALIZADOS
    # ========================================================

    return (
        X_train,
        X_test,
        X_min,
        data_range,
    )