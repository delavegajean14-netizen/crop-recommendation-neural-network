import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ============================================================
# CONSTRUCCIÓN DEL MODELO
# ============================================================

def build_model(input_shape, num_classes):

    # ========================================================
    # ARQUITECTURA DE LA RED NEURONAL
    # ========================================================

    model = models.Sequential([

        # Capa de entrada
        layers.Input(shape=(input_shape,)),

        # Primera capa oculta
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        # Segunda capa oculta
        layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.20),

        # Tercera capa oculta
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),

        # Capa de salida
        layers.Dense(
            num_classes,
            activation="softmax"
        )

    ])

    # ========================================================
    # CONFIGURACIÓN DEL OPTIMIZADOR
    # ========================================================

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001
    )

    # ========================================================
    # COMPILACIÓN DEL MODELO
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ========================================================
    # DEVOLVER MODELO
    # ========================================================

    return model