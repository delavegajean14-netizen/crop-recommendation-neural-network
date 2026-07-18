import os
import pandas as pd

# ============================================================
# RUTA DEL DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "Crop_recommendation.csv"
)


def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"No se encontró el dataset:\n{DATASET_PATH}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        encoding="latin1"
    )

    dataset.columns = dataset.columns.str.strip()
    
    return dataset

if __name__ == "__main__":
    dataset = load_dataset()

    print(dataset.head())