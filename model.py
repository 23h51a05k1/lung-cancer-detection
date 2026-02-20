"""
model.py
--------
Custom CNN architecture for lung cancer classification.

Architecture:
  Conv2D(32) → ReLU → MaxPool
  Conv2D(64) → ReLU → MaxPool
  Conv2D(128) → ReLU → MaxPool
  Dropout(0.4)
  Flatten
  Dense(128) → ReLU
  Dense(5)   → Softmax
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ── Class metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Normal",
    "Adenocarcinoma",
    "Squamous_Cell_Carcinoma",
    "Large_Cell_Carcinoma",
    "Small_Cell_Lung_Cancer",
]

CLASS_DESCRIPTIONS = {
    "Normal": (
        "No malignant tissue detected. The lung appears healthy with no visible "
        "nodules or masses. Regular screening is still recommended for high-risk individuals."
    ),
    "Adenocarcinoma": (
        "Adenocarcinoma is the most common type of lung cancer, typically arising "
        "in the peripheral lung as a ground-glass opacity or solid nodule. "
        "It originates from mucus-secreting gland cells and is often found in non-smokers."
    ),
    "Squamous_Cell_Carcinoma": (
        "Squamous Cell Carcinoma arises from the flat cells lining the bronchi, "
        "usually in the central lung near the hilum. It is strongly associated with "
        "smoking and may cause obstruction of the airway."
    ),
    "Large_Cell_Carcinoma": (
        "Large Cell Carcinoma is an undifferentiated cancer that can appear anywhere "
        "in the lung. It tends to grow rapidly and presents as a large peripheral mass, "
        "often with central necrosis."
    ),
    "Small_Cell_Lung_Cancer": (
        "Small Cell Lung Cancer (SCLC) is a highly aggressive neuroendocrine tumour "
        "that typically arises centrally and spreads early via the lymphatic system and "
        "bloodstream. It is almost exclusively associated with heavy smoking."
    ),
}

IMG_SIZE  = 224
N_CLASSES = len(CLASS_NAMES)


# ── Model builder ──────────────────────────────────────────────────────────────
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=N_CLASSES):
    """
    Build and return the custom CNN model.
    The last Conv2D block is named 'last_conv' for Grad-CAM retrieval.
    """
    inputs = layers.Input(shape=input_shape, name="input_ct")

    # Block 1 ─ 32 filters
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 ─ 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 ─ 128 filters  ← Grad-CAM target layer
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      name="last_conv")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Regularisation
    x = layers.Dropout(0.4, name="dropout")(x)

    # Classifier head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense1")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = models.Model(inputs, outputs, name="LungCancerCNN")
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
