"""
model.py
--------
CNN architecture + train/evaluate/save/load utilities for the acoustic
pump diagnostics classifier (normal vs abnormal, from log-mel spectrograms).

Used by:
- notebook/acoustic_pump_diagnostics.ipynb  (initial training + evaluation)
- api/retrain.py                            (retraining trigger)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from .preprocessing import N_MELS, N_FRAMES


def build_cnn(input_shape=(N_MELS, N_FRAMES, 1)) -> tf.keras.Model:
    """
    Small CNN for spectrogram classification. Deliberately compact:
    the MIMII pump subset is modest in size (thousands of 10s clips),
    so a deep network would just overfit -- same bias-variance lesson
    from the water pump summative (Random Forest default vs tuned).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # binary: P(abnormal)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_class_weights(y_train: np.ndarray) -> dict:
    """Inverse-frequency class weights -- abnormal is the minority class,
    same imbalance pattern as 'needs repair' in the water pump summative."""
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def train_model(X_train, y_train, X_val, y_val, epochs=30, batch_size=32, model=None):
    """Train (or continue training, if `model` is passed in -- used by retraining)."""
    if model is None:
        model = build_cnn()
    else:
        # Recompile before continuing training on a loaded model: a fresh
        # optimizer instance is required to bind to the just-loaded variables,
        # otherwise Keras raises "Unknown variable" on the first fit() call.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # smaller LR for fine-tuning
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

    class_weights = get_class_weights(y_train)

    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=2,
    )
    return model, history


def evaluate_model(model, X_test, y_test, threshold=0.5) -> dict:
    """Compute the full metric suite required by the rubric (4+ metrics)."""
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["normal", "abnormal"]
        ),
    }
    return metrics


def save_model(model: tf.keras.Model, path: str):
    model.save(path)


def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
