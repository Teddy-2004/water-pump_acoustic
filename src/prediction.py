"""
prediction.py
-------------
Single-clip inference wrapper. This is what api/main.py calls for the
/predict endpoint -- keeps the API route thin and testable.
"""

import numpy as np
from .preprocessing import file_to_features, CLASS_NAMES


def predict_file(model, filepath: str, threshold: float = 0.5) -> dict:
    """
    Run inference on a single .wav file.

    Returns:
        {
          "label": "normal" | "abnormal",
          "confidence": float in [0, 1],   # confidence in the predicted label
          "probability_abnormal": float in [0, 1],
        }
    """
    features = file_to_features(filepath)          # (N_MELS, N_FRAMES, 1)
    batch = np.expand_dims(features, axis=0)        # (1, N_MELS, N_FRAMES, 1)

    prob_abnormal = float(model.predict(batch, verbose=0).ravel()[0])
    label_idx = int(prob_abnormal >= threshold)
    label = CLASS_NAMES[label_idx]
    confidence = prob_abnormal if label_idx == 1 else (1 - prob_abnormal)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability_abnormal": round(prob_abnormal, 4),
    }
