"""
api/retrain.py
---------------
Background retraining job triggered by POST /retrain in api/main.py.

Flow:
    1. Load newly uploaded clips from data/incoming/<upload_id>/{normal,abnormal}/
    2. Preprocess them with src.preprocessing (identical pipeline to the notebook)
    3. Merge with a sample of existing data/train (so the model doesn't forget
       what it already knew -- avoids catastrophic forgetting on a small upload)
    4. Continue training the currently deployed model for a few epochs
    5. Evaluate on the held-out data/test set
    6. If the new model is not worse than the currently deployed one (by macro
       metrics), overwrite the deployed model file. Otherwise keep the old one
       and report the comparison -- retraining should never silently make
       production worse.
"""

import os
import json
import traceback
from datetime import datetime, timezone

import numpy as np

from src.preprocessing import load_dataset_from_folders
from src.model import load_model, train_model, evaluate_model, save_model

RETRAIN_STATUS = {"state": "idle", "started_at": None, "finished_at": None, "detail": None}

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
MAX_EXISTING_SAMPLE = 500  # cap how much old data we mix in, keeps retraining fast


def _load_incoming(incoming_dir: str):
    """Incoming dir may only have one of normal/abnormal populated -- handle both."""
    X_list, y_list = [], []
    for label_idx, cls in enumerate(["normal", "abnormal"]):
        cls_dir = os.path.join(incoming_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        # reuse load_dataset_from_folders by pointing it at a temp structure of just this class
        # simplest: call the same wav-walk logic directly
        from src.preprocessing import file_to_features
        import glob
        for wav_path in sorted(glob.glob(os.path.join(cls_dir, "*.wav"))):
            try:
                X_list.append(file_to_features(wav_path))
                y_list.append(label_idx)
            except Exception as e:
                print(f"[retrain] skipping {wav_path}: {e}")
    return np.array(X_list), np.array(y_list)


def run_retraining(incoming_dir: str, model_path: str, metrics_path: str, reload_model_callback):
    RETRAIN_STATUS.update({"state": "running", "started_at": datetime.now(timezone.utc).isoformat(), "detail": None})
    try:
        # 1. new data
        X_new, y_new = _load_incoming(incoming_dir)
        if len(X_new) == 0:
            raise ValueError("No valid .wav files found in the uploaded zip.")

        # 2. sample of existing training data, to avoid catastrophic forgetting
        X_old, y_old, _ = load_dataset_from_folders(TRAIN_DIR)
        if len(X_old) > MAX_EXISTING_SAMPLE:
            idx = np.random.choice(len(X_old), MAX_EXISTING_SAMPLE, replace=False)
            X_old, y_old = X_old[idx], y_old[idx]

        X_combined = np.concatenate([X_old, X_new], axis=0)
        y_combined = np.concatenate([y_old, y_new], axis=0)

        # simple internal train/val split for the fine-tuning run
        n_val = max(1, int(0.15 * len(X_combined)))
        perm = np.random.permutation(len(X_combined))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train, y_train = X_combined[train_idx], y_combined[train_idx]
        X_val, y_val = X_combined[val_idx], y_combined[val_idx]

        # 3. continue training the currently deployed model
        current_model = load_model(model_path)
        new_model, _ = train_model(X_train, y_train, X_val, y_val, epochs=10, model=current_model)

        # 4. evaluate candidate vs held-out test set
        X_test, y_test, _ = load_dataset_from_folders(TEST_DIR)
        new_metrics = evaluate_model(new_model, X_test, y_test)

        old_metrics = None
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                old_metrics = json.load(f)

        promote = True
        if old_metrics and "f1" in old_metrics:
            promote = new_metrics["f1"] >= old_metrics["f1"] - 0.01  # small tolerance

        if promote:
            save_model(new_model, model_path)
            reload_model_callback()
            with open(metrics_path, "w") as f:
                json.dump({
                    "accuracy": new_metrics["accuracy"],
                    "f1": new_metrics["f1"],
                    "precision": new_metrics["precision"],
                    "recall": new_metrics["recall"],
                    "roc_auc": new_metrics["roc_auc"],
                    "confusion_matrix": new_metrics["confusion_matrix"],
                    "n_train_clips": int(len(X_combined)),
                    "n_new_clips": int(len(X_new)),
                    "retrained_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)
            detail = f"Promoted new model. Test F1: {new_metrics['f1']:.4f} (previous: {old_metrics['f1']:.4f})" if old_metrics else \
                     f"Promoted new model. Test F1: {new_metrics['f1']:.4f}"
        else:
            detail = (f"New model UNDERPERFORMED (F1 {new_metrics['f1']:.4f} vs "
                      f"{old_metrics['f1']:.4f}) -- kept previous model in production.")

        RETRAIN_STATUS.update({
            "state": "idle",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "detail": detail,
        })

    except Exception as e:
        RETRAIN_STATUS.update({
            "state": "error",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "detail": f"{e}\n{traceback.format_exc()}",
        })
