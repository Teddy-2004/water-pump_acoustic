"""
preprocessing.py
-----------------
Audio -> log-mel spectrogram feature pipeline for the acoustic pump
diagnostics project (MIMII pump subset).

Used by:
- notebook/acoustic_pump_diagnostics.ipynb   (training)
- api/main.py                                (single-file prediction)
- api/retrain.py                             (bulk retraining)

Design notes
------------
- MIMII clips are 10s, mono, 16 kHz .wav files.
- We convert each clip to a fixed-size log-mel spectrogram "image"
  (N_MELS x N_FRAMES), which lets us reuse a standard CNN image
  classifier architecture instead of hand-crafted audio features.
- All parameters are centralized here so the notebook, API, and
  retraining code can never drift out of sync with each other.
"""

import os
import glob
import numpy as np
import librosa

# ----------------------------------------------------------------------
# Fixed audio / feature parameters. Change here ONLY -- every other
# module imports these constants rather than hardcoding values, so a
# retrained model always matches what the API expects.
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000        # MIMII native sample rate
CLIP_SECONDS = 10          # MIMII clips are 10s
N_MELS = 64                # mel bands (spectrogram height)
N_FFT = 1024
HOP_LENGTH = 512
N_FRAMES = 313              # ~= (SAMPLE_RATE * CLIP_SECONDS) / HOP_LENGTH, fixed for CNN input
CLASS_NAMES = ["normal", "abnormal"]  # index 0 / 1


def load_audio(filepath: str) -> np.ndarray:
    """Load a wav file, resample to SAMPLE_RATE, force mono, pad/trim to CLIP_SECONDS."""
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    target_len = SAMPLE_RATE * CLIP_SECONDS
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def audio_to_logmel(y: np.ndarray) -> np.ndarray:
    """Convert a raw waveform to a fixed-size log-mel spectrogram, shape (N_MELS, N_FRAMES, 1)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Fix the time axis to exactly N_FRAMES (pad/trim), so every example
    # is the same shape regardless of tiny librosa framing differences.
    if log_mel.shape[1] < N_FRAMES:
        pad_width = N_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :N_FRAMES]

    # Normalize to roughly [0, 1] for stable CNN training.
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    return log_mel[..., np.newaxis].astype(np.float32)  # (N_MELS, N_FRAMES, 1)


def file_to_features(filepath: str) -> np.ndarray:
    """Convenience wrapper: wav filepath -> model-ready spectrogram array."""
    y = load_audio(filepath)
    return audio_to_logmel(y)


def load_dataset_from_folders(root_dir: str):
    """
    Expects:
        root_dir/normal/*.wav
        root_dir/abnormal/*.wav

    Returns:
        X: np.ndarray, shape (N, N_MELS, N_FRAMES, 1)
        y: np.ndarray, shape (N,)  -- 0 = normal, 1 = abnormal
        filepaths: list[str] (same order as X/y, useful for error analysis)
    """
    X, y, filepaths = [], [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(root_dir, class_name)
        wav_files = sorted(glob.glob(os.path.join(class_dir, "*.wav")))
        for wav_path in wav_files:
            try:
                X.append(file_to_features(wav_path))
                y.append(label_idx)
                filepaths.append(wav_path)
            except Exception as e:
                print(f"[preprocessing] skipping {wav_path}: {e}")

    if not X:
        raise ValueError(
            f"No .wav files found under {root_dir}/normal or {root_dir}/abnormal. "
            "Check your data/ folder structure."
        )

    return np.array(X), np.array(y), filepaths
