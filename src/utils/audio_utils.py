import numpy as np
import torch
import librosa
import warnings


TARGET_SR = 16000


def load_audio(filepath, target_sr=TARGET_SR):
    """
    Load audio with librosa.
    Returns waveform shape: [1, T]
    """
    try:
        waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
    except Exception as exc:
        warnings.warn(f"Failed to load audio '{filepath}': {exc}. Using silence instead.")
        waveform = np.zeros((0,), dtype=np.float32)
        sr = target_sr

    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = torch.tensor(waveform).unsqueeze(0)  # [1, T]
    return waveform, sr


def pad_or_trim(waveform, target_length):
    """
    Make waveform exactly target_length samples.
    waveform shape: [1, T]
    """
    num_samples = waveform.shape[1]

    if num_samples > target_length:
        waveform = waveform[:, :target_length]
    elif num_samples < target_length:
        pad_amount = target_length - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    return waveform


def normalize_audio(waveform):
    """
    Peak normalize waveform to [-1, 1]
    """
    if waveform.numel() == 0:
        return waveform

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def preprocess_audio(filepath, target_sr=16000, duration_sec=2.0):
    """
    Full audio preprocessing pipeline.
    Returns waveform shape: [1, target_sr * duration_sec]
    """
    waveform, sr = load_audio(filepath, target_sr=target_sr)

    if waveform.numel() == 0 or waveform.shape[1] == 0:
        warnings.warn(f"Empty audio detected for '{filepath}'. Replacing with silence.")
        waveform = torch.zeros(1, 0, dtype=torch.float32)

    waveform = normalize_audio(waveform)

    target_length = int(target_sr * duration_sec)
    waveform = pad_or_trim(waveform, target_length)

    return waveform
