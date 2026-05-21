from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio_utils import preprocess_audio


class AudioDataset(Dataset):
    def __init__(self, csv_path, duration_sec=2.0):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.duration_sec = duration_sec

        if "filepath" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'filepath' and 'label' columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filepath = row["filepath"]
        label = int(row["label"])

        waveform = preprocess_audio(filepath, duration_sec=self.duration_sec)

        return {
            "waveform": waveform,   # shape [1, T]
            "label": torch.tensor(label, dtype=torch.long),
            "filepath": filepath
        }