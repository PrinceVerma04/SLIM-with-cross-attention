import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class PaperStyleEncoder(nn.Module):
    """
    Style encoder:
    Wav2Vec-XLSR finetuned for Speech Emotion Recognition
    Paper-selected layers: 0-10
    """
    def __init__(self):
        super().__init__()

        self.model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.model = Wav2Vec2Model.from_pretrained(
            self.model_name,
            output_hidden_states=True
        )

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

        self.selected_layers = list(range(0, 11))   # 0 to 10

    def forward(self, waveforms):
        """
        waveforms: [B, 1, T]
        returns:
            stacked hidden states: [B, K, F, T']
        """
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)  # [B, T]

        outputs = self.model(waveforms)
        hidden_states = outputs.hidden_states  # tuple of [B, T', F]

        selected = [hidden_states[i] for i in self.selected_layers]
        # stack => [K, B, T', F]
        stacked = torch.stack(selected, dim=0)

        # rearrange => [B, K, F, T']
        stacked = stacked.permute(1, 0, 3, 2)

        return stacked


class PaperLinguisticEncoder(nn.Module):
    """
    Linguistic encoder:
    Wav2Vec-XLSR finetuned for ASR
    Paper-selected layers: 14-21
    """
    def __init__(self):
        super().__init__()

        self.model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.model = Wav2Vec2Model.from_pretrained(
            self.model_name,
            output_hidden_states=True
        )

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

        self.selected_layers = list(range(14, 22))   # 14 to 21

    def forward(self, waveforms):
        """
        waveforms: [B, 1, T]
        returns:
            stacked hidden states: [B, K, F, T']
        """
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)  # [B, T]

        outputs = self.model(waveforms)
        hidden_states = outputs.hidden_states  # tuple of [B, T', F]

        selected = [hidden_states[i] for i in self.selected_layers]
        # stack => [K, B, T', F]
        stacked = torch.stack(selected, dim=0)

        # rearrange => [B, K, F, T']
        stacked = stacked.permute(1, 0, 3, 2)

        return stacked