import torch
import torch.nn as nn

from src.models.paper_ssl_encoders import PaperStyleEncoder, PaperLinguisticEncoder
from src.models.compression import CompressionModule
from src.models.cross_subspace_attention import CrossSubspaceAttention


class Stage1SLIM(nn.Module):
    """
    SLIM-CSAI Stage 1: learns style-linguistic dependency features via
    compression modules + cross-subspace attention.

    Trainable: style_compression, ling_compression, cross_attn
    Frozen:    style_encoder, ling_encoder
    """

    def __init__(self):
        super().__init__()

        self.style_encoder = PaperStyleEncoder()
        self.ling_encoder = PaperLinguisticEncoder()

        self.style_compression = CompressionModule(input_dim=1024, output_dim=256)
        self.ling_compression = CompressionModule(input_dim=1024, output_dim=256)

        # Novel: cross-subspace attention to model conditional dependency
        self.cross_attn = CrossSubspaceAttention(dim=256, num_heads=4, dropout=0.1)

    def forward(self, waveforms):
        """
        waveforms: [B, 1, T]
        Returns dict with dependency features and attention residuals.
        """
        with torch.no_grad():
            style_raw = self.style_encoder(waveforms)   # [B, 11, 1024, T']
            ling_raw = self.ling_encoder(waveforms)     # [B,  8, 1024, T']

        style_dep = self.style_compression(style_raw)   # [B, 256, T']
        ling_dep = self.ling_compression(ling_raw)      # [B, 256, T']

        # Cross-subspace attention: residuals are large for fake, small for real
        style_residual, ling_residual, attn_loss = self.cross_attn(style_dep, ling_dep)

        return {
            "style_raw": style_raw,
            "ling_raw": ling_raw,
            "style_dep": style_dep,
            "ling_dep": ling_dep,
            "style_residual": style_residual,
            "ling_residual": ling_residual,
            "attn_loss": attn_loss,
        }
