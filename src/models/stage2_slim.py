import torch
import torch.nn as nn

from src.models.paper_ssl_encoders import PaperStyleEncoder, PaperLinguisticEncoder
from src.models.compression import CompressionModule
from src.models.cross_subspace_attention import CrossSubspaceAttention


class AttentiveStatisticsPooling(nn.Module):
    """
    Input:  [B, F, T]
    Output: [B, 2F]   (weighted mean + weighted std)
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(input_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        attn = torch.softmax(self.attention(x), dim=2)      # [B, 1, T]
        mean = torch.sum(attn * x, dim=2)                   # [B, F]
        second_moment = torch.sum(attn * (x ** 2), dim=2)
        var = torch.clamp(second_moment - mean ** 2, min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)                # [B, 2F]


class Stage2SLIM(nn.Module):
    """
    SLIM-CSAI Stage 2.

    Feature streams concatenated before classifier:
      1. style_raw_vec   (256)  — ASP+MLP over raw style SSL embeddings
      2. ling_raw_vec    (256)  — ASP+MLP over raw linguistic SSL embeddings
      3. style_dep_vec   (256)  — mean-pooled compressed style dependency
      4. ling_dep_vec    (256)  — mean-pooled compressed linguistic dependency
      5. style_res_vec   (256)  — mean-pooled cross-attn residual (style ← ling)
      6. ling_res_vec    (256)  — mean-pooled cross-attn residual (ling ← style)
                                                             Total: 1536-dim

    Streams 5 & 6 are the CSAI novelty: large for fake, small for real.

    Trainable: style_asp, ling_asp, style_projection, ling_projection, classifier
    Frozen:    style_encoder, ling_encoder, compression modules, cross_attn
    """

    def __init__(self):
        super().__init__()

        self.style_encoder = PaperStyleEncoder()
        self.ling_encoder = PaperLinguisticEncoder()

        self.style_compression = CompressionModule(input_dim=1024, output_dim=256)
        self.ling_compression = CompressionModule(input_dim=1024, output_dim=256)

        self.cross_attn = CrossSubspaceAttention(dim=256, num_heads=4, dropout=0.1)

        # Raw SSL branch: ASP(1024) → [B, 2048] → MLP → [B, 256]
        self.style_asp = AttentiveStatisticsPooling(input_dim=1024)
        self.ling_asp = AttentiveStatisticsPooling(input_dim=1024)

        self.style_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
        )
        self.ling_projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
        )

        # Classifier: 1536 → 512 → 1
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(512, 1),
        )

    def _raw_pool(self, x, asp, proj):
        """x: [B, K, 1024, T] → [B, 256]"""
        x = x.mean(dim=1)      # [B, 1024, T]
        x = asp(x)             # [B, 2048]
        return proj(x)         # [B, 256]

    def _temporal_mean(self, x):
        """x: [B, D, T] → [B, D]"""
        return x.mean(dim=2)

    def forward(self, waveforms):
        """waveforms: [B, 1, T]"""

        with torch.no_grad():
            style_raw = self.style_encoder(waveforms)   # [B, 11, 1024, T']
            ling_raw = self.ling_encoder(waveforms)     # [B,  8, 1024, T']

        # Stage 1 frozen features
        style_dep = self.style_compression(style_raw)   # [B, 256, T']
        ling_dep = self.ling_compression(ling_raw)      # [B, 256, T']

        # CSAI residuals (frozen cross-attn)
        style_residual, ling_residual, _ = self.cross_attn(style_dep, ling_dep)

        # Raw SSL branch (trainable)
        style_raw_vec = self._raw_pool(style_raw, self.style_asp, self.style_projection)
        ling_raw_vec = self._raw_pool(ling_raw, self.ling_asp, self.ling_projection)

        # Pool all temporal features to vectors
        style_dep_vec = self._temporal_mean(style_dep)       # [B, 256]
        ling_dep_vec = self._temporal_mean(ling_dep)         # [B, 256]
        style_res_vec = self._temporal_mean(style_residual)  # [B, 256]
        ling_res_vec = self._temporal_mean(ling_residual)    # [B, 256]

        # Concatenate all 6 streams → 1536-dim
        final_feat = torch.cat([
            style_raw_vec, ling_raw_vec,
            style_dep_vec, ling_dep_vec,
            style_res_vec, ling_res_vec,
        ], dim=1)  # [B, 1536]

        logits = self.classifier(final_feat).squeeze(1)  # [B]

        return {
            "logits": logits,
            "style_raw_vec": style_raw_vec,
            "ling_raw_vec": ling_raw_vec,
            "style_dep_vec": style_dep_vec,
            "ling_dep_vec": ling_dep_vec,
            "style_res_vec": style_res_vec,
            "ling_res_vec": ling_res_vec,
            "final_feat": final_feat,
        }
