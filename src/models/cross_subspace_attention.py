import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSubspaceAttention(nn.Module):
    """
    SLIM-CSAI: Cross-Subspace Attention Inconsistency module.

    For real speech, style and linguistics are causally entangled — the model
    learns to reconstruct each subspace from the other (small residuals).
    For fake speech, TTS/VC systems sample style and content independently,
    so cross-attention fails to reconstruct → large residuals = mismatch signal.

    Input:  style_dep [B, D, T],  ling_dep [B, D, T]
    Output:
        style_residual [B, D, T]  — style not explained by linguistics
        ling_residual  [B, D, T]  — linguistics not explained by style
        attn_loss                 — Stage 1 reconstruction loss (minimized for real speech)
    """

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # style attends to linguistics: "what style does this linguistic context predict?"
        self.style_to_ling_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # linguistics attends to style: "what content does this style predict?"
        self.ling_to_style_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms for residual connections
        self.norm_style = nn.LayerNorm(dim)
        self.norm_ling = nn.LayerNorm(dim)

    def forward(self, style_dep, ling_dep):
        """
        style_dep: [B, D, T]
        ling_dep:  [B, D, T]
        Returns:
            style_residual: [B, D, T]
            ling_residual:  [B, D, T]
            attn_loss:      scalar (for Stage 1 training)
        """
        # MHA expects [B, T, D]
        s = style_dep.transpose(1, 2)   # [B, T, D]
        l = ling_dep.transpose(1, 2)    # [B, T, D]

        # style queries over linguistic keys/values → predicted style from linguistics
        style_pred, _ = self.style_to_ling_attn(query=s, key=l, value=l)
        style_pred = self.norm_style(style_pred)     # [B, T, D]

        # linguistics queries over style keys/values → predicted linguistics from style
        ling_pred, _ = self.ling_to_style_attn(query=l, key=s, value=s)
        ling_pred = self.norm_ling(ling_pred)        # [B, T, D]

        # Residuals: actual - predicted (large for fake, small for real)
        style_residual = (s - style_pred).transpose(1, 2)  # [B, D, T]
        ling_residual = (l - ling_pred).transpose(1, 2)    # [B, D, T]

        # Stage 1 loss: minimize residuals for real speech (push predictions to match actuals)
        attn_loss = (
            style_residual.pow(2).mean() +
            ling_residual.pow(2).mean()
        )

        return style_residual, ling_residual, attn_loss
