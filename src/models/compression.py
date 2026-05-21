import torch
import torch.nn as nn


class CompressionModule(nn.Module):
    """
    Bottleneck compression module per Fig. 7 and Appendix A.3 of the SLIM paper.

    Architecture: Pooling → MLP(1024→256) → [BN, LeakyReLU, Dropout] →
                  MLP(256→1024) → [BN, LeakyReLU, Dropout] →  (bottleneck recovery)
                  MLP(1024→256)                                 (projection head)

    Input:  [B, K, F, T]  where K=num layers, F=1024
    Output: [B, 256, T]
    """

    def __init__(self, input_dim=1024, output_dim=256, bn_dropout=0.1, fc_dropout=0.1):
        super().__init__()

        self.bottleneck = nn.Sequential(
            # Compress
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(bn_dropout),

            # Recover (bottleneck)
            nn.Linear(output_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(fc_dropout),
        )

        # Final projection head
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: [B, K, F, T]
        """
        B, K, F, T = x.shape

        # Average across transformer layer dimension
        x = x.mean(dim=1)          # [B, F, T]

        # Apply bottleneck per time step: reshape to [B*T, F]
        x = x.permute(0, 2, 1).contiguous().view(B * T, F)

        x = self.bottleneck(x)     # [B*T, F]
        x = self.projection(x)     # [B*T, 256]

        # Reshape back to [B, 256, T]
        x = x.view(B, T, -1).permute(0, 2, 1).contiguous()

        return x
