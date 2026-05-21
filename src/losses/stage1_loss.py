import torch
import torch.nn as nn


def stage1_loss(style_feat, ling_feat, lambda_intra=0.007):
    """
    Exact implementation of Algorithm 1 from the SLIM paper.

    Inputs: style_feat, ling_feat — both [B, 256, T] (temporally averaged dependency features)

    Steps:
      1. BatchNorm1d(affine=False) / batch_size  — normalize each subspace
      2. L_cross = ||x_style_norm - x_ling_norm||_F^2   (Frobenius norm, cross-subspace distance)
      3. L_intra = ||S_f^T S_f - I||_F^2 + ||L_f^T L_f - I||_F^2  (intra-subspace redundancy)
      4. L = L_cross + lambda * L_intra
    """
    batch_size = style_feat.shape[0]

    # Temporal average: [B, 256, T] -> [B, 256]
    s = style_feat.mean(dim=2)   # S_f
    l = ling_feat.mean(dim=2)    # L_f

    # BatchNorm1d (affine=False) then divide by batch_size
    bn = nn.BatchNorm1d(s.shape[1], affine=False).to(s.device)
    s_norm = bn(s) / batch_size
    l_norm = bn(l) / batch_size

    # L_cross: Frobenius norm of (s_norm - l_norm)
    D = torch.linalg.norm(s_norm - l_norm, ord='fro')
    l_cross = D ** 2

    # L_intra: off-diagonal correlation penalty
    # v = x^T * x  (correlation matrix [256, 256])
    v_ling = torch.mm(l_norm.T, l_norm)
    C_ling = torch.linalg.norm(v_ling - torch.eye(v_ling.shape[-1], device=v_ling.device), ord='fro')
    C_ling = C_ling ** 2

    v_style = torch.mm(s_norm.T, s_norm)
    C_style = torch.linalg.norm(v_style - torch.eye(v_style.shape[-1], device=v_style.device), ord='fro')
    C_style = C_style ** 2

    l_intra = C_style + C_ling
    total = l_cross + lambda_intra * l_intra

    return {
        "total_loss": total,
        "cross_loss": l_cross,
        "intra_loss": l_intra,
    }
