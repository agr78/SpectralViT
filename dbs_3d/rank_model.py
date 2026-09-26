"""Concat-token attention with a basis-dependent rank prior.

Spectral coefficients and clinical covariates are each one scalar token; a single
self-attention layer runs over the joint set, mean-pooled to a linear head, plus a
linear residual on the raw concatenated vector so the model can fall back to plain
concatenation.

The rank weight omega_i multiplies each spectral coefficient before embedding:

  none      omega_i = 1
  inv       omega_i = 1/i                  (the PCA form: eigenvalue decay)
  lap       omega_i = exp(-tau * lambda_i) (the Laplacian form)
  lap_tau   as above with tau learned
  learned   omega_i free, initialised at 1/i
"""
import torch
import torch.nn as nn


class RankAttn(nn.Module):
    def __init__(s, K, C, lam, weight='none', d=16, tau=1.0):
        super().__init__()
        s.K, s.C, s.weight = K, C, weight
        l = torch.tensor(lam, dtype=torch.float32)
        s.register_buffer('lam', l / (l.max() + 1e-12))
        i = torch.arange(1, K + 1, dtype=torch.float32)
        s.p = None
        if weight == 'none':
            s.register_buffer('w', torch.ones(K))
        elif weight == 'inv':
            s.register_buffer('w', 1.0 / i)
        elif weight == 'lap':
            s.register_buffer('w', torch.exp(-tau * (l / (l.max() + 1e-12))))
        elif weight == 'lap_tau':
            s.w = None; s.p = nn.Parameter(torch.tensor(float(tau)))
        elif weight == 'learned':
            s.w = nn.Parameter(1.0 / i)
        else:
            raise ValueError(weight)
        s.tok = nn.Linear(1, d)
        s.pos = nn.Parameter(torch.zeros(1, K + C, d))
        s.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 2, d * 2, 0.1, batch_first=True), 1)
        s.hd = nn.Linear(d, 1)
        s.lin = nn.Linear(K + C, 1)

    def forward(s, x, c):
        w = torch.exp(-nn.functional.softplus(s.p) * s.lam) if s.p is not None else s.w
        v = torch.cat([x * w, c], 1)
        z = s.tok(v.unsqueeze(-1)) + s.pos
        return s.hd(s.enc(z).mean(1)).squeeze(-1) + s.lin(v).squeeze(-1)
