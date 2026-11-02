"""
attention.py: Toroidal Phasor Attention and Concentric Phase Transformer (CPT)
==============================================================================
Implements Angular Phase Resonance Attention and CPTBlock.
Eliminates dot-product coordinate bloat via coherent phase resonance.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConcentricLinear, smoothed_radius


class ToroidalPositionalWinding(nn.Module):
    """
    Zero-Parameter Toroidal Harmonic Winding:
        theta_k(t) = omega_k * t
    Conferring strict SO(2)^K phase invariance across sequence positions.
    """

    def __init__(self, num_shells: int, max_seq_len: int = 4096):
        super().__init__()
        self.num_shells = num_shells
        # Frequencies: geometrically spaced across [1, max_freq]
        omega = torch.exp(
            torch.linspace(0, -math.log(10000.0), num_shells)
        )
        self.register_buffer("omega", omega)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            phase_offsets: (seq_len, num_shells)
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        return t * self.omega.unsqueeze(0)  # (L, num_shells)


class PhaseAttention(nn.Module):
    """
    Angular Phase Resonance Attention.
    Token affinity between query z_q and key z_k is computed via coherent optical phase resonance:
        R(z_q, z_k) = Re(z_q^H z_k) = sum_j (r_{j,q} * r_{j,k}) * cos(theta_{j,q} - theta_{j,k})
    """

    def __init__(
        self,
        embed_dim: int,
        num_shells: int = 16,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_shells = num_shells
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps

        # Projections to radial amplitudes r and angular phases theta
        # For each head: num_shells amplitudes and num_shells angles
        self.q_proj = nn.Linear(embed_dim, num_heads * num_shells * 2)
        self.k_proj = nn.Linear(embed_dim, num_heads * num_shells * 2)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.toroidal_pos = ToroidalPositionalWinding(num_shells=num_shells, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask (batch_size, seq_len, seq_len)
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, L, D = x.shape
        H = self.num_heads
        K = self.num_shells

        # 1. Project to radial and phase parameters
        q_params = self.q_proj(x).view(B, L, H, K, 2)
        k_params = self.k_proj(x).view(B, L, H, K, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        # Separate into amplitude r > 0 (via softplus) and phase theta in [-pi, pi]
        r_q = F.softplus(q_params[..., 0])  # (B, L, H, K)
        theta_q = q_params[..., 1]  # (B, L, H, K)

        r_k = F.softplus(k_params[..., 0])  # (B, L, H, K)
        theta_k = k_params[..., 1]  # (B, L, H, K)

        # 2. Add toroidal harmonic winding positional phases
        pos_phase = self.toroidal_pos(L, device=x.device).view(1, L, 1, K)
        theta_q = theta_q + pos_phase
        theta_k = theta_k + pos_phase

        # Transpose for batched multi-head attention: (B, H, L, K)
        r_q = r_q.permute(0, 2, 1, 3)
        theta_q = theta_q.permute(0, 2, 1, 3)
        r_k = r_k.permute(0, 2, 1, 3)
        theta_k = theta_k.permute(0, 2, 1, 3)

        # 3. Compute Phase Resonance Matrix:
        # For query position i and key position j:
        # R_h(i, j) = sum_k (r_{q,i,k} * r_{k,j,k}) * cos(theta_{q,i,k} - theta_{k,j,k})
        # Expand: cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        q_cos = r_q * torch.cos(theta_q)  # (B, H, L, K)
        q_sin = r_q * torch.sin(theta_q)  # (B, H, L, K)
        k_cos = r_k * torch.cos(theta_k)  # (B, H, L, K)
        k_sin = r_k * torch.sin(theta_k)  # (B, H, L, K)

        # Coherent wave interference: (B, H, L_q, L_k)
        resonance = torch.matmul(q_cos, k_cos.transpose(-1, -2)) + torch.matmul(q_sin, k_sin.transpose(-1, -2))
        scale = 1.0 / math.sqrt(K)
        scores = resonance * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Context aggregation
        out = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # (B, L, D)

        return self.out_proj(out)


class CPTBlock(nn.Module):
    """
    Concentric Phase Transformer (CPT) Block.
    Integrates PhaseAttention with a Concentric Shell FFN equipped with
    Dynamic Valence Conduction Shells.
    """

    def __init__(
        self,
        embed_dim: int,
        num_shells: int = 16,
        num_heads: int = 4,
        ffn_shells: int = 8,
        dropout: float = 0.0,
        enable_conduction: bool = True,
        r_conduction_0: float = 2.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = PhaseAttention(
            embed_dim=embed_dim,
            num_shells=num_shells,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(embed_dim)

        # Concentric Curvilinear FFN
        self.concentric_ffn1 = ConcentricLinear(
            in_features=embed_dim,
            out_features=embed_dim * 2,
            num_shells=ffn_shells,
            enable_conduction=enable_conduction,
            r_conduction_0=r_conduction_0,
        )
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Phase attention residual branch
        x = x + self.attn(self.ln1(x), mask=mask)

        # Concentric curvilinear FFN residual branch
        h = self.ln2(x)
        h = self.act(self.concentric_ffn1(h))
        h = self.dropout(self.ffn2(h))
        x = x + h
        return x
