"""
relational.py: Ring-Valued Matrix Operators and Relational Multi-Agent Layers
============================================================================
Implements Ring-Valued MatMul (Y = M ⊙ X) for native pairwise Euclidean metrics,
phase interference cos(phi_i - phi_j), and n-fold rotational puzzle constraints.
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import smoothed_radius


class RingMatMul(nn.Module):
    """
    Ring-Valued Matrix Operator Layer.
    Maps an ensemble of Q spatial 2D coordinates X = [x_1, ..., x_Q] to relational features
    using circular convolution, phase interference, and harmonic multipoles:
      - Pairwise squared distance: ||x_i - x_j||^2 = r_i^2 + r_j^2 - 2 r_i r_j cos(phi_i - phi_j)
      - Relative phase difference: cos(phi_i - phi_j), sin(phi_i - phi_j)
      - Pairwise radial distance kernels: exp(-||x_i - x_j||^2 / (2 sigma^2))
      - Rotational constraint multipoles: cos(m * theta_ij), sin(m * theta_ij)
    Solves multi-agent collision graphs, 4-Queens diagonals, and 8-puzzle braid parity.
    """

    def __init__(
        self,
        num_entities: int,
        out_features: int,
        num_distance_shells: int = 4,
        max_distance: float = 8.0,
        harmonic_modes: Tuple[int, ...] = (1, 2, 4),
        include_pairwise_distances: bool = True,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.out_features = out_features
        self.harmonic_modes = harmonic_modes
        self.include_pairwise_distances = include_pairwise_distances
        self.eps = eps

        # Distance shells for inter-entity proximity
        dist_shells = torch.linspace(0.0, max_distance, num_distance_shells)
        self.register_buffer("dist_shells", dist_shells)
        spacing = max_distance / max(1, num_distance_shells - 1)
        self.sigma_dist = float(spacing * 0.75)

        # Count features per pair (i < j):
        # Pairs: Q * (Q - 1) // 2
        num_pairs = num_entities * (num_entities - 1) // 2
        self.num_pairs = num_pairs

        # Pairwise features:
        # - Relative dx, dy (2)
        # - Distance r_ij (1)
        # - Distance shells (num_distance_shells)
        # - Phase diff: cos(phi_i - phi_j), sin(phi_i - phi_j) (2)
        # - Multipoles: 2 * len(harmonic_modes)
        feat_per_pair = 3 + num_distance_shells + 2 + 2 * len(harmonic_modes)
        self.feat_per_pair = feat_per_pair

        # Entity self features: (r_i, cos(phi_i), sin(phi_i))
        self.feat_per_entity = 3

        total_basis_dim = (num_entities * self.feat_per_entity) + (num_pairs * feat_per_pair)
        self.total_basis_dim = total_basis_dim

        self.linear = nn.Linear(total_basis_dim, out_features, bias=bias)

    def extract_relational_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., num_entities, 2) or (..., num_entities * 2)
        Returns:
            relational_features: Tensor of shape (..., total_basis_dim)
        """
        if x.shape[-1] == self.num_entities * 2 and x.ndim <= 3:
            x = x.view(*x.shape[:-1], self.num_entities, 2)

        batch_shape = x.shape[:-2]
        Q = self.num_entities

        # 1. Individual Entity Polar Coordinates
        r_ind = smoothed_radius(x, eps=self.eps)  # (..., Q, 1)
        phi_ind = torch.atan2(x[..., 1:2], x[..., 0:1])  # (..., Q, 1)
        entity_feats = torch.cat([r_ind, torch.cos(phi_ind), torch.sin(phi_ind)], dim=-1)
        entity_feats_flat = entity_feats.view(*batch_shape, -1)

        # 2. Pairwise Metric Interferences
        pair_feat_list = []
        for i in range(Q):
            for j in range(i + 1, Q):
                p_i = x[..., i, :]  # (..., 2)
                p_j = x[..., j, :]  # (..., 2)
                diff = p_j - p_i  # (..., 2)
                d_ij = smoothed_radius(diff, eps=self.eps)  # (..., 1)
                theta_ij = torch.atan2(diff[..., 1:2], diff[..., 0:1])  # (..., 1)

                # Distance shells
                diff_shells = d_ij - self.dist_shells.view(1, -1)
                shell_aff = torch.exp(-0.5 * (diff_shells / self.sigma_dist) ** 2)

                # Phase difference
                phi_i = phi_ind[..., i, :]
                phi_j = phi_ind[..., j, :]
                delta_phi = phi_i - phi_j
                phase_feats = torch.cat([torch.cos(delta_phi), torch.sin(delta_phi)], dim=-1)

                # Directional multipoles (e.g. 4-fold cos(4*theta) for diagonals)
                multipoles = []
                for m in self.harmonic_modes:
                    multipoles.extend([torch.cos(m * theta_ij), torch.sin(m * theta_ij)])
                multi_tensor = torch.cat(multipoles, dim=-1)

                p_feat = torch.cat([diff, d_ij, shell_aff, phase_feats, multi_tensor], dim=-1)
                pair_feat_list.append(p_feat)

        if len(pair_feat_list) > 0:
            pair_feats_flat = torch.cat(pair_feat_list, dim=-1)
            total_feats = torch.cat([entity_feats_flat, pair_feats_flat], dim=-1)
        else:
            total_feats = entity_feats_flat

        return total_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_relational_features(x)
        return self.linear(phi)

    @torch.no_grad()
    def fit_closed_form(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> float:
        """
        Solves optimal weights analytically via regularized normal equations in 1 step.
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        N = x.shape[0]

        phi = self.extract_relational_features(x)

        if self.linear.bias is not None:
            ones = torch.ones(N, 1, dtype=phi.dtype, device=phi.device)
            phi_aug = torch.cat([phi, ones], dim=1)
        else:
            phi_aug = phi

        K = phi_aug.shape[1]
        reg_eye = l2_reg * torch.eye(K, dtype=phi_aug.dtype, device=phi_aug.device)
        if self.linear.bias is not None:
            reg_eye[-1, -1] = 0.0

        H = torch.matmul(phi_aug.T, phi_aug) + N * reg_eye
        rhs = torch.matmul(phi_aug.T, y)

        try:
            weights_aug = torch.linalg.solve(H, rhs)
        except Exception:
            weights_aug = torch.matmul(torch.linalg.pinv(H), rhs)

        if self.linear.bias is not None:
            self.linear.weight.copy_(weights_aug[:-1, :].T)
            self.linear.bias.copy_(weights_aug[-1, :])
        else:
            self.linear.weight.copy_(weights_aug.T)

        preds = self.forward(x)
        mse = F.mse_loss(preds, y).item()
        return mse
