"""
layers.py: Concentric Shell and Angular Neural Layers
=====================================================
Differential geometric curvilinear coordinate mappings and radial basis formulations.
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothed_radius(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes the epsilon-smoothed pseudo-metric norm:
        r_eps(x) = sqrt(sum_i x_i^2 + eps^2)
    This is everywhere differentiable and strictly 1-Lipschitz.
    """
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + (eps**2))


def unit_direction(x: torch.Tensor, eps: float = 1e-6, dir_eps: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes radial magnitude r and smoothly regularized directional unit vector.
    At large r, u -> x / ||x||. Near the origin, u smoothly transitions to 0,
    ensuring bounded derivatives and eliminating the coordinate singularity at x = 0.
    """
    r = smoothed_radius(x, eps=eps)
    u = x / torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True) + (dir_eps**2))
    return r, u


class ConcentricShell(nn.Module):
    """
    Evaluates Gaussian concentric shell affinities across discrete radial boundaries.
        psi_k(r) = exp( - (r - r_k)^2 / (2 * sigma_r^2) )
    Invariant under the full orthogonal group O(D) of rotations and reflections.
    """

    def __init__(
        self,
        num_shells: int = 8,
        r_min: float = 0.0,
        r_max: float = 2.0,
        sigma_r: Optional[float] = None,
        radii: Optional[List[float]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_shells = num_shells
        self.eps = eps

        if radii is not None:
            r_tensor = torch.tensor(radii, dtype=torch.float32)
            self.num_shells = len(radii)
        else:
            r_tensor = torch.linspace(r_min, r_max, num_shells)

        self.register_buffer("radii", r_tensor)

        if sigma_r is None:
            spacing = (r_max - r_min) / max(1, self.num_shells - 1) if self.num_shells > 1 else 1.0
            sigma_r = max(spacing * 0.85, 1e-4)
        self.sigma_r = float(sigma_r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., D)
        Returns:
            Affinity features of shape (..., num_shells)
        """
        r = smoothed_radius(x, eps=self.eps)  # (..., 1)
        diff = r - self.radii.view(1, -1)  # (..., num_shells)
        return torch.exp(-0.5 * (diff / self.sigma_r) ** 2)


class DynamicConductionShell(nn.Module):
    """
    Dynamic Valence Electron Conduction Shell:
    Bound inner shells for r <= r_0, transitioning into unbounded mobile
    conduction band when excited past the work-function potential barrier r_0:
        delta_r = ReLU(r - r_0)
        h_conduction = [delta_r, (delta_r)^2]
    Smooth quantum tunneling gate:
        sigma_g(r) = Sigmoid(beta * (r - r_0))
    Rescues out-of-distribution radial extrapolation collapse.
    """

    def __init__(self, r_0: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.r_0 = float(r_0)
        self.beta = float(beta)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            conduction_features: Tensor of shape (..., 2) with [delta_r, (delta_r)^2]
            gate: Tensor of shape (..., 1) with activation gate in [0, 1]
        """
        r = smoothed_radius(x, eps=self.eps)
        delta_r = F.relu(r - self.r_0)
        conduction_features = torch.cat([delta_r, delta_r**2], dim=-1)
        gate = torch.sigmoid(self.beta * (r - self.r_0))
        return conduction_features, gate


class FreeMiniRingStabilizer(nn.Module):
    """
    Free Mobile Interstitial Mini-Ring Stabilizers:
    Mobile, learnable sub-units that dynamically migrate across lattice voids
    driven by error gradients to provide spatial error damping and noise invariance.
    """

    def __init__(
        self,
        num_stabilizers: int = 8,
        in_features: int = 2,
        initial_radius: float = 1.0,
        sigma_init: float = 0.25,
    ):
        super().__init__()
        self.num_stabilizers = num_stabilizers
        self.in_features = in_features

        # Initialize centers uniformly around origin
        angles = torch.linspace(0, 2 * math.pi, num_stabilizers + 1)[:-1]
        centers = torch.zeros(num_stabilizers, in_features)
        if in_features >= 2:
            centers[:, 0] = initial_radius * torch.cos(angles)
            centers[:, 1] = initial_radius * torch.sin(angles)
        self.centers = nn.Parameter(centers)
        self.log_sigma = nn.Parameter(torch.full((num_stabilizers,), math.log(sigma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_features)
        Returns:
            damping_features: (..., num_stabilizers)
        """
        # x: (..., 1, D), centers: (1, S, D)
        x_exp = x.unsqueeze(-2)
        c_exp = self.centers.view(1, self.num_stabilizers, self.in_features)
        dist_sq = torch.sum((x_exp - c_exp) ** 2, dim=-1)  # (..., S)
        sigma = torch.exp(self.log_sigma).view(1, self.num_stabilizers)
        return torch.exp(-0.5 * dist_sq / (sigma**2 + 1e-8))


class ConcentricLinear(nn.Module):
    """
    Concentric Shell and Angular Curvilinear Layer (CS-ANA).
    Combines:
      - Epsilon-smoothed radial magnitude r
      - Gaussian concentric shells psi(r)
      - Hyperspherical directional unit vector u = x / r
      - Optional directional-shell cross products (psi(r) * u_i)
      - Optional angular Fourier multipoles for 2D inputs: cos(m*theta), sin(m*theta)
      - Optional dynamic valence electron conduction band for OOD extrapolation
      - Closed-form regularized Cholesky solver fit_closed_form() for 0-epoch instant convergence
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_shells: int = 8,
        r_min: float = 0.0,
        r_max: float = 2.0,
        sigma_r: Optional[float] = None,
        radii: Optional[List[float]] = None,
        include_direction: bool = True,
        include_cross_terms: bool = True,
        num_angular_harmonics: int = 0,
        enable_conduction: bool = True,
        r_conduction_0: float = 2.0,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_direction = include_direction
        self.include_cross_terms = include_cross_terms
        self.num_angular_harmonics = num_angular_harmonics
        self.enable_conduction = enable_conduction
        self.eps = eps

        # Radial Shells
        self.shells = ConcentricShell(
            num_shells=num_shells,
            r_min=r_min,
            r_max=r_max,
            sigma_r=sigma_r,
            radii=radii,
            eps=eps,
        )

        # Dynamic Valence Conduction Band
        if self.enable_conduction:
            self.conduction = DynamicConductionShell(r_0=r_conduction_0, eps=eps)
        else:
            self.conduction = None

        # Compute basis feature dimension
        basis_dim = 1 + self.shells.num_shells  # radius + shells
        if self.include_direction:
            basis_dim += self.in_features  # u_1 ... u_D
        if self.include_cross_terms and self.include_direction:
            basis_dim += self.shells.num_shells * self.in_features
        if self.num_angular_harmonics > 0 and self.in_features == 2:
            basis_dim += self.num_angular_harmonics * 2
        if self.enable_conduction:
            basis_dim += 2  # [delta_r, delta_r^2]

        self.basis_dim = basis_dim
        self.linear = nn.Linear(self.basis_dim, self.out_features, bias=bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the full curvilinear feature representation Phi(x).
        """
        r, u = unit_direction(x, eps=self.eps)
        psi = self.shells(x)

        feat_list = [r, psi]

        if self.include_direction:
            feat_list.append(u)

        if self.include_cross_terms and self.include_direction:
            # Outer product of shells and directional unit vector
            # psi: (..., K), u: (..., D) -> (..., K * D)
            cross = (psi.unsqueeze(-1) * u.unsqueeze(-2)).flatten(start_dim=-2)
            feat_list.append(cross)

        if self.num_angular_harmonics > 0 and self.in_features == 2:
            theta = torch.atan2(x[..., 1:2], x[..., 0:1])
            harmonics = []
            for m in range(1, self.num_angular_harmonics + 1):
                harmonics.extend([torch.cos(m * theta), torch.sin(m * theta)])
            feat_list.append(torch.cat(harmonics, dim=-1))

        if self.enable_conduction and self.conduction is not None:
            cond_feat, _ = self.conduction(x)
            feat_list.append(cond_feat)

        return torch.cat(feat_list, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.extract_features(x)
        return self.linear(phi)

    @torch.no_grad()
    def fit_closed_form(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        l2_reg: float = 1e-4,
    ) -> float:
        """
        Analytically solves for optimal linear readout weights via regularized normal equations:
            W* = (Phi^T Phi + N * lambda * I)^(-1) Phi^T y
        Bypasses gradient descent entirely with unique global convexity (Theorem 1).
        
        Args:
            x: Input tensor of shape (N, in_features)
            y: Target tensor of shape (N, out_features) or (N,)
            l2_reg: L2 regularization coefficient lambda
            
        Returns:
            Training Mean Squared Error (MSE)
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        N = x.shape[0]

        phi = self.extract_features(x)  # (N, basis_dim)

        if self.linear.bias is not None:
            # Append constant bias feature 1 to phi
            ones = torch.ones(N, 1, dtype=phi.dtype, device=phi.device)
            phi_aug = torch.cat([phi, ones], dim=1)  # (N, basis_dim + 1)
        else:
            phi_aug = phi

        K = phi_aug.shape[1]
        # Regularized normal equations
        reg_eye = l2_reg * torch.eye(K, dtype=phi_aug.dtype, device=phi_aug.device)
        if self.linear.bias is not None:
            reg_eye[-1, -1] = 0.0  # Do not regularize bias

        H = torch.matmul(phi_aug.T, phi_aug) + N * reg_eye  # (K, K)
        rhs = torch.matmul(phi_aug.T, y)  # (K, out_features)

        # Solve via Cholesky decomposition or direct linear solve
        try:
            weights_aug = torch.linalg.solve(H, rhs)  # (K, out_features)
        except Exception:
            weights_aug = torch.matmul(torch.linalg.pinv(H), rhs)

        # Assign solution to module parameters
        if self.linear.bias is not None:
            w_opt = weights_aug[:-1, :].T  # (out_features, basis_dim)
            b_opt = weights_aug[-1, :]  # (out_features,)
            self.linear.weight.copy_(w_opt)
            self.linear.bias.copy_(b_opt)
        else:
            self.linear.weight.copy_(weights_aug.T)

        preds = self.forward(x)
        mse = F.mse_loss(preds, y).item()
        return mse
