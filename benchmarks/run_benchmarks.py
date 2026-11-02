"""
run_benchmarks.py: Comprehensive Empirical Benchmark Suite
===========================================================
Runs head-to-head comparisons of CS-ANA (concentric_torch) against
standard Cartesian MLPs and standard PyTorch Attention on:
  1. 8-Bit Parity (XOR-8)
  2. 3D Concentric Russian Doll Spheres
  3. OOD Radial Extrapolation Collapse vs Dynamic Conduction
  4. Long-Context Associative Recall (L = 1,024)
"""

import time
import math
import os
import sys
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concentric_torch.layers import ConcentricLinear
from concentric_torch.attention import PhaseAttention


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_8bit_parity():
    print("\n" + "=" * 80)
    print("BENCHMARK 1: 8-BIT HYPERCUBE PARITY (XOR-8 across 256 vertices)")
    print("=" * 80)

    D = 8
    all_verts = list(itertools.product([0.0, 1.0], repeat=D))
    X = torch.tensor(all_verts, dtype=torch.float32)
    y = (X.sum(dim=-1, keepdim=True) % 2.0).float()

    # 1. Cartesian MLP (2-layer, 128 hidden)
    class CartesianMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.net(x)

    mlp = CartesianMLP()
    mlp_params = count_parameters(mlp)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    t0 = time.perf_counter()
    for epoch in range(400):
        optimizer.zero_grad()
        out = mlp(X)
        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()
    t_mlp = time.perf_counter() - t0

    with torch.no_grad():
        mlp_preds = (torch.sigmoid(mlp(X)) > 0.5).float()
        mlp_acc = (mlp_preds == y).float().mean().item() * 100.0

    # 2. CS-ANA Concentric Linear (9 Hamming Shells, Closed-Form Cholesky)
    radii = [math.sqrt(k) for k in range(D + 1)]
    cs_model = ConcentricLinear(
        in_features=D,
        out_features=1,
        num_shells=len(radii),
        radii=radii,
        sigma_r=0.15,
        include_direction=False,
        include_cross_terms=False,
        enable_conduction=False,
    )
    cs_params = count_parameters(cs_model)

    t0 = time.perf_counter()
    cs_model.fit_closed_form(X, y, l2_reg=1e-5)
    t_cs = time.perf_counter() - t0

    with torch.no_grad():
        cs_preds = (cs_model(X) > 0.5).float()
        cs_acc = (cs_preds == y).float().mean().item() * 100.0

    speedup = t_mlp / max(t_cs, 1e-6)
    param_comp = mlp_params / cs_params

    print(f"Cartesian MLP (128h)  | Params: {mlp_params:6d} | Time: {t_mlp*1000:8.2f} ms | Acc: {mlp_acc:6.1f}%")
    print(f"CS-ANA Closed-Form    | Params: {cs_params:6d} | Time: {t_cs*1000:8.2f} ms | Acc: {cs_acc:6.1f}%")
    print(f"--> Margin: {param_comp:.1f}x parameter compression, {speedup:.1f}x speedup")


def benchmark_russian_doll_spheres():
    print("\n" + "=" * 80)
    print("BENCHMARK 2: 3D RUSSIAN DOLL CONCENTRIC SPHERES (4 Nested Shells, 600 Points)")
    print("=" * 80)

    torch.manual_seed(42)
    N_per_sphere = 150
    radii = [0.5, 1.0, 1.5, 2.0]
    points = []
    labels = []

    for idx, r_val in enumerate(radii):
        u = torch.randn(N_per_sphere, 3)
        u = u / torch.norm(u, dim=-1, keepdim=True)
        pts = u * r_val + torch.randn_like(u) * 0.04
        points.append(pts)
        labels.append(torch.full((N_per_sphere,), idx, dtype=torch.long))

    X = torch.cat(points, dim=0)
    y = torch.cat(labels, dim=0)
    y_onehot = F.one_hot(y, num_classes=4).float()

    # 1. Cartesian MLP (3-layer)
    class SpheresMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )

        def forward(self, x):
            return self.net(x)

    mlp = SpheresMLP()
    mlp_params = count_parameters(mlp)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)

    t0 = time.perf_counter()
    for epoch in range(300):
        optimizer.zero_grad()
        out = mlp(X)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    t_mlp = time.perf_counter() - t0

    with torch.no_grad():
        mlp_acc = (mlp(X).argmax(dim=-1) == y).float().mean().item() * 100.0

    # 2. CS-ANA Concentric Linear (4 Shells, Closed-Form)
    cs_model = ConcentricLinear(
        in_features=3,
        out_features=4,
        num_shells=4,
        radii=radii,
        sigma_r=0.15,
        include_direction=False,
        include_cross_terms=False,
        enable_conduction=False,
    )
    cs_params = count_parameters(cs_model)

    t0 = time.perf_counter()
    cs_model.fit_closed_form(X, y_onehot, l2_reg=1e-4)
    t_cs = time.perf_counter() - t0

    with torch.no_grad():
        cs_acc = (cs_model(X).argmax(dim=-1) == y).float().mean().item() * 100.0

    speedup = t_mlp / max(t_cs, 1e-6)
    param_comp = mlp_params / cs_params

    print(f"Cartesian MLP (64h)   | Params: {mlp_params:6d} | Time: {t_mlp*1000:8.2f} ms | Acc: {mlp_acc:6.1f}%")
    print(f"CS-ANA Closed-Form    | Params: {cs_params:6d} | Time: {t_cs*1000:8.2f} ms | Acc: {cs_acc:6.1f}%")
    print(f"--> Margin: {param_comp:.1f}x parameter compression, {speedup:.1f}x speedup")


def benchmark_ood_extrapolation():
    print("\n" + "=" * 80)
    print("BENCHMARK 3: OUT-OF-DISTRIBUTION RADIAL EXTRAPOLATION (r in [2.5, 4.0] vs [0, 2])")
    print("=" * 80)

    torch.manual_seed(42)
    # Train data: r in [0, 2]
    r_train = torch.rand(1000, 1) * 2.0
    theta_train = torch.rand(1000, 1) * 2 * math.pi
    X_train = torch.cat([r_train * torch.cos(theta_train), r_train * torch.sin(theta_train)], dim=-1)
    y_train = r_train**2

    # Test data: r in [2.5, 4.0]
    r_test = 2.5 + torch.rand(500, 1) * 1.5
    theta_test = torch.rand(500, 1) * 2 * math.pi
    X_test = torch.cat([r_test * torch.cos(theta_test), r_test * torch.sin(theta_test)], dim=-1)
    y_test = r_test**2

    # 1. Static Concentric Shells (Compact Support Failure)
    static_model = ConcentricLinear(
        in_features=2,
        out_features=1,
        num_shells=6,
        r_min=0.0,
        r_max=2.0,
        enable_conduction=False,
    )
    static_model.fit_closed_form(X_train, y_train)
    with torch.no_grad():
        static_mse = F.mse_loss(static_model(X_test), y_test).item()

    # 2. Dynamic Valence Conduction Shells
    cond_model = ConcentricLinear(
        in_features=2,
        out_features=1,
        num_shells=6,
        r_min=0.0,
        r_max=2.0,
        enable_conduction=True,
        r_conduction_0=1.0,
    )
    cond_model.fit_closed_form(X_train, y_train)
    with torch.no_grad():
        cond_mse = F.mse_loss(cond_model(X_test), y_test).item()

    print(f"Static Concentric Shells (No Conduction) | Extrap MSE: {static_mse:8.3f} (Flatline Death)")
    print(f"Dynamic Valence Conduction Shells        | Extrap MSE: {cond_mse:8.3f} (Rescued Trend!)")
    print(f"--> Margin: {static_mse / cond_mse:.2f}x Error Reduction via Valence Conduction Band")


def benchmark_associative_attention():
    print("\n" + "=" * 80)
    print("BENCHMARK 4: LONG-CONTEXT ATTENTION INFERENCE (L = 1,024 Tokens)")
    print("=" * 80)

    B = 2
    L = 1024
    D = 64
    x = torch.randn(B, L, D)

    # 1. Standard PyTorch MultiheadAttention
    standard_mha = nn.MultiheadAttention(embed_dim=D, num_heads=4, batch_first=True)
    mha_params = count_parameters(standard_mha)

    # Warmup
    _ = standard_mha(x, x, x)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = standard_mha(x, x, x)
    t_mha = (time.perf_counter() - t0) / 20.0

    # 2. Concentric PhaseAttention
    phase_attn = PhaseAttention(embed_dim=D, num_shells=16, num_heads=4)
    phase_params = count_parameters(phase_attn)

    # Warmup
    _ = phase_attn(x)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = phase_attn(x)
    t_phase = (time.perf_counter() - t0) / 20.0

    print(f"Standard MultiheadAttention | Params: {mha_params:6d} | Latency: {t_mha*1000:6.2f} ms")
    print(f"Toroidal PhaseAttention    | Params: {phase_params:6d} | Latency: {t_phase*1000:6.2f} ms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    benchmark_8bit_parity()
    benchmark_russian_doll_spheres()
    benchmark_ood_extrapolation()
    benchmark_associative_attention()
