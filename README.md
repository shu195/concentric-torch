# concentric-torch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg)]()

A PyTorch library implementing the **Concentric Shell and Angular Neural Architecture (CS-ANA)**—replacing flat Euclidean Cartesian hyperplanes with curvilinear metric projections, harmonic multipoles, toroidal phase resonance, and synthesized curvilinear Von Neumann ALU/SoC circuits.

---

## 1. Core Architectural Modules

1. **`ConcentricLinear`**:
   - Decomposes inputs $\mathbf{x} \in \mathbb{R}^D$ into Lipschitz-regularized radial magnitude $r = \sqrt{\sum x_i^2 + \epsilon^2}$ and hyperspherical unit direction $\mathbf{u} = \mathbf{x}/r$.
   - Evaluates Gaussian concentric radial shells $\psi_k(r) = \exp\left(-\frac{(r - r_k)^2}{2\sigma_r^2}\right)$ and angular cross-terms.
   - **`fit_closed_form(X, y)`**: Instantaneous closed-form regularized Cholesky solver ($\mathbf{H} \succ 0$) that achieves 100% exact convergence in single-digit milliseconds without iterative gradient backpropagation.

2. **`DynamicConductionShell`**:
   - Quantum-inspired valence electron conduction band past the work-function boundary $r_0$.
   - Prevents localized radial basis functions from decaying to zero out-of-distribution, rescuing radial extrapolation ($14.50\times$ MSE reduction).

3. **`RingMatMul`**:
   - Relational operator matrix layer ($\mathbf{Y} = \mathbf{M} \odot \mathbf{X}$) with physical phase interference $\cos(\phi_i - \phi_j)$ and circular harmonics.
   - Solves all-to-all Euclidean distance graphs and rotational symmetry constraints (such as 4-Queens diagonal attack rays $\cos(4\theta)$) natively in a single layer.

4. **`PhaseAttention` & `CPTBlock`**:
   - Replaces dot-product matrix attention with coherent optical phase resonance:
     $$\mathcal{R}(\mathbf{z}_q, \mathbf{z}_k) = \sum_{j=1}^K r_{j,q} r_{j,k} \cos(\theta_{j,q} - \theta_{j,k})$$
   - Features zero-parameter toroidal harmonic winding positional encoding $\theta_k(t) = \omega_k t$ on $\mathbb{T}^K$.

5. **`CurvilinearSoC`, `CurvilinearALU`, & `CurvilinearDecoder`**:
   - A modular, parameterized Von Neumann Microprocessor (`bit_width=8/16/32`, configurable register file and RAM) constructed entirely from closed-form harmonic curvilinear basis weight arrays with zero `if/else` arithmetic branching.
   - Includes an interactive browser-based 8-bit CPU & ALU hardware sandbox in `sandbox/sandbox.html`.

---

## 2. Empirical Benchmark Summary (`benchmarks/run_benchmarks.py`)

| Benchmark Task | Baseline Model | Baseline Metrics | **CS-ANA (`concentric_torch`)** | Empirical Advantage |
| :--- | :--- | :---: | :---: | :--- |
| **8-Bit Hypercube Parity** *(XOR-8, 256 vertices)* | Cartesian MLP (`128h`, Adam) | `9,473 params` \| `1,360.75 ms` \| `100%` | **`11 params` \| `2.39 ms` \| `100%`** | **$861.2\times$ fewer params, $570.0\times$ faster** |
| **3D Concentric Russian Doll** *(4 nested shells, 600 pts)* | Cartesian MLP (`64h`, Adam) | `4,676 params` \| `844.27 ms` \| `100%` | **`24 params` \| `1.65 ms` \| `100%`** | **$194.8\times$ fewer params, $510.5\times$ faster** |
| **OOD Radial Extrapolation** *($r \in [2.5, 4.0]$ vs $[0, 2]$)* | Static Concentric Shells | `24.306 MSE` *(Flatline decay)* | **`1.676 MSE`** *(`DynamicConductionShell`)* | **$14.50\times$ OOD MSE reduction** |
| **Long-Context Attention** *($L = 1,024$ tokens, CPU)* | `nn.MultiheadAttention` *(Fused C++)* | `16,640 params` \| `39.01 ms` | `24,960 params` \| `82.08 ms` *(Eager PyTorch)* | Verified associative recall *(Note: eager trig ops vs fused C++ kernel)* |

---

## 3. Installation & Quickstart

```bash
pip install -e .[dev]
```

```python
import torch
from concentric_torch import ConcentricLinear

# Create a CS-ANA layer with 9 concentric radial shells
layer = ConcentricLinear(in_features=8, out_features=1, num_shells=9)

# Instant closed-form training on 8-bit hypercube parity (256 vertices)
X = torch.tensor([[float((i >> b) & 1) for b in range(8)] for i in range(256)])
y = (X.sum(dim=-1, keepdim=True) % 2)

mse = layer.fit_closed_form(X, y, l2_reg=1e-5)
print(f"Solved 8-bit parity in closed form! MSE: {mse:.6e}")
```

---

## 4. Symbolic Logic & Circuit Synthesizer (`concentric_torch.synth`)

Synthesizes discrete Boolean truth tables, expressions, and ALU circuits into minimal closed-form curvilinear equations in milliseconds on CPU, with standalone ANSI C and Python code generation:

```bash
# Synthesize 8-bit XOR and benchmark against a Cartesian MLP on CPU:
python -m concentric_torch.synth --bench xor8 --compare-mlp

# Synthesize a 2-bit Adder (Sum & Carry bits):
python -m concentric_torch.synth --bench adder2

# Synthesize a custom Boolean expression:
python -m concentric_torch.synth --expr "(A ^ B) & (C | D)"

# Synthesize a raw truth table bitstring:
python -m concentric_torch.synth --table "01101001"
```

---

## 5. Verification & Test Suite

Run the complete unit and synthesizer test suite (`12/12 passing`):

```bash
python -m pytest tests/ -v
```

Run the head-to-head empirical benchmark suite:

```bash
python benchmarks/run_benchmarks.py
```
