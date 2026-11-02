"""
cli.py: Interactive Command-Line Interface for CS-ANA Circuit Synthesizer
========================================================================
Runs microsecond Boolean circuit synthesis and head-to-head CPU comparisons.
"""

import sys
import time
import math
import itertools
import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import CircuitSynthesizer


def run_mlp_comparison(X: torch.Tensor, Y: torch.Tensor) -> Tuple[int, float, float]:
    """Trains a standard Cartesian MLP with Adam on CPU to compare synthesis speed and parameters."""
    N, D = X.shape
    num_out = Y.shape[1]

    class CartesianMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(D, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_out),
            )

        def forward(self, x):
            return self.net(x)

    mlp = CartesianMLP()
    params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.02)

    t0 = time.perf_counter()
    for _ in range(250):
        opt.zero_grad()
        out = mlp(X)
        loss = F.binary_cross_entropy_with_logits(out, Y)
        loss.backward()
        opt.step()
    t_mlp_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        preds = (torch.sigmoid(mlp(X)) > 0.5).float()
        acc = (preds == Y).float().mean().item() * 100.0

    return params, t_mlp_ms, acc


def main():
    parser = argparse.ArgumentParser(
        description="concentric_synth: Microsecond Symbolic Logic & Circuit Synthesizer (CPU-Native)"
    )
    parser.add_argument("--bench", type=str, default=None, help="Benchmark: xor4, xor8, adder2, mux4, maj5")
    parser.add_argument("--expr", type=str, default=None, help="Boolean expression, e.g. '(A ^ B) & (C | D)'")
    parser.add_argument("--table", type=str, default=None, help="Truth table bitstring, e.g. '01101001'")
    parser.add_argument("--export", type=str, choices=["equation", "python", "c", "all"], default="all")
    parser.add_argument("--compare-mlp", action="store_true", help="Compare against Cartesian MLP Adam training on CPU")

    args = parser.parse_args()

    circuit = None

    if args.bench:
        circuit = CircuitSynthesizer.synthesize_benchmark(args.bench)
    elif args.expr:
        circuit = CircuitSynthesizer.synthesize_from_expression("custom_expr", args.expr)
    elif args.table:
        bitstring = args.table.strip()
        N = len(bitstring)
        D = int(math.log2(N))
        if 2**D != N:
            print(f"Error: Truth table length {N} must be a power of 2.")
            sys.exit(1)
        all_verts = list(itertools.product([0.0, 1.0], repeat=D))
        X = torch.tensor(all_verts, dtype=torch.float32)
        Y = torch.tensor([[float(b)] for b in bitstring], dtype=torch.float32)
        circuit = CircuitSynthesizer.synthesize_from_truth_table("custom_table", X, Y)
    else:
        # Default demo if no args provided: run XOR-8
        print("No input specified. Running default demo: 8-Bit Hypercube Parity (XOR-8)...")
        circuit = CircuitSynthesizer.synthesize_benchmark("xor8")

    # Render summary table
    print("\n" + "=" * 80)
    print(f"  CS-ANA CIRCUIT SYNTHESIS RESULTS: {circuit.name.upper()}")
    print("=" * 80)
    print(f"  Inputs: {circuit.num_inputs} | Outputs: {circuit.num_outputs} | States Checked: {2**circuit.num_inputs}")
    print(f"  Accuracy:            {circuit.accuracy:.2f}% (Exact Bitwise Match)")
    print(f"  Active Parameters:   {circuit.active_params} weights")
    print(f"  CPU Synthesis Time:  {circuit.synthesis_time_ms:.4f} ms ({circuit.synthesis_time_ms * 1000.0:.1f} microseconds)")

    if args.compare_mlp:
        print("\n  Benchmarking Cartesian MLP on CPU (Adam backpropagation)...")
        # Regenerate X, Y for MLP
        all_verts = list(itertools.product([0.0, 1.0], repeat=circuit.num_inputs))
        X = torch.tensor(all_verts, dtype=torch.float32)
        Y = circuit.evaluate(X)
        mlp_p, mlp_t, mlp_acc = run_mlp_comparison(X, Y)

        p_speed = mlp_t / max(circuit.synthesis_time_ms, 1e-6)
        p_comp = mlp_p / max(circuit.active_params, 1)
        print("-" * 80)
        print(f"  Cartesian MLP (CPU) | Params: {mlp_p:6d} | Time: {mlp_t:8.2f} ms | Acc: {mlp_acc:6.1f}%")
        print(f"  CS-ANA Closed-Form  | Params: {circuit.active_params:6d} | Time: {circuit.synthesis_time_ms:8.4f} ms | Acc: {circuit.accuracy:6.1f}%")
        print(f"  --> Speedup: {p_speed:.1f}x Faster | Param Compression: {p_comp:.1f}x Smaller")

    print("\n" + "-" * 80)
    print("  ANALYTICAL MATHEMATICAL FORMULA:")
    print("-" * 80)
    for out_idx in range(circuit.num_outputs):
        print(f"  {circuit.to_equation(out_idx)}")

    if args.export in ("python", "all"):
        print("\n" + "-" * 80)
        print("  STANDALONE PURE PYTHON CODE (Zero Dependencies):")
        print("-" * 80)
        print(circuit.to_python())

    if args.export in ("c", "all"):
        print("\n" + "-" * 80)
        print("  STANDALONE ANSI C CODE (<math.h>):")
        print("-" * 80)
        print(circuit.to_c())

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
