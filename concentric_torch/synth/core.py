"""
core.py: Core Symbolic Logic & Circuit Synthesizer Engine
=========================================================
Implements closed-form CS-ANA harmonic synthesis for discrete Boolean functions.
"""

import time
import math
import itertools
from typing import List, Tuple, Dict, Optional, Callable, Union
import torch
import torch.nn.functional as F


class SynthesizedCircuit:
    """Represents a compiled CS-ANA curvilinear circuit."""

    def __init__(
        self,
        name: str,
        num_inputs: int,
        num_outputs: int,
        feature_names: List[str],
        weights: torch.Tensor,  # (num_features, num_outputs)
        bias: Optional[torch.Tensor],  # (num_outputs,)
        synthesis_time_ms: float,
        accuracy: float,
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        radii: Optional[List[float]] = None,
        sigma_r: float = 0.15,
        use_phasor: bool = True,
    ):
        self.name = name
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.feature_names = feature_names
        self.weights = weights
        self.bias = bias
        self.synthesis_time_ms = synthesis_time_ms
        self.accuracy = accuracy
        self.feature_extractor = feature_extractor
        self.radii = radii
        self.sigma_r = sigma_r
        self.use_phasor = use_phasor

        # Count active non-zero parameters
        num_w = (weights.abs() > 1e-4).sum().item()
        num_b = (bias.abs() > 1e-4).sum().item() if bias is not None else 0
        self.active_params = int(num_w + num_b)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the circuit on binary input tensor x (..., num_inputs).
        Returns binary output predictions (..., num_outputs).
        """
        phi = self.feature_extractor(x)
        out = torch.matmul(phi, self.weights)
        if self.bias is not None:
            out = out + self.bias
        return (out > 0.5).float()

    def to_equation(self, output_idx: int = 0) -> str:
        """Returns the analytical mathematical equation for a given output bit."""
        terms = []
        w_col = self.weights[:, output_idx]
        for name, w in zip(self.feature_names, w_col):
            val = w.item()
            if abs(val) > 1e-4:
                terms.append(f"{val:+.4f} * {name}")
        b_val = self.bias[output_idx].item() if self.bias is not None else 0.0
        if abs(b_val) > 1e-4 or not terms:
            terms.append(f"{b_val:+.4f}")
        expr = " ".join(terms)
        return f"y_{output_idx} = round( {expr} )"

    def to_python(self) -> str:
        """Exports standalone, zero-dependency Python code."""
        lines = [
            f"# Auto-synthesized CS-ANA Circuit: {self.name}",
            f"# Active Parameters: {self.active_params} | Accuracy: {self.accuracy:.1f}%",
            f"# Compiled in: {self.synthesis_time_ms:.4f} ms on CPU",
            "import math",
            "",
            f"def compute_{self.name}(inputs):",
            f"    # inputs: list/tuple of {self.num_inputs} binary ints (0 or 1)",
            "    r = math.sqrt(sum(x**2 for x in inputs) + 1e-12)",
            "    phi = math.pi * sum(inputs)",
        ]

        # Extract features
        for i, name in enumerate(self.feature_names):
            if name.startswith("psi_") and "_x" not in name:
                r_k = float(name.split("_")[1])
                lines.append(f"    feat_{i} = math.exp(-0.5 * ((r - {r_k:.4f}) / {self.sigma_r:.4f})**2)  # {name}")
            elif "_x" in name and name.startswith("psi_"):
                # Shell cross term: psi_0.0000_x1
                parts = name.split("_")
                r_k = float(parts[1])
                idx = int(parts[2][1:])
                lines.append(f"    feat_{i} = math.exp(-0.5 * ((r - {r_k:.4f}) / {self.sigma_r:.4f})**2) * float(inputs[{idx}])  # {name}")
            elif name == "cos_phi":
                lines.append(f"    feat_{i} = math.cos(phi)  # {name}")
            elif name == "sin_phi":
                lines.append(f"    feat_{i} = math.sin(phi)  # {name}")
            elif name.startswith("x_"):
                idx = int(name.split("_")[1])
                lines.append(f"    feat_{i} = float(inputs[{idx}])  # {name}")
            elif name.startswith("pair_"):
                # pair_0_1
                parts = name.split("_")
                i1, i2 = int(parts[1]), int(parts[2])
                lines.append(f"    feat_{i} = float(inputs[{i1}] * inputs[{i2}])  # {name}")
            elif name.startswith("tri_"):
                # tri_0_1_2
                parts = name.split("_")
                i1, i2, i3 = int(parts[1]), int(parts[2]), int(parts[3])
                lines.append(f"    feat_{i} = float(inputs[{i1}] * inputs[{i2}] * inputs[{i3}])  # {name}")

        lines.append("")
        lines.append("    outputs = []")
        for out_idx in range(self.num_outputs):
            terms = []
            for feat_idx, w in enumerate(self.weights[:, out_idx]):
                val = w.item()
                if abs(val) > 1e-4:
                    terms.append(f"({val:+.6f} * feat_{feat_idx})")
            b_val = self.bias[out_idx].item() if self.bias is not None else 0.0
            terms_str = " + ".join(terms) if terms else "0.0"
            lines.append(f"    y_{out_idx} = {terms_str} {b_val:+.6f}")
            lines.append(f"    outputs.append(1 if y_{out_idx} > 0.5 else 0)")

        if self.num_outputs == 1:
            lines.append("    return outputs[0]")
        else:
            lines.append("    return tuple(outputs)")

        return "\n".join(lines)

    def to_c(self) -> str:
        """Exports standalone ANSI C function."""
        lines = [
            f"/* Auto-synthesized CS-ANA Circuit: {self.name} */",
            f"/* Active Parameters: {self.active_params} | Compile Time: {self.synthesis_time_ms:.4f} ms */",
            "#include <math.h>",
            "",
            f"void compute_{self.name}(const int inputs[{self.num_inputs}], int outputs[{self.num_outputs}]) {{",
            "    double sum_sq = 0.0;",
            "    double sum_lin = 0.0;",
            f"    for (int i = 0; i < {self.num_inputs}; i++) {{",
            "        sum_sq += (double)(inputs[i] * inputs[i]);",
            "        sum_lin += (double)inputs[i];",
            "    }",
            "    double r = sqrt(sum_sq + 1e-12);",
            "    double phi = 3.141592653589793 * sum_lin;",
            "",
        ]

        for i, name in enumerate(self.feature_names):
            if name.startswith("psi_") and "_x" not in name:
                r_k = float(name.split("_")[1])
                lines.append(f"    double feat_{i} = exp(-0.5 * pow((r - {r_k:.6f}) / {self.sigma_r:.6f}, 2.0));")
            elif "_x" in name and name.startswith("psi_"):
                parts = name.split("_")
                r_k = float(parts[1])
                idx = int(parts[2][1:])
                lines.append(f"    double feat_{i} = exp(-0.5 * pow((r - {r_k:.6f}) / {self.sigma_r:.6f}, 2.0)) * (double)inputs[{idx}];")
            elif name == "cos_phi":
                lines.append(f"    double feat_{i} = cos(phi);")
            elif name == "sin_phi":
                lines.append(f"    double feat_{i} = sin(phi);")
            elif name.startswith("x_"):
                idx = int(name.split("_")[1])
                lines.append(f"    double feat_{i} = (double)inputs[{idx}];")
            elif name.startswith("pair_"):
                parts = name.split("_")
                i1, i2 = int(parts[1]), int(parts[2])
                lines.append(f"    double feat_{i} = (double)(inputs[{i1}] * inputs[{i2}]);")
            elif name.startswith("tri_"):
                parts = name.split("_")
                i1, i2, i3 = int(parts[1]), int(parts[2]), int(parts[3])
                lines.append(f"    double feat_{i} = (double)(inputs[{i1}] * inputs[{i2}] * inputs[{i3}]);")

        lines.append("")
        for out_idx in range(self.num_outputs):
            terms = []
            for feat_idx, w in enumerate(self.weights[:, out_idx]):
                val = w.item()
                if abs(val) > 1e-4:
                    terms.append(f"({val:+.6f} * feat_{feat_idx})")
            b_val = self.bias[out_idx].item() if self.bias is not None else 0.0
            terms_str = " + ".join(terms) if terms else "0.0"
            lines.append(f"    double y_{out_idx} = {terms_str} {b_val:+.6f};")
            lines.append(f"    outputs[{out_idx}] = (y_{out_idx} > 0.5) ? 1 : 0;")

        lines.append("}")
        return "\n".join(lines)


class CircuitSynthesizer:
    """
    Synthesizes minimal CS-ANA curvilinear equations from Boolean truth tables or logic expressions.
    """

    @staticmethod
    def synthesize_from_truth_table(
        name: str,
        X: torch.Tensor,
        Y: torch.Tensor,
        use_phasor: bool = True,
        use_hamming_shells: bool = True,
        include_pairwise: bool = True,
        include_triplets: bool = True,
        l2_reg: float = 1e-5,
        prune_threshold: float = 1e-3,
    ) -> SynthesizedCircuit:
        """
        Solves optimal weights analytically via regularized normal equations on CPU.
        """
        t0 = time.perf_counter()

        X = X.float()
        Y = Y.float()

        N, D = X.shape
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        num_outputs = Y.shape[1]

        feature_names = []
        feature_tensors = []

        # 1. Linear pass-through inputs: x_0, ..., x_{D-1}
        for d in range(D):
            feature_names.append(f"x_{d}")
            feature_tensors.append(X[:, d : d + 1])

        # 2. Phasor Winding (SO(2) circular harmonics for parity and winding operations)
        if use_phasor:
            phi = math.pi * X.sum(dim=-1, keepdim=True)
            feature_names.append("cos_phi")
            feature_tensors.append(torch.cos(phi))
            feature_names.append("sin_phi")
            feature_tensors.append(torch.sin(phi))

        # 3. Hamming Concentric Shells (r_k = sqrt(k) for k = 0 ... D)
        radii = [math.sqrt(k) for k in range(D + 1)]
        sigma_r = 0.15
        if use_hamming_shells:
            r = torch.sqrt(torch.sum(X**2, dim=-1, keepdim=True) + 1e-12)
            for k in range(D + 1):
                r_k = radii[k]
                diff = r - r_k
                psi_k = torch.exp(-0.5 * (diff / sigma_r) ** 2)
                feature_names.append(f"psi_{r_k:.4f}")
                feature_tensors.append(psi_k)
                # Shell cross terms (receptive field modulation across inputs)
                for d in range(D):
                    feature_names.append(f"psi_{r_k:.4f}_x{d}")
                    feature_tensors.append(psi_k * X[:, d : d + 1])

        # 4. Pairwise multiplicative interactions
        if include_pairwise:
            for i in range(D):
                for j in range(i + 1, D):
                    feature_names.append(f"pair_{i}_{j}")
                    feature_tensors.append(X[:, i : i + 1] * X[:, j : j + 1])

        # 5. Triplet interactions (for MUX and 3rd order logic if D <= 6)
        if include_triplets and D <= 6:
            for i in range(D):
                for j in range(i + 1, D):
                    for k in range(j + 1, D):
                        feature_names.append(f"tri_{i}_{j}_{k}")
                        feature_tensors.append(X[:, i : i + 1] * X[:, j : j + 1] * X[:, k : k + 1])

        Phi = torch.cat(feature_tensors, dim=-1)  # (N, K)
        K_feat = Phi.shape[1]

        # Append constant bias
        ones = torch.ones(N, 1, dtype=Phi.dtype, device=Phi.device)
        Phi_aug = torch.cat([Phi, ones], dim=1)  # (N, K + 1)

        # Cholesky solve
        reg_eye = l2_reg * torch.eye(K_feat + 1, dtype=Phi.dtype, device=Phi.device)
        reg_eye[-1, -1] = 0.0  # Do not regularize bias

        H = torch.matmul(Phi_aug.T, Phi_aug) + N * reg_eye
        rhs = torch.matmul(Phi_aug.T, Y)

        try:
            weights_aug = torch.linalg.solve(H, rhs)
        except Exception:
            weights_aug = torch.matmul(torch.linalg.pinv(H), rhs)

        weights = weights_aug[:-1, :]
        bias = weights_aug[-1, :]

        # Prune small weights
        mask = weights.abs() > prune_threshold
        weights = weights * mask.float()

        synthesis_time_ms = (time.perf_counter() - t0) * 1000.0

        # Feature extractor closure
        def extract_features(x_in: torch.Tensor) -> torch.Tensor:
            f_list = []
            for d in range(D):
                f_list.append(x_in[..., d : d + 1])
            if use_phasor:
                phi_in = math.pi * x_in.sum(dim=-1, keepdim=True)
                f_list.append(torch.cos(phi_in))
                f_list.append(torch.sin(phi_in))
            if use_hamming_shells:
                r_in = torch.sqrt(torch.sum(x_in**2, dim=-1, keepdim=True) + 1e-12)
                for k in range(D + 1):
                    r_k = radii[k]
                    diff_in = r_in - r_k
                    psi_k_in = torch.exp(-0.5 * (diff_in / sigma_r) ** 2)
                    f_list.append(psi_k_in)
                    for d in range(D):
                        f_list.append(psi_k_in * x_in[..., d : d + 1])
            if include_pairwise:
                for i in range(D):
                    for j in range(i + 1, D):
                        f_list.append(x_in[..., i : i + 1] * x_in[..., j : j + 1])
            if include_triplets and D <= 6:
                for i in range(D):
                    for j in range(i + 1, D):
                        for k in range(j + 1, D):
                            f_list.append(x_in[..., i : i + 1] * x_in[..., j : j + 1] * x_in[..., k : k + 1])
            return torch.cat(f_list, dim=-1)

        # Verify accuracy across all N points
        preds = (torch.matmul(Phi, weights) + bias > 0.5).float()
        acc = (preds == Y).float().mean().item() * 100.0

        return SynthesizedCircuit(
            name=name,
            num_inputs=D,
            num_outputs=num_outputs,
            feature_names=feature_names,
            weights=weights,
            bias=bias,
            synthesis_time_ms=synthesis_time_ms,
            accuracy=acc,
            feature_extractor=extract_features,
            radii=radii,
            sigma_r=sigma_r,
            use_phasor=use_phasor,
        )

    @classmethod
    def synthesize_from_expression(
        cls,
        name: str,
        expr_str: str,
        var_names: Optional[List[str]] = None,
    ) -> SynthesizedCircuit:
        """
        Synthesizes a circuit from a Boolean Python expression string, e.g. '(A ^ B) & (C | ~D)'.
        """
        if var_names is None:
            import re
            tokens = sorted(list(set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expr_str))))
            var_names = [t for t in tokens if t not in {"and", "or", "not", "True", "False"}]

        D = len(var_names)
        all_verts = list(itertools.product([0, 1], repeat=D))
        X_list = []
        Y_list = []

        for vals in all_verts:
            env = dict(zip(var_names, vals))
            y_val = bool(eval(expr_str, {"__builtins__": None}, env))
            X_list.append(list(vals))
            Y_list.append([1.0 if y_val else 0.0])

        X = torch.tensor(X_list, dtype=torch.float32)
        Y = torch.tensor(Y_list, dtype=torch.float32)

        return cls.synthesize_from_truth_table(name, X, Y)

    @classmethod
    def synthesize_benchmark(cls, benchmark_name: str) -> SynthesizedCircuit:
        """Synthesizes canonical benchmarks: xor4, xor8, xor16, mux4, adder2, maj5."""
        benchmark_name = benchmark_name.lower()

        if benchmark_name.startswith("xor"):
            D = int(benchmark_name.replace("xor", ""))
            all_verts = list(itertools.product([0.0, 1.0], repeat=D))
            X = torch.tensor(all_verts, dtype=torch.float32)
            Y = (X.sum(dim=-1, keepdim=True) % 2.0).float()
            return cls.synthesize_from_truth_table(
                f"xor_{D}",
                X,
                Y,
                include_pairwise=(D <= 8),
                include_triplets=False,
            )

        elif benchmark_name == "mux4":
            # 2 address lines (s0, s1), 4 data lines (d0..d3) -> 6 inputs, 1 output
            all_verts = list(itertools.product([0.0, 1.0], repeat=6))
            X_list = []
            Y_list = []
            for v in all_verts:
                s0, s1, d0, d1, d2, d3 = v
                addr = int(s0 * 2 + s1)
                data = [d0, d1, d2, d3]
                X_list.append(v)
                Y_list.append([float(data[addr])])
            return cls.synthesize_from_truth_table("mux4", torch.tensor(X_list), torch.tensor(Y_list))

        elif benchmark_name == "adder2":
            # 2-bit Adder: A (2 bits), B (2 bits) -> Sum (3 bits)
            all_verts = list(itertools.product([0.0, 1.0], repeat=4))
            X_list = []
            Y_list = []
            for v in all_verts:
                a = int(v[0] * 2 + v[1])
                b = int(v[2] * 2 + v[3])
                res = a + b  # 0 .. 6
                bits = [float((res >> 2) & 1), float((res >> 1) & 1), float(res & 1)]
                X_list.append(v)
                Y_list.append(bits)
            return cls.synthesize_from_truth_table("adder2", torch.tensor(X_list), torch.tensor(Y_list))

        elif benchmark_name == "maj5":
            # 5-bit Majority
            all_verts = list(itertools.product([0.0, 1.0], repeat=5))
            X = torch.tensor(all_verts, dtype=torch.float32)
            Y = (X.sum(dim=-1, keepdim=True) >= 3.0).float()
            return cls.synthesize_from_truth_table("majority5", X, Y)

        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
