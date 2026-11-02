"""
concentric_circuit.py: Standalone, zero-dependency Curvilinear Circuit Module.

Supports:
- Arbitrary N-size weight arrays.
- Configurable decimal precision / quantization / rounding.
- High-level logic evaluation and raw analog energy output.
- Automatic feature basis expansion (bias, parity waves, concentric shells, multilinear terms).
- Direct compilation from Boolean expressions into N-number arrays.
"""

from __future__ import annotations
import math
import itertools
from typing import List, Sequence, Union, Optional, Tuple


class CurvilinearCircuit:
    """
    A standalone Curvilinear Circuit runner and quantizer.
    Takes an N-element number array (weights) and runs logic/math evaluations
    with configurable decimal places and precision modes.
    """

    def __init__(
        self,
        num_inputs: int,
        weights: Sequence[float],
        decimals: Optional[int] = None,
        threshold: float = 0.5,
        include_shells: bool = True,
        include_parity: bool = True,
        max_cross_order: int = 4,
    ):
        """
        Args:
            num_inputs: Number of boolean/analog input lines (A, B, C, ...).
            weights: The array of numbers (synaptic/basis weights).
            decimals: Number of decimal places to retain (e.g. 2 for 0.34, 0 for ints, None for full precision).
            threshold: Decision boundary threshold (default 0.5).
            include_shells: Whether concentric radius shells are included in basis.
            include_parity: Whether harmonic cos/sin parity wave is included.
            max_cross_order: Max order of cross-receptive interaction terms (2=pairwise, 3=triplets, etc.).
        """
        self.num_inputs = num_inputs
        self.raw_weights = [float(w) for w in weights]
        self.decimals = decimals
        self.threshold = threshold
        self.include_shells = include_shells
        self.include_parity = include_parity
        self.max_cross_order = max_cross_order

        # Shell centers at discrete Hamming sphere radii: sqrt(0), sqrt(1), ..., sqrt(N)
        self.shell_centers = [math.sqrt(k) for k in range(num_inputs + 1)] if include_shells else []
        self.shell_sigma = 0.35

        # Apply precision rounding if specified
        self.weights = self._apply_precision(self.raw_weights, self.decimals)

    @staticmethod
    def _apply_precision(weights: Sequence[float], decimals: Optional[int]) -> List[float]:
        if decimals is None:
            return list(weights)
        return [round(w, decimals) for w in weights]

    def set_decimals(self, decimals: Optional[int]) -> None:
        """Dynamically update decimal precision."""
        self.decimals = decimals
        self.weights = self._apply_precision(self.raw_weights, decimals)

    def set_weights(self, weights: Sequence[float]) -> None:
        """Load a new array of N numbers."""
        self.raw_weights = [float(w) for w in weights]
        self.weights = self._apply_precision(self.raw_weights, self.decimals)

    def compute_features(self, inputs: Sequence[float]) -> List[float]:
        """
        Computes the geometric basis features Phi(x).
        Order:
          [0]: Bias (1.0)
          [1-2]: cos(pi*sum(x)), sin(pi*sum(x)) (if parity enabled)
          [shells]: exp(- (r - r_k)^2 / (2 * sigma^2)) (if shells enabled)
          [1st order]: x_i
          [2nd order]: x_i * x_j
          ... up to max_cross_order
        """
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        feat: List[float] = [1.0]  # Bias

        # 1. Parity Phasor
        sum_x = sum(inputs)
        if self.include_parity:
            phi = (math.pi * sum_x) % (2.0 * math.pi)
            feat.append(math.cos(phi))
            feat.append(math.sin(phi))

        # 2. Concentric Radial Shells
        if self.include_shells:
            r = math.sqrt(sum(x * x for x in inputs) + 1e-12)
            inv_2s2 = 1.0 / (2.0 * self.shell_sigma * self.shell_sigma)
            for r0 in self.shell_centers:
                dr = r - r0
                feat.append(math.exp(-(dr * dr) * inv_2s2))

        # 3. Multilinear Cross Terms
        # Order 1: single inputs
        feat.extend(inputs)

        # Orders 2 to max_cross_order
        for order in range(2, min(self.max_cross_order, self.num_inputs) + 1):
            for combo in itertools.combinations(range(self.num_inputs), order):
                prod = 1.0
                for idx in combo:
                    prod *= inputs[idx]
                feat.append(prod)

        return feat

    def forward_raw(self, inputs: Sequence[float]) -> float:
        """
        Computes the raw continuous analog energy output: z = sum(w_i * phi_i).
        """
        feats = self.compute_features(inputs)
        n = min(len(feats), len(self.weights))
        return sum(feats[i] * self.weights[i] for i in range(n))

    def __call__(self, *inputs: Union[float, int, Sequence[float]]) -> int:
        """
        Evaluate logic circuit. Returns 1 or 0.
        Accepts either separate args: circuit(1, 0, 1, 0)
        or a list: circuit([1, 0, 1, 0]).
        """
        if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
            in_vals = [float(x) for x in inputs[0]]
        else:
            in_vals = [float(x) for x in inputs]

        z = self.forward_raw(in_vals)
        return 1 if z >= self.threshold else 0

    def evaluate_truth_table(self, ground_truth: Sequence[int]) -> Tuple[float, int, int]:
        """
        Evaluates accuracy against all 2^N binary input combinations.
        Returns: (accuracy_pct, correct_states, total_states)
        """
        total = 1 << self.num_inputs
        if len(ground_truth) != total:
            raise ValueError(f"Truth table must have {total} entries for {self.num_inputs} inputs")

        correct = 0
        for idx in range(total):
            in_bits = [(idx >> (self.num_inputs - 1 - i)) & 1 for i in range(self.num_inputs)]
            pred = self(in_bits)
            if pred == ground_truth[idx]:
                correct += 1

        acc = (correct / total) * 100.0
        return acc, correct, total

    def __repr__(self) -> str:
        dec_str = f"{self.decimals} decimals" if self.decimals is not None else "full precision"
        return f"CurvilinearCircuit(inputs={self.num_inputs}, num_weights={len(self.weights)}, precision={dec_str})"
