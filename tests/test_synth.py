"""
test_synth.py: Verification Test Suite for Circuit Synthesizer
==============================================================
Validates exact circuit compilation, microsecond timing, and generated code execution.
"""

import math
import itertools
import pytest
import torch

from concentric_torch.synth.core import CircuitSynthesizer


def test_synthesize_xor4_and_xor8():
    """Verify XOR-4 and XOR-8 synthesis achieves 100% exact accuracy."""
    c4 = CircuitSynthesizer.synthesize_benchmark("xor4")
    assert c4.accuracy == 100.0
    assert c4.synthesis_time_ms < 50.0

    c8 = CircuitSynthesizer.synthesize_benchmark("xor8")
    assert c8.accuracy == 100.0
    assert c8.synthesis_time_ms < 100.0


def test_synthesize_2bit_adder():
    """Verify 2-bit Ripple Carry Adder synthesis achieves 100% accuracy across all 3 bits."""
    c_adder = CircuitSynthesizer.synthesize_benchmark("adder2")
    assert c_adder.accuracy == 100.0
    assert c_adder.num_inputs == 4
    assert c_adder.num_outputs == 3


def test_synthesize_mux4():
    """Verify 4-channel addressable multiplexer synthesis."""
    c_mux = CircuitSynthesizer.synthesize_benchmark("mux4")
    assert c_mux.accuracy == 100.0
    assert c_mux.num_inputs == 6
    assert c_mux.num_outputs == 1


def test_synthesize_custom_expression():
    """Verify parsing and synthesis from Boolean formula string."""
    expr = "(A ^ B) & (C | D)"
    c_expr = CircuitSynthesizer.synthesize_from_expression("test_expr", expr, var_names=["A", "B", "C", "D"])
    assert c_expr.accuracy == 100.0
    assert c_expr.num_inputs == 4

    # Direct ground-truth test across all 16 inputs
    all_verts = list(itertools.product([0, 1], repeat=4))
    for v in all_verts:
        A, B, C, D = v
        expected = int((A ^ B) and (C or D))
        inp_tensor = torch.tensor([[float(x) for x in v]])
        pred = int(c_expr.evaluate(inp_tensor).item())
        assert pred == expected, f"Mismatch at {v}: got {pred}, expected {expected}"


def test_generated_python_code_execution():
    """Verify that circuit.to_python() generates valid executable code with 100% fidelity."""
    c = CircuitSynthesizer.synthesize_benchmark("xor4")
    py_code = c.to_python()

    # Execute generated code in a local scope
    local_scope = {}
    exec(py_code, local_scope, local_scope)
    compute_fn = local_scope.get("compute_xor_4")
    assert compute_fn is not None, "Generated function compute_xor_4 not found"

    # Test all 16 inputs
    for v in itertools.product([0, 1], repeat=4):
        expected = sum(v) % 2
        actual = compute_fn(list(v))
        assert actual == expected, f"Generated code gave wrong answer for {v}: {actual} vs {expected}"
