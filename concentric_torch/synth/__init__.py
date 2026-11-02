"""
concentric_torch.synth: Microsecond Symbolic Circuit & Logic Synthesizer
========================================================================
Compiles discrete Boolean functions and symbolic logic into minimal
closed-form curvilinear harmonic equations using CS-ANA principles.
"""

from .core import CircuitSynthesizer, SynthesizedCircuit

__all__ = ["CircuitSynthesizer", "SynthesizedCircuit"]
