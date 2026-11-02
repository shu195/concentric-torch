"""
concentric_torch: Curvilinear and Phasor Neural Architecture for PyTorch
========================================================================
Based on Concentric Shell and Angular Neural Architecture (CS-ANA) theory.
"""

from .layers import (
    ConcentricShell,
    DynamicConductionShell,
    ConcentricLinear,
    FreeMiniRingStabilizer,
)
from .relational import RingMatMul
from .attention import PhaseAttention, CPTBlock
from .circuit import CurvilinearCircuit
from .curvilinear_cpu_soc import CurvilinearSoC, CurvilinearALU, CurvilinearDecoder

__all__ = [
    "ConcentricShell",
    "DynamicConductionShell",
    "ConcentricLinear",
    "FreeMiniRingStabilizer",
    "RingMatMul",
    "PhaseAttention",
    "CPTBlock",
    "CurvilinearCircuit",
    "CurvilinearSoC",
    "CurvilinearALU",
    "CurvilinearDecoder",
]

__version__ = "0.1.0"
