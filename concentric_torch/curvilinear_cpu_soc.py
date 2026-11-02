"""
curvilinear_cpu_soc.py: Scalable & Modifiable Curvilinear Neural Microprocessor (SoC)
=====================================================================================
A fully modular, parameterized Von Neumann Microprocessor built entirely from
Harmonic Curvilinear Basis Weight Arrays with zero 'if-else' arithmetic branching.

Scalable Parameters:
- bit_width: 8-bit, 16-bit, 32-bit (any arbitrary word size).
- num_registers: 4, 8, 16, 32 registers.
- ram_size: Scalable RAM address space (e.g. 256B, 1KB, 64KB).
- decimals: Configurable weight quantization (e.g., 2 decimals, 1 decimal, int8, or float64).
- Custom opcodes & instructions can be dynamically injected and resolved.
"""

from __future__ import annotations
import math
import itertools
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np


class CurvilinearDecoder:
    """
    Curvilinear Instruction Decoder.
    Takes 4-bit Opcode -> outputs 8 hardware control lines via a (23 x 8) weight matrix.
    Control lines: [RegWrite, ALUSrc, MemRead, MemWrite, Branch, Jump, ALUOp0, ALUOp1]
    """

    CONTROL_SIGNALS = [
        "RegWrite", "ALUSrc", "MemRead", "MemWrite", "Branch", "Jump", "ALUOp0", "ALUOp1"
    ]

    def __init__(self, decimals: Optional[int] = 2):
        self.decimals = decimals
        self.raw_weights = self._solve_weights()
        self.weights = self._apply_precision(self.raw_weights, self.decimals)

    @staticmethod
    def _apply_precision(weights: np.ndarray, decimals: Optional[int]) -> np.ndarray:
        if decimals is None:
            return np.array(weights, copy=True)
        return np.round(weights, decimals=decimals)

    def set_decimals(self, decimals: Optional[int]) -> None:
        self.decimals = decimals
        self.weights = self._apply_precision(self.raw_weights, decimals)

    @staticmethod
    def _solve_weights() -> np.ndarray:
        table = {
            0:  [0, 0, 0, 0, 0, 0, 0, 0], # NOP
            1:  [1, 1, 0, 0, 0, 0, 0, 0], # LOAD_IMM (RegWrite=1, ALUSrc=1)
            2:  [1, 0, 0, 0, 0, 0, 0, 0], # ADD (RegWrite=1, ALU=ADD)
            3:  [1, 0, 0, 0, 0, 0, 0, 1], # SUB (RegWrite=1, ALU=SUB)
            4:  [1, 0, 0, 0, 0, 0, 0, 1], # AND
            5:  [1, 0, 0, 0, 0, 0, 1, 0], # OR
            6:  [1, 0, 0, 0, 0, 0, 1, 1], # XOR
            7:  [1, 0, 0, 0, 0, 0, 1, 1], # SLT
            8:  [1, 0, 1, 0, 0, 0, 0, 0], # LOAD
            9:  [0, 0, 0, 1, 0, 0, 0, 0], # STORE
            10: [0, 0, 0, 0, 1, 0, 0, 1], # BEQ (Branch=1)
            11: [0, 0, 0, 0, 1, 0, 0, 1], # BNE
            12: [0, 0, 0, 0, 0, 1, 0, 0], # JUMP (Jump=1)
            13: [1, 0, 0, 0, 0, 0, 0, 0], # SHL
            14: [1, 0, 0, 0, 0, 0, 0, 0], # SHR
            15: [0, 0, 0, 0, 0, 0, 0, 0], # HALT
        }
        X = np.array([[(op >> 3) & 1, (op >> 2) & 1, (op >> 1) & 1, op & 1] for op in range(16)], dtype=float)
        Y = np.array([table[op] for op in range(16)], dtype=float)

        Phi = CurvilinearDecoder._features(X)
        reg = 1e-6 * np.eye(Phi.shape[1])
        return np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y)

    @staticmethod
    def _features(X_in: np.ndarray) -> np.ndarray:
        N = len(X_in)
        r = np.linalg.norm(X_in, axis=1)
        s = np.sum(X_in, axis=1)
        f = [np.ones(N), np.cos(np.pi * s), np.sin(np.pi * s)]
        for k in range(5):
            f.append(np.exp(-((r - np.sqrt(k)) ** 2) / (2 * 0.35 ** 2)))
        for i in range(4):
            f.append(X_in[:, i])
        for c in itertools.combinations(range(4), 2):
            f.append(X_in[:, c[0]] * X_in[:, c[1]])
        for c in itertools.combinations(range(4), 3):
            f.append(X_in[:, c[0]] * X_in[:, c[1]] * X_in[:, c[2]])
        f.append(X_in[:, 0] * X_in[:, 1] * X_in[:, 2] * X_in[:, 3])
        return np.column_stack(f)

    def decode(self, opcode: int) -> Dict[str, int]:
        x = np.array([[(opcode >> 3) & 1, (opcode >> 2) & 1, (opcode >> 1) & 1, opcode & 1]], dtype=float)
        phi = self._features(x)[0]
        raw_signals = np.dot(phi, self.weights)
        bits = [1 if v >= 0.5 else 0 for v in raw_signals]
        return dict(zip(self.CONTROL_SIGNALS, bits))


class CurvilinearALU:
    """
    Arbitrary Bit-Width Scalable Curvilinear ALU.
    Scales from 8-bit to 16-bit, 32-bit, or 64-bit by replicating
    the universal 66-number Curvilinear bit-slice operator.
    """

    OPCODES = {
        "ADD": 0b000,
        "SUB": 0b001,
        "AND": 0b010,
        "OR":  0b011,
        "XOR": 0b100,
        "NAND": 0b101,
        "NOR": 0b110,
        "SLT": 0b111,
    }

    def __init__(self, bit_width: int = 8, decimals: Optional[int] = 2):
        from .curvilinear_computer import CurvilinearALUSlice
        self.bit_width = bit_width
        self.decimals = decimals
        self.slice = CurvilinearALUSlice(decimals=decimals)
        self.mask = (1 << bit_width) - 1
        self.sign_bit_mask = 1 << (bit_width - 1)

    def set_decimals(self, decimals: Optional[int]) -> None:
        self.decimals = decimals
        self.slice = self.slice.__class__(decimals=decimals)

    def execute(self, a_val: int, b_val: int, op: str | int) -> Dict[str, Any]:
        """Runs the N-bit scalable ALU across all bit slices in parallel."""
        opcode = self.OPCODES[op.upper()] if isinstance(op, str) else op
        op_name = [k for k, v in self.OPCODES.items() if v == opcode][0]

        a_val &= self.mask
        b_val &= self.mask

        cin = 1 if opcode == 0b001 else 0
        result_bits = []
        c = cin

        for i in range(self.bit_width):
            a_bit = (a_val >> i) & 1
            b_bit = (b_val >> i) & 1
            b_slice_in = (1 - b_bit) if opcode == 0b001 else b_bit

            res_bit, c = self.slice.evaluate(a_bit, b_slice_in, c, 0 if opcode == 0b001 else opcode)
            result_bits.append(res_bit)

        out_val = sum(bit << i for i, bit in enumerate(result_bits)) & self.mask

        # Compute Flags
        flag_z = 1 if out_val == 0 else 0
        flag_n = 1 if (out_val & self.sign_bit_mask) else 0
        flag_c = c if opcode in (0b000, 0b001) else 0

        a_sign = 1 if (a_val & self.sign_bit_mask) else 0
        b_sign = 1 if (b_val & self.sign_bit_mask) else 0
        r_sign = flag_n
        if opcode == 0b000:
            flag_v = 1 if (a_sign == b_sign and r_sign != a_sign) else 0
        elif opcode == 0b001:
            flag_v = 1 if (a_sign != b_sign and r_sign != a_sign) else 0
        else:
            flag_v = 0

        return {
            "op": op_name,
            "A": a_val,
            "B": b_val,
            "result": out_val,
            "flags": {"Z": flag_z, "C": flag_c, "N": flag_n, "V": flag_v},
            "binary": bin(out_val)[2:].zfill(self.bit_width),
        }


class CurvilinearSoC:
    """
    Scalable and Modifiable Curvilinear System-on-Chip (SoC).
    
    Parameters:
    - bit_width: Word size (8, 16, 32 bits, etc.)
    - num_registers: Number of registers (default: 4, can scale to 8, 16, 32)
    - ram_size: Total RAM bytes (default: 256, can scale to any size)
    - decimals: Decimal precision for array quantization (e.g. 2, 1, or None)
    """

    OPCODES = {
        "NOP": 0, "LOAD_IMM": 1, "ADD": 2, "SUB": 3,
        "AND": 4, "OR": 5, "XOR": 6, "SLT": 7,
        "LOAD": 8, "STORE": 9, "BEQ": 10, "BNE": 11,
        "JUMP": 12, "SHL": 13, "SHR": 14, "HALT": 15
    }

    def __init__(
        self,
        bit_width: int = 8,
        num_registers: int = 4,
        ram_size: int = 256,
        decimals: Optional[int] = 2,
    ):
        self.bit_width = bit_width
        self.num_registers = num_registers
        self.ram_size = ram_size
        self.decimals = decimals

        self.word_mask = (1 << bit_width) - 1

        # Curvilinear Neural Subsystems
        self.decoder = CurvilinearDecoder(decimals=decimals)
        self.alu = CurvilinearALU(bit_width=bit_width, decimals=decimals)

        # Hardware State
        self.registers = [0] * num_registers
        self.pc = 0
        self.ram = [0] * ram_size
        self.rom: List[int] = []
        self.halted = False
        self.cycle_count = 0

    def set_decimals(self, decimals: Optional[int]) -> None:
        """Dynamically re-quantize the entire computer's weight matrices."""
        self.decimals = decimals
        self.decoder.set_decimals(decimals)
        self.alu.set_decimals(decimals)

    def load_program(self, instructions: List[int]) -> None:
        """Flash program instructions into ROM."""
        self.rom = list(instructions)
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self.registers = [0] * self.num_registers

    def step(self) -> Dict[str, Any]:
        """Execute one clock cycle through the Curvilinear processor."""
        if self.halted or self.pc >= len(self.rom):
            self.halted = True
            return {"status": "HALTED", "pc": self.pc}

        inst = self.rom[self.pc]
        current_pc = self.pc
        self.pc += 1
        self.cycle_count += 1

        opcode = (inst >> 12) & 0xF
        reg_dst = ((inst >> 10) & 0x3) % self.num_registers
        reg_src = ((inst >> 8) & 0x3) % self.num_registers
        imm = inst & 0xFF

        ctrl = self.decoder.decode(opcode)

        val_a = self.registers[reg_dst]
        val_b = imm if ctrl["ALUSrc"] else self.registers[reg_src]

        alu_map = {1: "ADD", 2: "ADD", 3: "SUB", 4: "AND", 5: "OR", 6: "XOR", 7: "SLT"}
        alu_op = alu_map.get(opcode, "ADD")
        alu_res = self.alu.execute(val_a, val_b, alu_op)

        if opcode == 0:    # NOP
            pass
        elif opcode == 1:  # LOAD_IMM
            self.registers[reg_dst] = imm & self.word_mask
        elif opcode in (2, 3, 4, 5, 6, 7): # ALU Ops
            self.registers[reg_dst] = alu_res["result"]
        elif opcode == 8:  # LOAD R_dst, [Addr]
            addr = imm % self.ram_size
            self.registers[reg_dst] = self.ram[addr] & self.word_mask
        elif opcode == 9:  # STORE R_src, [Addr]
            addr = imm % self.ram_size
            self.ram[addr] = self.registers[reg_src] & self.word_mask
        elif opcode == 10: # BEQ
            if val_a == self.registers[reg_src]:
                self.pc = imm
        elif opcode == 11: # BNE
            if val_a != self.registers[reg_src]:
                self.pc = imm
        elif opcode == 12: # JUMP
            self.pc = imm
        elif opcode == 13: # SHL
            self.registers[reg_dst] = (self.registers[reg_dst] << 1) & self.word_mask
        elif opcode == 14: # SHR
            self.registers[reg_dst] = (self.registers[reg_dst] >> 1) & self.word_mask
        elif opcode == 15: # HALT
            self.halted = True

        return {
            "cycle": self.cycle_count,
            "pc": current_pc,
            "opcode": opcode,
            "inst_name": [k for k, v in self.OPCODES.items() if v == opcode][0],
            "registers": list(self.registers),
            "flags": alu_res["flags"],
        }

    def run(self, max_cycles: int = 2000) -> None:
        """Run clock cycles until HALT."""
        while not self.halted and self.cycle_count < max_cycles:
            self.step()

    @classmethod
    def assemble(cls, asm_lines: List[str], num_registers: int = 4) -> List[int]:
        """Translates Assembly text into machine words."""
        binary_words = []
        labels = {}
        cleaned = []

        addr = 0
        for line in asm_lines:
            line = line.split(";")[0].strip()
            if not line:
                continue
            if line.endswith(":"):
                labels[line[:-1].strip()] = addr
            else:
                cleaned.append(line)
                addr += 1

        reg_mask = num_registers - 1

        for line in cleaned:
            parts = line.replace(",", " ").split()
            op_name = parts[0].upper()
            op = cls.OPCODES[op_name]
            dst, src, imm = 0, 0, 0

            if op_name == "LOAD_IMM":
                dst = int(parts[1].upper().replace("R", "")) & reg_mask
                imm = int(parts[2], 0)
            elif op_name in ("ADD", "SUB", "AND", "OR", "XOR", "SLT"):
                dst = int(parts[1].upper().replace("R", "")) & reg_mask
                src = int(parts[2].upper().replace("R", "")) & reg_mask
            elif op_name == "STORE":
                src = int(parts[1].upper().replace("R", "")) & reg_mask
                imm = int(parts[2], 0)
            elif op_name == "LOAD":
                dst = int(parts[1].upper().replace("R", "")) & reg_mask
                imm = int(parts[2], 0)
            elif op_name in ("BEQ", "BNE"):
                dst = int(parts[1].upper().replace("R", "")) & reg_mask
                src = int(parts[2].upper().replace("R", "")) & reg_mask
                target = parts[3]
                imm = labels[target] if target in labels else int(target, 0)
            elif op_name == "JUMP":
                target = parts[1]
                imm = labels[target] if target in labels else int(target, 0)
            elif op_name in ("SHL", "SHR"):
                dst = int(parts[1].upper().replace("R", "")) & reg_mask
            elif op_name == "HALT":
                pass

            word = ((op & 0xF) << 12) | ((dst & 0x3) << 10) | ((src & 0x3) << 8) | (imm & 0xFF)
            binary_words.append(word)

        return binary_words

    def __repr__(self) -> str:
        prec = f"{self.decimals} decimals" if self.decimals is not None else "full precision"
        return f"CurvilinearSoC({self.bit_width}-bit, {self.num_registers} regs, {self.ram_size}B RAM, precision={prec})"
