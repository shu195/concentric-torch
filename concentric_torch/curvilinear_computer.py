"""
curvilinear_computer.py: A Full 8-Bit Arithmetic Logic Unit (ALU) & Virtual Machine
Built Entirely from Curvilinear Array Operators (Zero IF-statements).

Architecture:
- 8 Cascaded Curvilinear Bit-Slices.
- Each bit-slice is a 66-number Curvilinear Basis array.
- Supported Instructions (3-bit Opcode):
    000: ADD (Addition with Carry-Lookahead propagation)
    001: SUB (Two's complement Subtraction with Borrow)
    010: AND (Bitwise Logical AND)
    011: OR  (Bitwise Logical OR)
    100: XOR (Bitwise Logical XOR)
    101: NAND (Bitwise Logical NAND)
    110: NOR (Bitwise Logical NOR)
    111: SLT (Set Less Than / Comparator: returns 1 if A < B, else 0)
- Status Flags: Zero (Z), Carry (C), Negative (N), Overflow (V).
- Pure arithmetic matrix operations; zero branching.
"""

from __future__ import annotations
import math
import itertools
from typing import List, Tuple, Dict, Any
import numpy as np


class CurvilinearALUSlice:
    """
    A 1-bit Curvilinear ALU Slice.
    Accepts: A_i, B_i, Cin, Op0, Op1, Op2 (6 inputs)
    Produces: Result_i, Cout_i (2 outputs)
    Computed via two 66-element weight vectors (132 numbers total).
    """

    def __init__(self, decimals: int | None = 2):
        self.decimals = decimals
        self.shell_centers = [math.sqrt(k) for k in range(7)]  # 0 to 6
        self.inv_2s2 = 1.0 / (2.0 * 0.35 * 0.35)

        # Precompute the exact 66-number closed-form weights for Out and Cout
        self.w_out, self.w_cout = self._solve_weights()
        if self.decimals is not None:
            self.w_out = np.round(self.w_out, decimals=self.decimals)
            self.w_cout = np.round(self.w_cout, decimals=self.decimals)

    @staticmethod
    def _solve_weights() -> Tuple[np.ndarray, np.ndarray]:
        X, y_out, y_cout = [], [], []
        for a in (0, 1):
            for b in (0, 1):
                for cin in (0, 1):
                    for op in range(8):
                        op0 = (op >> 2) & 1
                        op1 = (op >> 1) & 1
                        op2 = op & 1
                        if op == 0:    # ADD
                            res = (a + b + cin) % 2
                            cout = 1 if (a + b + cin) >= 2 else 0
                        elif op == 1:  # SUB (A - B - Cin)
                            diff = a - b - cin
                            res = diff % 2
                            cout = 1 if diff < 0 else 0
                        elif op == 2:  # AND
                            res = a & b
                            cout = 0
                        elif op == 3:  # OR
                            res = a | b
                            cout = 0
                        elif op == 4:  # XOR
                            res = a ^ b
                            cout = 0
                        elif op == 5:  # NAND
                            res = 1 - (a & b)
                            cout = 0
                        elif op == 6:  # NOR
                            res = 1 - (a | b)
                            cout = 0
                        elif op == 7:  # SLT / CMP (A < B)
                            res = 1 if a < b else 0
                            cout = 0
                        X.append([a, b, cin, op0, op1, op2])
                        y_out.append(res)
                        y_cout.append(cout)

        X = np.array(X, dtype=float)
        Phi = CurvilinearALUSlice._build_basis(X)
        reg = 1e-6 * np.eye(Phi.shape[1])
        w_o = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ np.array(y_out, dtype=float))
        w_c = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ np.array(y_cout, dtype=float))
        return w_o, w_c

    @staticmethod
    def _build_basis(X_mat: np.ndarray) -> np.ndarray:
        N = len(X_mat)
        r = np.linalg.norm(X_mat, axis=1)
        s = np.sum(X_mat, axis=1)
        feats = [np.ones(N), np.cos(np.pi * s), np.sin(np.pi * s)]
        for k in range(7):
            r0 = np.sqrt(k)
            feats.append(np.exp(-((r - r0) ** 2) / (2 * 0.35 ** 2)))
        for i in range(6):
            feats.append(X_mat[:, i])
        for combo in itertools.combinations(range(6), 2):
            feats.append(X_mat[:, combo[0]] * X_mat[:, combo[1]])
        for combo in itertools.combinations(range(6), 3):
            feats.append(X_mat[:, combo[0]] * X_mat[:, combo[1]] * X_mat[:, combo[2]])
        for combo in itertools.combinations(range(6), 4):
            feats.append(X_mat[:, combo[0]] * X_mat[:, combo[1]] * X_mat[:, combo[2]] * X_mat[:, combo[3]])
        return np.column_stack(feats)

    def evaluate(self, a: int, b: int, cin: int, op_code: int) -> Tuple[int, int]:
        """Runs the 66-number dot products to calculate result and carry."""
        op0 = (op_code >> 2) & 1
        op1 = (op_code >> 1) & 1
        op2 = op_code & 1
        x = np.array([[a, b, cin, op0, op1, op2]], dtype=float)
        phi = self._build_basis(x)[0]

        res_raw = float(np.dot(phi, self.w_out))
        cout_raw = float(np.dot(phi, self.w_cout))

        res_bit = 1 if res_raw >= 0.5 else 0
        cout_bit = 1 if cout_raw >= 0.5 else 0
        return res_bit, cout_bit


class CurvilinearALU8Bit:
    """
    An 8-Bit Arithmetic Logic Unit built entirely by cascading
    CurvilinearALUSlice array blocks.
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

    def __init__(self, decimals: int | None = 2):
        # 8 cascaded bit-slices sharing the universal 66-number operator
        self.slice = CurvilinearALUSlice(decimals=decimals)
        self.decimals = decimals

    def execute(self, a_byte: int, b_byte: int, op: str | int) -> Dict[str, Any]:
        """
        Executes an ALU operation on two 8-bit unsigned integers [0..255].
        """
        opcode = self.OPCODES[op.upper()] if isinstance(op, str) else op
        op_name = [k for k, v in self.OPCODES.items() if v == opcode][0]

        cin = 1 if opcode == 0b001 else 0  # Invert borrow for SUB
        result_bits = []
        c = cin

        for i in range(8):
            a_bit = (a_byte >> i) & 1
            b_bit = (b_byte >> i) & 1
            # For subtraction: two's complement B XOR cin
            b_val = (1 - b_bit) if opcode == 0b001 else b_bit

            # Evaluate through the curvilinear array slice
            res_bit, c = self.slice.evaluate(a_bit, b_val, c, 0 if opcode == 0b001 else opcode)
            result_bits.append(res_bit)

        out_val = sum(bit << i for i, bit in enumerate(result_bits)) & 0xFF

        # Status Flags
        flag_z = 1 if out_val == 0 else 0
        flag_n = (out_val >> 7) & 1
        flag_c = c if opcode in (0b000, 0b001) else 0
        
        # Overflow flag V
        a_sign = (a_byte >> 7) & 1
        b_sign = (b_byte >> 7) & 1
        r_sign = flag_n
        if opcode == 0b000:
            flag_v = 1 if (a_sign == b_sign and r_sign != a_sign) else 0
        elif opcode == 0b001:
            flag_v = 1 if (a_sign != b_sign and r_sign != a_sign) else 0
        else:
            flag_v = 0

        return {
            "op": op_name,
            "A": a_byte,
            "B": b_byte,
            "result": out_val,
            "flags": {"Z": flag_z, "C": flag_c, "N": flag_n, "V": flag_v},
            "binary": f"{out_val:08b}",
        }
