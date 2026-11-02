"""Matrix-chain (tensor-train / MPS) representation of an 8-bit adder vs raw table and flat array."""
import itertools
import numpy as np

W = 8; n = 2 * W
def adder_outputs():
    # inputs enumerated as bits of index i (MSB first) in the variable order given later
    pass

def tt_svd(T, tol=1e-9):
    """Exact-ish TT decomposition of tensor T with shape (2,)*n. Returns cores."""
    cores, r_prev = [], 1
    C = T.reshape(1, -1)
    for k in range(n - 1):
        C = C.reshape(r_prev * 2, -1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        r = int((S > tol * S[0]).sum())
        cores.append(U[:, :r].reshape(r_prev, 2, r))
        C = S[:r, None] * Vt[:r]; r_prev = r
    cores.append(C.reshape(r_prev, 2, 1))
    return cores

def tt_eval_all(cores):
    out = cores[0][0]                    # (2, r)
    for c in cores[1:]:
        out = np.einsum("ar,rbs->abs", out.reshape(-1, c.shape[0]), c).reshape(-1, c.shape[2])
    return out.ravel()

def run(order_name, var_order):
    # var_order: list of (operand, bit) for each tensor axis, axis 0 = most significant index bit
    states = np.array(list(itertools.product([0, 1], repeat=n)))
    a = np.zeros(len(states), int); b = np.zeros(len(states), int)
    for ax, (op, bit) in enumerate(var_order):
        (a if op == "a" else b)[:] += states[:, ax] << bit
    s = a + b
    total_params, total_raw, ranks_all, ok = 0, 0, [], True
    for ob in range(W + 1):
        y = ((s >> ob) & 1).astype(float)
        cores = tt_svd(y.reshape((2,) * n))
        pred = tt_eval_all(cores) >= 0.5
        ok &= bool(np.all(pred == y.astype(bool)))
        total_params += sum(c.size for c in cores); total_raw += y.size
        ranks_all.append(max(c.shape[2] for c in cores))
    print(f"{order_name:28s} max bond rank per output bit: {ranks_all}")
    print(f"{'':28s} total numbers: {total_params:7d}  (raw tables: {total_raw}) | exact on all 65536 inputs: {ok}")
    return total_params

interleaved = [(op, k) for k in range(W - 1, -1, -1) for op in ("a", "b")]   # a7 b7 a6 b6 ... a0 b0 (MSB first)
interleaved_lsb = [(op, k) for k in range(W) for op in ("a", "b")]           # a0 b0 a1 b1 ... (LSB first, carry flows along chain)
separated = [("a", k) for k in range(W)] + [("b", k) for k in range(W)]      # a0..a7 b0..b7
print("8-bit adder, 16 inputs, 9 output bits, each output = 65,536-entry truth table\n")
p1 = run("interleaved LSB-first", interleaved_lsb)
p2 = run("interleaved MSB-first", interleaved)
p3 = run("separated (all a, then all b)", separated)
print("\nfor reference: 8 flat 66-weight bit-slices x 2 outputs =", 8 * 2 * 66, "numbers (but needs the 6-input slice design by hand)")

rng = np.random.default_rng(0)
yr = rng.integers(0, 2, 2 ** n).astype(float)
cr = tt_svd(yr.reshape((2,) * n))
print("RANDOM 16-input function: max bond rank", max(c.shape[2] for c in cr), "| numbers", sum(c.size for c in cr), "vs raw", yr.size)
