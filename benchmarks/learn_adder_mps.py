"""Learn a 32-bit adder bit from random samples with a matrix chain (MPS, bond r) fitted by ALS vs an MLP."""
import sys, time
import numpy as np, torch, torch.nn as nn

W = 32; n = 2 * W; MASK = (1 << W) - 1
def to_x(a, b):
    X = np.zeros((len(a), n), np.int64)
    for i in range(W):                      # interleaved, LSB first: a0 b0 a1 b1 ...
        X[:, 2 * i] = (a >> i) & 1; X[:, 2 * i + 1] = (b >> i) & 1
    return X
def rand_pairs(rng, m):
    return rng.integers(0, 1 << W, m, dtype=np.int64), rng.integers(0, 1 << W, m, dtype=np.int64)
def long_carry_pairs(rng, m):
    a = rng.integers(0, 1 << W, m, dtype=np.int64)
    k = rng.integers(0, W, m)
    b = ((~a) & MASK) ^ (np.int64(1) << k)  # a+b is all ones except bit k flipped -> long carry/borrow-like chains
    b = np.where(rng.random(m) < 0.5, b, (~a) & MASK)
    return a, b & MASK

class MPS:
    def __init__(s, r, seed):
        rng = np.random.default_rng(seed); s.r = r
        s.G = np.zeros((n, 2, r, r))
        for k in range(n):
            for v in range(2):
                q, _ = np.linalg.qr(rng.normal(size=(r, r))); s.G[k, v] = q   # norm-preserving init
        s.l0 = np.zeros(r); s.l0[0] = 1; s.rN = np.zeros(r); s.rN[0] = 1
    def predict(s, X):
        L = np.broadcast_to(s.l0, (len(X), s.r))
        for k in range(n): L = np.einsum("sa,sab->sb", L, s.G[k][X[:, k]])
        return L @ s.rN
    def fit(s, X, y, sweeps=12, lam=1e-4, log=None):
        M, r = len(X), s.r
        for sw in range(sweeps):
            for direction in (1, -1):
                if direction == 1:
                    R = np.zeros((n + 1, M, r)); R[n] = s.rN
                    for k in range(n - 1, -1, -1): R[k] = np.einsum("sab,sb->sa", s.G[k][X[:, k]], R[k + 1])
                    L = np.broadcast_to(s.l0, (M, r)); order = range(n)
                else:
                    L_all = np.zeros((n + 1, M, r)); L_all[0] = s.l0
                    for k in range(n): L_all[k + 1] = np.einsum("sa,sab->sb", L_all[k], s.G[k][X[:, k]])
                    R_cur = np.broadcast_to(s.rN, (M, r)); order = range(n - 1, -1, -1)
                for k in order:
                    Lk = L if direction == 1 else L_all[k]
                    Rk = R[k + 1] if direction == 1 else R_cur
                    for v in (0, 1):
                        idx = X[:, k] == v
                        if idx.sum() == 0: continue
                        F = (Lk[idx][:, :, None] * Rk[idx][:, None, :]).reshape(-1, r * r)
                        sol = np.linalg.solve(F.T @ F + lam * np.eye(r * r), F.T @ y[idx])
                        s.G[k, v] = sol.reshape(r, r)
                    if direction == 1: L = np.einsum("sa,sab->sb", L, s.G[k][X[:, k]])
                    else: R_cur = np.einsum("sab,sb->sa", s.G[k][X[:, k]], R_cur)
            if log: log(sw)

def mlp_fit_eval(Xtr, ytr, tests, seed, epochs=1500):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)[:, None]
    for _ in range(epochs):
        opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(Xt), yt).backward(); opt.step()
    with torch.no_grad():
        return [float(np.mean((m(torch.tensor(X, dtype=torch.float32)) > 0).numpy().ravel() == y)) for X, y in tests]

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    r = 4
    print(f"32-bit adder, 64 inputs, interleaved order. Matrix chain bond r={r} -> {n*2*r*r} numbers per output bit\n")
    for ob in (0, 8, 16, 31, 32):
        for M in (300, 2000):
            a, b = rand_pairs(rng, M); Xtr = to_x(a, b); ytr = (((a + b) >> ob) & 1).astype(float)
            a2, b2 = rand_pairs(rng, 4000); Xte = to_x(a2, b2); yte = ((((a2 + b2) >> ob) & 1)).astype(bool)
            a3, b3 = long_carry_pairs(rng, 4000); Xlc = to_x(a3, b3); ylc = ((((a3 + b3) >> ob) & 1)).astype(bool)
            t0 = time.time(); best = None
            for seed in range(3):
                mp = MPS(r, seed); mp.fit(Xtr, ytr, sweeps=10)
                tr_acc = np.mean((mp.predict(Xtr) >= 0.5) == ytr.astype(bool))
                if best is None or tr_acc > best[0]: best = (tr_acc, mp)
            mp = best[1]; dt = time.time() - t0
            ev = lambda X, y: float(np.mean((mp.predict(X) >= 0.5) == y))
            ml = mlp_fit_eval(Xtr, ytr, [(Xte, yte), (Xlc, ylc)], 0)
            print(f"out bit {ob:2d} | {M:4d} train samples | chain: train {best[0]:.3f} random-test {ev(Xte, yte):.3f} long-carry-test {ev(Xlc, ylc):.3f} ({dt:.1f}s) | MLP: random-test {ml[0]:.3f} long-carry-test {ml[1]:.3f}")
            sys.stdout.flush()
