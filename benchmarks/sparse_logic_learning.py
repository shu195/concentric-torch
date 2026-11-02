"""Can the number-array (Walsh/multilinear) basis learn hidden logic from few samples? vs MLP."""
import itertools, time
import numpy as np, torch, torch.nn as nn

n = 10; N = 1 << n
X = np.array(list(itertools.product([0, 1], repeat=n)), float)
H = np.array([[1.0]])
for _ in range(n):
    H = np.kron(H, np.array([[1.0, 1.0], [1.0, -1.0]]))   # rows = states (MSB first), cols = characters
H /= np.sqrt(N)                                            # orthonormal

def bit(i): return X[:, i].astype(int)
targets = {
 "parity of 3 bits":        lambda: bit(1) ^ bit(4) ^ bit(7),
 "majority of 5 bits":      lambda: (X[:, [0, 2, 3, 6, 9]].sum(1) >= 3).astype(int),
 "(a^b)&(c|d)":             lambda: (bit(0) ^ bit(1)) & (bit(2) | bit(3)),
 "3-bit mux (sel=2 bits)":  lambda: np.where(bit(0) * 2 + bit(1) == 0, bit(2), np.where(bit(0) * 2 + bit(1) == 1, bit(3), np.where(bit(0) * 2 + bit(1) == 2, bit(4), bit(5)))),
 "RANDOM (control)":        lambda: np.random.default_rng(5).integers(0, 2, N),
}

def lasso(A, y, lam, it=400):
    L = np.linalg.norm(A, 2) ** 2; w = np.zeros(A.shape[1])
    for _ in range(it):
        g = A.T @ (A @ w - y); w = w - g / L
        w = np.sign(w) * np.maximum(np.abs(w) - lam / L, 0)
    return w

def fit_sparse(tr, y):
    A = H[tr]; yy = 2.0 * y[tr] - 1
    k = len(tr) * 4 // 5; a, b = np.arange(k), np.arange(k, len(tr))
    best = None
    for lam in [0.01, 0.03, 0.1, 0.3]:
        w = lasso(A[a], yy[a], lam); acc = np.mean(np.sign(A[b] @ w) == yy[b])
        if best is None or acc > best[0]: best = (acc, lam)
    return lasso(A, yy, best[1]), best[1]

def fit_mlp(tr, te, y, seed):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(m.parameters(), 1e-2, weight_decay=1e-4)
    Xt = torch.tensor(X[tr], dtype=torch.float32); yt = torch.tensor(y[tr], dtype=torch.float32)[:, None]
    for _ in range(1500):
        opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(m(Xt), yt).backward(); opt.step()
    with torch.no_grad(): p = (m(torch.tensor(X[te], dtype=torch.float32)) > 0).numpy().ravel().astype(int)
    return np.mean(p == y[te])

frac = 0.10
print(f"n={n} inputs, {N} states, training on {int(frac*N)} samples ({frac:.0%}), tested on the other {N-int(frac*N)}\n")
print(f"{'function':26s} {'sparse-array':>13s} {'MLP':>7s}  {'nonzero weights':>16s}")
for name, f in targets.items():
    y = np.asarray(f()).astype(int); sa, ml, nz = [], [], []
    for seed in range(5):
        idx = np.random.default_rng(seed).permutation(N); tr, te = idx[:int(frac * N)], idx[int(frac * N):]
        w, lam = fit_sparse(tr, y)
        sa.append(np.mean(np.sign(H[te] @ w) == 2 * y[te] - 1)); nz.append(int((np.abs(w) > 0.02).sum()))
        ml.append(fit_mlp(tr, te, y, seed))
    print(f"{name:26s} {np.mean(sa):13.3f} {np.mean(ml):7.3f}  {np.mean(nz):16.1f}")
