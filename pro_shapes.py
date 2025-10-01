# pro_shapes.py
# Quantum vs Classical on puzzle-like shapes (XOR, circle, plus, checker)
# - NumPy 2.0 safe (no arr.ptp()).
# - Fast defaults + optional snapshots.
# - Batched, no-grad quantum plotting to avoid RAM spikes.

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pennylane as qml
from pennylane import numpy as pnp

# ============================
# Config (tweak freely)
# ============================
TARGET    = "xor"      # "xor" | "circle" | "plus" | "checker"
N_SAMPLES = 300
RES       = 80         # grid resolution (fast: 60–100)
STEPS     = 20         # training steps (fast: 15–40)
LAYERS    = 1          # 1 or 2; deeper = slower
BATCH     = 2048       # batch size for plotting predictions
LR        = 0.30       # learning rate
SNAPSHOTS = 1          # 0, 1, 2, or 4 boundary snapshots in FIGURE 3
ABLATE_ENTANGLER = False  # True removes CNOTs (shows why entanglement matters)
SEED = 0

# ============================
# Small utils
# ============================
def normalize_01(arr):
    arr = np.asarray(arr, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    rng = mx - mn
    if rng <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / rng

def estimate_calls(n_samples, steps, res, n_snapshots=0):
    train_lb = steps * n_samples
    plot_calls = res * res * (1 + n_snapshots)  # final + snapshots
    return train_lb, plot_calls, train_lb + plot_calls

# ============================
# Datasets (puzzle-like targets)
# ============================
rng = np.random.default_rng(SEED)

def make_xor(n=200, jitter=0.05):
    X0 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y0 = np.array([0,1,1,0], dtype=int)
    reps = max(1, n // 4)
    X = np.repeat(X0, reps, axis=0).astype(float)
    y = np.repeat(y0, reps)
    X += rng.uniform(-jitter, jitter, X.shape)
    X = X * 2 - 1   # scale to [-1,1]^2
    return X, y

def make_circle(n=400, r=0.6):
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:,0]**2 + X[:,1]**2) < r**2).astype(int)
    return X, y

def make_plus(n=400, bar=0.15):
    X = rng.uniform(-1,1,(n,2))
    y = ((np.abs(X[:,0]) < bar) | (np.abs(X[:,1]) < bar)).astype(int)
    return X, y

def make_checker(n=400, cells=4):
    X = rng.uniform(-1,1,(n,2))
    u = ((X[:,0]+1)/2 * cells).astype(int)
    v = ((X[:,1]+1)/2 * cells).astype(int)
    y = ((u+v) % 2).astype(int)
    return X, y

def make_data(kind, n=N_SAMPLES):
    if kind == "xor":     return make_xor(n)
    if kind == "circle":  return make_circle(n)
    if kind == "plus":    return make_plus(n)
    if kind == "checker": return make_checker(n)
    raise ValueError("Unknown TARGET")

X, y = make_data(TARGET, N_SAMPLES)

# ============================
# Classical baselines
# ============================
svm = SVC(kernel="rbf", gamma="scale").fit(X, y)               # strong baseline
lin = LogisticRegression(max_iter=200).fit(X, y)               # weak baseline

def svm_prob_like(grid):
    df = svm.decision_function(grid)
    return normalize_01(df)

def lin_prob_like(grid):
    return lin.predict_proba(grid)[:, 1]

# ============================
# Quantum model (PennyLane)
# ============================
n_qubits = 2
dev_train = qml.device("default.qubit", wires=n_qubits)
dev_plot  = qml.device("default.qubit", wires=n_qubits)

def encode(x):
    qml.AngleEmbedding(x, wires=[0,1], rotation="Y")
    if not ABLATE_ENTANGLER:
        qml.CNOT(wires=[0,1])
    qml.RZ(x[0]*x[1]*np.pi, wires=1)

def variational(weights):
    for layer in weights:
        for w, wire in zip(layer, range(n_qubits)):
            qml.RX(w[0], wires=wire)
            qml.RY(w[1], wires=wire)
            qml.RZ(w[2], wires=wire)
        if not ABLATE_ENTANGLER:
            qml.CNOT(wires=[0,1])

@qml.qnode(dev_train, interface="autograd")
def circuit_train(x, weights):
    encode(x)
    variational(weights)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev_plot, diff_method=None)
def circuit_nograd(x, weights):
    encode(x)
    variational(weights)
    return qml.expval(qml.PauliZ(0))

def q_proba_train(Xin, weights):
    outs = pnp.array([circuit_train(x, weights) for x in Xin])
    return (outs + 1)/2

def q_proba_batched(Xin, weights, batch=BATCH):
    out = np.empty(len(Xin), dtype=float)
    for i in range(0, len(Xin), batch):
        chunk = Xin[i:i+batch]
        vals = [(circuit_nograd(x, weights)+1)/2 for x in chunk]
        out[i:i+batch] = vals
    return out

# ============================
# Train VQC (fast & stable)
# ============================
pnp.random.seed(SEED)
weights = 0.1 * pnp.random.randn(LAYERS, n_qubits, 3)

Xb = pnp.array(X, requires_grad=False)
yb = pnp.array(y, requires_grad=False)

def loss_fn(w):
    p = q_proba_train(Xb, w)
    eps = 1e-8
    return -pnp.mean(yb*pnp.log(p+eps) + (1-yb)*pnp.log(1-p+eps))

opt = qml.GradientDescentOptimizer(stepsize=LR)
loss_hist = []

train_lb, plot_calls, total_lb = estimate_calls(N_SAMPLES, STEPS, RES, SNAPSHOTS)
print(f"[estimate] train≥{train_lb:,}  plots≈{plot_calls:,}  total≥{total_lb:,} circuit calls")
print("[note] gradients add constant-factor overhead during training; plots use no-grad QNode.")

t0 = time.time()
for t in range(STEPS):
    weights = opt.step(loss_fn, weights)
    if (t+1) % max(1, STEPS//10) == 0:
        L = float(loss_fn(weights))
        loss_hist.append(L)
train_time = time.time() - t0

# ============================
# Plot helpers
# ============================
def plot_boundary(ax, predict_fn, X, y, title, res=RES):
    xx, yy = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(float)
    Z = predict_fn(grid).reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=25, alpha=0.65)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=18, linewidths=0.3)
    ax.set_title(title)
    return im

def plot_quantum(ax, weights, title, res=RES):
    xx, yy = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(float)
    Z = q_proba_batched(grid, weights, batch=BATCH).reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=25, alpha=0.65)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=18, linewidths=0.3)
    ax.set_title(title)
    return im

# ============================
# FIGURE 1: Baselines vs Quantum
# ============================
fig, axs = plt.subplots(1, 3, figsize=(14,4.5))
plot_boundary(axs[0], svm_prob_like, X, y, "SVM (RBF) — strong baseline")
plot_boundary(axs[1], lin_prob_like, X, y, "Logistic (linear) — weak baseline")
plot_quantum(axs[2], weights, f"Quantum VQC — L={LAYERS}, steps={STEPS}")
plt.tight_layout()
plt.show()

# ============================
# FIGURE 2: Training curve
# ============================
if loss_hist:
    plt.figure(figsize=(5,3))
    xs = np.linspace(0, STEPS, len(loss_hist))
    plt.plot(xs, loss_hist, marker="o")
    plt.xlabel("Training steps"); plt.ylabel("Batch loss")
    plt.title(f"VQC training (time ~{train_time:.1f}s)")
    plt.tight_layout()
    plt.show()

# ============================
# FIGURE 3: Boundary evolution (optional snapshots)
# ============================
if SNAPSHOTS > 0:
    # crude synthetic snapshots via scaling weights (cheap, illustrative)
    scales_map = {
        1: [1.0],
        2: [0.5, 1.0],
        4: [0.25, 0.5, 0.75, 1.0]
    }
    scales = scales_map.get(SNAPSHOTS, [1.0])
    snap_weights = [weights * s for s in scales]

    fig, axs = plt.subplots(1, len(snap_weights), figsize=(4*len(snap_weights), 3.6))
    if len(snap_weights) == 1:
        axs = [axs]
    titles = [f"snapshot {i+1}" for i in range(len(snap_weights))]
    for ax, w, ttl in zip(axs, snap_weights, titles):
        plot_quantum(ax, w, f"VQC boundary: {ttl}")
    plt.tight_layout()
    plt.show()

# ============================
# Quick report
# ============================
acc_svm = (svm.predict(X) == y).mean()
acc_lin = (lin.predict(X) == y).mean()
q_preds = (q_proba_batched(X, weights) >= 0.5).astype(int)
acc_q   = (q_preds == y).mean()

print("\n=== Quick Report ===")
print(f"Target: {TARGET} | Samples: {N_SAMPLES} | Steps: {STEPS} | Layers: {LAYERS} | Entangler ablated: {ABLATE_ENTANGLER}")
print(f"SVM (RBF) acc:        {acc_svm:.3f}")
print(f"Logistic (linear) acc:{acc_lin:.3f}")
print(f"Quantum VQC acc:      {acc_q:.3f}")
print("Note: SVM is a strong classical baseline; logistic is the honest 'weak' baseline.")
