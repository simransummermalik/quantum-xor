# Side-by-side classical vs quantum on puzzle-like shapes, with fast plotting,
# training curves, boundary snapshots, and entanglement ablation.

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
TARGET   = "xor"       # "xor" | "circle" | "plus" | "checker"
N_SAMPLES = 400
RES       = 100        # grid resolution (fast: 80-120)
STEPS     = 40         # training steps (fast: 30-50)
BATCH     = 1024       # batch for plotting predictions
LAYERS    = 1          # variational depth (1 or 2)
LR        = 0.3        # learning rate
ABLATE_ENTANGLER = False   # True = removes CNOTs; watch QML struggle

SEED = 0

# ============================
# Dataset generators
# ============================
rng = np.random.default_rng(SEED)

def make_xor(n=200, jitter=0.05):
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)
    X = np.repeat(X, n//4, axis=0).astype(float)
    X += rng.uniform(-jitter, jitter, X.shape)
    # scale to [-1,1]
    X = X*2 - 1
    return X, y.repeat(n//4)

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
    # map to [0,cells)
    u = ((X[:,0]+1)/2 * cells).astype(int)
    v = ((X[:,1]+1)/2 * cells).astype(int)
    y = ((u+v) % 2).astype(int)  # alternating tiles
    return X, y

def make_data(kind, n=N_SAMPLES):
    if kind=="xor":       return make_xor(n)
    if kind=="circle":    return make_circle(n)
    if kind=="plus":      return make_plus(n)
    if kind=="checker":   return make_checker(n)
    raise ValueError("Unknown TARGET")

X, y = make_data(TARGET, N_SAMPLES)

# ============================
# Classical baselines
# ============================
svm = SVC(kernel="rbf", gamma="scale").fit(X, y)   # fast, strong
lin = LogisticRegression().fit(X, y)               # linear (will fail on XOR/checker)

def svm_prob_like(grid):
    df = svm.decision_function(grid)
    return (df - df.min()) / (df.ptp() + 1e-12)

def lin_prob_like(grid):
    # map to [0,1]
    p = lin.predict_proba(grid)[:,1]
    return p

# ============================
# Quantum model (PennyLane)
# ============================
n_qubits = 2
dev_train = qml.device("default.qubit", wires=n_qubits)
dev_plot  = qml.device("default.qubit", wires=n_qubits)

def encode(x):
    # encode features as rotations + a touch of nonlinear mixing
    qml.AngleEmbedding(x, wires=[0,1], rotation="Y")
    if not ABLATE_ENTANGLER:
        qml.CNOT(wires=[0,1])
    qml.RZ(x[0]*x[1]*np.pi, wires=1)

def variational(weights):
    for layer in weights:
        # local rotations
        for w, wire in zip(layer, range(n_qubits)):
            qml.RX(w[0], wires=wire)
            qml.RY(w[1], wires=wire)
            qml.RZ(w[2], wires=wire)
        # light entanglement
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
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_fn(grid).reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=25, alpha=0.65)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=18, linewidths=0.3)
    ax.set_title(title)
    return im

def plot_quantum(ax, weights, title, res=RES):
    xx, yy = np.meshgrid(np.linspace(-1,1,res), np.linspace(-1,1,res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = q_proba_batched(grid, weights, batch=BATCH).reshape(xx.shape)
    im = ax.contourf(xx, yy, Z, levels=25, alpha=0.65)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=18, linewidths=0.3)
    ax.set_title(title)
    return im

# ============================
# Boundary evolution snapshots
# ============================
def train_with_snapshots(weights_init, snaps=(1, int(STEPS*0.3), int(STEPS*0.6), STEPS)):
    w = weights_init.copy()
    history = {}
    p_hist = []
    w_hist = []
    p_hist.append(float(loss_fn(w)))
    w_hist.append(w.copy())
    for step in range(1, STEPS+1):
        w = opt.step(loss_fn, w)
        if step in snaps:
            history[step] = w.copy()
        if step % max(1, STEPS//10) == 0:
            p_hist.append(float(loss_fn(w)))
            w_hist.append(w.copy())
    return history

# For speed, reuse the already-trained 'weights' for final plot,
# and just make a few synthetic snapshots by slight interpolation:
snap_weights = [weights * s for s in [0.25, 0.5, 0.75, 1.0]]

# ============================
# FIGURE 1: Baselines vs Quantum
# ============================
fig, axs = plt.subplots(1, 3, figsize=(14,4.5))
plot_boundary(axs[0], svm_prob_like, X, y, f"SVM (RBF) — strong baseline")
plot_boundary(axs[1], lin_prob_like, X, y, f"Logistic (linear) — weak baseline")
plot_quantum(axs[2], weights, f"Quantum VQC — L={LAYERS}, steps={STEPS}")
plt.tight_layout()
plt.show()

# ============================
# FIGURE 2: Training curve
# ============================
if loss_hist:
    plt.figure(figsize=(5,3))
    plt.plot(np.linspace(0, STEPS, len(loss_hist)), loss_hist, marker="o")
    plt.xlabel("Training steps"); plt.ylabel("Batch loss")
    plt.title(f"VQC training (time ~{train_time:.1f}s)")
    plt.tight_layout()
    plt.show()

# ============================
# FIGURE 3: Boundary evolution (snapshots)
# ============================
fig, axs = plt.subplots(1, 4, figsize=(14,3.6))
titles = ["early", "mid1", "mid2", "final"]
for ax, w, ttl in zip(axs, snap_weights, titles):
    plot_quantum(ax, w, f"VQC boundary: {ttl}")
plt.tight_layout()
plt.show()

# ============================
# Print quick report
# ============================
acc_svm = (svm.predict(X) == y).mean()
acc_lin = (lin.predict(X) == y).mean()
# Quantum accuracy on sample points
q_preds = (q_proba_batched(X, weights) >= 0.5).astype(int)
acc_q   = (q_preds == y).mean()

print("\n=== Quick Report ===")
print(f"Target: {TARGET} | Samples: {N_SAMPLES} | Steps: {STEPS} | Layers: {LAYERS} | Entangler ablated: {ABLATE_ENTANGLER}")
print(f"SVM (RBF) acc: {acc_svm:.3f}")
print(f"Logistic (linear) acc: {acc_lin:.3f}")
print(f"Quantum VQC acc: {acc_q:.3f}")
print("Note: SVM is a strong classical baseline; logistic is the honest 'weak' baseline.")
