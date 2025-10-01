import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as pnp

# ---- 1) Make a target "shape" ----
# Circle: label = 1 if inside radius, else 0
def make_circle_data(n=300, radius=0.6):
    X = np.random.uniform(-1, 1, (n, 2))
    y = (X[:,0]**2 + X[:,1]**2 < radius**2).astype(int)
    return X, y

X, y = make_circle_data()

# ---- 2) Classical ML: SVM with RBF kernel ----
svm = SVC(kernel="rbf", gamma="scale", probability=True).fit(X, y)

# ---- 3) Quantum ML: 2-qubit VQC ----
n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

def encode(x):
    qml.AngleEmbedding(x, wires=[0,1], rotation="Y")
    qml.CNOT(wires=[0,1])
    qml.RZ(x[0]*x[1]*np.pi, wires=1)

def variational(weights):
    for layer in weights:
        for w, wire in zip(layer, range(n_qubits)):
            qml.RX(w[0], wires=wire)
            qml.RY(w[1], wires=wire)
            qml.RZ(w[2], wires=wire)
        qml.CNOT(wires=[0,1])

@qml.qnode(dev, interface="autograd")
def circuit(x, weights):
    encode(x)
    variational(weights)
    return qml.expval(qml.PauliZ(0))

def q_predict_proba(X, weights):
    outs = pnp.array([circuit(x, weights) for x in X])
    return (outs + 1) / 2  # map [-1,1] → [0,1]

# Train VQC
pnp.random.seed(0)
n_layers = 2
weights = 0.1 * pnp.random.randn(n_layers, n_qubits, 3)
opt = qml.GradientDescentOptimizer(stepsize=0.3)

Xb = pnp.array(X, requires_grad=False)
yb = pnp.array(y, requires_grad=False)

def loss(weights):
    p = q_predict_proba(Xb, weights)
    eps = 1e-8
    return -pnp.mean(yb*pnp.log(p+eps) + (1-yb)*pnp.log(1-p+eps))

for step in range(60):
    weights = opt.step(loss, weights)

# ---- 4) Plot decision boundaries ----
def plot_boundary(ax, predict_fn, X, y, title, weights=None):
    xx, yy = np.meshgrid(np.linspace(-1,1,200), np.linspace(-1,1,200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if weights is not None:
        Z = predict_fn(grid, weights)
    else:
        Z = predict_fn(grid)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=50, alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y, edgecolors="k", s=25)
    ax.set_title(title)

fig, axs = plt.subplots(1,2, figsize=(10,5))
plot_boundary(axs[0], lambda g: svm.predict_proba(g)[:,1], X, y, "Classical SVM (Circle)")
plot_boundary(axs[1], q_predict_proba, X, y, "Quantum VQC (Circle)", weights=weights)
plt.show()
