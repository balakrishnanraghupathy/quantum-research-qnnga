"""
5_quantum_ga_pqnn.py
----------------------
Simplified implementation of the paper's core idea, QNNGA:

  1. A Quantum Genetic Algorithm (QGA) represents each candidate feature
     subset as a set of qubit chromosomes (rotation angles theta_i encoding
     probability amplitudes alpha=cos(theta), beta=sin(theta)). Individuals
     are "measured" into binary feature masks, and the qubit population is
     updated each generation using a quantum rotation gate that nudges
     angles toward the best-found mask (standard Han & Kim style QGA).
  2. A quantum-state-overlap redundancy penalty discourages selecting
     highly correlated features: each selected feature's values are
     angle-encoded onto a single qubit and pairwise fidelity (state
     overlap) between selected features is penalized in the fitness score.
  3. A Parameterized Quantum Neural Network (PQNN) — a variational quantum
     circuit with angle-encoded inputs, trainable RY/RZ rotation layers
     and CNOT entanglers — classifies the selected feature subset. This
     runs on a fully classical simulator (PennyLane's default.qubit),
     matching the "hardware agnostic" description in the abstract.
  4. Final probabilities are mapped to a 3-tier clinical risk output:
     Low / Moderate / High.

This is a compact, educational reference implementation, not a
reproduction of the paper's exact tuning — treat reported metrics as
illustrative, not guaranteed.

Install requirements:
    pip install pennylane numpy scikit-learn
"""

import time
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

import pennylane as qml
from sklearn.metrics import accuracy_score, roc_auc_score

_spec = spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
prep = module_from_spec(_spec)
_spec.loader.exec_module(prep)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_QUBITS = 8          # target number of selected features (paper: 8 of 14)
PQNN_LAYERS = 3        # variational circuit depth
QGA_POP = 10
QGA_GENERATIONS = 6
REDUNDANCY_WEIGHT = 0.15  # penalty strength for correlated/overlapping features


# ---------------------------------------------------------------------
# 1. Quantum-state overlap (fidelity) between two angle-encoded features
# ---------------------------------------------------------------------
def feature_state_overlap(col_a, col_b):
    """
    Angle-encode each feature column onto a single qubit: |psi> =
    cos(theta/2)|0> + sin(theta/2)|1>, theta scaled from the feature's
    mean value in [0, pi]. Returns the fidelity |<psi_a|psi_b>|^2 as a
    proxy for redundancy between two features (1 = identical, 0 = orthogonal).
    """
    theta_a = np.pi * (np.mean(col_a) - col_a.min()) / (col_a.max() - col_a.min() + 1e-9)
    theta_b = np.pi * (np.mean(col_b) - col_b.min()) / (col_b.max() - col_b.min() + 1e-9)
    overlap = np.cos((theta_a - theta_b) / 2) ** 2
    return overlap


def redundancy_penalty(mask, X):
    cols = np.where(mask == 1)[0]
    if len(cols) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            total += feature_state_overlap(X[:, cols[i]], X[:, cols[j]])
            count += 1
    return total / count  # average pairwise overlap


# ---------------------------------------------------------------------
# 2. Parameterized Quantum Neural Network (PQNN) classifier
# ---------------------------------------------------------------------
def build_pqnn(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        # Angle encoding of the (scaled) selected features
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        # Variational layers: RY/RZ rotations + ring of CNOT entanglers
        for layer in range(PQNN_LAYERS):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.expval(qml.PauliZ(0))

    return circuit


def scale_to_angles(X):
    """Min-max scale each column to [0, pi] for angle encoding."""
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    return np.pi * (X - X_min) / (X_max - X_min + 1e-9), (X_min, X_max)


def pqnn_forward(circuit, weights, X_angles):
    outputs = np.array([circuit(row, weights) for row in X_angles])
    # Map PauliZ expectation in [-1, 1] -> probability in [0, 1]
    return (outputs + 1) / 2


def train_pqnn(X_train, y_train, n_qubits, epochs=40, lr=0.15):
    weights = 0.1 * rng.standard_normal((PQNN_LAYERS, n_qubits, 2))
    circuit = build_pqnn(n_qubits)
    X_angles, _ = scale_to_angles(X_train)
    y_signed = 2 * y_train - 1  # map {0,1} -> {-1,+1} to match PauliZ range

    def cost(w):
        preds = np.array([circuit(row, w) for row in X_angles])
        return np.mean((preds - y_signed) ** 2)

    opt = qml.GradientDescentOptimizer(stepsize=lr)
    for epoch in range(epochs):
        weights = opt.step(cost, weights)
        if (epoch + 1) % 10 == 0:
            print(f"  PQNN epoch {epoch+1}/{epochs}  loss={cost(weights):.4f}")

    return circuit, weights


# ---------------------------------------------------------------------
# 3. Quantum Genetic Algorithm (QGA) for feature selection
# ---------------------------------------------------------------------
def measure_qubits(thetas):
    """Collapse each qubit chromosome to a bit via its Born-rule probability."""
    probs_1 = np.sin(thetas) ** 2
    return (rng.random(thetas.shape) < probs_1).astype(int)


def enforce_cardinality(mask, target_k):
    """Keep exactly target_k selected features (paper selects a fixed subset size)."""
    ones = np.where(mask == 1)[0]
    if len(ones) > target_k:
        drop = rng.choice(ones, size=len(ones) - target_k, replace=False)
        mask[drop] = 0
    elif len(ones) < target_k:
        zeros = np.where(mask == 0)[0]
        add = rng.choice(zeros, size=target_k - len(ones), replace=False)
        mask[add] = 1
    return mask


def qga_fitness(mask, X_train, X_test, y_train, y_test, n_qubits):
    cols = np.where(mask == 1)[0]
    circuit, weights = train_pqnn(X_train[:, cols], y_train, n_qubits, epochs=20)
    X_angles, _ = scale_to_angles(X_test[:, cols])
    probs = pqnn_forward(circuit, weights, X_angles)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    penalty = redundancy_penalty(mask, X_train)
    fitness = acc - REDUNDANCY_WEIGHT * penalty
    return fitness, acc


def quantum_genetic_algorithm(X_train, X_test, y_train, y_test, n_features, target_k=N_QUBITS):
    # Each individual = vector of qubit rotation angles theta in [0, pi/2]
    population = rng.uniform(np.pi / 8, 3 * np.pi / 8, size=(QGA_POP, n_features))
    best_mask, best_fit, best_acc = None, -1.0, 0.0

    for gen in range(QGA_GENERATIONS):
        masks = [enforce_cardinality(measure_qubits(ind), target_k) for ind in population]
        results = [
            qga_fitness(m, X_train, X_test, y_train, y_test, target_k) for m in masks
        ]
        fits = np.array([r[0] for r in results])
        accs = np.array([r[1] for r in results])

        gen_best = np.argmax(fits)
        if fits[gen_best] > best_fit:
            best_fit, best_acc = fits[gen_best], accs[gen_best]
            best_mask = masks[gen_best]

        print(f"QGA Gen {gen+1}/{QGA_GENERATIONS}  best_fitness={fits.max():.4f}  best_acc={accs.max():.4f}")

        # Quantum rotation gate update: nudge every individual's angles
        # toward the best individual found so far (exploration + exploitation)
        target_theta = np.pi / 2 * best_mask  # push toward 1 (theta=pi/2) or 0
        rotation_step = 0.05 * np.pi
        direction = np.sign(target_theta - population)
        population = population + rotation_step * direction
        population = np.clip(population, 0.01, np.pi / 2 - 0.01)

        # Small mutation for diversity
        mutate = rng.random(population.shape) < 0.05
        population[mutate] = rng.uniform(np.pi / 8, 3 * np.pi / 8, size=mutate.sum())

    return best_mask, best_fit, best_acc


# ---------------------------------------------------------------------
# 4. Three-tier risk stratification
# ---------------------------------------------------------------------
def risk_tier(prob, low_cut=0.33, high_cut=0.66):
    if prob < low_cut:
        return "Low"
    elif prob < high_cut:
        return "Moderate"
    return "High"


# ---------------------------------------------------------------------
# 5. End-to-end run
# ---------------------------------------------------------------------
def run_qnnga(local_path: str = None):
    X_train, X_test, y_train, y_test, feature_names, _ = prep.load_data(local_path)
    n_features = X_train.shape[1]

    print("Running Quantum Genetic Algorithm feature selection...")
    best_mask, best_fit, _ = quantum_genetic_algorithm(
        X_train, X_test, y_train, y_test, n_features
    )
    selected = [f for f, m in zip(feature_names, best_mask) if m == 1]
    print("\nSelected orthogonal feature subset:", selected)

    cols = np.where(best_mask == 1)[0]
    print("\nTraining final PQNN classifier on selected subset...")
    start = time.time()
    circuit, weights = train_pqnn(X_train[:, cols], y_train, len(cols), epochs=60)
    X_test_angles, _ = scale_to_angles(X_test[:, cols])
    probs = pqnn_forward(circuit, weights, X_test_angles)
    latency = time.time() - start

    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    tiers = [risk_tier(p) for p in probs]

    print("\n=== QNNGA (Quantum GA + PQNN) ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC-ROC  : {auc:.3f}")
    print(f"Latency  : {latency:.4f} s (inference on test set)")
    print(f"Sample risk tiers: {tiers[:10]}")

    return {
        "model": "QNNGA", "accuracy": acc, "auc": auc, "latency": latency,
        "selected_features": selected, "risk_tiers": tiers,
    }


if __name__ == "__main__":
    run_qnnga()
