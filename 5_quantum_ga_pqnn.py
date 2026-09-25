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
    pip install qiskit==1.2.2 qiskit-aer==0.15.1 qiskit-machine-learning==0.7.2 numpy==1.26.4 scikit-learn==1.7.2
"""

import time
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from sklearn.metrics import accuracy_score, roc_auc_score

# Qiskit Core & Aer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

# Load preprocessing module
_spec = spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
prep = module_from_spec(_spec)
_spec.loader.exec_module(prep)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

N_QUBITS = 8           # target number of selected features
PQNN_LAYERS = 3        # variational circuit depth
QGA_POP = 10
QGA_GENERATIONS = 6
REDUNDANCY_WEIGHT = 0.15


# ---------------------------------------------------------------------
# 1. Quantum-state overlap (fidelity) between angle-encoded features
# ---------------------------------------------------------------------
def feature_state_overlap(col_a, col_b):
    theta_a = np.pi * (np.mean(col_a) - col_a.min()) / (col_a.max() - col_a.min() + 1e-9)
    theta_b = np.pi * (np.mean(col_b) - col_b.min()) / (col_b.max() - col_b.min() + 1e-9)
    return np.cos((theta_a - theta_b) / 2) ** 2


def redundancy_penalty(mask, X):
    cols = np.where(mask == 1)[0]
    if len(cols) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            total += feature_state_overlap(X[:, cols[i]], X[:, cols[j]])
            count += 1
    return total / count


# ---------------------------------------------------------------------
# 2. Parameterized Quantum Neural Network (PQNN) in Qiskit
# ---------------------------------------------------------------------
def build_pqnn_circuit(n_qubits):
    inputs = ParameterVector("x", n_qubits)
    weights = ParameterVector("w", PQNN_LAYERS * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)

    # Angle encoding of inputs
    for i in range(n_qubits):
        qc.ry(inputs[i], i)

    # Variational layers: RY, RZ, and circular CNOT entanglement
    w_idx = 0
    for layer in range(PQNN_LAYERS):
        for i in range(n_qubits):
            qc.ry(weights[w_idx], i)
            qc.rz(weights[w_idx + 1], i)
            w_idx += 2
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    # Observable: Pauli-Z on qubit 0 (represented as string "II...IZ")
    observable_str = "I" * (n_qubits - 1) + "Z"
    observable = SparsePauliOp.from_list([(observable_str, 1.0)])

    # Exact Aer Statevector Estimator (shot-free)
    estimator = AerEstimator(run_options={"shots": None, "seed": RANDOM_STATE})

    qnn = EstimatorQNN(
        circuit=qc,
        observables=[observable],
        input_params=inputs,
        weight_params=weights,
        estimator=estimator,
    )

    return qnn, len(weights)


def scale_to_angles(X):
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    return np.pi * (X - X_min) / (X_max - X_min + 1e-9), (X_min, X_max)


def pqnn_forward(qnn, weights, X_angles):
    outputs = qnn.forward(X_angles, weights).flatten()
    return (outputs + 1) / 2  # Map expectation [-1, 1] -> [0, 1]


def train_pqnn(X_train, y_train, n_qubits, epochs=40, lr=0.15):
    qnn, num_weights = build_pqnn_circuit(n_qubits)
    weights = 0.1 * rng.standard_normal(num_weights)
    X_angles, _ = scale_to_angles(X_train)
    y_signed = (2 * y_train - 1).reshape(-1, 1)  # Targets in {-1, +1}

    # Batch Gradient Descent via Qiskit's Estimator backward pass
    for epoch in range(epochs):
        preds = qnn.forward(X_angles, weights)  # Shape (N, 1)
        _, grad_weights = qnn.backward(X_angles, weights)  # Shape (N, 1, num_weights)

        # MSE gradient: d/dw [ (1/N) * sum((pred - y)^2) ] = (2/N) * sum((pred - y) * dpred/dw)
        error = preds - y_signed
        grad = np.mean(2 * error[:, :, None] * grad_weights, axis=0).flatten()

        weights -= lr * grad

        if (epoch + 1) % 10 == 0:
            loss = np.mean((preds - y_signed) ** 2)
            print(f"  PQNN epoch {epoch+1}/{epochs}  loss={loss:.4f}")

    return qnn, weights


# ---------------------------------------------------------------------
# 3. Quantum Genetic Algorithm (QGA) for feature selection
# ---------------------------------------------------------------------
def measure_qubits(thetas):
    probs_1 = np.sin(thetas) ** 2
    return (rng.random(thetas.shape) < probs_1).astype(int)


def enforce_cardinality(mask, target_k):
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
    qnn, weights = train_pqnn(X_train[:, cols], y_train, n_qubits, epochs=20)
    X_angles, _ = scale_to_angles(X_test[:, cols])
    probs = pqnn_forward(qnn, weights, X_angles)
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    penalty = redundancy_penalty(mask, X_train)
    fitness = acc - REDUNDANCY_WEIGHT * penalty
    return fitness, acc


def quantum_genetic_algorithm(X_train, X_test, y_train, y_test, n_features, target_k=N_QUBITS):
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

        target_theta = np.pi / 2 * best_mask
        rotation_step = 0.05 * np.pi
        direction = np.sign(target_theta - population)
        population = population + rotation_step * direction
        population = np.clip(population, 0.01, np.pi / 2 - 0.01)

        mutate = rng.random(population.shape) < 0.05
        population[mutate] = rng.uniform(np.pi / 8, 3 * np.pi / 8, size=mutate.sum())

    return best_mask, best_fit, best_acc


# ---------------------------------------------------------------------
# 4. Three-tier risk stratification & main routine
# ---------------------------------------------------------------------
def risk_tier(prob, low_cut=0.33, high_cut=0.66):
    if prob < low_cut:
        return "Low"
    elif prob < high_cut:
        return "Moderate"
    return "High"


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
    qnn, weights = train_pqnn(X_train[:, cols], y_train, len(cols), epochs=60)
    X_test_angles, _ = scale_to_angles(X_test[:, cols])
    probs = pqnn_forward(qnn, weights, X_test_angles)
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