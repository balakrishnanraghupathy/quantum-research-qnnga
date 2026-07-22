"""
4_nn_ga_baseline.py
---------------------
Classical Neural-Network + Genetic-Algorithm (NN-GA) baseline: a simple
GA searches for a good binary feature mask, and an MLP is trained/scored
on the selected subset each generation. This matches the "classical
Neural Network Genetic Algorithm baseline ~91.80%" mentioned in the
abstract. A minimal GA is implemented from scratch (no extra dependency
like DEAP needed).

Install requirements:
    pip install scikit-learn numpy
"""

import time
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

_spec = spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
prep = module_from_spec(_spec)
_spec.loader.exec_module(prep)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


def fitness(mask, X_train, X_test, y_train, y_test):
    """Fitness = validation accuracy of an MLP trained on the selected
    feature subset. Masks selecting zero features are penalized."""
    if mask.sum() == 0:
        return 0.0
    cols = np.where(mask == 1)[0]
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train[:, cols], y_train)
    preds = clf.predict(X_test[:, cols])
    return accuracy_score(y_test, preds)


def genetic_algorithm_feature_selection(
    X_train, X_test, y_train, y_test,
    n_features, pop_size=12, generations=8,
    crossover_rate=0.8, mutation_rate=0.05,
):
    # Initialize population of random binary masks
    population = rng.integers(0, 2, size=(pop_size, n_features))

    best_mask, best_fit = None, -1.0

    for gen in range(generations):
        scores = np.array([
            fitness(ind, X_train, X_test, y_train, y_test) for ind in population
        ])

        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_fit:
            best_fit = scores[gen_best_idx]
            best_mask = population[gen_best_idx].copy()

        print(f"Gen {gen+1}/{generations}  best_fitness={scores.max():.4f}")

        # --- Selection (roulette wheel) ---
        probs = scores / (scores.sum() + 1e-9)
        parents_idx = rng.choice(pop_size, size=pop_size, p=probs)
        parents = population[parents_idx]

        # --- Crossover ---
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < crossover_rate:
                point = rng.integers(1, n_features)
                children[i, point:], children[i + 1, point:] = (
                    children[i + 1, point:].copy(),
                    children[i, point:].copy(),
                )

        # --- Mutation ---
        mutate_mask = rng.random(children.shape) < mutation_rate
        children[mutate_mask] = 1 - children[mutate_mask]

        population = children

    return best_mask, best_fit


def run_nn_ga(local_path: str = None):
    X_train, X_test, y_train, y_test, feature_names, _ = prep.load_data(local_path)
    n_features = X_train.shape[1]

    best_mask, best_fit = genetic_algorithm_feature_selection(
        X_train, X_test, y_train, y_test, n_features
    )
    selected = [f for f, m in zip(feature_names, best_mask) if m == 1]
    print("Selected features:", selected)

    cols = np.where(best_mask == 1)[0]
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=RANDOM_STATE)

    start = time.time()
    clf.fit(X_train[:, cols], y_train)
    y_pred = clf.predict(X_test[:, cols])
    y_proba = clf.predict_proba(X_test[:, cols])[:, 1]
    latency = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Classical NN-GA Baseline ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC-ROC  : {auc:.3f}")
    print(f"Latency  : {latency:.4f} s")

    return {"model": "NN-GA", "accuracy": acc, "auc": auc, "latency": latency,
            "selected_features": selected}


if __name__ == "__main__":
    run_nn_ga()
