"""
6_main_compare_results.py
----------------------------
Orchestrates every model in this project and prints a comparison table,
mirroring the abstract's results table:

    Naive Bayes            ~73-74%
    Bagging / Stacking     ~74%
    Classical NN-GA        ~91-92%
    QNNGA (Quantum GA+PQNN) ~95%  <- proposed method

Usage:
    python 6_main_compare_results.py [path_to_dataset_csv]

If no local dataset path is given, it will attempt to download the UCI
Cleveland Heart Disease dataset directly (requires internet access).
"""

import sys
from importlib.util import spec_from_file_location, module_from_spec


def _load(module_name, path):
    spec = spec_from_file_location(module_name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("data_preprocessing", "1_data_preprocessing.py")
nb_mod = _load("naive_bayes_baseline", "2_naive_bayes_baseline.py")
ens_mod = _load("bagging_stacking_baseline", "3_bagging_stacking_baseline.py")
nnga_mod = _load("nn_ga_baseline", "4_nn_ga_baseline.py")
qnnga_mod = _load("quantum_ga_pqnn", "5_quantum_ga_pqnn.py")


def main():
    local_path = sys.argv[1] if len(sys.argv) > 1 else None

    results = []

    print("\n########## Naive Bayes ##########")
    results.append(nb_mod.run_naive_bayes(local_path))

    print("\n########## Bagging / Stacking ##########")
    X_train, X_test, y_train, y_test, feature_names, _ = prep.load_data(local_path)
    results.append(ens_mod.run_bagging(X_train, X_test, y_train, y_test))
    results.append(ens_mod.run_stacking(X_train, X_test, y_train, y_test))

    print("\n########## Classical NN + GA ##########")
    results.append(nnga_mod.run_nn_ga(local_path))

    print("\n########## QNNGA (Quantum GA + PQNN) ##########")
    results.append(qnnga_mod.run_qnnga(local_path))

    print("\n\n===================== SUMMARY =====================")
    print(f"{'Model':<12}{'Accuracy':>12}{'AUC-ROC':>12}{'Latency(s)':>14}")
    for r in results:
        print(f"{r['model']:<12}{r['accuracy']*100:>11.2f}%{r['auc']:>12.3f}{r['latency']:>14.4f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
