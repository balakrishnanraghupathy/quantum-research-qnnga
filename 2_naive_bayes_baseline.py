"""
2_naive_bayes_baseline.py
--------------------------
Classical Naive Bayes baseline for CVD risk classification,
matching the "Naive Bayes ~73.77%" baseline mentioned in the abstract.

Install requirements:
    pip install scikit-learn
"""

import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from importlib.util import spec_from_file_location, module_from_spec

# Load 1_data_preprocessing.py by path since it starts with a digit
# (not a valid plain `import` name in Python).
_spec = spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
prep = module_from_spec(_spec)
_spec.loader.exec_module(prep)


def run_naive_bayes(local_path: str = None):
    X_train, X_test, y_train, y_test, feature_names, _ = prep.load_data(local_path)

    model = GaussianNB()

    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== Naive Bayes Baseline ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC-ROC  : {auc:.3f}")
    print(f"Latency  : {latency:.4f} s")
    print(classification_report(y_test, y_pred, target_names=["No CVD", "CVD"]))

    return {"model": "NaiveBayes", "accuracy": acc, "auc": auc, "latency": latency}


if __name__ == "__main__":
    run_naive_bayes()
