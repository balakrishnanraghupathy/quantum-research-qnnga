"""
3_bagging_stacking_baseline.py
--------------------------------
Classical ensemble baselines: Bagging and Stacking, matching the
"bagging / ensemble stacking ~74.00%" baselines mentioned in the abstract.

Install requirements:
    pip install scikit-learn
"""

import time
from importlib.util import spec_from_file_location, module_from_spec

from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

_spec = spec_from_file_location("data_preprocessing", "1_data_preprocessing.py")
prep = module_from_spec(_spec)
_spec.loader.exec_module(prep)


def run_bagging(X_train, X_test, y_train, y_test):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, random_state=42),
        n_estimators=50,
        random_state=42,
    )
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("=== Bagging Classifier ===")
    print(f"Accuracy : {acc*100:.2f}%  AUC-ROC: {auc:.3f}  Latency: {latency:.4f}s")
    return {"model": "Bagging", "accuracy": acc, "auc": auc, "latency": latency}


def run_stacking(X_train, X_test, y_train, y_test):
    base_learners = [
        ("nb", GaussianNB()),
        ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
    ]
    model = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
    )
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency = time.time() - start

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print("=== Stacking Ensemble ===")
    print(f"Accuracy : {acc*100:.2f}%  AUC-ROC: {auc:.3f}  Latency: {latency:.4f}s")
    return {"model": "Stacking", "accuracy": acc, "auc": auc, "latency": latency}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, _ = prep.load_data()
    run_bagging(X_train, X_test, y_train, y_test)
    run_stacking(X_train, X_test, y_train, y_test)
