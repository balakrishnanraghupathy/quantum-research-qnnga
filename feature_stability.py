"""
feature_stability.py - Feature-selection stability analysis for the QGA (and classical GA).

INPUT  : runs.csv, one row per feature-selection run, columns:
            method    QGA or GA
            repeat    repeat index (seed of the fold split)
            fold      fold index within the repeat
            selected  selected attributes separated by ';'   e.g. age;cp;trestbps;chol;thalach;exang;oldpeak;ca
         Each run must use the TRAINING part of its fold only (as in Section 4.11).

OUTPUT : output/TableS_feature_stability.csv   stability indices per method
         output/TableS_selection_frequency.csv selection frequency of every attribute
         output/FigS_feature_stability.pdf/.png selection-frequency figure (same style as main figures)

Usage  : python feature_stability.py runs.csv
"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import make_figures as mf  # shared style, palette and save()

# Candidate predictors searched by the QGA. The standard UCI Cleveland file has 13 predictors
# plus the target ('num'/'target'); edit this list if your chromosome length differs.
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
REFERENCE = {"age", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "ca"}  # Table 11


def to_matrix(subsets):
    Z = np.zeros((len(subsets), len(FEATURES)), dtype=int)
    for i, s in enumerate(subsets):
        for f in s:
            Z[i, FEATURES.index(f)] = 1
    return Z


def nogueira(Z):
    """Stability estimator of Nogueira, Sechidis & Brown (JMLR 2018). 1 = identical subsets in
    every run; ~0 = no more agreement than random selection of the same size."""
    M, d = Z.shape
    p = Z.mean(0)
    s2 = M / (M - 1) * p * (1 - p)
    kbar = Z.sum(1).mean()
    return 1 - s2.mean() / ((kbar / d) * (1 - kbar / d))


def nogueira_ci(Z, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [nogueira(Z[rng.integers(0, len(Z), len(Z))]) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def kuncheva(a, b, d):
    k = len(a)
    if len(b) != k or k in (0, d):
        return np.nan
    return (len(a & b) * d - k * k) / (k * (d - k))


def analyse(df, method):
    subsets = [set(s.split(";")) for s in df.loc[df.method == method, "selected"]]
    Z = to_matrix(subsets)
    lo, hi = nogueira_ci(Z)
    pairs = list(combinations(subsets, 2))
    sizes = Z.sum(1)
    row = dict(method=method, runs=len(subsets),
               subset_size=f"{sizes.mean():.2f} +/- {sizes.std(ddof=1):.2f}",
               nogueira=round(nogueira(Z), 3), nogueira_95ci=f"[{lo:.3f}, {hi:.3f}]",
               mean_jaccard=round(np.mean([jaccard(a, b) for a, b in pairs]), 3),
               mean_kuncheva=round(np.nanmean([kuncheva(a, b, len(FEATURES)) for a, b in pairs]), 3),
               exact_reference_subset_pct=round(100 * np.mean([s == REFERENCE for s in subsets]), 1),
               mean_jaccard_to_reference=round(np.mean([jaccard(s, REFERENCE) for s in subsets]), 3))
    return row, Z.mean(0)


def plot_frequency(freqs):
    order = np.argsort(-freqs.get("QGA", next(iter(freqs.values()))))
    fig, ax = plt.subplots(figsize=(mf.DOUBLE * 0.75, 2.4))
    x = np.arange(len(FEATURES)); bw = 0.38 if len(freqs) == 2 else 0.6
    colors = {"QGA": mf.OI["verm"], "GA": mf.OI["grey"]}
    for k, (m, f) in enumerate(freqs.items()):
        off = (k - (len(freqs) - 1) / 2) * bw
        ax.bar(x + off, f[order] * 100, bw, color=colors.get(m, mf.OI["blue"]),
               label="Proposed QGA" if m == "QGA" else "Classical GA")
    ax.axhline(50, color="#555555", lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([FEATURES[i] for i in order], rotation=45, ha="right")
    for t in ax.get_xticklabels():
        if t.get_text() in REFERENCE:
            t.set_fontweight("bold")
    ax.set_ylim(0, 100); ax.set_ylabel("Selection frequency (%)"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    mf.save(fig, "FigS_feature_stability")


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    rows, freqs = [], {}
    for m in [x for x in ["QGA", "GA"] if x in set(df.method)]:
        r, f = analyse(df, m); rows.append(r); freqs[m] = f
    pd.DataFrame(rows).to_csv(mf.OUT / "TableS_feature_stability.csv", index=False)
    pd.DataFrame({m: np.round(f * 100, 1) for m, f in freqs.items()}, index=FEATURES) \
        .rename_axis("attribute").to_csv(mf.OUT / "TableS_selection_frequency.csv")
    plot_frequency(freqs)
    print(pd.DataFrame(rows).T.to_string(header=False))
