"""
rho_estimator.py -- recover the error-field correlation rho that the RAW one-hot-residual estimator can't.

THE PROBLEM (measured, not assumed)
-----------------------------------
In lineage_experiment.py the raw Pearson correlation of flattened one-hot residuals moved only 0.359 -> 0.373
across conditions whose behavior differed materially (final error 0.259 -> 0.326). Cause: a one-hot residual
carries a structural -1 at the gold index on every erring item, so two error fields share a baseline
correlation that swamps the "same wrong label on the same item" signal we actually want -- especially when one
field is a CONTINUOUS probability residual (the student) and the other is DISCRETE (the reference).

This module provides gold-centered / error-restricted estimators and a validation harness that constructs
GROUND-TRUTH rho and reports which estimator recovers it to +/-0.1. The winner becomes the estimator the
correlated/decorrelated matrix and H6 use.

numpy-only.
"""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def _onehot(a, k):
    o = np.zeros((len(a), k)); o[np.arange(len(a)), a] = 1.0; return o


def error_field(pred, y_true, k, prob=None, drop_gold=True):
    """Residual error field. If `prob` (n,k) given, use it (continuous student); else one-hot(pred).
       drop_gold zeroes the gold-class column, removing the structural -1 baseline."""
    G = _onehot(y_true, k)
    P = prob if prob is not None else _onehot(pred, k)
    e = P - G
    if drop_gold:
        e = e.copy(); e[np.arange(len(y_true)), y_true] = 0.0
    return e


def rho_raw(a_lab, b_lab, y, k, a_prob=None, b_prob=None):
    """Baseline: Pearson on full flattened one-hot residuals (the inadequate estimator)."""
    ea = error_field(a_lab, y, k, prob=a_prob, drop_gold=False).ravel()
    eb = error_field(b_lab, y, k, prob=b_prob, drop_gold=False).ravel()
    return _corr(ea, eb)


def rho_gold_centered(a_lab, b_lab, y, k, a_prob=None, b_prob=None):
    """Pearson on residuals with the gold column dropped (removes the shared -1 baseline)."""
    ea = error_field(a_lab, y, k, prob=a_prob, drop_gold=True).ravel()
    eb = error_field(b_lab, y, k, prob=b_prob, drop_gold=True).ravel()
    return _corr(ea, eb)


def rho_error_restricted(a_lab, b_lab, y, k, a_prob=None, b_prob=None):
    """Gold-dropped, restricted to items where at least one source errs (drops co-correct items)."""
    a_pred = a_lab if a_prob is None else a_prob.argmax(1)
    b_pred = b_lab if b_prob is None else b_prob.argmax(1)
    mask = (a_pred != y) | (b_pred != y)
    if mask.sum() < 3:
        return 0.0
    ea = error_field(a_lab, y, k, prob=a_prob, drop_gold=True)[mask].ravel()
    eb = error_field(b_lab, y, k, prob=b_prob, drop_gold=True)[mask].ravel()
    return _corr(ea, eb)


def rho_jointly_wrong_kappa(a_lab, b_lab, y, k, a_prob=None, b_prob=None):
    """Among items where BOTH err, chance-corrected agreement on the WRONG label (Cohen's kappa).
       Purely error-direction; immune to the gold baseline. Best for discrete-vs-discrete."""
    a_pred = a_lab if a_prob is None else a_prob.argmax(1)
    b_pred = b_lab if b_prob is None else b_prob.argmax(1)
    both = (a_pred != y) & (b_pred != y)
    if both.sum() < 3:
        return 0.0
    aw, bw = a_pred[both], b_pred[both]
    po = float((aw == bw).mean())
    # chance agreement from marginal wrong-label distributions
    classes = np.arange(k)
    pa = np.array([(aw == c).mean() for c in classes])
    pb = np.array([(bw == c).mean() for c in classes])
    pe = float((pa * pb).sum())
    return (po - pe) / (1 - pe + EPS)


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < EPS or b.std() < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# -------------------------------------------------------------------------------------------
# Validation harness: construct GROUND-TRUTH rho, report recovery per estimator
# -------------------------------------------------------------------------------------------
def _make_pair(n, k, gold_err_a, gold_err_b, shared_frac, rng, b_continuous=False):
    """Build two label fields with known gold errors and a known SHARED-ERROR fraction (the ground-truth rho).
       shared_frac = fraction of a's errors that b reproduces exactly (same item, same wrong label)."""
    y = rng.integers(0, k, n)
    # a's errors
    a = y.copy()
    na = int(gold_err_a * n)
    a_err_idx = rng.choice(n, na, replace=False)
    for i in a_err_idx:
        a[i] = (y[i] + rng.integers(1, k)) % k
    # b: reproduce shared_frac of a's errors exactly; place the rest on other items
    b = y.copy()
    nb = int(gold_err_b * n)
    n_shared = int(shared_frac * min(na, nb))
    shared = rng.choice(a_err_idx, n_shared, replace=False) if n_shared else np.array([], int)
    b[shared] = a[shared]
    pool = np.setdiff1d(np.arange(n), a_err_idx)
    n_rest = nb - n_shared
    rest = rng.choice(pool, min(n_rest, len(pool)), replace=False) if n_rest > 0 else np.array([], int)
    for i in rest:
        b[i] = (y[i] + rng.integers(1, k)) % k
    b_prob = None
    if b_continuous:
        # turn b into a soft probability field (student-like): peak on b's label, noise elsewhere
        b_prob = np.full((n, k), 0.1 / (k - 1))
        b_prob[np.arange(n), b] = 0.9
        b_prob += rng.random((n, k)) * 0.05
        b_prob /= b_prob.sum(1, keepdims=True)
    # ground-truth rho = shared errors / geometric mean of error counts (a symmetric shared-error rate)
    gt = n_shared / max(np.sqrt(na * nb), 1)
    return y, a, b, b_prob, gt


def validate(n=4000, k=4, seeds=5):
    rng = np.random.default_rng(0)
    ests = {"raw": rho_raw, "gold_centered": rho_gold_centered,
            "error_restricted": rho_error_restricted, "jointly_wrong_kappa": rho_jointly_wrong_kappa}
    print("=== recover GROUND-TRUTH rho (a=discrete, b=CONTINUOUS student-like) -- the failing case ===")
    print(f"{'gt_rho':>7s} | " + " | ".join(f"{name:>16s}" for name in ests))
    errs = {name: [] for name in ests}
    for target in [0.0, 0.3, 0.6, 0.9]:
        row_gt, row = [], {name: [] for name in ests}
        for s in range(seeds):
            y, a, b, b_prob, gt = _make_pair(n, k, 0.20, 0.20, target, np.random.default_rng(s),
                                             b_continuous=True)
            row_gt.append(gt)
            for name, fn in ests.items():
                v = fn(a, b, y, k, b_prob=b_prob)
                row[name].append(v); errs[name].append(abs(v - gt))
        gt_m = np.mean(row_gt)
        print(f"{gt_m:7.3f} | " + " | ".join(f"{np.mean(row[name]):16.3f}" for name in ests))
    print("\nmean |estimate - ground_truth| across all cells (lower is better; target < 0.10):")
    for name in ests:
        print(f"  {name:20s} {np.mean(errs[name]):.3f}")
    best = min(errs, key=lambda nm: np.mean(errs[nm]))
    print(f"\n[winner] {best} (mean abs error {np.mean(errs[best]):.3f})")
    return best


if __name__ == "__main__":
    validate()
