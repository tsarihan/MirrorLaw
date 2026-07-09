"""
Calibration sweep: grounding the 1/4 graduation threshold empirically.

The threshold is borrowed from metrology's 4:1 Test Accuracy Ratio (TAR): a reference
standard is deemed adequate when its uncertainty is <= 1/4 of the quantity being
calibrated. We make the analogue concrete in the Mirror-Law setting.

Setup (linear, realizable): truth f*=Xw*. A learner starts with true error field e0
(||e0||=E0). A reference carries a SYSTEMATIC error field delta with ||delta|| = r*E0
(Regime C). Distillation trains the learner to MATCH the reference (minimize the
observable residual R_g = ||f_theta - f_ref||^2 = ||e_theta - delta||^2). In the
realizable case the learner clones the reference error, e_theta -> delta, so:

  final true error fraction  ||e_theta||^2 / ||e0||^2  ->  r^2     (an irreducible FLOOR)
  observable residual        R_g                       ->  0       (for EVERY r)

Hence: r=1/4 -> r^2 = 1/16 ~ 6% residual contamination (the "graduated" reference);
r=1 -> break-even (distillation neither helps nor hurts); r>1 -> distillation HARMS.
And R_g->0 regardless of r, so r is invisible without grading the reference on
ground truth -- the same ground-truth-contact requirement as ensembling and Prop 4.
"""
import numpy as np
d, N, S = 50, 4000, 12
RS = np.linspace(0.0, 1.25, 14)
STEPS, LR = 400, 0.5

def run(seed, r):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d)); X /= np.sqrt(d)
    wstar = rng.standard_normal(d)
    unit = lambda z: z / np.linalg.norm(z)
    E0 = 1.0
    e0_dir = unit(rng.standard_normal(d)); w = wstar + E0 * e0_dir       # learner init (true error E0)
    delta_dir = unit(rng.standard_normal(d)); b = (r * E0) * delta_dir   # reference systematic error
    wref = wstar + b
    yref = X @ wref                                                       # distillation target = reference outputs
    # gradient descent to match reference (observable objective)
    for _ in range(STEPS):
        grad = X.T @ (X @ w - yref) / N
        w = w - LR * grad
    true_err = np.linalg.norm(w - wstar)**2                              # needs oracle
    init_err = E0**2
    Rg = np.mean((X @ w - yref)**2)                                       # observable residual
    return true_err / init_err, Rg

print(f"d={d} N={N} S={S} seeds | sweeping r=||delta||/||e0||\n")
print(f"{'r':>6} {'true-err frac (mean±95%CI)':>28} {'r^2':>8} {'obs R_g':>12}")
frac_mean = []
for r in RS:
    fr = np.array([run(s, r)[0] for s in range(S)])
    rg = np.mean([run(s, r)[1] for s in range(S)])
    ci = 1.96 * fr.std(ddof=1) / np.sqrt(S)
    frac_mean.append(fr.mean())
    tag = ""
    if abs(r-0.25) < 0.05 or (r>0.20 and r<0.30): tag = "  <- ~1/4 graduation (~6%)"
    print(f"{r:>6.3f} {fr.mean():>16.4f} ± {ci:<7.4f} {r**2:>8.4f} {rg:>12.2e}{tag}")
print("\n[Analysis] true-err fraction tracks r^2 (the cloned-bias floor); R_g ~ 0 for all r")
print("[Analysis] r=1/4 -> ~6% residual; r=1 -> break-even; r>1 -> distillation harms")

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.6, 3.8))
rr = np.linspace(0, 1.25, 200)
ax.plot(rr, rr**2, "-", color="#888", lw=1.5, label=r"$r^2$ (theory floor)")
ax.plot(RS, frac_mean, "o", color="#1f4e79", ms=5, label="trained learner (12 seeds)")
ax.axvline(0.25, color="#2e8b57", ls="--", lw=1.2); ax.text(0.27, 0.85, "4:1 TAR\n(r=1/4)", color="#2e8b57", fontsize=8.5)
ax.axhline(1/16, color="#2e8b57", ls=":", lw=1.0); ax.text(0.9, 1/16+0.02, r"$1/16\approx6\%$", color="#2e8b57", fontsize=9)
ax.axvline(1.0, color="#c0392b", ls="--", lw=1.2); ax.text(1.02, 0.2, "break-even\n($r=1$)", color="#c0392b", fontsize=8.5)
ax.fill_between([1.0, 1.25], 0, 1.6, color="#c0392b", alpha=0.06)
ax.set_xlabel(r"reference/learner error ratio $r=\|\delta\|/\|e_\theta^0\|$")
ax.set_ylabel(r"residual true error $\|e_\theta\|^2/\|e_\theta^0\|^2$")
ax.set_title("Calibration: the graduation threshold is a 4:1 convention", fontsize=10.5)
ax.set_xlim(0, 1.25); ax.set_ylim(0, 1.6); ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig("fig13_calibration.pdf", bbox_inches="tight"); plt.close(fig)
print("\nSaved fig13_calibration.pdf")
