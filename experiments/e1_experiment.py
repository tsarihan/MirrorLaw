"""
E1 for "The Mirror Law": empirical demonstration of the three regimes,
deceptive descent, the calibration relation R*_inf = ||b||^2, and the
two-teacher detection check. Linear model (theory exact) + MLP confirmation.
CPU, ~seconds. Fully reproducible (seeded).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ----------------------------------------------------------------------
# Linear realizable setup: learner, truth, and bias all linear in x.
#   f*(x) = w*.x ,  g(x) = (w*+beta).x + noise ,  learner = w.x  (MSE to g)
#   => R*  = ||w - w*||^2 (Sigma~I) ,  R_g floors at noise variance
#   => Regime C: w -> w*+beta, so R* -> ||beta||^2 = ||b||^2  (deceptive descent)
# ----------------------------------------------------------------------
d, N, T, lr = 20, 4000, 150, 0.3
X = rng.standard_normal((N, d))
wstar = rng.standard_normal(d); wstar *= 2.0 / np.linalg.norm(wstar)   # ||w*|| = 2

def run_linear(beta_norm, sigma, seed):
    r = np.random.default_rng(seed)
    beta = r.standard_normal(d)
    beta = beta * (beta_norm / np.linalg.norm(beta)) if beta_norm > 0 else beta * 0.0
    g = X @ (wstar + beta) + sigma * r.standard_normal(N)
    w = np.zeros(d); Rg = np.empty(T); Rs = np.empty(T)
    for t in range(T):
        pred = X @ w
        w -= lr * (2.0 / N) * (X.T @ (pred - g))
        Rg[t] = np.mean((X @ w - g) ** 2)
        Rs[t] = np.mean((X @ w - X @ wstar) ** 2)
    return Rg, Rs, beta, w

RgA, RsA, _, _ = run_linear(0.0, 0.0, 1)   # A: faithful
RgB, RsB, _, _ = run_linear(0.0, 1.0, 2)   # B: noisy   (sigma^2 = 1)
RgC, RsC, betaC, wC = run_linear(1.0, 0.0, 3)   # C: biased  (||b||^2 = 1)

print("=== Linear, asymptotic risks (final step) ===")
print(f"A faithful : R_g={RgA[-1]:.4e}  R*={RsA[-1]:.4e}")
print(f"B noisy    : R_g={RgB[-1]:.4e}  R*={RsB[-1]:.4e}   (sigma^2=1.0)")
print(f"C biased   : R_g={RgC[-1]:.4e}  R*={RsC[-1]:.4e}   (||b||^2={np.sum(betaC**2):.4f})")

# ----------------------------------------------------------------------
# Calibration relation: sweep ||b|| and check R*_inf == ||b||^2 (slope 1).
# ----------------------------------------------------------------------
bnorms = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
Rstar_inf = np.array([run_linear(bn, 0.0, 100 + i)[1][-1] for i, bn in enumerate(bnorms)])
benergy = bnorms ** 2
slope = float(np.dot(benergy, Rstar_inf) / np.dot(benergy, benergy))   # through-origin LS
print("\n=== Calibration sweep: R*_inf vs ||b||^2 ===")
for be, ri in zip(benergy, Rstar_inf):
    print(f"  ||b||^2={be:5.3f}  ->  R*_inf={ri:7.4f}")
print(f"  through-origin slope = {slope:.4f}  (theory: 1.0)")

# ----------------------------------------------------------------------
# Detection: two teachers with INDEPENDENT bias directions, both Regime C.
# Each teacher's R_g -> ~0, yet their systematic disagreement is large.
# ----------------------------------------------------------------------
_, _, beta1, w1 = run_linear(1.0, 0.0, 21)
_, _, beta2, w2 = run_linear(1.0, 0.0, 22)
disagree = np.sqrt(np.mean((X @ w1 - X @ w2) ** 2))   # RMS teacher-vs-teacher disagreement
print("\n=== Detection (two decorrelated biased teachers) ===")
print(f"  RMS disagreement between teachers = {disagree:.4f}")
print(f"  (each learner's loss against its own teacher ~ {RgC[-1]:.2e})")

# ----------------------------------------------------------------------
# MLP confirmation (hand-coded 2-layer tanh net, momentum GD).
#   f* = fixed random MLP ; bias b(x) = c*sigmoid(w0.x) ; teacher g = f*+b
#   learner = MLP trained by MSE to g.  Expect R_g small, R* -> ||b||^2.
# ----------------------------------------------------------------------
d2, H, N2, STEPS = 10, 64, 2000, 6000
X2 = rng.standard_normal((N2, d2))

def mlp_init(seed):
    r = np.random.default_rng(seed)
    return [r.standard_normal((d2, H)) / np.sqrt(d2), np.zeros(H),
            r.standard_normal((H, 1)) / np.sqrt(H), np.zeros(1)]

def mlp_fwd(p, Xin):
    W1, b1, W2, b2 = p
    z1 = Xin @ W1 + b1; h1 = np.tanh(z1)
    return (h1 @ W2 + b2)[:, 0], h1

pstar = mlp_init(777)
ystar, _ = mlp_fwd(pstar, X2)
w0 = rng.standard_normal(d2)
bias = 1.3 * (1.0 / (1.0 + np.exp(-(X2 @ w0))))     # localized systematic offset
g2 = ystar + bias
bias_energy = float(np.mean(bias ** 2))

p = mlp_init(13)
vel = [np.zeros_like(a) for a in p]; mom, lrm = 0.9, 0.03
idx, RgM, RsM = [], [], []
for t in range(STEPS):
    out, h1 = mlp_fwd(p, X2)
    r = out - g2
    dout = (2.0 / N2) * r
    W1, b1, W2, b2 = p
    dW2 = h1.T @ dout[:, None]; db2 = np.array([dout.sum()])
    dz1 = (dout[:, None] @ W2.T) * (1 - h1 ** 2)
    dW1 = X2.T @ dz1; db1 = dz1.sum(0)
    for i, gr in enumerate([dW1, db1, dW2, db2]):
        vel[i] = mom * vel[i] - lrm * gr; p[i] = p[i] + vel[i]
    if t % 40 == 0 or t == STEPS - 1:
        idx.append(t); RgM.append(np.mean(r ** 2)); RsM.append(np.mean((out - ystar) ** 2))
idx = np.array(idx); RgM = np.array(RgM); RsM = np.array(RsM)
print("\n=== MLP confirmation (Regime C) ===")
print(f"  ||b||^2 = {bias_energy:.4f}")
print(f"  final R_g = {RgM[-1]:.4e}   final R* = {RsM[-1]:.4e}")

# ======================================================================
# FIGURE 1: three regimes (linear), R_g (dashed) vs R* (solid)
# ======================================================================
steps = np.arange(1, T + 1)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
panels = [("Regime A: faithful reference", RgA, RsA, None),
          ("Regime B: noisy reference", RgB, RsB, 1.0),
          ("Regime C: biased reference", RgC, RsC, 1.0)]
for a, (title, Rg, Rs, floor) in zip(ax, panels):
    a.semilogy(steps, np.clip(Rs, 1e-12, None), color="#1f4e79", lw=2.2, label=r"$R^\star$ (vs truth)")
    a.semilogy(steps, np.clip(Rg, 1e-12, None), color="#c0392b", lw=2.0, ls="--", label=r"$R_g$ (vs reference)")
    if title.startswith("Regime B"):
        a.axhline(1.0, color="#888", lw=1.0, ls=":"); a.text(T * 0.45, 1.25, r"$\sigma^2$", color="#555")
    if title.startswith("Regime C"):
        a.axhline(1.0, color="#888", lw=1.0, ls=":"); a.text(T * 0.4, 1.25, r"$\|b\|^2$", color="#555")
    a.set_title(title, fontsize=10); a.set_xlabel("gradient-descent step")
    a.grid(True, which="both", alpha=0.15)
ax[0].set_ylabel("risk (log scale)"); ax[0].legend(loc="lower left", fontsize=8.5, framealpha=0.9)
ax[2].annotate("inherited bias", xy=(T, 1.0), xytext=(T * 0.55, 0.06),
               arrowprops=dict(arrowstyle="->", color="#444"), color="#444", fontsize=8.5)
fig.suptitle("Deceptive descent: noise washes out (B) but systematic bias is cloned (C)", fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig("fig1_regimes.pdf", bbox_inches="tight"); plt.close(fig)

# ======================================================================
# FIGURE 2: (a) calibration R*_inf = ||b||^2 ; (b) MLP confirmation
# ======================================================================
fig2, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.4))
lim = benergy.max() * 1.12
axa.plot([0, lim], [0, lim], color="#888", ls=":", lw=1.2, label=r"$y=x$ (theory)")
axa.scatter(benergy, Rstar_inf, color="#1f4e79", s=42, zorder=3, label="measured")
axa.set_xlabel(r"reference bias energy $\|b\|^2$")
axa.set_ylabel(r"learner asymptotic true risk $R^\star_\infty$")
axa.set_title(f"Calibration relation (slope = {slope:.3f})", fontsize=10)
axa.set_xlim(0, lim); axa.set_ylim(0, lim); axa.legend(fontsize=8.5); axa.grid(alpha=0.15)

axb.semilogy(idx, np.clip(RsM, 1e-12, None), color="#1f4e79", lw=2.2, label=r"$R^\star$ (vs truth)")
axb.semilogy(idx, np.clip(RgM, 1e-12, None), color="#c0392b", lw=2.0, ls="--", label=r"$R_g$ (vs reference)")
axb.axhline(bias_energy, color="#888", lw=1.0, ls=":")
axb.text(idx[-1] * 0.45, bias_energy * 1.2, r"$\|b\|^2$", color="#555")
axb.set_xlabel("gradient-descent step"); axb.set_ylabel("risk (log scale)")
axb.set_title("MLP confirmation (Regime C, nonlinear)", fontsize=10)
axb.legend(fontsize=8.5, loc="lower left"); axb.grid(True, which="both", alpha=0.15)
fig2.tight_layout(); fig2.savefig("fig2_calibration_mlp.pdf", bbox_inches="tight"); plt.close(fig2)

print("\nSaved fig1_regimes.pdf and fig2_calibration_mlp.pdf")
