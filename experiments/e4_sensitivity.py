"""
E4 (sensitivity analysis) for "The Mirror Law".

Regime B is benign *in expectation under resampling and capacity control*: a
zero-mean (unbiased) noisy reference washes out and the learner recovers the
truth. The paper flags that this benignness degrades toward Regime-C-like
behavior as data becomes scarce and capacity unbounded (finite-data
memorization of label noise). We quantify that boundary along the three
parameters reviewers asked about: capacity, training-set size, and noise rate,
and show that weight-decay-style regularization mitigates it.

Setup (random-feature ridge regression, the standard memorization setting):
  x ~ N(0, I_d), d=20.  Fixed random map phi(x) = tanh(x W + b), W in R^{d x Pmax}.
  Truth f*(x) = phi_{:k}(x).theta*  (realizable for capacity >= k), unit variance.
  Reference (Regime B, b=0): g(x_i) = f*(x_i) + sigma*eps_i, eps_i ~ N(0,1),
    fixed per training point.  Learner: ridge fit with p features to N noisy
    labels (lam=0 => min-norm interpolation). Capacity = p.
  R* = test MSE to f* (true risk).

CPU, ~1 minute, fully reproducible (seeded).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d, PMAX, k, SEEDS, NTEST = 20, 1200, 16, 6, 4000
_fr = np.random.default_rng(12345)
W = _fr.standard_normal((d, PMAX)) / np.sqrt(d)
b = _fr.standard_normal(PMAX)
theta_star = np.zeros(PMAX); theta_star[:k] = _fr.standard_normal(k)

def phi(X, p): return np.tanh(X @ W[:, :p] + b[:p])
def phi_full(X): return np.tanh(X @ W + b)
_Xn = _fr.standard_normal((20000, d)); F_SCALE = (phi_full(_Xn) @ theta_star).std()
def f_star(X): return (phi_full(X) @ theta_star) / F_SCALE
_rt = np.random.default_rng(999); Xtest = _rt.standard_normal((NTEST, d)); ftest = f_star(Xtest)

def run(N, p, sigma, seed, lam_rel=0.0):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, d)); y = f_star(X) + sigma * r.standard_normal(N)
    Phi = phi(X, p)
    if lam_rel == 0.0:
        c, *_ = np.linalg.lstsq(Phi, y, rcond=None)          # min-norm interpolation
    else:
        G = Phi.T @ Phi; lam = lam_rel * np.mean(np.diag(G))
        c = np.linalg.solve(G + lam * np.eye(p), Phi.T @ y)  # ridge (weight decay)
    return float(np.mean((phi(Xtest, p) @ c - ftest) ** 2))

def sweep(values, args, lam_rel=0.0):
    return np.array([np.mean([run(*args(v), seed=s, lam_rel=lam_rel) for s in range(SEEDS)]) for v in values])

# (a) sample size: fixed capacity, light ridge for a clean boundary
p_a, sig_a = 300, 0.7
Ns = np.array([50,100,150,200,300,400,600,1000,2000,4000,8000])
Ra = sweep(Ns, lambda N: (N, p_a, sig_a), lam_rel=0.03)

# (b) capacity: ridgeless (memorization) vs ridge (mitigated)
N_b, sig_b = 400, 0.7
Ps = np.array([8,16,32,64,128,200,300,400,560,800,1200])
Rb_0 = sweep(Ps, lambda p: (N_b, p, sig_b), lam_rel=0.0)
Rb_r = sweep(Ps, lambda p: (N_b, p, sig_b), lam_rel=0.10)

# (c) noise: interpolating (p>N, memorizes) vs averaging (p<<N)
sig = np.array([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
Rc_int = sweep(sig, lambda s: (200, 600, s), lam_rel=0.0)
Rc_avg = sweep(sig, lambda s: (3000, 64, s), lam_rel=0.0)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
CT, CR, C2 = "#1f4e79", "#3c8c5a", "#c0504d"

ax[0].plot(Ns, Ra, "o-", color=CT)
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("training-set size $N$ (log)"); ax[0].set_ylabel(r"true risk $R^\star$ (log)")
ax[0].set_title(f"(a) more data averages noise out\n($p={p_a}$, $\\sigma={sig_a}$)", fontsize=9.5)
ax[0].grid(alpha=0.15, which="both")

ax[1].plot(Ps, Rb_0, "o-", color=CT, label="min-norm (no weight decay)")
ax[1].plot(Ps, Rb_r, "s--", color=CR, label="ridge (weight decay)")
ax[1].axvline(N_b, color="#888", ls=":", lw=1)
ax[1].annotate(f"$p\\approx N={N_b}$", (N_b, max(Rb_0)*0.5), xytext=(N_b*1.2, max(Rb_0)*0.5),
               fontsize=8, color="#555", arrowprops=dict(arrowstyle="->", color="#888"))
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("learner capacity $p$ (log)"); ax[1].set_ylabel(r"true risk $R^\star$ (log)")
ax[1].set_title(f"(b) capacity memorizes noise (Regime B$\\to$C-like);\nweight decay mitigates ($N={N_b}$, $\\sigma={sig_b}$)", fontsize=9.5)
ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=0.15, which="both")

ax[2].plot(sig, Rc_int, "o-", color=C2, label="interpolating ($p>N$): memorized")
ax[2].plot(sig, Rc_avg, "s-", color=CT, label="averaging ($p\\ll N$): washed out")
ax[2].set_xlabel(r"noise level $\sigma$"); ax[2].set_ylabel(r"true risk $R^\star$")
ax[2].set_title("(c) noise costs true error\nonly when memorized", fontsize=9.5)
ax[2].legend(fontsize=8, loc="upper left"); ax[2].grid(alpha=0.15)

fig.suptitle("Regime B sensitivity: benign noise-averaging degrades into memorization with scarce data, excess capacity, and no regularization",
             fontsize=10.5, y=1.04)
fig.tight_layout(); fig.savefig("fig7_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)

print("(a) N   :", list(Ns)); print("    R*  :", [round(v,3) for v in Ra])
print("    R* at N=%d = %.3f ; at N=%d = %.3f" % (Ns[0], Ra[0], Ns[-1], Ra[-1]))
print("(b) p   :", list(Ps))
print("    R* min-norm:", [round(v,3) for v in Rb_0])
print("    R* ridge   :", [round(v,3) for v in Rb_r])
i64 = list(Ps).index(64); ipk = int(np.argmax(Rb_0))
print("    min-norm: p=64->%.3f, peak %.3f at p=%d ; ridge at that p ->%.3f" % (Rb_0[i64], Rb_0.max(), Ps[ipk], Rb_r[ipk]))
print("(c) sigma:", list(sig))
print("    interp :", [round(v,3) for v in Rc_int]); print("    averag :", [round(v,3) for v in Rc_avg])
print("    sigma=1.0: interp R*=%.3f vs averaging R*=%.3f" % (Rc_int[list(sig).index(1.0)], Rc_avg[list(sig).index(1.0)]))
