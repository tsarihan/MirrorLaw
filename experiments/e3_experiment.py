"""
E3 for "The Mirror Law": reward-model over-optimization as deceptive descent.
A policy is optimized against a PROXY reward r = q* + lambda*(nuisance), where
q* is the true (gold) quality and the proxy carries a systematic bias toward a
nuisance direction. Optimizing the proxy drives measured reward up while true
quality rises then falls (Goodhart / reward hacking) -- the RLHF instance of
Regime C. Gaussian policy N(mu, sigma^2 I) vs reference N(0, sigma^2 I), KL
penalty beta; the toy reward is differentiable so we optimize mu by gradient
ascent directly. CPU, instant, seeded.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d = 8
v = rng.standard_normal(d); v /= np.linalg.norm(v)          # legitimate quality direction
u = rng.standard_normal(d); u -= (u @ v) * v; u /= np.linalg.norm(u)   # nuisance direction (orthogonal)
a, lam, sigma, beta, lr, STEPS = 1.0, 1.2, 1.0, 0.01, 0.04, 500

q_true = lambda mu: a * (mu @ v) - 0.5 * (mu @ mu)          # gold quality, peaks at mu = a*v
r_prox = lambda mu: q_true(mu) + lam * (mu @ u)             # proxy = gold + systematic nuisance bias
kl     = lambda mu: (mu @ mu) / (2 * sigma ** 2)            # KL(N(mu,s^2I) || N(0,s^2I))

mu = np.zeros(d)
hist = {"step": [], "proxy": [], "true": [], "kl": []}
for t in range(STEPS + 1):
    hist["step"].append(t); hist["proxy"].append(r_prox(mu)); hist["true"].append(q_true(mu)); hist["kl"].append(kl(mu))
    grad = a * v + lam * u - (1.0 + beta / sigma ** 2) * mu   # d/dmu [ r_prox - beta*KL ]
    mu = mu + lr * grad
for k in hist: hist[k] = np.array(hist[k])

peak_i = int(np.argmax(hist["true"]))
print("=== E3: reward-model over-optimization ===")
print(f"true quality: start {hist['true'][0]:+.3f}  peak {hist['true'][peak_i]:+.3f} (step {hist['step'][peak_i]}, KL {hist['kl'][peak_i]:.3f})  final {hist['true'][-1]:+.3f}")
print(f"proxy reward: start {hist['proxy'][0]:+.3f}  ->  final {hist['proxy'][-1]:+.3f}  (monotonically increasing)")
print(f"quality lost to over-optimization (peak - final): {hist['true'][peak_i] - hist['true'][-1]:.3f}")
print(f"final KL from reference: {hist['kl'][-1]:.3f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))
# (a) vs optimization step
axA.plot(hist["step"], hist["proxy"], color="#c0392b", lw=2.0, ls="--", label=r"proxy reward $\mathbb{E}[r]$ (optimized)")
axA.plot(hist["step"], hist["true"], color="#1f4e79", lw=2.2, label=r"true quality $\mathbb{E}[q^\star]$ (held out)")
axA.axvline(hist["step"][peak_i], color="#888", ls=":", lw=1.0)
axA.annotate("true quality peaks,\nthen over-optimization", (hist["step"][peak_i], hist["true"][peak_i]),
             xytext=(hist["step"][peak_i] + 90, hist["true"][peak_i] + 0.15),
             arrowprops=dict(arrowstyle="->", color="#444"), fontsize=8, color="#444")
axA.set_xlabel("policy-optimization step"); axA.set_ylabel("reward / quality")
axA.set_title("Reward hacking over training", fontsize=10.5); axA.legend(fontsize=8.5, loc="center right"); axA.grid(alpha=0.15)
# (b) vs KL from reference (the over-optimization frontier)
order = np.argsort(hist["kl"])
axB.plot(hist["kl"][order], hist["proxy"][order], color="#c0392b", lw=2.0, ls="--", label=r"proxy reward $\mathbb{E}[r]$")
axB.plot(hist["kl"][order], hist["true"][order], color="#1f4e79", lw=2.2, label=r"true quality $\mathbb{E}[q^\star]$")
axB.scatter([hist["kl"][peak_i]], [hist["true"][peak_i]], color="#1f4e79", zorder=5, s=30)
axB.axvspan(hist["kl"][peak_i], hist["kl"].max() * 1.02, color="#c0392b", alpha=0.06)
axB.text(hist["kl"][peak_i] * 1.05 + 0.35, hist["true"].min() + 0.05, "over-optimized\n(quality declines)", fontsize=8, color="#a33")
axB.set_xlabel(r"KL$(\pi_\theta \,\|\, \pi_\mathrm{ref})$"); axB.set_ylabel("reward / quality")
axB.set_title("Over-optimization frontier", fontsize=10.5); axB.legend(fontsize=8.5, loc="center right"); axB.grid(alpha=0.15)
fig.suptitle("E3: optimizing a biased proxy reward inflates measured reward while true quality falls", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig("fig4_reward_hacking.pdf", bbox_inches="tight"); plt.close(fig)
print("Saved fig4_reward_hacking.pdf")
