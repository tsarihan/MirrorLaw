"""E3 robustness: (a) the true-quality loss from over-optimizing a biased proxy
grows smoothly with the bias magnitude lambda; (b) a stronger KL leash (beta)
graded-ly mitigates the hacking. Linear policy dynamics, analytic, instant."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(11)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d = 8
v = rng.standard_normal(d); v /= np.linalg.norm(v)
u = rng.standard_normal(d); u -= (u@v)*v; u /= np.linalg.norm(u)
sigma, lr, STEPS = 1.0, 0.04, 2000
q_true = lambda mu: (mu@v) - 0.5*(mu@mu)
kl = lambda mu: (mu@mu)/(2*sigma**2)

def run(lam, beta):
    mu = np.zeros(d); peak = -1e9; traj_true=[]
    for t in range(STEPS+1):
        qt = q_true(mu); traj_true.append(qt); peak = max(peak, qt)
        mu = mu + lr*(v + lam*u - (1.0+beta/sigma**2)*mu)
    return peak, q_true(mu), kl(mu)   # peak true, converged true, converged KL

# (a) sweep bias magnitude lambda (beta fixed small)
lams = np.array([0.0,0.4,0.8,1.2,1.6,2.0]); beta_a = 0.01
peak_l, conv_l, lost_l = [], [], []
for lam in lams:
    pk, cv, _ = run(lam, beta_a); peak_l.append(pk); conv_l.append(cv); lost_l.append(pk-cv)
print("=== (a) vs lambda (beta=0.01) ===")
for lam,pk,cv,lo in zip(lams,peak_l,conv_l,lost_l):
    print(f"  lambda={lam:.1f}  peak_true={pk:+.3f}  converged_true={cv:+.3f}  quality_lost={lo:.3f}")

# (b) sweep KL penalty beta (lambda fixed)
betas = np.array([0.005,0.01,0.02,0.05,0.1,0.2,0.5]); lam_b = 1.2
conv_b, kl_b = [], []
for beta in betas:
    _, cv, klf = run(lam_b, beta); conv_b.append(cv); kl_b.append(klf)
print("=== (b) vs beta (lambda=1.2) ===")
for beta,cv,kf in zip(betas,conv_b,kl_b):
    print(f"  beta={beta:.3f}  converged_true={cv:+.3f}  converged_KL={kf:.3f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))
axA.plot(lams, peak_l, color="#1f4e79", marker="^", lw=1.8, ms=5, label="peak true quality")
axA.plot(lams, conv_l, color="#c0392b", marker="s", lw=1.8, ms=5, label="converged true quality")
axA.fill_between(lams, conv_l, peak_l, color="#c0392b", alpha=0.08)
axA.set_xlabel(r"proxy bias magnitude $\lambda$"); axA.set_ylabel("true quality")
axA.set_title("(a) loss grows smoothly with bias", fontsize=10); axA.legend(fontsize=8.5); axA.grid(alpha=0.15)
axB.plot(kl_b, conv_b, color="#1f4e79", marker="o", lw=1.8, ms=5)
for beta,kf,cv in zip(betas,kl_b,conv_b):
    axB.annotate(f"$\\beta$={beta:g}", (kf,cv), textcoords="offset points", xytext=(4,-9), fontsize=7, color="#555")
axB.set_xlabel(r"converged KL$(\pi_\theta\,\|\,\pi_\mathrm{ref})$"); axB.set_ylabel("converged true quality")
axB.set_title("(b) KL leash graded-ly mitigates hacking", fontsize=10); axB.grid(alpha=0.15)
fig.suptitle("E3 robustness: over-optimization is smooth in bias and mitigable by the KL penalty", fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig("fig6_e3_robustness.pdf", bbox_inches="tight"); plt.close(fig)
print("Saved fig6_e3_robustness.pdf")
