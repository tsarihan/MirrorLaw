"""
E5 sensitivity for "The Mirror Law": dose-response of reward-model
over-optimization to the systematic bias in the preference signal.

The mirror law predicts that what the reward model clones -- and what the policy
then over-optimizes -- scales with the bias present in the preference signal. We
sweep the preference-bias strength and measure two things at each level:
  (1) how much the learned reward model has cloned the bias
      (correlation of RM scores with the nuisance direction, beyond gold), and
  (2) how severe the resulting over-optimization is
      (true-quality lost from peak to convergence under fixed-beta PPO).

Both should rise from ~0 at zero bias to large at high bias: bias in -> bias out.

This is the genuine-pipeline analogue of E1's calibration sweep. CPU, seeded.
Produces fig9_rlhf_sensitivity.pdf.
"""
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

DX, DA, SIGMA, PREF_NOISE = 8, 4, 0.20, 0.30
device = "cpu"
g = torch.Generator().manual_seed(7)
Wstar = torch.randn(DX, DA, generator=g) / np.sqrt(DX)
u = torch.randn(DA, generator=g); u = u / u.norm()

def a_star(x): return x @ Wstar
def gold(x, a): return -((a - a_star(x)) ** 2).sum(-1)
def mlp(din, dout, h=64):
    return nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, dout))

# shared SFT reference policy (same regardless of preference bias)
mu_ref = mlp(DX, DA); opt = torch.optim.Adam(mu_ref.parameters(), lr=3e-3)
Xsft = torch.randn(4000, DX)
for _ in range(45):
    opt.zero_grad(); (((mu_ref(Xsft) - a_star(Xsft)) ** 2).mean()).backward(); opt.step()
for p in mu_ref.parameters(): p.requires_grad_(False)
REF = {k: v.clone() for k, v in mu_ref.state_dict().items()}

def logprob(mu, a): return (-0.5 * ((a - mu) / SIGMA) ** 2 - np.log(SIGMA) - 0.5*np.log(2*np.pi)).sum(-1)

def run_pipeline(bias, seed, beta=0.05, npref=12000, rm_steps=700, ppo_steps=150, batch=640):
    torch.manual_seed(seed)
    # --- preferences from a bias-strength `bias` oracle ---
    xp = torch.randn(npref, DX); base = mu_ref(xp)
    a1 = base + 1.1*torch.randn(npref, DA); a2 = base + 1.1*torch.randn(npref, DA)
    def proxy(x, a): return gold(x, a) + bias * (a @ u)
    with torch.no_grad():
        p1 = torch.sigmoid((proxy(xp, a1) - proxy(xp, a2)) / PREF_NOISE)
        w = torch.rand(npref) < p1
    aw = torch.where(w[:, None], a1, a2); al = torch.where(w[:, None], a2, a1)
    # --- Bradley-Terry reward model ---
    RM = mlp(DX + DA, 1); o = torch.optim.Adam(RM.parameters(), lr=2e-3)
    for _ in range(rm_steps):
        o.zero_grad()
        rw = RM(torch.cat([xp, aw], -1)).squeeze(-1); rl = RM(torch.cat([xp, al], -1)).squeeze(-1)
        (-torch.log(torch.sigmoid(rw - rl) + 1e-9).mean()).backward(); o.step()
    for p in RM.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        xx = torch.randn(2000, DX); aa = mu_ref(xx) + 0.5*torch.randn(2000, DA)
        rp = RM(torch.cat([xx, aa], -1)).squeeze(-1).numpy()
        # partial correlation of RM with nuisance, controlling for gold (how much *bias* it cloned)
        gd = gold(xx, aa).numpy(); nu = (aa @ u).numpy()
        def resid(y, x): 
            b = np.polyfit(x, y, 1); return y - (b[0]*x + b[1])
        rm_bias_corr = np.corrcoef(resid(rp, gd), resid(nu, gd))[0, 1]
    # --- PPO against the learned reward (fixed beta) ---
    pol = mlp(DX, DA); pol.load_state_dict(REF); op = torch.optim.Adam(pol.parameters(), lr=8e-3)
    golds = []
    for t in range(ppo_steps):
        x = torch.randn(batch, DX)
        with torch.no_grad():
            mu_old = pol(x); a = mu_old + SIGMA*torch.randn(batch, DA); lpo = logprob(mu_old, a)
            mr = mlp(DX, DA); mr.load_state_dict(REF); mref = mr(x)
            rm = RM(torch.cat([x, a], -1)).squeeze(-1)
            kl = 0.5*(((a - mref)**2 - (a - mu_old)**2)/SIGMA**2).sum(-1)
            adv = (rm - beta*kl); adv = (adv - adv.mean())/(adv.std()+1e-6)
        for _ in range(6):
            mu = pol(x); lp = logprob(mu, a); ratio = torch.exp(lp - lpo)
            surr = torch.minimum(ratio*adv, torch.clamp(ratio, 0.8, 1.2)*adv)
            op.zero_grad(); (-surr.mean()).backward(); op.step()
        with torch.no_grad():
            xe = torch.randn(3000, DX); ae = pol(xe) + SIGMA*torch.randn(3000, DA)
            golds.append(gold(xe, ae).mean().item())
    golds = np.array(golds)
    return rm_bias_corr, float(golds.max()), float(golds[-1]), float(golds.max() - golds[-1])

biases = [0.0, 0.5, 1.0, 1.5, 2.2, 3.0]
SEEDS = [0, 1]
corr, peak, final, drop = [], [], [], []
for b in biases:
    rows = [run_pipeline(b, s) for s in SEEDS]
    corr.append(np.mean([r[0] for r in rows])); peak.append(np.mean([r[1] for r in rows]))
    final.append(np.mean([r[2] for r in rows])); drop.append(np.mean([r[3] for r in rows]))
    print("bias=%.1f: RM-bias-corr=%.3f  peak_gold=%.3f  final_gold=%.3f  over-opt drop=%.3f"
          % (b, corr[-1], peak[-1], final[-1], drop[-1]))

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(biases, corr, "o-", color="#1f4e79", lw=2)
ax[0].set_xlabel("preference-signal bias strength"); ax[0].set_ylabel("RM bias-cloning\n(partial corr. with nuisance | gold)")
ax[0].set_title("(a) the reward model clones more bias\nas the preference bias grows", fontsize=9.5)
ax[0].grid(alpha=0.15)

ax[1].plot(biases, drop, "o-", color="#c0504d", lw=2, label="true quality lost (peak $-$ final)")
ax[1].set_xlabel("preference-signal bias strength"); ax[1].set_ylabel("over-optimization severity")
ax[1].set_title("(b) over-optimization scales with\nthe preference bias (bias in $\\to$ bias out)", fontsize=9.5)
ax[1].grid(alpha=0.15); ax[1].legend(fontsize=8.5, loc="upper left")

fig.suptitle("E5 sensitivity: reward-model bias-cloning and over-optimization both scale with the systematic bias in the preference signal",
             fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig("fig9_rlhf_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)
print("\nwrote fig9_rlhf_sensitivity.pdf")
print("summary: at zero preference bias, over-opt drop = %.3f (no over-optimization);" % drop[0])
print("         at strong bias (%.1f), drop = %.3f -- the Mirror Law as a dose-response." % (biases[-1], drop[-1]))
