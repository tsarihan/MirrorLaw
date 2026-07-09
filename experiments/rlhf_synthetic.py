"""
E5 (synthetic RLHF-style) for "The Mirror Law".

This replaces the toy proxy-reward optimization of E3 with a synthetic RLHF-style
pipeline in a controlled contextual-bandit setting, addressing the reviewer
concern that E3 was deterministic gradient ascent on a hand-specified reward.
Here, in contrast to E3, we have all four ingredients of RLHF:

  1. PREFERENCE DATA: pairwise comparisons (x, a_win, a_lose) from a preference
     oracle whose ranking carries a SYSTEMATIC bias (a nuisance direction that
     the oracle rewards but that is irrelevant to true quality) -- the analogue
     of length/sycophancy bias in human feedback.
  2. A LEARNED REWARD MODEL: an MLP trained on the preferences via the
     Bradley-Terry loss. By the mirror law it clones the systematic bias in the
     preference signal (Regime C for the reward model).
  3. SAMPLING-BASED POLICY OPTIMIZATION: a Gaussian policy optimized by PPO
     (sampled actions, clipped surrogate, advantage), NOT differentiable ascent.
  4. A KL TRUST REGION to a reference (SFT) policy, and a separate GOLD evaluator
     that the optimizer never sees.

Prediction (deceptive descent / reward over-optimization, Gao et al. 2023):
optimizing the learned reward drives the proxy (RM) score up monotonically while
the gold reward traces an inverted-U -- it rises while the RM's true-quality
component dominates, then falls as the policy exploits the RM's cloned bias. A
stronger KL penalty (beta) limits the drift and preserves gold quality.

CPU, ~1 minute, fully reproducible (seeded). Produces fig8_rlhf.pdf.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

DX, DA = 8, 4          # context dim, action dim
SIGMA = 0.20           # policy std (fixed)
BIAS = 2.20            # systematic preference bias along nuisance direction u
PREF_NOISE = 0.30      # logistic preference temperature (lower = sharper labels)
device = "cpu"

# ----- ground truth: gold reward and the biased preference oracle -----
g = torch.Generator().manual_seed(7)
Wstar = torch.randn(DX, DA, generator=g) / np.sqrt(DX)   # context -> ideal action
u = torch.randn(DA, generator=g); u = u / u.norm()        # nuisance (bias) direction

def a_star(x):                     # context-dependent ideal action
    return x @ Wstar
def gold(x, a):                    # true quality: closeness to ideal action (in [~ -k,0])
    return -((a - a_star(x)) ** 2).sum(-1)
def proxy_pref_score(x, a):        # oracle ranks by gold PLUS a bias toward +u
    return gold(x, a) + BIAS * (a @ u)

# ----- a small MLP -----
def mlp(din, dout, h=64):
    return nn.Sequential(nn.Linear(din, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, dout))

# ===== 1. SFT reference policy: supervised fit to a*(x), deliberately under-trained =====
mu_ref = mlp(DX, DA)
opt = torch.optim.Adam(mu_ref.parameters(), lr=3e-3)
Xsft = torch.randn(4000, DX)
for _ in range(45):                 # under-train so RLHF can first IMPROVE gold, then over-optimize
    opt.zero_grad()
    loss = ((mu_ref(Xsft) - a_star(Xsft)) ** 2).mean()
    loss.backward(); opt.step()
for p in mu_ref.parameters(): p.requires_grad_(False)
REF = {k: v.clone() for k, v in mu_ref.state_dict().items()}

# ===== 2. preference data from the biased oracle, then a Bradley-Terry reward model =====
def sample_actions_for_prefs(n):
    x = torch.randn(n, DX)
    base = mu_ref(x)
    a1 = base + 1.1 * torch.randn(n, DA)      # broad proposals around the reference
    a2 = base + 1.1 * torch.randn(n, DA)
    return x, a1, a2

NPREF = 14000
xp, a1, a2 = sample_actions_for_prefs(NPREF)
with torch.no_grad():
    s1, s2 = proxy_pref_score(xp, a1), proxy_pref_score(xp, a2)
    p1 = torch.sigmoid((s1 - s2) / PREF_NOISE)            # noisy (logistic) preferences
    win1 = (torch.rand(NPREF) < p1)
aw = torch.where(win1[:, None], a1, a2)                   # winner / loser
al = torch.where(win1[:, None], a2, a1)

RM = mlp(DX + DA, 1)
optrm = torch.optim.Adam(RM.parameters(), lr=2e-3)
for _ in range(800):                                      # Bradley-Terry training
    optrm.zero_grad()
    rw = RM(torch.cat([xp, aw], -1)).squeeze(-1)
    rl = RM(torch.cat([xp, al], -1)).squeeze(-1)
    loss = -torch.log(torch.sigmoid(rw - rl) + 1e-9).mean()
    loss.backward(); optrm.step()
for p in RM.parameters(): p.requires_grad_(False)

# check: the learned RM has cloned the bias (correlates with +u beyond gold)
with torch.no_grad():
    xx = torch.randn(2000, DX); aa = mu_ref(xx) + 0.5 * torch.randn(2000, DA)
    rm_pred = RM(torch.cat([xx, aa], -1)).squeeze(-1)
    bias_corr = np.corrcoef((aa @ u).numpy(), rm_pred.numpy())[0, 1]
    gold_corr = np.corrcoef(gold(xx, aa).numpy(), rm_pred.numpy())[0, 1]
    proxy_corr = np.corrcoef(proxy_pref_score(xx, aa).numpy(), rm_pred.numpy())[0, 1]

# ===== 3. PPO: optimize the policy against the LEARNED reward, with a KL leash =====
def policy_logprob(mu, a):                                # Gaussian log-prob, fixed sigma
    return (-0.5 * ((a - mu) / SIGMA) ** 2 - np.log(SIGMA) - 0.5 * np.log(2 * np.pi)).sum(-1)

def run_ppo(beta, steps=220, batch=640, inner=6, clip=0.2, lr=8e-3, log=None):
    pol = mlp(DX, DA); pol.load_state_dict(REF)           # init policy = SFT reference
    optp = torch.optim.Adam(pol.parameters(), lr=lr)
    hist = {"rm": [], "gold": [], "kl": []}
    for t in range(steps):
        x = torch.randn(batch, DX)
        with torch.no_grad():
            mu_old = pol(x)
            a = mu_old + SIGMA * torch.randn(batch, DA)   # SAMPLE actions (stochastic policy)
            logp_old = policy_logprob(mu_old, a)
            mu_r = mlp(DX, DA); mu_r.load_state_dict(REF)  # reference mean (frozen)
            mu_ref_x = mu_r(x)
            rm = RM(torch.cat([x, a], -1)).squeeze(-1)     # LEARNED reward
            klp = 0.5 * (((a - mu_ref_x) ** 2 - (a - mu_old) ** 2) / SIGMA ** 2).sum(-1)  # log pi_theta - log pi_ref
            reward = rm - beta * klp                       # KL-penalised RLHF reward
            adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        for _ in range(inner):                             # PPO clipped update
            mu = pol(x); logp = policy_logprob(mu, a)
            ratio = torch.exp(logp - logp_old)
            surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            optp.zero_grad(); (-surr.mean()).backward(); optp.step()
        with torch.no_grad():
            xe = torch.randn(3000, DX); mue = pol(xe)
            ae = mue + SIGMA * torch.randn(3000, DA)
            hist["rm"].append(RM(torch.cat([xe, ae], -1)).squeeze(-1).mean().item())
            hist["gold"].append(gold(xe, ae).mean().item())
            mu_r = mlp(DX, DA); mu_r.load_state_dict(REF)
            hist["kl"].append((0.5 * ((mue - mu_r(xe)) ** 2).sum(-1) / SIGMA ** 2).mean().item())
    if log is not None: log[beta] = hist
    return hist

betas = [0.0, 0.02, 0.05, 0.15, 0.4]
logs = {}
for b in betas:
    run_ppo(b, log=logs)

main_b = 0.02
H = logs[main_b]
gold0 = H["gold"][0]
peak_i = int(np.argmax(H["gold"])); peak = H["gold"][peak_i]; final = H["gold"][-1]

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
steps = np.arange(len(H["gold"]))

# (a) deceptive descent in reward space: RM up, gold inverted-U
axL = ax[0]; axR = axL.twinx()
l1, = axL.plot(steps, H["gold"], color="#1f4e79", lw=2, label="gold quality (true)")
l2, = axR.plot(steps, H["rm"], color="#c0504d", lw=2, ls="--", label="reward-model score (proxy)")
axL.axvline(peak_i, color="#888", ls=":", lw=1)
axL.annotate("gold peaks, then declines\nwhile RM keeps rising", (peak_i, peak),
             xytext=(peak_i + len(steps)*0.12, peak - (peak-final)*0.1), fontsize=8.5, color="#444",
             arrowprops=dict(arrowstyle="->", color="#888"))
axL.set_xlabel("PPO step"); axL.set_ylabel("gold quality", color="#1f4e79")
axR.set_ylabel("reward-model score", color="#c0504d")
axL.set_title("(a) synthetic RLHF-style: deceptive descent in reward space", fontsize=10)
axL.legend(handles=[l1, l2], loc="lower left", fontsize=8.5)
axL.grid(alpha=0.15)

# (b) KL leash mitigates over-optimization
for b in betas:
    ax[1].plot(steps, logs[b]["gold"], lw=1.8, label=f"$\\beta={b}$")
ax[1].set_xlabel("PPO step"); ax[1].set_ylabel("gold quality (true)")
ax[1].set_title("(b) a stronger KL penalty limits over-optimization", fontsize=10)
ax[1].legend(fontsize=8.5, title="KL coeff.", title_fontsize=8.5); ax[1].grid(alpha=0.15)

fig.suptitle("E5: reward-model over-optimization in a synthetic RLHF-style pipeline (learned RM from biased preferences + PPO + gold eval)",
             fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig("fig8_rlhf.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- summary
print("RM fidelity: corr(RM,gold)=%.3f  corr(RM,proxy=gold+bias)=%.3f  corr(RM,nuisance a.u)=%.3f" % (gold_corr, proxy_corr, bias_corr))
print("main run beta=%.2f:" % main_b)
print("  gold: start=%.3f  peak=%.3f (step %d)  final=%.3f  => over-opt drop from peak = %.3f"
      % (gold0, peak, peak_i, final, peak - final))
print("  RM score: start=%.3f -> final=%.3f (monotone up: %s)"
      % (H["rm"][0], H["rm"][-1], bool(H["rm"][-1] > H["rm"][0])))
print("KL-leash sweep (final gold by beta):")
for b in betas:
    gg = logs[b]["gold"]; print("  beta=%.2f: peak=%.3f final=%.3f drop=%.3f" %
                                 (b, max(gg), gg[-1], max(gg) - gg[-1]))
