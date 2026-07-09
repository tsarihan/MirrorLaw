"""
E6 (recursive distillation / model collapse) for "The Mirror Law".

A new, checkable prediction the qualitative collapse literature does not make:
collapse is *noise contraction*, and it does not remove systematic bias.
Decomposing a recursively-generated chain into bias and noise, the mirror law
predicts that the NOISE component contracts geometrically across generations
(the classic variance loss) while the SYSTEMATIC BIAS is retained (a single
biased source) or ACCUMULATES ~linearly (a per-generation artifact) -- so the
error becomes bias-dominated rather than vanishing.

We use the canonical model-collapse recursion (Shumailov et al., 2024): estimate
the mean and variance of a distribution, then resample from the estimate, for
many generations -- which provably contracts the variance -- and layer the mirror
law's bias term on top. Each generation draws N samples from the previous
generation's N(mu, sigma^2), estimates mu (plus an optional per-generation
artifact delta) and sigma, and passes them on. We average many chains for smooth
curves. CPU, seeded. Produces fig10_collapse.pdf.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
MU_STAR, SIG0, N, GENS, CHAINS = 0.0, 1.0, 10, 20, 400

def chains(c0, delta, seed):
    r = np.random.default_rng(seed)
    mu = np.full(CHAINS, MU_STAR + c0); sig = np.full(CHAINS, SIG0)
    BIAS = np.zeros((GENS, CHAINS)); NOISE = np.zeros((GENS, CHAINS))
    for k in range(GENS):
        BIAS[k] = mu - MU_STAR; NOISE[k] = sig
        s = r.standard_normal((CHAINS, N)) * sig[:, None] + mu[:, None]   # resample from prev estimate
        mu = s.mean(1) + delta                                            # + per-generation artifact
        sig = s.std(1)                                                    # biased estimator -> contracts
    return np.abs(BIAS).mean(1), NOISE.mean(1)

gens = np.arange(GENS)
bA, nA = chains(c0=0.6, delta=0.0, seed=0)     # (a) single biased source
bB, nB = chains(c0=0.0, delta=0.04, seed=1)    # (b) per-generation artifact

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
C_B, C_N, C_T = "#c0504d", "#1f4e79", "#7f7f7f"

for a, (b, n, ttl) in zip(ax, [(bA, nA, "(a) single biased source:\nnoise dies, bias survives"),
                                (bB, nB, "(b) per-generation artifact:\nbias accumulates ~linearly")]):
    a.plot(gens, b, "o-", color=C_B, label="systematic bias $|b|$")
    a.plot(gens, n, "s-", color=C_N, label="noise std $\\sigma$ (contracts)")
    a.plot(gens, np.sqrt(b**2 + n**2), "^--", color=C_T, lw=1.3, label="total RMS error")
    a.set_xlabel("generation"); a.set_ylabel("error component"); a.set_title(ttl, fontsize=9.5)
    a.grid(alpha=0.15)
ax[0].legend(fontsize=8.3, loc="center right")
ax[1].plot(gens, 0.04*gens, ":", color=C_B, lw=1, alpha=0.6, label="linear $k\\cdot\\delta$")
ax[1].legend(fontsize=8.3, loc="upper left")

fig.suptitle("E6: model collapse decomposed --- recursive generation contracts noise but retains or accumulates systematic bias",
             fontsize=10.5, y=1.02)
fig.tight_layout(); fig.savefig("fig10_collapse.pdf", bbox_inches="tight"); plt.close(fig)

print("(a) single source:   noise %.3f -> %.3f (%.1fx contraction); bias %.3f -> %.3f (retained)"
      % (nA[0], nA[-1], nA[0]/nA[-1], bA[0], bA[-1]))
print("(b) per-gen artifact: noise %.3f -> %.3f; bias %.3f -> %.3f (linear ref k*delta=%.3f)"
      % (nB[0], nB[-1], bB[0], bB[-1], 0.04*(GENS-1)))
print("=> collapse removes the noise; the systematic bias survives or grows. wrote fig10_collapse.pdf")
