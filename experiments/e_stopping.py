"""
E7 (mitigation, cont.): an epsilon-stopping rule for decorrelated ensembling, and
how many teachers K are needed -- with the ground-truth-free estimator and its
fundamental limitation (the correlated-bias floor is invisible to disagreement).

Model (linear, Regime C): teacher k clones bias b_k with energy E||b_k||^2 = beta^2
and pairwise correlation rho = E<b_i,b_j>/beta^2. Construct b_k = sqrt(rho)*u +
sqrt(1-rho)*v_k (u shared unit dir, v_k independent unit dirs; near-orthogonal in
high d => ||b_k||^2 ~ 1, <b_i,b_j> ~ rho).

Predictions verified:
  R*_ens(K) = beta^2 [ rho + (1-rho)/K ]        (ensemble true risk)
  floor     = rho*beta^2                          (irreducible; survives K->inf)
  disagreement/2 -> beta^2 (1-rho)               (OBSERVABLE; estimates reducible part only)
  K*(eps) (decorrelated) = ceil(beta^2/eps)
"""
import numpy as np, math
d, N = 60, 6000        # high d so u, v_k are near-orthogonal
KS = list(range(1, 13))
RHOS = [0.0, 0.1, 0.25, 0.5]
S = 12
EPS = 0.10

def run_seed(seed, rho):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, d))
    unit = lambda z: z/np.linalg.norm(z)
    u = unit(r.standard_normal(d))
    V = [unit(r.standard_normal(d)) for _ in range(max(KS))]
    B = [math.sqrt(rho)*u + math.sqrt(1-rho)*v for v in V]     # bias k (student clones it: f_k = f* + b_k)
    # outputs relative to truth: f_k - f* = X @ b_k
    F = [X @ b for b in B]
    Rstar = {}; disag = {}
    for K in KS:
        bbar = np.mean(B[:K], 0)
        Rstar[K] = float(np.mean((X @ bbar)**2))               # true risk (needs ground truth)
        if K >= 2:
            ds = [float(np.mean((F[i]-F[j])**2)) for i in range(K) for j in range(i+1, K)]
            disag[K] = float(np.mean(ds))                      # OBSERVABLE mean pairwise disagreement
    return Rstar, disag

print(f"d={d}, N={N}, S={S} seeds, target eps={EPS}\n")
print(f"{'rho':>5} {'floor=rho*b^2':>12} | {'K':>2} {'R*_ens (true)':>22} {'formula':>9} {'disag/2 (obs)':>16}")
for rho in RHOS:
    RS = {K: [] for K in KS}; DS = {K: [] for K in KS if K >= 2}
    for s in range(S):
        rr, dd = run_seed(s, rho)
        for K in KS: RS[K].append(rr[K])
        for K in dd: DS[K].append(dd[K])
    # K* analytic (decorrelated) and the floor
    floor = rho  # beta^2=1
    for K in [1, 2, 4, 8, 12]:
        m = np.mean(RS[K]); formula = rho + (1-rho)/K
        dh = f"{np.mean(DS[K])/2:.3f}" if K in DS else "  -- "
        print(f"{rho:>5} {floor:>12.3f} | {K:>2} {m:>14.4f}        {formula:>9.4f} {dh:>16}")
    # epsilon-stopping: true K needed vs disagreement-based estimate
    trueK = next((K for K in KS if np.mean(RS[K]) <= EPS), None)
    betahat2 = np.mean(DS[2])/2 if 2 in DS else float('nan')   # estimate beta^2(1-rho) from K=2 disagreement
    Khat = math.ceil(betahat2/EPS) if betahat2 > 0 else None
    Kstar_dec = math.ceil(1.0/EPS)  # if it were fully decorrelated, beta^2=1
    achievable = EPS > floor + 1e-9
    print(f"      -> eps={EPS}: true K needed = {trueK if trueK else '>%d (floor %.2f)'%(max(KS),floor)};"
          f"  disagreement-based K_hat = {Khat} (cancels reducible part to eps,"
          f" but TRUE inherited bias then = floor+eps = {floor+EPS:.2f});"
          f" eps achievable? {achievable}\n")
print("Takeaways: R*_ens matches rho+(1-rho)/K; disag/2 -> 1-rho (sees only the reducible bias);")
print("the floor rho is invisible to disagreement, so K_hat over-certifies when rho>0 -> grade teachers to get beta^2.")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
COL = {0.0:"#1f4e79", 0.1:"#2e86c1", 0.25:"#e08e0b", 0.5:"#c0392b"}
fig, ax = plt.subplots(figsize=(5.6, 3.8))
Ks = np.array(KS)
for rho in RHOS:
    means = []
    for K in KS:
        vals = [run_seed(s, rho)[0][K] for s in range(S)]
        means.append(np.mean(vals))
    ax.plot(Ks, means, "-o", color=COL[rho], lw=2, ms=4, label=fr"$\rho={rho}$")
    ax.axhline(rho, color=COL[rho], ls=":", lw=1.0, alpha=0.7)   # floor rho*beta^2
ax.axhline(EPS, color="k", ls="--", lw=1.1); ax.text(11.2, EPS+0.012, r"target $\epsilon$", fontsize=8)
ax.text(11.4, 0.5+0.012, r"floors $\rho\|b\|^2$", color="#c0392b", fontsize=8, ha="right")
ax.set_xlabel(r"number of decorrelated teachers $K$"); ax.set_ylabel(r"ensemble true risk $R^\star_\mathrm{ens}$")
ax.set_title(r"$\epsilon$-stopping: $R^\star_\mathrm{ens}=\|b\|^2[\rho+(1-\rho)/K]$", fontsize=10.5)
ax.legend(fontsize=8.5, title="bias correlation"); ax.grid(alpha=0.15); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig("fig12_stopping.pdf", bbox_inches="tight"); plt.close(fig)
print("\nSaved fig12_stopping.pdf")
