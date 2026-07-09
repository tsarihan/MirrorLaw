"""
E7 (mitigation): does DECORRELATING references reduce inherited bias?
Directly tests the paper's central recommendation. Theory prediction (linear,
realizable, Regime C): a student trained on teacher k clones its bias, w_k -> w*+b_k,
so an ensemble of K students averages the biases: bias_ens = mean_k b_k.
 - DECORRELATED biases (independent unit vectors): ||mean b_k||^2 ~ ||b||^2 / K
   => ensemble true risk falls as 1/K.
 - CORRELATED biases (shared direction): mean b_k = b => no reduction (R* stays ||b||^2).
So ensembling helps ONLY to the extent the references are decorrelated -- exactly
Proposition 4's claim that a decorrelated second mirror is what exposes/cancels bias.
Multi-seed, CPU, seconds.
"""
import numpy as np, json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d, N, T, lr = 20, 4000, 220, 0.3
KS = list(range(1, 9))

def student_w(X, wstar, beta):
    g = X @ (wstar + beta); w = np.zeros(d)
    for _ in range(T): w -= lr*(2.0/N)*(X.T @ (X @ w - g))
    return w

def run_seed(seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, d)); wstar = r.standard_normal(d); wstar *= 2.0/np.linalg.norm(wstar)
    B = [(lambda b: b/np.linalg.norm(b))(r.standard_normal(d)) for _ in range(max(KS))]   # independent unit biases
    w_dec = [student_w(X, wstar, b) for b in B]
    w_cor = student_w(X, wstar, B[0])    # shared-bias student (all correlated students identical)
    f0 = X @ wstar
    dec = {K: float(np.mean((X @ np.mean(w_dec[:K], 0) - f0)**2)) for K in KS}
    cor = {K: float(np.mean((X @ w_cor - f0)**2)) for K in KS}   # ensemble of identical students = itself
    return dec, cor

S = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DEC = {K: [] for K in KS}; COR = {K: [] for K in KS}
for s in range(S):
    dec, cor = run_seed(s)
    for K in KS: DEC[K].append(dec[K]); COR[K].append(cor[K])

def ci(xs):
    xs = np.asarray(xs); m = xs.mean(); sd = xs.std(ddof=1)
    t = {11:2.201,12:2.179,9:2.262}.get(len(xs)-1, 1.96); return m, sd, t*sd/np.sqrt(len(xs))

print(f"=== Mitigation by decorrelated ensembling (S={S} seeds; single-teacher R* = ||b||^2 = 1) ===")
print(f"{'K':>3}  {'decorrelated R*':>22}   {'correlated R*':>20}   {'theory 1/K':>10}")
dec_m=[]; cor_m=[]
for K in KS:
    md, sdd, hd = ci(DEC[K]); mc, sdc, hc = ci(COR[K]); dec_m.append(md); cor_m.append(mc)
    print(f"{K:>3}  {md:7.4f} ± {hd:.4f} (95%CI)   {mc:6.4f} ± {hc:.4f}   {1.0/K:>10.4f}")

fig, ax = plt.subplots(figsize=(5.2, 3.6))
Ks = np.array(KS)
ax.plot(Ks, 1.0/Ks, color="#888", ls=":", lw=1.3, label=r"theory $\|b\|^2/K$ (decorrelated)")
md=[ci(DEC[K]) for K in KS]; mc=[ci(COR[K]) for K in KS]
ax.errorbar(Ks, [m for m,_,h in md], yerr=[h for _,_,h in md], marker="o", color="#1f4e79", lw=2, capsize=3, label="decorrelated teachers")
ax.errorbar(Ks, [m for m,_,h in mc], yerr=[h for _,_,h in mc], marker="s", color="#c0392b", lw=2, capsize=3, label="correlated teachers (shared bias)")
ax.set_xlabel("number of teachers $K$ in the ensemble"); ax.set_ylabel(r"ensemble true risk $R^\star$")
ax.set_title("Decorrelated ensembling cancels inherited bias", fontsize=10.5)
ax.legend(fontsize=8.5); ax.grid(alpha=0.15); ax.set_ylim(0, 1.1)
fig.tight_layout(); fig.savefig("fig11_mitigation.pdf", bbox_inches="tight"); plt.close(fig)
print("\nSaved fig11_mitigation.pdf")
print(f"Single teacher: R*={dec_m[0]:.3f}.  8 decorrelated: R*={dec_m[7]:.4f} ({dec_m[0]/dec_m[7]:.1f}x reduction).  8 correlated: R*={cor_m[7]:.3f} (no reduction).")
