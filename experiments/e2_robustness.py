"""E2 robustness: sweep the teacher's spurious-correlation strength rho_teacher
and show the distilled student tracks the teacher's bias (OOD accuracy declines
with it) while clean-label and symmetric-noise students stay robust -- i.e. the
effect is smooth and monotone, not a knife-edge of one setting."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d_core, d_spur = 6, 6
mu_core, sig_core, mu_spur, sig_spur = 1.0, 1.0, 2.6, 0.45
wc = rng.standard_normal(d_core); wc /= np.linalg.norm(wc)
def make(n, rho, seed):
    r = np.random.default_rng(seed); y = (r.random(n) < 0.5).astype(float)
    s = np.where(r.random(n) < rho, y, 1.0 - y)
    core = r.standard_normal((n, d_core)) * sig_core + mu_core * (2*y-1)[:, None] * wc
    spur = r.standard_normal((n, d_spur)) * sig_spur + mu_spur * (2*s-1)[:, None]
    return np.concatenate([core, spur], 1), y, s

N = 4000
Xpool, ypool, _ = make(N, 0.65, 2)
Xood, yood, _ = make(3000, 0.50, 4)
# shared normalization fit later per-teacher-set union; use pool stats (teacher applied to pool)
def fwd(p, X):
    W1,b1,W2,b2 = p; h1 = np.tanh(X@W1+b1); lo=(h1@W2+b2)[:,0]; return 1/(1+np.exp(-np.clip(lo,-30,30))), h1
def init(seed, hin, din):
    r = np.random.default_rng(seed)
    return [r.standard_normal((din,hin))/np.sqrt(din), np.zeros(hin), r.standard_normal((hin,1))/np.sqrt(hin), np.zeros(1)]
def train(X, t, seed, steps, hin):
    p=init(seed,hin,X.shape[1]); vel=[np.zeros_like(a) for a in p]; n=X.shape[0]
    for _ in range(steps):
        pr,h1=fwd(p,X); dl=(pr-t)/n; W1,b1,W2,b2=p
        dW2=h1.T@dl[:,None]; db2=np.array([dl.sum()]); dz1=(dl[:,None]@W2.T)*(1-h1**2)
        dW1=X.T@dz1; db1=dz1.sum(0)
        for i,g in enumerate([dW1,db1,dW2,db2]): vel[i]=0.9*vel[i]-0.05*g; p[i]=p[i]+vel[i]
    return p
acc=lambda p,X,y: float(np.mean((fwd(p,X)[0]>0.5)==(y>0.5)))

rhos = [0.70, 0.76, 0.82, 0.88, 0.94, 0.99]
res = {k: [] for k in ["teacher","distill","clean","noisy"]}
for rt in rhos:
    Xt, yt, _ = make(N, rt, 1)
    mu = np.concatenate([Xt,Xpool],0).mean(0); sd = np.concatenate([Xt,Xpool],0).std(0)+1e-8
    Xtn, Xpn, Xon = (Xt-mu)/sd, (Xpool-mu)/sd, (Xood-mu)/sd
    teacher = train(Xtn, yt, 101, 1200, 8)
    soft = fwd(teacher, Xpn)[0]
    flip = rng.random(N) < 0.20; yn = np.where(flip, 1-ypool, ypool)
    dst = train(Xpn, soft, 202, 2500, 48)
    cln = train(Xpn, ypool, 203, 2500, 48)
    noi = train(Xpn, yn, 204, 2500, 48)
    res["teacher"].append(acc(teacher, Xon, yood)); res["distill"].append(acc(dst, Xon, yood))
    res["clean"].append(acc(cln, Xon, yood)); res["noisy"].append(acc(noi, Xon, yood))
    print(f"rho_teacher={rt:.2f}  OOD  teacher={res['teacher'][-1]:.3f}  distill={res['distill'][-1]:.3f}  clean={res['clean'][-1]:.3f}  noisy={res['noisy'][-1]:.3f}")

fig, ax = plt.subplots(figsize=(6.4, 4.0))
sty = {"teacher":("#555","o","-","teacher (biased reference)"),
       "distill":("#c0392b","s","--","distilled student"),
       "clean":("#1f4e79","^","-","clean-label student"),
       "noisy":("#2e8b57","D",":","symmetric-noise student")}
for k,(c,m,ls,lab) in sty.items():
    ax.plot(rhos, res[k], color=c, marker=m, ls=ls, lw=1.8, ms=5, label=lab)
ax.axhline(0.5, color="#aaa", ls=":", lw=0.8); ax.text(0.70, 0.515, "chance", color="#888", fontsize=8)
ax.set_xlabel(r"teacher spurious-correlation strength $\rho_\mathrm{teacher}$"); ax.set_ylabel("OOD accuracy")
ax.set_title("E2 robustness: bias transfer scales smoothly with teacher bias", fontsize=10)
ax.set_ylim(0.45, 0.9); ax.legend(fontsize=8.5, loc="center left"); ax.grid(alpha=0.15)
fig.tight_layout(); fig.savefig("fig5_e2_robustness.pdf", bbox_inches="tight"); plt.close(fig)
print("Saved fig5_e2_robustness.pdf")
