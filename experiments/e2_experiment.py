"""
E2 for "The Mirror Law": classification under a spurious correlation.
A teacher is trained on a STRONGLY spurious set and acquires an emergent
systematic bias (over-reliance on an easy shortcut feature). Students are then
trained on a SEPARATE, weakly-correlated pool whose true labels would teach the
robust feature. We test whether the teacher's bias transfers via distillation
(Regime C) while symmetric label noise does not (Regime B), and whether
agreement-with-the-teacher conceals the inherited bias that OOD accuracy reveals.
Hand-coded binary MLP, CPU, seconds, seeded.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

d_core, d_spur = 6, 6
mu_core, sig_core = 1.0, 1.0
mu_spur, sig_spur = 2.6, 0.45        # spurious: clean, high-margin shortcut (attractive)
wc = rng.standard_normal(d_core); wc /= np.linalg.norm(wc)

def make_data(n, rho, seed):
    r = np.random.default_rng(seed)
    y = (r.random(n) < 0.5).astype(float)
    s = np.where(r.random(n) < rho, y, 1.0 - y)
    core = r.standard_normal((n, d_core)) * sig_core + mu_core * (2 * y - 1)[:, None] * wc
    spur = r.standard_normal((n, d_spur)) * sig_spur + mu_spur * (2 * s - 1)[:, None]
    return np.concatenate([core, spur], axis=1), y, s

Xteach, yteach, _ = make_data(6000, 0.97, 1)   # teacher set: very strong shortcut
Xpool,  ypool,  spool = make_data(6000, 0.65, 2)  # student pool: weak correlation
Xin,    yin,    sin  = make_data(3000, 0.65, 3)  # in-distribution test (pool dist)
Xood,   yood,   sood = make_data(3000, 0.50, 4)  # OOD test (spurious uninformative)
allX = np.concatenate([Xteach, Xpool], 0); mu = allX.mean(0); sd = allX.std(0) + 1e-8
Xteach, Xpool, Xin, Xood = [(Z - mu) / sd for Z in (Xteach, Xpool, Xin, Xood)]

H = 64
def init(seed, hin):
    r = np.random.default_rng(seed)
    return [r.standard_normal((Xpool.shape[1], hin)) / np.sqrt(Xpool.shape[1]), np.zeros(hin),
            r.standard_normal((hin, 1)) / np.sqrt(hin), np.zeros(1)]

def fwd(p, X):
    W1, b1, W2, b2 = p
    h1 = np.tanh(X @ W1 + b1)
    logit = (h1 @ W2 + b2)[:, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))), h1

def train(X, target, seed, steps=4000, lr=0.05, mom=0.9, hin=H):
    p = init(seed, hin); vel = [np.zeros_like(a) for a in p]; N = X.shape[0]
    for t in range(steps):
        pr, h1 = fwd(p, X); dlogit = (pr - target) / N
        W1, b1, W2, b2 = p
        dW2 = h1.T @ dlogit[:, None]; db2 = np.array([dlogit.sum()])
        dz1 = (dlogit[:, None] @ W2.T) * (1 - h1 ** 2)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        for i, g in enumerate([dW1, db1, dW2, db2]):
            vel[i] = mom * vel[i] - lr * g; p[i] = p[i] + vel[i]
    return p

acc = lambda p, X, y: float(np.mean((fwd(p, X)[0] > 0.5) == (y > 0.5)))
def minority_acc(p, X, y, s):
    m = (s != y); pr = fwd(p, X)[0]; return float(np.mean((pr[m] > 0.5) == (y[m] > 0.5)))
agree = lambda pa, pb, X: float(np.mean((fwd(pa, X)[0] > 0.5) == (fwd(pb, X)[0] > 0.5)))

# Teacher: low capacity + early stop -> locks onto the easy shortcut (emergent bias)
teacher = train(Xteach, yteach, seed=101, steps=1200, hin=8)
teach_soft = fwd(teacher, Xpool)[0]

flip = rng.random(Xpool.shape[0]) < 0.20
ynoisy = np.where(flip, 1.0 - ypool, ypool)
student_distill = train(Xpool, teach_soft, seed=202)
student_clean   = train(Xpool, ypool,      seed=203)
student_noisy   = train(Xpool, ynoisy,     seed=204)

def logistic_oracle(cols, steps=3000, lr=0.1):
    Xc = Xpool[:, cols]; w = np.zeros(Xc.shape[1]); b = 0.0
    for _ in range(steps):
        pr = 1 / (1 + np.exp(-np.clip(Xc @ w + b, -30, 30))); g = pr - ypool
        w -= lr * (Xc.T @ g) / len(ypool); b -= lr * g.mean()
    return w, b
oracc = lambda cols, w, b, X, y: float(np.mean(((1/(1+np.exp(-np.clip(X[:, cols] @ w + b, -30, 30)))) > 0.5) == (y > 0.5)))
core_cols = list(range(d_core)); spur_cols = list(range(d_core, d_core + d_spur))
wc_, bc_ = logistic_oracle(core_cols); ws_, bs_ = logistic_oracle(spur_cols)

models = {"Teacher": teacher, "Student\n(distilled)": student_distill,
          "Student\n(clean labels)": student_clean, "Student\n(symmetric noise)": student_noisy}
print("model                         in-dist   OOD-all   OOD-minority   agree-teacher(OOD)")
rows = {}
for name, p in models.items():
    r = (acc(p, Xin, yin), acc(p, Xood, yood), minority_acc(p, Xood, yood, sood), agree(p, teacher, Xood))
    rows[name] = r
    print(f"{name.replace(chr(10),' '):28s}  {r[0]:6.3f}   {r[1]:6.3f}    {r[2]:6.3f}        {r[3]:6.3f}")
print("\nOracles:  core-only OOD %.3f | spur-only OOD %.3f" % (oracc(core_cols, wc_, bc_, Xood, yood), oracc(spur_cols, ws_, bs_, Xood, yood)))

labels = list(models.keys())
in_d = [rows[k][0] for k in labels]; ood_d = [rows[k][1] for k in labels]; min_d = [rows[k][2] for k in labels]
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(9.2, 4.0))
bars = [ax.bar(x - w, in_d, w, label="in-distribution acc.", color="#9bb8d3"),
        ax.bar(x, ood_d, w, label="OOD acc. (spurious broken)", color="#1f4e79"),
        ax.bar(x + w, min_d, w, label="OOD minority-group acc.", color="#c0392b")]
ax.axhline(0.5, color="#888", ls=":", lw=1.0); ax.text(len(labels)-0.55, 0.515, "chance", color="#666", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
ax.set_title("E2: an emergent teacher bias transfers under distillation, not under symmetric noise", fontsize=10.5)
ax.legend(loc="lower center", ncol=3, fontsize=8.5, framealpha=0.9)
for bb in bars:
    for r in bb:
        ax.annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()), ha="center", va="bottom", fontsize=7.0)
fig.tight_layout(); fig.savefig("fig3_spurious.pdf", bbox_inches="tight"); plt.close(fig)
print("Saved fig3_spurious.pdf")
