"""
Multi-seed robustness for E1 and E2 (addresses 'seed-to-seed variation unreported').
Checkpointed: appends each seed's metrics to e1e2_seeds.jsonl so it can run in chunks.
Usage:
  python e1e2_multiseed.py 0 4      # run seeds [0,4) -> append to jsonl
  python e1e2_multiseed.py summarize
Settings are moderately reduced from the single-run scripts for speed; the point is
variance characterization, and the means track the single-seed numbers.
"""
import sys, json, os, numpy as np
JSONL = "e1e2_seeds.jsonl"

def e1_linear(seed, d=20, N=4000, T=150, lr=0.3):
    r = np.random.default_rng(seed)
    X = r.standard_normal((N, d)); wstar = r.standard_normal(d); wstar *= 2.0/np.linalg.norm(wstar)
    def run(bn, sigma, sd):
        rr = np.random.default_rng(sd)
        beta = rr.standard_normal(d); beta = beta*(bn/np.linalg.norm(beta)) if bn > 0 else beta*0.0
        g = X @ (wstar + beta) + sigma*rr.standard_normal(N); w = np.zeros(d)
        for _ in range(T): w -= lr*(2.0/N)*(X.T @ (X @ w - g))
        return float(np.mean((X@w-g)**2)), float(np.mean((X@w-X@wstar)**2)), beta, w
    _, RsB, _, _ = run(0.0, 1.0, seed*10+2)
    _, RsC, bC, _ = run(1.0, 0.0, seed*10+3)
    bn = np.array([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]); be = bn**2
    Rstar = np.array([run(b,0.0,seed*131+i)[1] for i,b in enumerate(bn)])
    slope = float(np.dot(be,Rstar)/np.dot(be,be))
    _,_,b1,w1 = run(1.0,0.0,seed*7+21); _,_,b2,w2 = run(1.0,0.0,seed*7+22)
    dis = float(np.sqrt(np.mean((X@w1-X@w2)**2)))
    return [RsB, RsC/float(np.sum(bC**2)), slope, dis]

def e1_mlp(seed, d=10, H=64, N=1500, STEPS=2500):
    r = np.random.default_rng(seed*977+1); X = r.standard_normal((N, d))
    def mk(s): rr=np.random.default_rng(s); return [rr.standard_normal((d,H))/np.sqrt(d),np.zeros(H),rr.standard_normal((H,1))/np.sqrt(H),np.zeros(1)]
    def fwd(p,Xin): W1,b1,W2,b2=p; h=np.tanh(Xin@W1+b1); return (h@W2+b2)[:,0],h
    pstar=mk(seed*31+7); ystar,_=fwd(pstar,X)
    w0=r.standard_normal(d); bias=1.3/(1.0+np.exp(-(X@w0))); g=ystar+bias; be=float(np.mean(bias**2))
    p=mk(seed*53+13); vel=[np.zeros_like(a) for a in p]
    for t in range(STEPS):
        out,h=fwd(p,X); dout=(2.0/N)*(out-g); W1,b1,W2,b2=p
        dW2=h.T@dout[:,None]; db2=np.array([dout.sum()]); dz1=(dout[:,None]@W2.T)*(1-h**2)
        dW1=X.T@dz1; db1=dz1.sum(0)
        for i,gr in enumerate([dW1,db1,dW2,db2]): vel[i]=0.9*vel[i]-0.03*gr; p[i]=p[i]+vel[i]
    out,_=fwd(p,X); return [float(np.mean((out-g)**2)), float(np.mean((out-ystar)**2))/be]

def e2(seed, steps_student=2200):
    r = np.random.default_rng(seed*89+5); dc, ds = 6, 6
    wc = r.standard_normal(dc); wc /= np.linalg.norm(wc)
    def make(n, rho, sd):
        rr=np.random.default_rng(sd); y=(rr.random(n)<0.5).astype(float); s=np.where(rr.random(n)<rho,y,1-y)
        core=rr.standard_normal((n,dc))*1.0+1.0*(2*y-1)[:,None]*wc
        spur=rr.standard_normal((n,ds))*0.45+2.6*(2*s-1)[:,None]
        return np.concatenate([core,spur],1), y, s
    Xt,yt,_=make(4000,0.97,seed*4+1); Xp,yp,sp=make(4000,0.65,seed*4+2)
    Xi,yi,si=make(3000,0.65,seed*4+3); Xo,yo,so=make(3000,0.50,seed*4+4)
    aX=np.concatenate([Xt,Xp],0); mu=aX.mean(0); sd_=aX.std(0)+1e-8
    Xt,Xp,Xi,Xo=[(Z-mu)/sd_ for Z in (Xt,Xp,Xi,Xo)]
    def init(s,h): rr=np.random.default_rng(s); return [rr.standard_normal((Xp.shape[1],h))/np.sqrt(Xp.shape[1]),np.zeros(h),rr.standard_normal((h,1))/np.sqrt(h),np.zeros(1)]
    def fwd(p,X): W1,b1,W2,b2=p; h=np.tanh(X@W1+b1); return 1/(1+np.exp(-np.clip((h@W2+b2)[:,0],-30,30))),h
    def train(X,tg,s,steps,lr=0.05,h=64):
        p=init(s,h); vel=[np.zeros_like(a) for a in p]; N=X.shape[0]
        for _ in range(steps):
            pr,hh=fwd(p,X); dl=(pr-tg)/N; W1,b1,W2,b2=p
            dW2=hh.T@dl[:,None]; db2=np.array([dl.sum()]); dz1=(dl[:,None]@W2.T)*(1-hh**2)
            dW1=X.T@dz1; db1=dz1.sum(0)
            for i,g in enumerate([dW1,db1,dW2,db2]): vel[i]=0.9*vel[i]-lr*g; p[i]=p[i]+vel[i]
        return p
    accf=lambda p,X,y: float(np.mean((fwd(p,X)[0]>0.5)==(y>0.5)))
    minf=lambda p,X,y,s:(lambda m:float(np.mean((fwd(p,X)[0][m]>0.5)==(y[m]>0.5))))(s!=y)
    agf=lambda pa,pb,X: float(np.mean((fwd(pa,X)[0]>0.5)==(fwd(pb,X)[0]>0.5)))
    teach=train(Xt,yt,seed*4+101,900,h=8); ts=fwd(teach,Xp)[0]
    fl=np.random.default_rng(seed*4+9).random(Xp.shape[0])<0.20; yn=np.where(fl,1-yp,yp)
    sd_d=train(Xp,ts,seed*4+202,steps_student); sd_c=train(Xp,yp,seed*4+203,steps_student); sd_n=train(Xp,yn,seed*4+204,steps_student)
    out={}
    for nm,p in [("teacher",teach),("distilled",sd_d),("clean",sd_c),("noisy",sd_n)]:
        out[nm]=[accf(p,Xo,yo), minf(p,Xo,yo,so), agf(p,teach,Xo)]
    return out

def tcrit(df):
    table={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
           10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,19:2.093,29:2.045}
    return table.get(df,1.96)

def summarize():
    rows=[json.loads(l) for l in open(JSONL)]
    seeds=sorted({r["seed"] for r in rows}); rows=[next(r for r in rows if r["seed"]==s) for s in seeds]
    n=len(rows); print(f"=== Multi-seed robustness (S={n} seeds: {seeds}) ===\n")
    def summ(name, xs, theory=None):
        xs=np.asarray(xs,float); m=xs.mean(); sd=xs.std(ddof=1); half=tcrit(n-1)*sd/np.sqrt(n)
        th=f"   (theory {theory})" if theory is not None else ""
        print(f"  {name:36s} {m:8.4f} ± {sd:7.4f}   95% CI [{m-half:.4f}, {m+half:.4f}]{th}")
    L=np.array([r["e1L"] for r in rows]); M=np.array([r["e1M"] for r in rows])
    print("--- E1 linear ---")
    summ("Regime B final R* (noise washes)", L[:,0], "~0")
    summ("Regime C  R*/||b||^2 (bias cloned)", L[:,1], 1.0)
    summ("calibration slope", L[:,2], 1.0)
    summ("detection RMS disagreement", L[:,3], "~1.4")
    print("--- E1 MLP confirmation ---")
    summ("final R_g", M[:,0], "~0"); summ("R*/||b||^2 plateau", M[:,1], 1.0)
    print("--- E2 (OOD acc / OOD-minority / agree-teacher) ---")
    for k in ["teacher","distilled","clean","noisy"]:
        A=np.array([r["e2"][k] for r in rows])
        summ(f"{k}: OOD acc", A[:,0]); summ(f"{k}: OOD-minority acc", A[:,1]); summ(f"{k}: agree-teacher", A[:,2])

if __name__ == "__main__":
    if sys.argv[1] == "summarize":
        summarize()
    else:
        a, b = int(sys.argv[1]), int(sys.argv[2])
        done = set()
        if os.path.exists(JSONL): done = {json.loads(l)["seed"] for l in open(JSONL)}
        for s in range(a, b):
            if s in done: print(f"seed {s} already done", flush=True); continue
            rec = {"seed": s, "e1L": e1_linear(s), "e1M": e1_mlp(s), "e2": e2(s)}
            with open(JSONL, "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"seed {s} done", flush=True)
