"""
calibration_r.py -- the real-model calibration ratio r = reference_error / initial_policy_error,
and a check that post-training residual gold error tracks the r^2 contamination floor (paper Sec. 4).
CPU only: reference gold comes from grading each prefs file's `chosen` answers with the gold scorer;
initial policy gold is read from a baseline run's step-0 snapshot; final gold from each run's last step.
Keeps three objects distinct: TRAINING REFERENCE (these prefs), held-out GOLD (gold_score), BIAS PROBE.
"""
import argparse, glob, json, os
from task import gold_score

def ref_gold(path):
    recs = [json.loads(l) for l in open(path)]
    return (sum(gold_score(r["question"], r["chosen"]) for r in recs) / len(recs)) if recs else None

def step0_gold(path):
    return json.load(open(path)).get("history", {}).get("gold_acc", [None])[0]

def final_gold(path):
    g = json.load(open(path)).get("history", {}).get("gold_acc", [])
    return g[-1] if g else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefs-dir", default="prefs")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--baseline-run", default=None)
    args = ap.parse_args()
    cands = ([args.baseline_run] if args.baseline_run else []) + sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    base = next((c for c in cands if c and os.path.exists(c)), None)
    init_gold = step0_gold(base) if base else None
    init_err = (1 - init_gold) if init_gold is not None else None
    print(f"base policy gold (step 0) = {init_gold}  ->  init_err = {init_err}\n")
    print(f"{'condition':>22} {'ref_gold':>9} {'ref_err':>8} {'r=ref/init':>11} {'final_gold':>11} {'resid_err':>10} {'r^2*init':>9}")
    for pf in sorted(glob.glob(os.path.join(args.prefs_dir, "prefs_*.jsonl"))):
        cond = os.path.basename(pf)[len("prefs_"):-len(".jsonl")]
        rg = ref_gold(pf)
        if rg is None or not init_err: continue
        ref_err = 1 - rg; r = ref_err / init_err
        run = os.path.join(args.runs_dir, f"{cond}_seed0.json")
        fg = final_gold(run) if os.path.exists(run) else None
        resid = (1 - fg) if fg is not None else None
        pred = (r ** 2) * init_err
        print(f"{cond:>22} {rg:>9.3f} {ref_err:>8.3f} {r:>11.3f} {str(round(fg,3) if fg is not None else '--'):>11} {str(round(resid,3) if resid is not None else '--'):>10} {pred:>9.3f}")
    print("\n[Analysis] r<1 reference helps, r>1 hurts; residual gold error should track ~ r^2 * init_err (cloned-bias floor).")

if __name__ == "__main__":
    main()
