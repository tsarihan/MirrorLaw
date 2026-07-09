"""
score_context.py -- turn context_probe completions into the TRUE-effective-context deliverable.

Input: the probe JSONL (from context_probe.py) plus a completions file (JSONL with {"prompt_id"|index, or
matched order, and "completion": "..."}). Scores exact-match per (task, length), then reports the
NEEDLE-vs-MULTI_HOP gap = the model's honest effective context.

Usage:
    python score_context.py --probe ctx.jsonl --completions out.jsonl
    # completions aligned by order to probe, OR each line has "gold"/"completion".
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict


def norm(s):
    m = re.findall(r"-?\d+", str(s))
    return m[-1] if m else str(s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--completions", required=True, help="JSONL aligned by order; each has 'completion'")
    ap.add_argument("--pass-threshold", type=float, default=0.5, help="accuracy to count a cell as 'passing'")
    a = ap.parse_args()
    probe = [json.loads(l) for l in open(a.probe)]
    comp = [json.loads(l) for l in open(a.completions)]
    if len(comp) != len(probe):
        print(f"[warn] {len(comp)} completions vs {len(probe)} probes; scoring the min overlap")
    n = min(len(comp), len(probe))

    cell = defaultdict(lambda: [0, 0])   # (task, length) -> [correct, total]
    for i in range(n):
        p = probe[i]; c = comp[i].get("completion", "")
        ok = norm(c) == norm(p["gold"])
        key = (p["task"], p["length_tokens"])
        cell[key][0] += int(ok); cell[key][1] += 1

    tasks = sorted({t for (t, _) in cell})
    lengths = sorted({L for (_, L) in cell})
    print("Accuracy per (task, length):")
    hdr = f"{'length':>8s} | " + " | ".join(f"{t:>14s}" for t in tasks)
    print(hdr); print("-" * len(hdr))
    acc = {}
    for L in lengths:
        line = f"{L:8d} | "
        for t in tasks:
            cc, tt = cell.get((t, L), [0, 0])
            v = cc / tt if tt else float("nan")
            acc[(t, L)] = v
            line += f"{v:14.3f} | "
        print(line)

    # effective context per task = longest length that still passes
    print(f"\nEffective context (longest length with accuracy >= {a.pass_threshold}):")
    eff = {}
    for t in tasks:
        passing = [L for L in lengths if acc.get((t, L), 0) >= a.pass_threshold]
        eff[t] = max(passing) if passing else 0
        print(f"  {t:16s}: {eff[t]}")
    if "needle" in eff and ("multi_hop" in eff or "variable_track" in eff):
        hard = max(eff.get("multi_hop", 0), eff.get("variable_track", 0))
        print(f"\n>>> BAND-AID GAP: needle passes to {eff['needle']} but reasoning only to {hard}")
        print(f">>> TRUE effective context (reasoning) = {hard};  advertised-ish (needle) = {eff['needle']}")


if __name__ == "__main__":
    main()
