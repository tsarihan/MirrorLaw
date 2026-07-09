"""
context_probe.py -- measure a model's TRUE effective context, not its advertised length.

H8: the field multiplies APPARENT context (RoPE-scaling, hybrids, sparse attention) while real capability
lags. Needle-in-a-haystack alone is INSUFFICIENT ("context length alone hurts despite perfect retrieval").
This generator builds three probe families at controlled context lengths so you can measure the gap:

  needle          : single fact retrieval (the easy, often-passed test)
  multi_hop       : answer requires chaining K facts scattered across the context (the honest test)
  variable_track  : track a variable's value through M reassignments (state, not retrieval)

THE DELIVERABLE is the GAP: the length at which `needle` still passes minus the length at which `multi_hop`
/ `variable_track` break. That gap is the band-aid — the honest effective context of the deployed model.

Emits JSONL prompts with gold answers; model-agnostic (score any LLM's completion). numpy/stdlib only.
"""
from __future__ import annotations
import argparse, json, random


FILLER = ("The archive contains many records. Routine maintenance was performed. "
          "Weather that day was unremarkable. The committee met as scheduled. "
          "Supplies were restocked. The report was filed without incident. ")


def _pad(rng, approx_tokens):
    """approx filler to reach a target token count (~1.3 words/token rough)."""
    words = int(approx_tokens * 1.3)
    out = []
    while sum(len(s.split()) for s in out) < words:
        out.append(FILLER)
    return " ".join(out)


def needle_item(rng, length_tokens):
    key = rng.randint(1000, 9999)
    val = rng.randint(1000, 9999)
    fact = f"IMPORTANT: the access code for vault {key} is {val}. "
    pos = rng.random()
    pad = _pad(rng, length_tokens)
    cut = int(len(pad) * pos)
    ctx = pad[:cut] + fact + pad[cut:]
    return {"task": "needle", "length_tokens": length_tokens,
            "prompt": ctx + f"\nQuestion: what is the access code for vault {key}?",
            "gold": str(val)}


def multi_hop_item(rng, length_tokens, hops=3):
    """K facts that must be chained: A->B, B->C, ... ; question asks the end of the chain."""
    ids = [rng.randint(1000, 9999) for _ in range(hops + 1)]
    facts = [f"Record {ids[i]} points to record {ids[i+1]}. " for i in range(hops)]
    rng.shuffle(facts)
    pad = _pad(rng, length_tokens)
    # scatter facts across the padding
    chunks = pad.split(". ")
    for f in facts:
        chunks.insert(rng.randint(0, len(chunks)), f)
    ctx = ". ".join(chunks)
    return {"task": "multi_hop", "length_tokens": length_tokens, "hops": hops,
            "prompt": ctx + f"\nQuestion: following the chain of pointers, which record does {ids[0]} "
                            f"ultimately lead to after {hops} hops?",
            "gold": str(ids[-1])}


def variable_track_item(rng, length_tokens, updates=5):
    """A variable is reassigned `updates` times through the context; report its FINAL value."""
    name = rng.choice(["counter", "balance", "level", "score"])
    vals = [rng.randint(10, 99) for _ in range(updates)]
    stmts = [f"Set {name} to {v}. " for v in vals]
    pad = _pad(rng, length_tokens)
    chunks = pad.split(". ")
    # insert updates IN ORDER at increasing positions (order matters for the final value)
    positions = sorted(rng.sample(range(len(chunks)), min(updates, len(chunks))))
    for p, s in zip(positions, stmts):
        chunks[p] = s + chunks[p]
    ctx = ". ".join(chunks)
    return {"task": "variable_track", "length_tokens": length_tokens, "updates": updates,
            "prompt": ctx + f"\nQuestion: what is the final value of {name}?",
            "gold": str(vals[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./context_probe.jsonl")
    ap.add_argument("--lengths", default="1000,4000,16000,64000,256000",
                    help="approx context lengths (tokens) to probe")
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    items = []
    for L in lengths:
        for _ in range(a.per_cell):
            items.append(needle_item(rng, L))
            items.append(multi_hop_item(rng, L, a.hops))
            items.append(variable_track_item(rng, L, a.updates))
    with open(a.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"[wrote] {a.out}  ({len(items)} items: needle/multi_hop/variable_track × {lengths})")
    print("SCORE: exact-match the model's completion vs gold, per (task, length).")
    print("DELIVERABLE: plot accuracy vs length per task. The length where needle still passes but")
    print("             multi_hop/variable_track break = the model's TRUE effective context (the band-aid gap).")


if __name__ == "__main__":
    main()
