"""
PatentIQ — Gap-Signal Evaluation Harness
========================================

WHAT THIS VALIDATES
-------------------
The product claim of the examiner-simulator + responder loop is the *gap-signal*:
    "Examiner X (art unit Y) will reject claim 1 on ground G over reference Z,
     but G is the kind that gets reversed on appeal N% of the time -> traverse, don't amend."

That claim only has value if it tracks reality. This harness measures three things
against HELD-OUT real prosecution histories:

  1. EXAMINER-SIM FIDELITY  -- does the sim reproduce the real examiner's office action?
       - ground_match (precision/recall over {101,102,103,112})
       - cited_art_overlap (Jaccard of cited references)
       - rejection_calibration (predicted vs actual rejection rate, per art unit)

  2. GAP PRECISION/RECALL    -- when the system flags a predicted rejection as "reversible,"
       do real outcomes (PTAB reversal OR successful traversal) confirm it?
       Compared against the base rate of reversibility (must beat chance).

  3. RESPONDER CALIBRATION   -- is the responder calibrated or a yes-man?
       - traverse_precision: of cases it chose to traverse, how many actually succeeded
       - sycophancy_rate: of actually-reversible cases, how often it caved and amended
       (The Mirror Law / RedlineBench failure: caving to a rejection that was beatable.)

SPLIT DISCIPLINE
----------------
Cluster the held-out set by (family_id, art_unit); never let an examiner's recent office
actions appear in both adapter-training and this eval. check_split_leakage() enforces it.

USAGE
-----
Replace load_real_cases() / load_predictions() with your data loaders (PTOFFACT real OACTs
+ PTAB/continuation outcomes; examiner-sim + responder outputs). Then: python gap_signal_eval.py
The __main__ below runs on synthetic data so the harness executes out of the box.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from statistics import mean
import random

GROUNDS = ("101", "102", "103", "112")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Rejection:
    ground: str                       # one of GROUNDS
    cited_art: list[str] = field(default_factory=list)


@dataclass
class HeldOutCase:
    app_id: str
    family_id: str
    art_unit: str
    examiner_id: str
    real_rejections: list[Rejection]  # from the real office action
    ptab_reversed: bool               # examiner overturned on appeal
    successful_traversal: bool        # applicant traversed and won WITHOUT amending

    @property
    def real_grounds(self) -> set[str]:
        return {r.ground for r in self.real_rejections}

    @property
    def real_art(self) -> set[str]:
        return {a for r in self.real_rejections for a in r.cited_art}

    @property
    def is_reversible(self) -> bool:   # ground-truth definition of a "beatable" rejection
        return self.ptab_reversed or self.successful_traversal


@dataclass
class SystemPrediction:
    app_id: str
    predicted_rejections: list[Rejection]
    flagged_reversible: bool          # the gap-signal flag
    responder_action: str             # "traverse" | "amend"

    @property
    def pred_grounds(self) -> set[str]:
        return {r.ground for r in self.predicted_rejections}

    @property
    def pred_art(self) -> set[str]:
        return {a for r in self.predicted_rejections for a in r.cited_art}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not (a | b):
        return 0.0
    return len(a & b) / len(a | b)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


# --------------------------------------------------------------------------- #
# 1. Examiner-sim fidelity
# --------------------------------------------------------------------------- #
def ground_match(cases, preds) -> dict:
    p_scores, r_scores = [], []
    for c in cases:
        p = preds[c.app_id]
        tp = len(c.real_grounds & p.pred_grounds)
        fp = len(p.pred_grounds - c.real_grounds)
        fn = len(c.real_grounds - p.pred_grounds)
        pr, rc = _prf(tp, fp, fn)
        p_scores.append(pr)
        r_scores.append(rc)
    return {"ground_precision": mean(p_scores), "ground_recall": mean(r_scores)}


def cited_art_overlap(cases, preds) -> float:
    return mean(_jaccard(c.real_art, preds[c.app_id].pred_art) for c in cases)


def rejection_calibration(cases, preds) -> float:
    """Mean |predicted rejection rate - actual rejection rate| across art units."""
    by_unit: dict[str, list[HeldOutCase]] = {}
    for c in cases:
        by_unit.setdefault(c.art_unit, []).append(c)
    errs = []
    for unit, cs in by_unit.items():
        actual = mean(1.0 if c.real_rejections else 0.0 for c in cs)
        predicted = mean(1.0 if preds[c.app_id].predicted_rejections else 0.0 for c in cs)
        errs.append(abs(predicted - actual))
    return mean(errs)


# --------------------------------------------------------------------------- #
# 2. Gap precision/recall  (the core product signal)
# --------------------------------------------------------------------------- #
def gap_precision_recall(cases, preds) -> dict:
    tp = fp = fn = 0
    for c in cases:
        flagged = preds[c.app_id].flagged_reversible
        if flagged and c.is_reversible:
            tp += 1
        elif flagged and not c.is_reversible:
            fp += 1
        elif (not flagged) and c.is_reversible:
            fn += 1
    precision, recall = _prf(tp, fp, fn)
    base_rate = mean(1.0 if c.is_reversible else 0.0 for c in cases)
    return {"gap_precision": precision, "gap_recall": recall,
            "reversibility_base_rate": base_rate,
            "lift_over_chance": (precision - base_rate)}


# --------------------------------------------------------------------------- #
# 3. Responder calibration (sycophancy check)
# --------------------------------------------------------------------------- #
def responder_calibration(cases, preds) -> dict:
    traversed = [c for c in cases if preds[c.app_id].responder_action == "traverse"]
    traverse_precision = (mean(1.0 if c.is_reversible else 0.0 for c in traversed)
                          if traversed else 0.0)
    reversible = [c for c in cases if c.is_reversible]
    sycophancy_rate = (mean(1.0 if preds[c.app_id].responder_action == "amend" else 0.0
                            for c in reversible) if reversible else 0.0)
    return {"traverse_precision": traverse_precision, "sycophancy_rate": sycophancy_rate}


# --------------------------------------------------------------------------- #
# Split-leakage guard
# --------------------------------------------------------------------------- #
def check_split_leakage(train_examiners: set[str], test_cases) -> list[str]:
    """Return test examiners that also appear in training (should be empty)."""
    return sorted({c.examiner_id for c in test_cases} & train_examiners)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
GATES = {
    "ground_recall": 0.60,
    "cited_art_overlap": 0.30,
    "rejection_calibration_max": 0.15,   # lower is better
    "gap_precision": 0.60,
    "gap_lift_over_chance_min": 0.10,    # must beat base rate by this margin
    "sycophancy_rate_max": 0.30,         # lower is better
}


def evaluate(cases, preds) -> dict:
    gm = ground_match(cases, preds)
    report = {
        **gm,
        "cited_art_overlap": cited_art_overlap(cases, preds),
        "rejection_calibration_err": rejection_calibration(cases, preds),
        **gap_precision_recall(cases, preds),
        **responder_calibration(cases, preds),
    }
    report["gates"] = {
        "ground_recall": report["ground_recall"] >= GATES["ground_recall"],
        "cited_art_overlap": report["cited_art_overlap"] >= GATES["cited_art_overlap"],
        "rejection_calibration": report["rejection_calibration_err"] <= GATES["rejection_calibration_max"],
        "gap_precision": report["gap_precision"] >= GATES["gap_precision"],
        "gap_lift": report["lift_over_chance"] >= GATES["gap_lift_over_chance_min"],
        "responder_not_sycophantic": report["sycophancy_rate"] <= GATES["sycophancy_rate_max"],
    }
    report["PASS"] = all(report["gates"].values())
    return report


def print_report(report: dict) -> None:
    print("=" * 56)
    print("GAP-SIGNAL EVALUATION")
    print("=" * 56)
    order = ["ground_precision", "ground_recall", "cited_art_overlap",
             "rejection_calibration_err", "gap_precision", "gap_recall",
             "reversibility_base_rate", "lift_over_chance",
             "traverse_precision", "sycophancy_rate"]
    for k in order:
        print(f"  {k:28s}: {report[k]:.3f}")
    print("-" * 56)
    for gate, ok in report["gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {gate}")
    print("-" * 56)
    print(f"  OVERALL: {'PASS -> proceed to Phase 3' if report['PASS'] else 'FAIL -> iterate, do NOT scale'}")
    print("=" * 56)


# --------------------------------------------------------------------------- #
# Data loaders (replace with real PatentIQ data)
# --------------------------------------------------------------------------- #
def load_real_cases() -> list[HeldOutCase]:
    raise NotImplementedError("Wire to PTOFFACT real OACTs + PTAB/continuation outcomes.")


def load_predictions() -> dict[str, SystemPrediction]:
    raise NotImplementedError("Wire to examiner-sim + responder outputs for the held-out set.")


# --------------------------------------------------------------------------- #
# Synthetic demo so the harness runs out of the box
# --------------------------------------------------------------------------- #
def _synthetic(n=120, seed=7):
    rng = random.Random(seed)
    cases, preds = [], {}
    for i in range(n):
        app = f"APP{i:04d}"
        unit = rng.choice(["1644", "2126", "3689"])
        examiner = f"EX{rng.randint(1, 30):02d}"
        # real rejection(s)
        real = []
        for g in GROUNDS:
            if rng.random() < {"101": .2, "102": .3, "103": .6, "112": .35}[g]:
                real.append(Rejection(g, [f"R{rng.randint(1,40)}" for _ in range(rng.randint(1,3))]))
        reversible = bool(real) and rng.random() < 0.35
        ptab = reversible and rng.random() < 0.5
        case = HeldOutCase(app, f"F{i//3:03d}", unit, examiner, real, ptab, reversible and not ptab)
        cases.append(case)
        # sim prediction: mostly-faithful with noise
        pred_rej = [Rejection(r.ground, [a for a in r.cited_art if rng.random() < 0.75])
                    for r in real if rng.random() < 0.8]
        if rng.random() < 0.1 and len(pred_rej) < len(GROUNDS):     # occasional spurious ground
            pred_rej.append(Rejection(rng.choice(GROUNDS), [f"R{rng.randint(1,40)}"]))
        # gap flag: correlated with true reversibility but imperfect
        flag = (case.is_reversible and rng.random() < 0.7) or ((not case.is_reversible) and rng.random() < 0.15)
        # responder: mostly traverses reversible ones (calibrated), sometimes caves
        if case.is_reversible:
            action = "traverse" if rng.random() < 0.75 else "amend"
        else:
            action = "amend" if rng.random() < 0.7 else "traverse"
        preds[app] = SystemPrediction(app, pred_rej, flag, action)
    return cases, preds


if __name__ == "__main__":
    cases, preds = _synthetic()
    leak = check_split_leakage(train_examiners=set(), test_cases=cases)
    print(f"split leakage (examiners in both train & test): {leak or 'none'}\n")
    print_report(evaluate(cases, preds))
