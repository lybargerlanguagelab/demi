"""Synthetic example from Section 3.2 of the DEMI paper, run on both versions.

This script needs no database and no All of Us access. It builds the paper's
small synthetic knowledgebase as an in-memory pairwise table, then:

1. runs the article's algorithm (paper/demi_v17.py) and checks every
   intermediate value against the numbers printed in the article; and
2. runs the current algorithm (demi.py, version 23) on the same table and
   prints its values next to the article's, without expecting them to match.

Population of 1,000 patients; outcome Y present in 200; feature Z1 present in
400 and Z2 in 300; 150 patients have both; Z1 precedes Z2 in 140 of them and
follows it in 10. Counts against Y: Z1 (a, b, c, d) = (120, 280, 80, 520);
Z2 (a, b, c, d) = (90, 210, 110, 590).

Published values (rounded to three decimals):
  odds ratios 2.786 and 2.299; total effects T = 1.024 and 0.832
  excess probability 0.075; direct effects D = 0.962 and 0.832
  correction factor Lambda = 2.288; posterior odds 0.657; probability 0.397

Run from the repository root:
    python examples/synthetic_example.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import demi as demi_current  # noqa: E402  (version 23)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demi_paper = _load(os.path.join(ROOT, "paper", "demi_v17.py"), "demi_v17")

Y, Z1, Z2 = 0, 1, 2
NAMES = {Y: "Y", Z1: "Z1", Z2: "Z2"}


def build_edges(or_function):
    """Pairwise table for the synthetic knowledgebase.

    or_function(a, b, c, d) supplies the adjusted odds ratio for the
    feature-outcome rows (each version has its own rule).
    """
    feature_rows = {
        Z1: dict(a=120, b=280, c=80, d=520),
        Z2: dict(a=90, b=210, c=110, d=590),
    }
    rows = []
    for z, k in feature_rows.items():
        a, b, c, d = k["a"], k["b"], k["c"], k["d"]
        odds_ratio = or_function(a, b, c, d)
        rows.append({
            "target_concept_code_int": Y, "concept_code_int": z,
            "n_code_target": a, "n_code_no_target": b,
            "n_target_no_code": c, "n_no_code_no_target": d,
            "n_code": a + b, "n_target": a + c, "n_no_target": b + d,
            "n_code_before_target": 0, "n_target_before_code": 0,
            "total_effects": odds_ratio, "total_effects_log": math.log(odds_ratio),
        })
    # Feature-feature rows, both directions: co-occurrence and precedence.
    both, z1_only, z2_only, neither = 150, 250, 150, 450
    z1_before_z2, z2_before_z1 = 140, 10
    rows.append({
        "target_concept_code_int": Z2, "concept_code_int": Z1,
        "n_code_target": both, "n_code_no_target": z1_only,
        "n_target_no_code": z2_only, "n_no_code_no_target": neither,
        "n_code": 400, "n_target": 300, "n_no_target": 700,
        "n_code_before_target": z1_before_z2, "n_target_before_code": z2_before_z1,
        "total_effects": np.nan, "total_effects_log": np.nan,
    })
    rows.append({
        "target_concept_code_int": Z1, "concept_code_int": Z2,
        "n_code_target": both, "n_code_no_target": z2_only,
        "n_target_no_code": z1_only, "n_no_code_no_target": neither,
        "n_code": 300, "n_target": 400, "n_no_target": 600,
        "n_code_before_target": z2_before_z1, "n_target_before_code": z1_before_z2,
        "total_effects": np.nan, "total_effects_log": np.nan,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. The article's algorithm (version 17), checked against the published values
# ---------------------------------------------------------------------------
print("== Article's algorithm (paper/demi_v17.py) against Section 3.2 ==")
edges17 = build_edges(demi_paper.compute_or_v17_scalar)
r17 = demi_paper.run_demi(
    edges17, force_outcome=Y, db_path=None,
    use_code_group_mapping=False, use_db_temporal_order=False,
    lambda_min_count=15, epv=5, user_cap=10,
)
order = [NAMES[e] for e in r17["temporal_order"]]
T, D = r17["total_log_or"], r17["direct_log_or"]
p_z2 = float(r17["predictor_prevalence"].loc[Z2])
weight = float(r17["weights"][Z1].loc[Z2])
Lambda = float(r17["correction_factor"])
prior_odds = float(r17["prior_odds"])
posterior_odds = prior_odds * math.exp(float(D.loc[Z1]) + float(D.loc[Z2])) / Lambda
p17 = float(r17["predict_proba"]({Z1: 1, Z2: 1}))

checks = [
    ("temporal order", " -> ".join(order), "Z1 -> Z2 -> Y", None),
    ("odds ratio Z1", float(r17["total_or"].loc[Z1]), 2.786, 1e-3),
    ("odds ratio Z2", float(r17["total_or"].loc[Z2]), 2.299, 1e-3),
    ("total effect T Z1", float(T.loc[Z1]), 1.024, 1e-3),
    ("total effect T Z2", float(T.loc[Z2]), 0.832, 1e-3),
    ("P(Z2 | Z1)", weight, 0.375, 1e-3),
    ("prevalence of Z2", p_z2, 0.300, 1e-3),
    ("excess probability", weight - p_z2, 0.075, 1e-3),
    ("direct effect D Z1", float(D.loc[Z1]), 0.962, 1e-3),
    ("direct effect D Z2", float(D.loc[Z2]), 0.832, 1e-3),
    ("prior odds", prior_odds, 0.25, 1e-9),
    ("correction factor Lambda", Lambda, 2.288, 1e-3),
    ("posterior odds", posterior_odds, 0.657, 1e-3),
    ("predicted probability", p17, 0.397, 1e-3),
]
failures = 0
for label, got, expected, tol in checks:
    ok = (got == expected) if tol is None else abs(got - expected) <= tol
    shown = got if tol is None else f"{got:.4f}"
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {label:26s} got {shown}  expected {expected}")
if failures:
    print(f"\n{failures} check(s) differ from the published values.")
    sys.exit(1)
print("\nAll values match Section 3.2 of the article.")

# ---------------------------------------------------------------------------
# 2. The current algorithm (version 23) on the same table, for comparison
# ---------------------------------------------------------------------------
print("\n== Current algorithm (demi.py, version 23) on the same table ==")
print("Version 23 scores differently (see CHANGELOG.md); no match is expected.")
edges23 = build_edges(demi_current.compute_or_v23_scalar)
r23 = demi_current.run_demi(edges23, force_outcome=Y, require_global_temporal_order=False)
eff = r23["effect_df"].set_index("predictor_int")
p23 = float(r23["predict_proba"]({Z1: 1, Z2: 1}))
rows = [
    ("odds ratio Z1", float(r17["total_or"].loc[Z1]), float(eff.loc[Z1, "or_val"])),
    ("odds ratio Z2", float(r17["total_or"].loc[Z2]), float(eff.loc[Z2, "or_val"])),
    ("direct effect Z1 (v23: log OR when Z1 = 1)", float(D.loc[Z1]), float(eff.loc[Z1, "direct_log_or_z1"])),
    ("direct effect Z2 (v23: log OR when Z2 = 1)", float(D.loc[Z2]), float(eff.loc[Z2, "direct_log_or_z1"])),
    ("predicted probability, both present", p17, p23),
]
print(f"{'':44s} {'v17 (article)':>14s} {'v23':>10s}")
for label, a, b in rows:
    print(f"{label:44s} {a:14.4f} {b:10.4f}")
print(f"\nv23 scoring rule: {r23['scoring_rule']}; probability method: {r23['probability_method']}; "
      f"calibration: {r23['calibration_status']}")
