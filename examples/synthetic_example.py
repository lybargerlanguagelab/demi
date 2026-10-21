"""Synthetic example from Section 3.2 of the DEMI paper, run end to end.

This script needs no database and no All of Us access. It builds the paper's
small synthetic knowledgebase as an in-memory pairwise table, runs the DEMI
library on it, and compares every intermediate value with the numbers printed
in the article. Use it to confirm that an installation reproduces the
published computation.

Population of 1,000 patients; outcome Y present in 200; feature Z1 present in
400 and Z2 in 300; 150 patients have both; Z1 precedes Z2 in 140 of them and
follows it in 10. Counts against Y: Z1 (a, b, c, d) = (120, 280, 80, 520);
Z2 (a, b, c, d) = (90, 210, 110, 590).

Expected (as published, rounded to three decimals):
  odds ratios 2.786 and 2.299; total effects T = 1.024 and 0.832
  excess probability 0.075; direct effects D = 0.962 and 0.832
  correction factor Lambda = 2.288; posterior odds 0.657; probability 0.397

Run from the repository root:
    python examples/synthetic_example.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import demi_v17 as demi  # noqa: E402

Y, Z1, Z2 = 0, 1, 2
NAMES = {Y: "Y", Z1: "Z1", Z2: "Z2"}

# Feature-outcome rows: target = Y, code = feature.
# a = feature and outcome, b = feature only, c = outcome only, d = neither.
feature_rows = {
    Z1: dict(a=120, b=280, c=80, d=520),
    Z2: dict(a=90, b=210, c=110, d=590),
}

rows = []
for z, k in feature_rows.items():
    a, b, c, d = k["a"], k["b"], k["c"], k["d"]
    odds_ratio = demi.compute_or_v17_scalar(a, b, c, d)
    rows.append({
        "target_concept_code_int": Y, "concept_code_int": z,
        "n_code_target": a, "n_code_no_target": b,
        "n_target_no_code": c, "n_no_code_no_target": d,
        "n_code": a + b, "n_target": a + c, "n_no_target": b + d,
        "n_code_before_target": 0, "n_target_before_code": 0,
        "total_effects": odds_ratio, "total_effects_log": math.log(odds_ratio),
    })

# Feature-feature rows, both directions, carrying co-occurrence and precedence.
# target = Z2, code = Z1 supplies the mediation weight P(Z2 = 1 | Z1 = 1).
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
edges = pd.DataFrame(rows)

result = demi.run_demi(
    edges,
    force_outcome=Y,
    db_path=None,            # no database: local temporal order, no hierarchy filter
    use_code_group_mapping=False,
    use_db_temporal_order=False,
    lambda_min_count=15,
    epv=5,
    user_cap=10,
)

order = [NAMES[e] for e in result["temporal_order"]]
T = result["total_log_or"]
D = result["direct_log_or"]
p_z2 = float(result["predictor_prevalence"].loc[Z2])
weight = float(result["weights"][Z1].loc[Z2])
excess = weight - p_z2
Lambda = float(result["correction_factor"])
prior_odds = float(result["prior_odds"])
posterior_odds = prior_odds * math.exp(float(D.loc[Z1]) + float(D.loc[Z2])) / Lambda
probability = float(result["predict_proba"]({Z1: 1, Z2: 1}))

checks = [
    ("temporal order", " -> ".join(order), "Z1 -> Z2 -> Y", None),
    ("odds ratio Z1", float(result["total_or"].loc[Z1]), 2.786, 1e-3),
    ("odds ratio Z2", float(result["total_or"].loc[Z2]), 2.299, 1e-3),
    ("total effect T Z1", float(T.loc[Z1]), 1.024, 1e-3),
    ("total effect T Z2", float(T.loc[Z2]), 0.832, 1e-3),
    ("P(Z2 | Z1)", weight, 0.375, 1e-3),
    ("prevalence of Z2", p_z2, 0.300, 1e-3),
    ("excess probability", excess, 0.075, 1e-3),
    ("direct effect D Z1", float(D.loc[Z1]), 0.962, 1e-3),
    ("direct effect D Z2", float(D.loc[Z2]), 0.832, 1e-3),
    ("prior odds", prior_odds, 0.25, 1e-9),
    ("correction factor Lambda", Lambda, 2.288, 1e-3),
    ("posterior odds", posterior_odds, 0.657, 1e-3),
    ("predicted probability", probability, 0.397, 1e-3),
]

failures = 0
for label, got, expected, tol in checks:
    if tol is None:
        ok = got == expected
        shown = got
    else:
        ok = abs(got - expected) <= tol
        shown = f"{got:.4f}"
    failures += 0 if ok else 1
    print(f"{'PASS' if ok else 'FAIL'}  {label:26s} got {shown}  expected {expected}")

if failures:
    print(f"\n{failures} check(s) differ from the published values.")
    sys.exit(1)
print("\nAll values match Section 3.2 of the paper.")
