# DEMI v17: the algorithm code that accompanied the manuscript of Alemi et al.
# (2026), doi:10.1016/j.jbi.2026.105111, through peer review as part of its
# supplementary Appendix. The code was removed from the supplement before
# publication so that it is distributed only here, under the license in this
# repository; the published Appendix contains the assumptions and proof only.
# Kept unchanged as the code of record for the results reported in the article;
# only this header comment was added. paper/demi_v17.py is the same algorithm
# as a library; demi.py in the repository root is the current, later version.
# See paper/REPRODUCIBILITY.md.
#
# Copyright (c) 2026 George Mason University. Licensed under the George Mason
# University Noncommercial Use License; see LICENSE and NOTICE.

import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

def _compute_or_vectorized(edf: pd.DataFrame) -> np.ndarray:
    """
    Vectorized DEMI v17 odds ratio calculation.

    Parameters
    ----------
    edf : pd.DataFrame
        Must contain the four 2x2 table count columns:
        n_code_target (a), n_code_no_target (b),
        n_target_no_code (c), n_no_code_no_target (d).

    Returns
    -------
    np.ndarray
        Exp-scale odds ratios. Returns NaN where a == 0 and c == 0.
    """
    a = edf.get("n_code_target",       pd.Series(np.nan, index=edf.index)).to_numpy(float)
    b = edf.get("n_code_no_target",    pd.Series(np.nan, index=edf.index)).to_numpy(float)
    c = edf.get("n_target_no_code",    pd.Series(np.nan, index=edf.index)).to_numpy(float)
    d = edf.get("n_no_code_no_target", pd.Series(np.nan, index=edf.index)).to_numpy(float)

    thresh = 30.0
    ab = a + b
    ac = a + c
    out = np.full(len(a), np.nan, dtype=float)

    any_nan = np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(d)
    valid = ~any_nan

    # Undefined when there are no outcome-positive patients in either code stratum.
    defined = valid & ~((a == 0) & (c == 0))

    both_large = defined & (ab >= thresh) & (ac >= thresh)
    small = defined & ~both_large

    # a == 0
    a0 = defined & (a == 0)
    a0_b0 = a0 & (b == 0)
    a0_ok = a0 & (b > 0)
    out[a0_b0] = 1.0 / (c[a0_b0] + 1.0)
    min_bc = np.minimum(b, c)
    safe_a0 = a0_ok & (min_bc > 0)
    out[safe_a0] = 1.0 / min_bc[safe_a0]

    # Both samples large, a > 0
    bl_a = both_large & (a > 0)
    m = bl_a & (b == 0) & (c == 0)
    out[m] = a[m] + 1.0
    m = bl_a & (b == 0)
    out[m] = ac[m] / (b[m] + 1.0)
    m = bl_a & (c == 0)
    out[m] = ab[m] / (c[m] + 1.0)
    m = bl_a & (b > 0) & (c > 0)
    out[m] = (a[m] * d[m]) / (b[m] * c[m])

    # One or both samples small, a > 0
    sm_a = small & (a > 0)
    m = sm_a & (b == 0) & (c == 0)
    out[m] = a[m] + 1.0
    m = sm_a & (b == 0) & (c > 0)
    out[m] = ac[m] / (b[m] + 1.0)
    m = sm_a & (c == 0) & (b > 0)
    out[m] = ab[m] / (c[m] + 1.0)
    m = sm_a & (b > 0) & (c > 0)
    raw = (a[m] * d[m]) / (b[m] * c[m])
    or1 = ab[m] / c[m]
    or2 = ac[m] / b[m]
    out[m] = np.where(raw >= 1.0, np.maximum(or1, or2), np.minimum(or1, or2))

    return out

def _compute_pref(
    outcome_rows: pd.DataFrame,
    pref_override: Optional[float] = None,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Compute outcome prevalence and prior odds from rows involving the outcome.

    Returns
    -------
    tuple
        (pref, prior_odds, n_y1, n_y0)
    """
    n_y1: Optional[float] = None
    n_y0: Optional[float] = None

    if len(outcome_rows) > 0:
        row = outcome_rows.iloc[0]
        a = float(row.get("n_code_target", 0) or 0)
        b = float(row.get("n_code_no_target", 0) or 0)
        c = float(row.get("n_target_no_code", 0) or 0)
        d = float(row.get("n_no_code_no_target", 0) or 0)
        n_y1 = a + c
        n_y0 = b + d

    if pref_override is not None:
        pref = float(pref_override)
        n_y1_eff = pref
        n_y0_eff = 1.0 - pref
    elif n_y1 is not None and n_y0 is not None and (n_y1 + n_y0) > 0:
        pref = n_y1 / (n_y1 + n_y0)
        n_y1_eff = n_y1
        n_y0_eff = n_y0
    else:
        pref = 0.5
        n_y1_eff = 1.0
        n_y0_eff = 1.0

    prior_odds = n_y1_eff / n_y0_eff if n_y0_eff > 0 else 1.0
    return pref, prior_odds, n_y1, n_y0

def _partition_kplus_kminus(
    nodes: List[int],
    T_val: pd.Series,
) -> Tuple[List[int], List[int]]:
    """
    Partition nodes by the sign of their total effect.
    """
    k_plus = [k for k in nodes if np.isfinite(T_val.loc[k]) and T_val.loc[k] > 0.0]
    k_minus = [k for k in nodes if np.isfinite(T_val.loc[k]) and T_val.loc[k] < 0.0]
    return k_plus, k_minus

def _run_recursion_within_group(
    group: List[int],
    T_val: pd.Series,
    lambda_l: Dict[int, pd.Series],
    sign: int,
    prev_j: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Backward recursion for one sign group in DEMI v17.

    For each predictor k, the indirect term is the sum over all downstream
    same-sign predictors j of:

        lambda_excess_{k,j} * D_{j,Y}

    where

        lambda_excess_{k,j} = P(Z_j=1 | Z_k=1) - P(Z_j=1)

    The direct effect is then:

        D_{k,Y} = max(0, T_{k,Y} - indirect)   for sign = +1
        D_{k,Y} = min(0, T_{k,Y} - indirect)   for sign = -1
    """
    if not group:
        return pd.Series(dtype=float)

    d_series = pd.Series(np.nan, index=group, dtype=float)

    for i in range(len(group) - 1, -1, -1):
        k = group[i]
        downstream = group[i + 1:]

        indirect = 0.0
        lam = lambda_l.get(k, pd.Series(dtype=float))

        for j in downstream:
            lam_kj = float(lam.get(j, 0.0))
            if prev_j is not None and lam_kj != 0.0:
                lam_kj -= float(prev_j.get(j, 0.0))
            d_j = float(d_series.get(j, np.nan))
            if np.isfinite(d_j):
                indirect += lam_kj * d_j

        raw = float(T_val.loc[k]) - indirect
        d_series.loc[k] = max(0.0, raw) if sign > 0 else min(0.0, raw)

    return d_series

def select_top_predictors(
    T_series: pd.Series,
    n_cases: int,
    *,
    epv: int = 5,
    user_cap: int = 500,
) -> pd.Index:
    """
    EPV-balanced dual-tail predictor selection for DEMI v17.
    """
    t = T_series.dropna()
    t = t[t != 0.0]

    n_max = min(max(n_cases // epv, 1), user_cap)
    n_each = n_max // 2

    harmful = t[t > 0].sort_values(ascending=False)
    protective = t[t < 0].sort_values(ascending=True)

    n_harm = min(n_each, len(harmful))
    n_prot = min(n_each, len(protective))

    spare = n_max - n_harm - n_prot
    if spare > 0:
        if n_harm < n_each:
            n_prot = min(len(protective), n_prot + spare)
        elif n_prot < n_each:
            n_harm = min(len(harmful), n_harm + spare)

    keep = list(harmful.index[:n_harm]) + list(protective.index[:n_prot])

    if len(keep) < n_max:
        remaining = t.drop(index=keep, errors="ignore")
        if len(remaining) > 0:
            extra = remaining.abs().sort_values(ascending=False).index[: n_max - len(keep)]
            keep += list(extra)

    return pd.Index(keep)

def build_snomed_ancestor_sets(
    relationship_path: str,
    concept_ids: Optional[Set[int]] = None,
) -> Dict[int, Set[int]]:
    """
    Build transitive ancestor sets from a SNOMED CT RF2 relationship file.

    Accepts either the Full RF2 file or the Snapshot RF2 file. For Full RF2,
    the most recent row per relationship id is kept before filtering active
    isa relationships.
    """
    isa = 116680003

    rel = pd.read_csv(
        relationship_path,
        sep="\t",
        usecols=["id", "effectiveTime", "active", "typeId", "sourceId", "destinationId"],
        dtype=str,
    )
    rel["_t"] = rel["effectiveTime"].astype(int)
    rel = rel.loc[rel.groupby("id")["_t"].idxmax()].copy()
    rel = rel[(rel["active"] == "1") & (rel["typeId"] == str(isa))]

    rel["sourceId"] = rel["sourceId"].astype(int)
    rel["destinationId"] = rel["destinationId"].astype(int)

    if concept_ids is not None:
        concept_ids = {int(x) for x in concept_ids}
        rel = rel[
            rel["sourceId"].isin(concept_ids) | rel["destinationId"].isin(concept_ids)
        ].copy()

    parents_of: Dict[int, Set[int]] = {}
    for child, parent in rel[["sourceId", "destinationId"]].itertuples(index=False):
        parents_of.setdefault(int(child), set()).add(int(parent))

    all_nodes: Set[int] = set(parents_of.keys())
    for parents in parents_of.values():
        all_nodes.update(parents)

    memo: Dict[int, Set[int]] = {}

    def ancestors(node: int) -> Set[int]:
        if node in memo:
            return memo[node]
        direct = parents_of.get(node, set())
        out: Set[int] = set(direct)
        for p in direct:
            out.update(ancestors(p))
        memo[node] = out
        return out

    target_nodes = all_nodes if concept_ids is None else set(int(x) for x in concept_ids)
    return {int(node): ancestors(int(node)) for node in target_nodes}

def snomed_ancestor_exclusion_mask(
    concept_codes: pd.Series,
    concept_code_strs: Optional[pd.Series],
    relationship_path: str,
) -> pd.Series:
    """
    Return a boolean mask excluding any SNOMED concept that is a transitive
    ancestor of another SNOMED concept already present in the data.
    """
    if concept_code_strs is not None:
        is_snomed = concept_code_strs.str.contains("SNOMED", na=False, regex=False)
    else:
        is_snomed = pd.Series(True, index=concept_codes.index)

    snomed_int_codes: Set[int] = set(
        concept_codes[is_snomed].dropna().astype(int).unique()
    )

    if not snomed_int_codes:
        return pd.Series(False, index=concept_codes.index)

    try:
        ancestors = build_snomed_ancestor_sets(relationship_path, snomed_int_codes)
    except Exception as e:
        warnings.warn(
            f"[DEMI] Could not load SNOMED relationship file: {e}. "
            "Proceeding without SNOMED ancestor exclusion.",
            RuntimeWarning,
        )
        return pd.Series(False, index=concept_codes.index)

    to_exclude: Set[int] = set()
    for code_a in snomed_int_codes:
        for code_b in snomed_int_codes:
            if code_a == code_b:
                continue
            if code_a in ancestors.get(code_b, set()):
                to_exclude.add(code_a)
                break

    return concept_codes.astype("Int64").isin(list(to_exclude))

def analyze_causal_sequence_py(
    df_in: Union[str, pd.DataFrame],
    *,
    force_outcome: int,
    lambda_min_count: int = 15,
    epv: int = 5,
    user_cap: int = 500,
    snomed_relationship_path: Optional[str] = None,
    pref_override: Optional[float] = None,
    debug_total_effect: bool = False,
) -> Dict[str, Any]:
    """
    DEMI v17 causal decomposition on a pre-filtered DataFrame.

    Steps
    -----
    1. Remove Q&A items and rollup nodes.
    2. Build temporal order from pairwise precedence scores.
    3. Compute total effects T_{k,Y} = log(OR_v17).
    4. Select predictors with EPV-balanced dual-tail selection.
    5. Compute lambda values and direct effects by backward recursion.
    6. Apply the exact v17 intercept correction.
    """
    df = pd.read_csv(df_in) if isinstance(df_in, str) else df_in.copy()

    for col in ["target_concept_code_int", "concept_code_int"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in [
        "n_code_target", "n_code", "n_target", "n_no_target",
        "n_code_before_target", "n_target_before_code",
        "n_code_no_target", "n_target_no_code", "n_no_code_no_target",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outcome = int(force_outcome)

    name_col: Optional[str] = None
    for candidate in ("concept_name", "standard_concept_name"):
        if candidate in df.columns:
            name_col = candidate
            break

    # Q&A exclusion
    if name_col is not None:
        qa_mask = (
            df[name_col].str.startswith("Question:", na=False)
            | df[name_col].str.contains("Answer:", na=False, regex=False)
        )
        n_qa = int(df.loc[qa_mask, "concept_code_int"].nunique())
        df = df[~qa_mask].copy()
    else:
        n_qa = 0

    # Rollup exclusion
    n_rollup_snomed = 0
    n_rollup_pattern = 0

    if snomed_relationship_path is not None and name_col is not None:
        code_str_col = df["concept_code"] if "concept_code" in df.columns else None
        snomed_exclude = snomed_ancestor_exclusion_mask(
            df["concept_code_int"],
            code_str_col,
            snomed_relationship_path,
        )
        snomed_exclude = snomed_exclude & (df["concept_code_int"] != outcome)
        n_rollup_snomed = int(snomed_exclude.sum())
        df = df[~snomed_exclude].copy()

    if name_col is not None:
        current_name_col = name_col if name_col in df.columns else None
        if current_name_col is not None:
            name = df[current_name_col].fillna("")

            rule1_pattern = (
                r'\bfinding\b|\bfindings\b|\bdisorder of\b|^disorder of\b'
                r'|\bstructure\b|\bobservation\b|\bcondition\b|^disease$'
            )
            rule1_mask = name.str.contains(rule1_pattern, case=False, regex=True, na=False)

            generic_ending = (
                r'\bdisorder\b|\bdisease\b|\bsyndrome\b'
                r'|\bcondition\b|\bproblem\b|\bsymptom\b'
            )
            has_generic = name.str.contains(generic_ending, case=False, regex=True, na=False)
            if "n_target_no_code" in df.columns and "n_code_target" in df.columns:
                ac_col = df["n_code_target"].fillna(0) + df["n_target_no_code"].fillna(0)
                c_rate = df["n_target_no_code"].fillna(0) / ac_col.replace(0, np.nan)
                rule2_mask = has_generic & (c_rate < 0.03)
            else:
                rule2_mask = pd.Series(False, index=df.index)

            rollup_mask = (rule1_mask | rule2_mask) & (df["concept_code_int"] != outcome)
            if snomed_relationship_path is not None and "concept_code" in df.columns:
                already_handled = df["concept_code"].str.contains(
                    "SNOMED", na=False, regex=False
                )
                rollup_mask = rollup_mask & ~already_handled

            n_rollup_pattern = int(rollup_mask.sum())
            df = df[~rollup_mask].copy()

    n_rollup = n_rollup_snomed + n_rollup_pattern

    edge_cols = [
        "n_target_before_code", "n_code_before_target",
        "n_code_target", "n_code", "n_target", "n_no_target",
        "n_code_no_target", "n_target_no_code", "n_no_code_no_target",
    ]
    present = [c for c in edge_cols if c in df.columns]
    edge_index = df.groupby(
        ["target_concept_code_int", "concept_code_int"], as_index=True
    )[present].sum()

    outcome_rows = df[df["target_concept_code_int"] == outcome]
    pref, prior_odds_data, n_y1, n_y0 = _compute_pref(
        outcome_rows,
        pref_override=pref_override,
    )

    # Step 1: temporal ordering
    all_events = sorted(
        set(int(x) for x in df["target_concept_code_int"].dropna().unique())
        | set(int(x) for x in df["concept_code_int"].dropna().unique())
    )
    scores: Dict[int, float] = {}
    for zi in all_events:
        try:
            pairs = edge_index.xs(zi, level="target_concept_code_int")
            tbc = pairs["n_target_before_code"].fillna(0.0) if "n_target_before_code" in pairs else 0.0
            cbt = pairs["n_code_before_target"].fillna(0.0) if "n_code_before_target" in pairs else 0.0
            scores[zi] = float((tbc - cbt).sum() if hasattr(tbc, "sum") else 0.0)
        except KeyError:
            scores[zi] = 0.0

    sorted_scores = pd.Series(scores).sort_values(ascending=False)
    temporal_order = [int(ev) for ev in sorted_scores.index if ev != outcome] + [outcome]
    nodes = temporal_order[:-1]
    pos_by_event = {ev: i for i, ev in enumerate(temporal_order)}

    # Step 2: total effects
    e_rows, e_keys = [], []
    for k in nodes:
        key = (outcome, int(k))
        if key in edge_index.index:
            e_rows.append(edge_index.loc[key])
            e_keys.append(k)

    total_or = pd.Series(1.0, index=nodes, dtype=float)
    t_val = pd.Series(0.0, index=nodes, dtype=float)

    if e_rows:
        edf = pd.DataFrame(e_rows, index=e_keys)
        or_vals = _compute_or_vectorized(edf)
        for k, ov in zip(e_keys, or_vals):
            if np.isfinite(ov) and ov > 0:
                total_or.loc[k] = float(ov)
                t_val.loc[k] = float(np.log(ov))

        if debug_total_effect:
            for k, ov in zip(e_keys, or_vals):
                r = edf.loc[k]
                print(f"\n--- DEBUG T_val k={k} ---")
                print(
                    f"a={r.get('n_code_target',0):.0f}  b={r.get('n_code_no_target',0):.0f}  "
                    f"c={r.get('n_target_no_code',0):.0f}  d={r.get('n_no_code_no_target',0):.0f}"
                )
                print(f"OR={ov:.6f}  ln(OR)={np.log(max(ov, 1e-300)):.6f}")

    # Step 2b: predictor selection
    selected_idx = select_top_predictors(
        t_val,
        n_cases=int(n_y1) if n_y1 is not None else 1,
        epv=epv,
        user_cap=user_cap,
    )
    selected_set = set(selected_idx.tolist())
    nodes = [n for n in nodes if n in selected_set]
    pos_by_event = {ev: i for i, ev in enumerate(temporal_order)}

    # Step 3: lambda values and marginal prevalences
    n_total = float(n_y1 + n_y0) if (n_y1 is not None and n_y0 is not None) else 0.0

    prev_j_dict: Dict[int, float] = {}
    if n_total > 0:
        for j in nodes:
            key_jy = (outcome, int(j))
            key_yj = (int(j), outcome)
            n_j = 0.0
            for key in (key_jy, key_yj):
                if key in edge_index.index:
                    row = edge_index.loc[key]
                    n_j = float(row.get("n_code", 0.0))
                    if n_j == 0.0:
                        n_j = (
                            float(row.get("n_code_target", 0.0))
                            + float(row.get("n_code_no_target", 0.0))
                        )
                    if n_j > 0:
                        break
            prev_j_dict[j] = n_j / n_total if n_j > 0 else 0.0
    prev_j_series = pd.Series(prev_j_dict, dtype=float)

    lambda_l: Dict[int, pd.Series] = {}
    for k in nodes:
        pos_k = pos_by_event[k]
        downstream = [int(ev) for ev in temporal_order[pos_k + 1 : -1]]
        lam: Dict[int, float] = {}
        for j in downstream:
            key = (int(j), int(k))
            if key not in edge_index.index:
                lam[j] = 0.0
                continue
            row = edge_index.loc[key]
            n_kj = float(row.get("n_code_target", 0.0))
            n_k = float(row.get("n_code", 0.0))
            if n_k == 0.0:
                n_k = (
                    float(row.get("n_code_target", 0.0))
                    + float(row.get("n_code_no_target", 0.0))
                )
            lam[j] = (
                0.0 if n_k < lambda_min_count
                else float(np.clip(n_kj / n_k if n_k > 0 else 0.0, 0.0, 1.0))
            )
        lambda_l[k] = pd.Series(lam, dtype=float)

    k_plus, k_minus = _partition_kplus_kminus(nodes, t_val)

    d_plus = _run_recursion_within_group(k_plus, t_val, lambda_l, sign=+1, prev_j=prev_j_series)
    d_minus = _run_recursion_within_group(k_minus, t_val, lambda_l, sign=-1, prev_j=prev_j_series)

    d_val = pd.Series(0.0, index=nodes, dtype=float)
    for k, v in d_plus.items():
        d_val.loc[k] = v if np.isfinite(v) else float(t_val.loc[k])
    for k, v in d_minus.items():
        d_val.loc[k] = v if np.isfinite(v) else float(t_val.loc[k])

    # Step 4: exact intercept correction
    p_marginal = (
        float(n_y1) / float(n_y1 + n_y0)
        if (n_y1 is not None and n_y0 is not None and (n_y1 + n_y0) > 0)
        else 0.5
    )
    prior_odds = p_marginal / (1.0 - p_marginal)
    beta_0_prior = float(np.log(prior_odds))

    log_e_phi = 0.0
    for k in (list(k_plus) + list(k_minus)):
        p_k = float(prev_j_series.get(k, 0.0))
        d_k = float(d_val.get(k, 0.0))
        if not np.isfinite(d_k) or p_k < 0.0 or p_k > 1.0:
            continue
        if p_k == 0.0:
            continue
        if p_k == 1.0:
            log_e_phi += d_k
            continue
        log_e_phi += float(np.logaddexp(np.log(p_k) + d_k, np.log1p(-p_k)))

    e_phi = float(np.exp(log_e_phi))
    beta_0 = beta_0_prior - log_e_phi

    beta_vals = d_val.astype(float)
    odds_ratios = np.exp(beta_vals)

    coef_df = pd.DataFrame({
        "predictor_int": list(beta_vals.index) + ["(intercept)"],
        "predictor": list(beta_vals.index) + ["(intercept)"],
        "beta": list(beta_vals.values) + [beta_0],
        "odds_ratio": list(odds_ratios.values) + [float(np.exp(beta_0))],
        "group": (
            ["K+" if k in k_plus else "K-" if k in k_minus else "excluded" for k in beta_vals.index]
            + ["intercept"]
        ),
    })

    predictors = list(beta_vals.index)
    beta_vec = beta_vals.values

    def predict_proba(z):
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))

        if isinstance(z, pd.DataFrame):
            return sigmoid(
                beta_0
                + z.reindex(columns=predictors, fill_value=0.0).astype(float).to_numpy()
                @ beta_vec
            )
        if isinstance(z, (dict, pd.Series)):
            return float(
                sigmoid(beta_0 + np.array([float(z.get(p, 0.0)) for p in predictors]) @ beta_vec)
            )
        arr = np.asarray(z, dtype=float)
        if arr.ndim == 1:
            return float(sigmoid(beta_0 + arr @ beta_vec))
        if arr.ndim == 2:
            return sigmoid(beta_0 + arr @ beta_vec)
        raise ValueError("Z must be 1D or 2D array-like")

    baseline_prob = p_marginal

    k_to_total = pd.DataFrame({
        "target_concept_code_int": outcome,
        "concept_code_int": [int(k) for k in nodes],
        "total": total_or.reindex(nodes).values,
        "total_log": t_val.reindex(nodes).values,
        "direct": d_val.reindex(nodes).values,
        "group": [
            "K+" if k in k_plus else "K-" if k in k_minus else "excluded"
            for k in nodes
        ],
    })

    return {
        "sorted_scores": sorted_scores,
        "temporal_order": temporal_order,
        "K_plus": k_plus,
        "K_minus": k_minus,
        "total_or": total_or,
        "T_val": t_val,
        "D_val": d_val,
        "lambda_l": lambda_l,
        "prev_j": prev_j_series,
        "coef_df": coef_df,
        "beta_0": beta_0,
        "beta_0_prior": beta_0_prior,
        "E_phi": e_phi,
        "prior_odds": prior_odds,
        "baseline_probability": baseline_prob,
        "pref_used_for_adjustment": pref,
        "n_outcome_yes": n_y1,
        "n_outcome_no": n_y0,
        "n_qa_excluded": n_qa,
        "n_rollup_excluded": n_rollup,
        "n_rollup_snomed": n_rollup_snomed,
        "n_rollup_pattern": n_rollup_pattern,
        "beta": pd.Series(beta_vec, index=predictors, dtype=float),
        "logit_predictors": predictors,
        "predict_proba": predict_proba,
        "k_to_total": k_to_total,
    }
