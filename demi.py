"""DEMI: Direct Effects Multiplicative Inference (algorithm version 23).

Scores a binary outcome from pairwise relationships among clinical concepts.
This module is the algorithm only: it takes a pandas DataFrame of pairwise
rows and returns scoring state with ``predict_proba`` and ``predict_score``
functions. It has no database code. Within the lab's deployed system this
same logic runs inside a service wrapper that supplies global temporal
scores, hierarchy filtering, and a calibration table from a database; those
inputs are optional arguments here, and without them the module uses local
temporal ordering and reports probabilities as prior log-odds plus score.

Version 23 differs from the version evaluated in Alemi et al. (2026,
doi:10.1016/j.jbi.2026.105111); see CHANGELOG.md and paper/REPRODUCIBILITY.md.
The article's algorithm is in paper/.

Entry points:
    run_demi(edges, force_outcome=outcome_id, require_global_temporal_order=False)
    result["predict_proba"]({predictor_id: 1, ...})

Copyright (c) 2026 George Mason University. Licensed under the George Mason
University Noncommercial Use License; see LICENSE and NOTICE.
"""

import math as _math
import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd


SMALL_THRESHOLD = 53
# Pearson chi-square critical value, 1 df, two-sided alpha = 0.05 (1.96^2).
# A predictor receives a net-count vote only when N * phi^2 exceeds this,
# i.e. |phi| > 1.96 / sqrt(N).  Same dead zone as the DEMI knowledgebase.
CHI_SQUARE_CRITICAL_DEFAULT = 3.841458820694124
SCORING_RULE_NET_COUNT = "net_count_phi_chi_square"
SCORING_RULE_DIRECT_EFFECTS = "v23_noisy_or_three_state_D1_D0_Dmissing"
PROBABILITY_METHOD_PRIOR_SCORE = "prior_log_odds_plus_score"
PROBABILITY_METHOD_CALIBRATION = "three_anchor_calibration"
NOISY_OR_ZERO_THRESHOLD = 1e-8
_PI_OVER_SQRT3 = _math.pi / _math.sqrt(3.0)

# Keep SQL parameter counts conservative across both SQLite and MySQL-backed
# service deployments.  A rectangular pairwise query uses two chunks, so a
# maximum width of 450 stays at or below 900 bound values.
MAX_SQL_IN_CHUNK_SIZE = 450

DEMI_COUNT_COLUMNS = (
    "target_concept_code_int",
    "concept_code_int",
    "n_code_target",
    "n_code_no_target",
    "n_target_no_code",
    "n_no_code_no_target",
    "n_code",
    "n_target",
    "n_no_target",
    "n_code_before_target",
    "n_target_before_code",
    "total_effects",
    "total_effects_log",
    "total_effects_abs",
    "phi",
    "phi_abs",
    "phi_rank",
    "direction",
)

DEMI_PAIRWISE_COLUMNS = (
    "target_concept_code_int",
    "concept_code_int",
    "n_code_target",
    "n_code_no_target",
    "n_target_no_code",
    "n_no_code_no_target",
    "n_no_target",
    "phi",
)


def _chunks(values: Iterable[Any], size: int) -> Iterator[List[Any]]:
    """Yield bounded lists without materializing any database result rows."""
    chunk: List[Any] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

DEMI_V23_CALIBRATION_TABLE_COLUMNS = [
    "target_id",
    "sample_type",
    "prior",
    "OR_max",
    "OR_min",
    "UCL_prob",
    "LCL_prob",
    "score_p01",
    "score_median",
    "score_p99",
    "n_selected",
    "n_target",
    "N_target_total",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert numeric-ish values while treating missing data as default."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _finite_optional_float(value: Any) -> Optional[float]:
    """Read one stored numeric value without allocating a pandas Series."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if _math.isfinite(numeric) else None


def _compute_outcome_prevalence(
    outcome_rows: pd.DataFrame,
    pref_override: Optional[float] = None,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Compute target prevalence and retain the resolved marginal counts."""
    n_y1 = None
    n_y0 = None
    if len(outcome_rows) > 0:
        df = outcome_rows.copy()
        for col in [
            "n_target",
            "n_no_target",
            "n_code_target",
            "n_code_no_target",
            "n_target_no_code",
            "n_no_code_no_target",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "n_target" in df.columns and "n_no_target" in df.columns:
            marg = df[["n_target", "n_no_target"]].dropna().copy()
            if not marg.empty:
                grouped = (
                    marg.groupby(["n_target", "n_no_target"])
                    .size()
                    .reset_index(name="n_rows")
                )
                grouped["N_total"] = grouped["n_target"] + grouped["n_no_target"]
                chosen = grouped.sort_values(
                    ["n_target", "N_total", "n_rows"],
                    ascending=[False, False, False],
                ).iloc[0]
                n_y1 = float(chosen["n_target"])
                n_y0 = float(chosen["n_no_target"])

        required = [
            "n_code_target",
            "n_code_no_target",
            "n_target_no_code",
            "n_no_code_no_target",
        ]
        if (n_y1 is None or n_y0 is None) and all(c in df.columns for c in required):
            df["y1_cells"] = df["n_code_target"].fillna(0) + df[
                "n_target_no_code"
            ].fillna(0)
            df["y0_cells"] = df["n_code_no_target"].fillna(0) + df[
                "n_no_code_no_target"
            ].fillna(0)
            df["N_total"] = df["y1_cells"] + df["y0_cells"]
            grouped = (
                df.groupby(["y1_cells", "y0_cells", "N_total"])
                .size()
                .reset_index(name="n_rows")
            )
            chosen = grouped.sort_values(
                ["y1_cells", "N_total", "n_rows"],
                ascending=[False, False, False],
            ).iloc[0]
            n_y1 = float(chosen["y1_cells"])
            n_y0 = float(chosen["y0_cells"])

    if pref_override is not None:
        prevalence = float(pref_override)
    elif n_y1 is not None and n_y0 is not None and (n_y1 + n_y0) > 0:
        prevalence = n_y1 / (n_y1 + n_y0)
    else:
        prevalence = 0.5
    return prevalence, n_y1, n_y0


def _read_calibration_table_obj(table: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Read a v23 calibration table from a DataFrame, CSV, or Excel path."""
    if isinstance(table, pd.DataFrame):
        if table.attrs.get("_demi_v23_calibration_normalized") is True:
            return table
        df = table.copy()
    else:
        path = str(table)
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, sheet_name=0)
        else:
            df = pd.read_csv(path)
    if "target_id" not in df.columns and "concept_code_int" in df.columns:
        df = df.rename(columns={"concept_code_int": "target_id"})
    if "N_target_total" not in df.columns and "N_total" in df.columns:
        # DEMI DB v5 stores this field as N_total.  Keep N_target_total as the
        # algorithm's canonical internal name so the remaining calibration
        # logic and public snapshot contract stay unchanged.
        df = df.rename(columns={"N_total": "N_target_total"})
    if "target_id" in df.columns:
        df = df[df["target_id"].notna()].copy()
        df = df[
            ~df["target_id"].astype(str).str.startswith("Instructions:", na=False)
        ].copy()
    df.attrs["_demi_v23_calibration_normalized"] = True
    return df


def read_calibration(
    target_id: Any,
    table_path: Optional[Union[str, pd.DataFrame]] = None,
) -> Optional[Dict[str, Any]]:
    """Return calibration anchors for one target, or None if unavailable."""
    if table_path is None or (isinstance(table_path, str) and not table_path):
        return None
    if isinstance(table_path, str) and not os.path.exists(table_path):
        return None

    table = _read_calibration_table_obj(table_path)
    missing = [c for c in DEMI_V23_CALIBRATION_TABLE_COLUMNS if c not in table.columns]
    if missing:
        return None

    row = table[table["target_id"].astype(str) == str(target_id)]
    if row.empty:
        try:
            target_int = int(target_id)
            numeric_ids = pd.to_numeric(table["target_id"], errors="coerce").astype(
                "Int64"
            )
            row = table[numeric_ids == target_int]
        except Exception:
            pass
    if row.empty:
        return None

    raw = row.iloc[0]
    out: Dict[str, Any] = {col: raw[col] for col in DEMI_V23_CALIBRATION_TABLE_COLUMNS}
    out["target_id"] = str(out["target_id"])
    out["sample_type"] = str(out["sample_type"])
    for col in [
        "prior",
        "OR_max",
        "OR_min",
        "UCL_prob",
        "LCL_prob",
        "score_p01",
        "score_median",
        "score_p99",
    ]:
        out[col] = float(out[col])
    for col in ["n_selected", "n_target", "N_target_total"]:
        out[col] = int(float(out[col]))
    return out


def three_anchor_transform(score: float, cal: Dict[str, Any]) -> float:
    """Map a raw DEMI score to probability using three calibration anchors."""
    lcl = float(cal["LCL_prob"])
    ucl = float(cal["UCL_prob"])
    prior = float(cal["prior"])
    score_p01 = float(cal["score_p01"])
    score_median = float(cal["score_median"])
    score_p99 = float(cal["score_p99"])
    score = max(score_p01, min(score_p99, float(score)))
    if score <= score_median:
        if score_median == score_p01:
            return prior
        t = (score - score_p01) / (score_median - score_p01)
        return float(lcl + t * (prior - lcl))
    if score_p99 == score_median:
        return prior
    t = (score - score_median) / (score_p99 - score_median)
    return float(prior + t * (ucl - prior))


def _row_to_abcd(
    row: Union[pd.Series, Dict[str, Any]],
) -> Tuple[float, float, float, float]:
    """Extract a, b, c, d cells from a DEMI count row."""
    a = _safe_float(row.get("n_code_target", 0.0))
    b = _safe_float(row.get("n_code_no_target", 0.0))
    if "n_target_no_code" in row:
        c = _safe_float(row.get("n_target_no_code", 0.0))
    else:
        c = max(_safe_float(row.get("n_target", 0.0)) - a, 0.0)
    if "n_no_code_no_target" in row:
        d = _safe_float(row.get("n_no_code_no_target", 0.0))
    else:
        d = max(_safe_float(row.get("n_no_target", 0.0)) - b, 0.0)
    return a, b, c, d


def _phi_coefficient(a: float, b: float, c: float, d: float) -> float:
    """Phi correlation for a 2x2 count table, clipped to [-1, 1]."""
    denom = _math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0.0:
        return 0.0
    return float(np.clip((a * d - b * c) / denom, -1.0, 1.0))


def _phi_to_log_or(phi: float, n1: float, n2: float) -> float:
    """Convert phi to log odds ratio using the v23 logistic-scale formula."""
    n1 = float(n1)
    n2 = float(n2)
    total = n1 + n2
    if total <= 0.0:
        return 0.0
    h = n1 / total
    if h <= 0.0 or h >= 1.0:
        return 0.0
    phi = float(np.clip(phi, -0.9999, 0.9999))
    denom = _math.sqrt(h * (1.0 - h) * (1.0 - phi**2))
    if denom < 1e-12:
        return 0.0
    return float(phi * _PI_OVER_SQRT3 / denom)


def _phi_from_edge_row(row: Union[pd.Series, Dict[str, Any]]) -> float:
    if "phi" in row:
        try:
            stored_phi = float(row.get("phi"))
            if _math.isfinite(stored_phi) and abs(stored_phi) <= 1.0:
                return stored_phi
        except (TypeError, ValueError):
            pass
    a, b, c, d = _row_to_abcd(row)
    return _phi_coefficient(a, b, c, d)


def _parse_stored_direction(value: Any) -> Optional[int]:
    """Normalize a valid stored total-effect direction, or return None."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        labels = {
            "risk": 1,
            "positive": 1,
            "positive_direction": 1,
            "+": 1,
            "+1": 1,
            "protective": -1,
            "negative": -1,
            "negative_direction": -1,
            "-": -1,
            "-1": -1,
            "neutral": 0,
            "none": 0,
            "0": 0,
        }
        parsed = labels.get(value.strip().casefold())
        if parsed is not None:
            return parsed
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number not in (-1.0, 0.0, 1.0):
        return None
    return int(number)


def _compute_T0(a: float, b: float, c: float, d: float) -> float:
    """Total log effect for measured absence of a predictor."""
    c = float(c)
    d = float(d)
    ac = float(a) + c
    bd = float(b) + d
    if c <= 0 or d <= 0 or ac <= 0 or bd <= 0:
        return 0.0
    return float(_math.log(c * bd) - _math.log(d * ac))


_MAX_FINITE_FLOAT = float(np.finfo(np.float64).max)
_MAX_FINITE_LOG_OR = float(_math.log(_MAX_FINITE_FLOAT))

def _finite_or_from_log_or(log_or: float) -> Optional[float]:
    """Return a positive finite OR while preserving the sign of ``log_or``."""
    value = float(log_or)
    if not _math.isfinite(value):
        return None
    clipped = min(max(value, -_MAX_FINITE_LOG_OR), _MAX_FINITE_LOG_OR)
    return float(_math.exp(clipped))

def _finite_or_array_from_log_or(values: Any) -> np.ndarray:
    """Vectorized finite OR representation; NaN inputs remain NaN."""
    array = np.asarray(values, dtype=float)
    return np.exp(np.clip(array, -_MAX_FINITE_LOG_OR, _MAX_FINITE_LOG_OR))

def _v23_log_or_value_and_direction(
    a: float,
    b: float,
    c: float,
    d: float,
    *,
    small_threshold: int = SMALL_THRESHOLD,
    sample_type: Optional[str] = None,
) -> Tuple[Optional[float], int]:
    """
    DEMI v23 log-OR formula.

    This is the authoritative internal implementation.  It returns log-OR
    directly so an extreme but finite phi-derived value never needs to pass
    through ``exp(log_or)`` and back through ``log(OR)``.

    Large samples use the phi-based formula whenever any cell is zero:

        log OR = _phi_to_log_or( phi(a,b,c,d), n1=a+b, n2=c+d )

    Small samples use zero-cell edge calculations only to set direction. They
    do not use the edge OR magnitude in count-score calculations.

    Degenerate exclusions:
        a+b = 0  (no exposed patients)
        a+c = 0  (no outcome-positive patients)
        b+d = 0  (no outcome-negative patients)
        c+d = 0  (no unexposed patients)
    Note: d=0 is no longer excluded. Large samples compute its OR via phi;
    small samples retain only its direction.

    All-positive cells use the run-level sample_type when it is provided.
    In DEMI v23 runs, sample_type is based on total target-positive cases,
    not on feature-specific a+b or a+c counts.
    """
    a = int(a); b = int(b); c = int(c); d = int(d)

    # Degenerate marginal exclusions
    if a + b == 0 or a + c == 0 or b + d == 0 or c + d == 0:
        return None, 0

    # Direction
    if a == 0:
        direction = -1
    elif b == 0 or c == 0:
        direction = +1
    else:
        direction = +1 if (a * d) > (b * c) else (-1 if (a * d) < (b * c) else 0)

    if sample_type is None:
        # Backward-compatible direct-call behavior when no run-level sample_type
        # is available. The main DEMI path passes sample_type explicitly below.
        use_small_count_formula = (a + c < small_threshold)
    else:
        use_small_count_formula = (str(sample_type).lower() == "small")

    # Any zero cell is an edge case.
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Small samples use only edge direction, not edge OR magnitude.
        if use_small_count_formula:
            return None, direction
        phi_val = _phi_coefficient(float(a), float(b), float(c), float(d))
        log_or  = _phi_to_log_or(phi_val, n1=float(a + b), n2=float(c + d))
        if not _math.isfinite(log_or):
            return None, 0
        return float(log_or), direction

    # All cells positive.
    if use_small_count_formula:
        raw_or = (a * d) / (b * c)
        or1    = float(a + b) / c
        or2    = float(a + c) / b
        value  = max(or1, or2) if raw_or >= 1.0 else min(or1, or2)
        return float(_math.log(value)), direction
    return float(
        _math.log(float(a)) + _math.log(float(d))
        - _math.log(float(b)) - _math.log(float(c))
    ), direction

def v23_or_value_and_direction(
    a: float,
    b: float,
    c: float,
    d: float,
    *,
    small_threshold: int = SMALL_THRESHOLD,
    sample_type: Optional[str] = None,
) -> Tuple[Optional[float], int]:
    """
    Return the DEMI v23 OR and direction.

    The underlying log-OR is retained without truncation by the internal
    calculation.  Only this OR-scale compatibility value is clipped to the
    largest or smallest positive finite float when exponentiation would
    otherwise overflow or underflow.
    """
    log_or, direction = _v23_log_or_value_and_direction(
        a,
        b,
        c,
        d,
        small_threshold=small_threshold,
        sample_type=sample_type,
    )
    if log_or is None:
        return None, direction
    return _finite_or_from_log_or(log_or), direction


def compute_or_v23_scalar(
    n_exposed_outcome: float,
    n_exposed_no_outcome: float,
    n_unexposed_outcome: float,
    n_unexposed_no_outcome: float,
    *,
    sample_type: Optional[str] = None,
) -> float:
    value, _ = v23_or_value_and_direction(
        n_exposed_outcome,
        n_exposed_no_outcome,
        n_unexposed_outcome,
        n_unexposed_no_outcome,
        sample_type=sample_type,
    )
    return float(value) if value is not None and value > 0 else np.nan


def compute_or_vectorized(df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
    """Compute v23 odds ratios for each row in a DataFrame."""
    sample_type = kwargs.get("sample_type")
    values: List[float] = []
    for _, row in df.iterrows():
        a, b, c, d = _row_to_abcd(row)
        values.append(compute_or_v23_scalar(a, b, c, d, sample_type=sample_type))
    return np.asarray(values, dtype=float)


def _noisy_or_indirect_weight(
    predictor: int,
    downstream_mediators: List[int],
    outcome: int,
    edge_index: pd.DataFrame,
) -> float:
    """Compute v23 noisy-OR indirect weight for one predictor."""
    if not downstream_mediators:
        return 0.0
    product = 1.0
    for mediator in downstream_mediators:
        key_km = (int(mediator), int(predictor))
        if key_km in edge_index.index:
            phi_kj = _phi_from_edge_row(edge_index.loc[key_km])
        else:
            reverse_key = (int(predictor), int(mediator))
            if reverse_key in edge_index.index:
                a2, b2, c2, d2 = _row_to_abcd(edge_index.loc[reverse_key])
                phi_kj = _phi_coefficient(a2, c2, b2, d2)
            else:
                phi_kj = 0.0

        key_my = (int(outcome), int(mediator))
        phi_jy = (
            _phi_from_edge_row(edge_index.loc[key_my])
            if key_my in edge_index.index
            else 0.0
        )
        term = 1.0 - phi_kj**2 * phi_jy**2
        product *= max(0.0, min(1.0, term))
    return 1.0 - product


def _run_noisy_or_direct_effects(
    nodes: List[int],
    temporal_order: List[int],
    outcome: int,
    edge_index: pd.DataFrame,
    total_log_or: pd.Series,
    *,
    zero_threshold: float = NOISY_OR_ZERO_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """Compute D1 and D0 direct effects for the v23 large branch."""
    selected_set = set(nodes)
    pos_by_event = {event: i for i, event in enumerate(temporal_order)}
    direct_log_or_z1 = pd.Series(np.nan, index=nodes, dtype=float)
    direct_log_or_z0 = pd.Series(np.nan, index=nodes, dtype=float)

    for predictor in [n for n in temporal_order if n in selected_set]:
        total_effect = float(total_log_or.get(predictor, np.nan))
        if not np.isfinite(total_effect) or abs(total_effect) <= zero_threshold:
            direct_log_or_z1.loc[predictor] = 0.0
            direct_log_or_z0.loc[predictor] = 0.0
            continue

        pos_k = pos_by_event.get(predictor, -1)
        downstream = [
            int(event)
            for event in temporal_order[pos_k + 1 :]
            if int(event) != int(outcome) and int(event) in selected_set
        ]
        indirect_weight = _noisy_or_indirect_weight(
            predictor, downstream, outcome, edge_index
        )
        indirect_effect = total_effect * indirect_weight
        key = (int(outcome), int(predictor))
        t0 = (
            _compute_T0(*_row_to_abcd(edge_index.loc[key]))
            if key in edge_index.index
            else 0.0
        )
        direct_log_or_z1.loc[predictor] = float(total_effect - indirect_effect)
        direct_log_or_z0.loc[predictor] = float(t0 + indirect_effect)

    return direct_log_or_z1, direct_log_or_z0


def _run_noisy_or_direct_effects_streaming(
    nodes: List[int],
    temporal_order: List[int],
    outcome: int,
    outcome_edge_index: pd.DataFrame,
    total_log_or: pd.Series,
    pairwise_block_reader: Callable[[List[int], List[int]], pd.DataFrame],
    *,
    chunk_size: int = MAX_SQL_IN_CHUNK_SIZE,
    zero_threshold: float = NOISY_OR_ZERO_THRESHOLD,
) -> Tuple[pd.Series, pd.Series]:
    """Accumulate the unchanged noisy-OR product one database block at a time.

    The preferred edge orientation remains ``(downstream, upstream)``.  The
    reverse row is used only when that preferred row is unavailable, exactly as
    in ``_noisy_or_indirect_weight``.  Direct effects are still calculated here;
    no direct-effect database column is assumed.
    """
    requested_nodes = [int(x) for x in nodes]
    selected_set = set(requested_nodes)
    ordered_nodes = [
        int(x)
        for x in temporal_order
        if int(x) in selected_set and int(x) != int(outcome)
    ]
    n_nodes = len(ordered_nodes)
    actual_chunk_size = max(1, min(int(chunk_size), MAX_SQL_IN_CHUNK_SIZE))
    if not ordered_nodes:
        empty = pd.Series(dtype=float).reindex(requested_nodes)
        return empty, empty.copy()

    phi_to_outcome = np.zeros(n_nodes, dtype=float)
    t0 = np.zeros(n_nodes, dtype=float)
    for position, mediator in enumerate(ordered_nodes):
        key = (int(outcome), int(mediator))
        if key not in outcome_edge_index.index:
            continue
        row = outcome_edge_index.loc[key]
        phi_to_outcome[position] = _phi_from_edge_row(row)
        t0[position] = _compute_T0(*_row_to_abcd(row))

    node_chunks = [
        ordered_nodes[start : start + actual_chunk_size]
        for start in range(0, n_nodes, actual_chunk_size)
    ]
    position_by_node = {node: i for i, node in enumerate(ordered_nodes)}
    no_transmission_product = np.ones(n_nodes, dtype=float)

    def load_phi_map(
        target_chunk: List[int],
        concept_chunk: List[int],
    ) -> Dict[Tuple[int, int], float]:
        block = pairwise_block_reader(target_chunk, concept_chunk)
        if block.empty:
            return {}
        prepared = _prepare_demi_dataframe_v23(block)
        if prepared.empty:
            return {}
        edge_index = _build_edge_index(prepared)
        return {
            (int(target_id), int(concept_id)): _phi_from_edge_row(row)
            for (target_id, concept_id), row in edge_index.iterrows()
        }

    for upstream_block_index, upstream_chunk in enumerate(node_chunks):
        for downstream_block_index in range(
            upstream_block_index,
            len(node_chunks),
        ):
            downstream_chunk = node_chunks[downstream_block_index]
            preferred_phi = load_phi_map(downstream_chunk, upstream_chunk)
            reverse_phi = (
                preferred_phi
                if upstream_block_index == downstream_block_index
                else load_phi_map(upstream_chunk, downstream_chunk)
            )

            for upstream in upstream_chunk:
                upstream_position = position_by_node[upstream]
                for downstream in downstream_chunk:
                    downstream_position = position_by_node[downstream]
                    if downstream_position <= upstream_position:
                        continue
                    preferred_key = (int(downstream), int(upstream))
                    reverse_key = (int(upstream), int(downstream))
                    phi_between = preferred_phi.get(preferred_key)
                    if phi_between is None:
                        phi_between = reverse_phi.get(reverse_key)
                    if phi_between is None:
                        continue
                    transmitted = (
                        float(phi_between) ** 2
                        * float(phi_to_outcome[downstream_position]) ** 2
                    )
                    term = 1.0 - transmitted
                    no_transmission_product[upstream_position] *= max(
                        0.0, min(1.0, term)
                    )
    indirect_weight = 1.0 - no_transmission_product
    total = pd.to_numeric(
        total_log_or.reindex(ordered_nodes), errors="coerce"
    ).to_numpy(dtype=float)
    direct_z1 = np.zeros(n_nodes, dtype=float)
    direct_z0 = np.zeros(n_nodes, dtype=float)
    active = np.isfinite(total) & (np.abs(total) > float(zero_threshold))
    indirect = total[active] * indirect_weight[active]
    direct_z1[active] = total[active] - indirect
    direct_z0[active] = t0[active] + indirect
    return (
        pd.Series(direct_z1, index=ordered_nodes, dtype=float).reindex(
            requested_nodes
        ),
        pd.Series(direct_z0, index=ordered_nodes, dtype=float).reindex(
            requested_nodes
        ),
    )


def _small_sample_present_score_weights(selected_df: pd.DataFrame) -> np.ndarray:
    """Return exported v23 net-count weights; never fall back to the OR band."""
    if "net_count_weight" not in selected_df.columns:
        raise KeyError("Net-count scoring requires net_count_weight.")
    return pd.to_numeric(selected_df["net_count_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _net_count_weights(
    phi_values: Any,
    chi_square_values: Any,
    *,
    chi_square_critical: float = CHI_SQUARE_CRITICAL_DEFAULT,
) -> np.ndarray:
    """Return +1 / -1 / 0 votes: sign(phi) when chi-square is significant.

    A predictor in the dead zone (chi-square <= critical value) gets 0.
    The sign of phi equals the sign of ad - bc, so zero-cell tables need no
    special rule: a = 0 votes protective only when the expected count makes
    the absence surprising.
    """
    phi_arr = pd.to_numeric(pd.Series(phi_values), errors="coerce").to_numpy(
        dtype=float
    )
    chi_arr = pd.to_numeric(pd.Series(chi_square_values), errors="coerce").to_numpy(
        dtype=float
    )
    significant = (
        np.isfinite(phi_arr)
        & np.isfinite(chi_arr)
        & (chi_arr > float(chi_square_critical))
    )
    weights = np.zeros(len(phi_arr), dtype=float)
    weights[significant] = np.sign(phi_arr[significant])
    return weights


def select_features(
    feature_records: pd.DataFrame,
    *,
    sample_type: str,
) -> pd.DataFrame:
    """Retain all v23-eligible features without SAFE, EPV, or top-N caps."""
    empty_cols = [
        "predictor_int",
        "a",
        "b",
        "c",
        "d",
        "or_val",
        "direction",
        "abs_log_or",
        "is_edge",
    ]
    if feature_records is None or feature_records.empty:
        return pd.DataFrame(columns=empty_cols)

    sample_type = str(sample_type).lower()

    eligible = feature_records.copy()
    if all(col in eligible.columns for col in ("a", "b", "c", "d")):
        a = eligible["a"].fillna(0).astype(float)
        b = eligible["b"].fillna(0).astype(float)
        c = eligible["c"].fillna(0).astype(float)
        d = eligible["d"].fillna(0).astype(float)
        eligible = eligible[
            ~((a + b == 0) | (a + c == 0) | (b + d == 0) | (c + d == 0))
        ].copy()

    # Small targets retain every eligible row, including neutral/zero-weight
    # rows, so the model table represents the complete eligible feature set.
    valid_direction = eligible["direction"] != 0
    if sample_type != "small":
        eligible = eligible[
            valid_direction
            & eligible["abs_log_or"].notna()
            & np.isfinite(eligible["abs_log_or"].astype(float))
        ].copy()

    if eligible.empty:
        return eligible.reset_index(drop=True)

    if sample_type == "small":
        result = eligible.sort_values(
            ["abs_log_or", "predictor_int"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    else:
        non_edge = eligible[~eligible["is_edge"].astype(bool)].sort_values(
            ["abs_log_or", "predictor_int"], ascending=[False, True]
        )
        edge = eligible[eligible["is_edge"].astype(bool)].sort_values(
            ["abs_log_or", "predictor_int"], ascending=[False, True]
        )
        result = pd.concat([non_edge, edge], axis=0).reset_index(drop=True)
    return result


def _predictor_column_for_int(patient_frame: pd.DataFrame, predictor_int: Any) -> Optional[Any]:
    candidates = [predictor_int, str(predictor_int)]
    try:
        candidates.append(int(predictor_int))
    except Exception:
        pass
    seen = set()
    for candidate in candidates:
        key = (type(candidate).__name__, str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate in patient_frame.columns:
            return candidate
    return None


def _get_score_weights(selected_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return D1 and D0 score weights for selected features.

    Net-count models carry their +1 / -1 / 0 votes in ``net_count_weight``.
    Only present predictors vote, so the absent-state weight is zero.
    """
    if "net_count_weight" in selected_df.columns:
        w1 = (
            pd.to_numeric(selected_df["net_count_weight"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        return w1, np.zeros(len(selected_df), dtype=float)
    is_large = (
        "sample_type" in selected_df.columns
        and len(selected_df) > 0
        and str(selected_df["sample_type"].iloc[0]).lower() == "large"
    )
    if is_large:
        missing = {"direct_log_or_z1", "direct_log_or_z0"} - set(selected_df.columns)
        if missing:
            raise KeyError(f"Large-target scoring requires {sorted(missing)}")
        w1 = selected_df["direct_log_or_z1"].to_numpy(dtype=float)
        w0 = selected_df["direct_log_or_z0"].to_numpy(dtype=float)
    else:
        w1 = _small_sample_present_score_weights(selected_df)
        w0 = np.zeros(len(selected_df), dtype=float)
    return w1, w0


def _score_one_patient_from_mapping(
    selected_df: pd.DataFrame,
    patient_features: Union[Dict[Any, Any], pd.Series],
    measured_features: Optional[Union[Set[Any], Dict[Any, Any]]] = None,
) -> Optional[float]:
    """Score one patient dict/Series with v23 three-state semantics."""
    weights_z1, weights_z0 = _get_score_weights(selected_df)
    total = 0.0
    touched = False
    for i, (_, row) in enumerate(selected_df.iterrows()):
        predictor_int = row["predictor_int"]
        value = patient_features.get(predictor_int, None)
        if value is None:
            value = patient_features.get(str(predictor_int), None)
        if value is None:
            continue
        try:
            present = float(value) != 0.0 and not pd.isna(value)
        except Exception:
            present = bool(value)
        if present:
            w1_i = float(weights_z1[i])
            total += w1_i
            touched = touched or w1_i != 0.0
        else:
            w0_i = float(weights_z0[i])
            if measured_features is None:
                total += w0_i
                touched = touched or w0_i != 0.0
            elif (
                predictor_int in measured_features or str(predictor_int) in measured_features
            ):
                total += w0_i
                touched = touched or w0_i != 0.0
    if not touched:
        return None
    return float(total)


def predict_score_only(
    selected_df: pd.DataFrame,
    patient_features: Union[
        Dict[Any, Any], pd.Series, pd.DataFrame, np.ndarray, List[Any]
    ],
    measured_features: Optional[Union[Set[Any], pd.DataFrame, np.ndarray]] = None,
) -> Union[Optional[float], np.ndarray]:
    """Return one or many raw DEMI v23 scores."""
    if selected_df is None or selected_df.empty:
        return None

    weights_z1, weights_z0 = _get_score_weights(selected_df)
    if isinstance(patient_features, pd.DataFrame):
        n_rows = len(patient_features)
        scores = np.zeros(n_rows, dtype=float)
        touched = np.zeros(n_rows, dtype=bool)
        for i, (_, row) in enumerate(selected_df.iterrows()):
            col = _predictor_column_for_int(patient_features, row["predictor_int"])
            if col is None:
                continue
            vals = patient_features[col].fillna(0).astype(float).to_numpy()
            present = vals != 0.0
            w1_i = float(weights_z1[i])
            scores += present.astype(float) * w1_i
            if w1_i != 0.0:
                touched |= present
            w0_i = float(weights_z0[i])
            if measured_features is None:
                scores += (~present).astype(float) * w0_i
                if w0_i != 0.0:
                    touched |= np.ones(n_rows, dtype=bool)
            elif isinstance(measured_features, pd.DataFrame):
                m_col = _predictor_column_for_int(measured_features, row["predictor_int"])
                measured = (
                    measured_features[m_col].astype(bool).to_numpy()
                    if m_col is not None
                    else np.zeros(n_rows, dtype=bool)
                )
                scores += ((~present) & measured).astype(float) * w0_i
                if w0_i != 0.0:
                    touched |= measured
            elif isinstance(measured_features, np.ndarray):
                measured = measured_features[:, i].astype(bool)
                scores += ((~present) & measured).astype(float) * w0_i
                if w0_i != 0.0:
                    touched |= measured
        scores[~touched] = np.nan
        return scores

    if isinstance(patient_features, (dict, pd.Series)):
        measured = (
            measured_features
            if isinstance(measured_features, (set, frozenset, type(None)))
            else None
        )
        return _score_one_patient_from_mapping(selected_df, patient_features, measured)

    arr = np.asarray(patient_features, dtype=float)
    if arr.ndim == 1:
        if len(arr) != len(weights_z1):
            raise ValueError("For array input, length must match selected predictors.")
        present = arr != 0.0
        absent_measured = (
            ~present & measured_features.astype(bool)
            if isinstance(measured_features, np.ndarray)
            else ~present
        )
        active = (
            (present & (weights_z1 != 0.0))
            | (absent_measured & (weights_z0 != 0.0))
        )
        if not np.any(active):
            return None
        return float(
            np.dot(present.astype(float), weights_z1)
            + np.dot(absent_measured.astype(float), weights_z0)
        )

    if arr.ndim == 2:
        if arr.shape[1] != len(weights_z1):
            raise ValueError(
                "For 2D array input, columns must match selected predictors."
            )
        present = arr != 0.0
        absent_measured = (
            ~present & measured_features.astype(bool)
            if isinstance(measured_features, np.ndarray)
            else ~present
        )
        scores = (
            present.astype(float) @ weights_z1
            + absent_measured.astype(float) @ weights_z0
        )
        touched = (
            (present & (weights_z1 != 0.0))
            | (absent_measured & (weights_z0 != 0.0))
        ).any(axis=1)
        scores[~touched] = np.nan
        return scores.astype(float)

    raise ValueError(
        "patient_features must be dict, Series, DataFrame, 1D array, or 2D array."
    )


def predict_patient(
    selected_df: pd.DataFrame,
    patient_features: Union[
        Dict[Any, Any], pd.Series, pd.DataFrame, np.ndarray, List[Any]
    ],
    cal: Optional[Dict[str, Any]] = None,
    measured_features: Optional[Union[Set[Any], pd.DataFrame, np.ndarray]] = None,
    *,
    prior_log_odds: Optional[float] = None,
    scoring_rule: Optional[str] = None,
) -> Union[Optional[float], np.ndarray]:
    """Convert a raw score once, using its scoring rule's probability scale.

    Weighted large targets require count-based ``prior_log_odds`` and use
    sigmoid(prior_log_odds + score). No-evidence scores contribute zero.
    Net-count targets retain three-anchor calibration, including its prior.
    The existing four positional arguments remain supported; standalone
    weighted callers must additionally supply ``prior_log_odds``.
    """
    inferred_rule = None
    if selected_df is not None:
        if "net_count_weight" in selected_df.columns:
            inferred_rule = SCORING_RULE_NET_COUNT
        elif not selected_df.empty:
            if "scoring_rule" in selected_df.columns:
                inferred_rule = str(selected_df["scoring_rule"].iloc[0])
            elif "sample_type" in selected_df.columns:
                inferred_rule = (
                    SCORING_RULE_DIRECT_EFFECTS
                    if str(selected_df["sample_type"].iloc[0]).lower() == "large"
                    else SCORING_RULE_NET_COUNT
                )
    rule = scoring_rule or inferred_rule
    if rule is None:
        raise ValueError("Provide scoring_rule when selected_df has no scoring metadata.")
    if rule not in {SCORING_RULE_DIRECT_EFFECTS, SCORING_RULE_NET_COUNT}:
        raise ValueError("Unsupported DEMI v23 scoring_rule: %r" % rule)
    if rule == SCORING_RULE_DIRECT_EFFECTS and inferred_rule == SCORING_RULE_NET_COUNT:
        raise ValueError("Net-count scores cannot use the weighted logistic conversion.")

    score = predict_score_only(selected_df, patient_features, measured_features)
    if rule == SCORING_RULE_DIRECT_EFFECTS:
        if prior_log_odds is None or not np.isfinite(prior_log_odds):
            raise ValueError("Weighted prediction requires finite prior_log_odds.")
        if isinstance(score, np.ndarray):
            # predict_score_only marks rows with no usable evidence as NaN.
            evidence_score = np.where(np.isnan(score), 0.0, score)
        elif score is None:
            # Preserve the row count for empty models on batch input.
            if isinstance(patient_features, pd.DataFrame):
                evidence_score = np.zeros(len(patient_features), dtype=float)
            else:
                values = np.asarray(patient_features)
                evidence_score = (
                    np.zeros(values.shape[0], dtype=float)
                    if values.ndim == 2 else 0.0
                )
        else:
            evidence_score = float(score)
        log_odds = float(prior_log_odds) + evidence_score
        probability = np.exp(-np.logaddexp(0.0, -log_odds))
        return probability if isinstance(probability, np.ndarray) else float(probability)

    if score is None:
        return None
    if cal is None:
        if isinstance(score, np.ndarray):
            return np.full(score.shape, np.nan, dtype=float)
        return None
    if isinstance(score, np.ndarray):
        return np.array(
            [
                np.nan if np.isnan(s) else three_anchor_transform(float(s), cal)
                for s in score
            ],
            dtype=float,
        )
    return three_anchor_transform(float(score), cal)


def _prepare_demi_dataframe_v23(
    df_in: Union[str, pd.DataFrame],
) -> pd.DataFrame:
    """Normalize count columns and remove degenerate 2x2 rows."""
    df = pd.read_csv(df_in) if isinstance(df_in, str) else df_in.copy()
    for col in ["target_concept_code_int", "concept_code_int"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col!r}")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in [
        "n_code_target",
        "n_code",
        "n_target",
        "n_no_target",
        "n_code_before_target",
        "n_target_before_code",
        "n_code_no_target",
        "n_target_no_code",
        "n_no_code_no_target",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required_cells = [
        "n_code_target",
        "n_code_no_target",
        "n_target_no_code",
        "n_no_code_no_target",
    ]
    if all(col in df.columns for col in required_cells):
        a = df["n_code_target"].fillna(0)
        b = df["n_code_no_target"].fillna(0)
        c = df["n_target_no_code"].fillna(0)
        d = df["n_no_code_no_target"].fillna(0)
        degenerate = (a + b == 0) | (a + c == 0) | (b + d == 0) | (c + d == 0)
        df = df[~degenerate].copy()

    return df


def _apply_large_rollup_exclusion(
    df: pd.DataFrame,
    *,
    outcome: int,
) -> pd.DataFrame:
    """Apply v23 name-pattern rollup filtering for large targets."""
    df = df.copy()
    name_col = None
    for candidate in ("concept_name", "standard_concept_name"):
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        return df

    name = df[name_col].fillna("")
    broad_pattern = (
        r"\bfinding\b|\bfindings\b|\bdisorder\b|\bdisorder of\b|^disorder of\b"
        r"|\bdisease\b|^disease$|\bcondition\b|\bstructure\b|\bobservation\b"
    )
    rule1_mask = name.str.contains(broad_pattern, case=False, regex=True, na=False)
    generic_ending = (
        r"\bdisorder\b|\bdisease\b|\bsyndrome\b"
        r"|\bcondition\b|\bproblem\b|\bsymptom\b|\bfinding\b"
    )
    has_generic = name.str.contains(generic_ending, case=False, regex=True, na=False)
    if "n_target_no_code" in df.columns and "n_code_target" in df.columns:
        ac_col = df["n_code_target"].fillna(0) + df["n_target_no_code"].fillna(0)
        c_rate = df["n_target_no_code"].fillna(0) / ac_col.replace(0, np.nan)
        rule2_mask = has_generic & (c_rate < 0.03)
    else:
        rule2_mask = pd.Series(False, index=df.index)
    rollup_mask = (rule1_mask | rule2_mask) & (df["concept_code_int"] != int(outcome))
    return df[~rollup_mask].copy()


def _build_edge_index(df: pd.DataFrame) -> pd.DataFrame:
    edge_cols = [
        "n_target_before_code",
        "n_code_before_target",
        "n_code_target",
        "n_code",
        "n_target",
        "n_no_target",
        "n_code_no_target",
        "n_target_no_code",
        "n_no_code_no_target",
        "total_effects",
        "total_effects_log",
        "total_effects_abs",
        "phi",
        "phi_abs",
        "phi_rank",
        "direction",
    ]
    present = [col for col in edge_cols if col in df.columns]
    keys = ["target_concept_code_int", "concept_code_int"]
    if not df.duplicated(keys).any():
        return df.set_index(keys)[present]
    precomputed = {
        "total_effects",
        "total_effects_log",
        "total_effects_abs",
        "phi",
        "phi_abs",
        "phi_rank",
        "direction",
    }
    aggregation = {
        column: ("first" if column in precomputed else "sum")
        for column in present
    }
    return df.groupby(keys, as_index=True).agg(aggregation)


def _compute_temporal_order(
    df: pd.DataFrame,
    edge_index: pd.DataFrame,
    outcome: int,
) -> List[int]:
    """Compatibility-only temporal order calculated from the supplied rows."""
    all_events = sorted(
        set(int(x) for x in df["target_concept_code_int"].dropna().unique())
        | set(int(x) for x in df["concept_code_int"].dropna().unique())
    )
    scores: Dict[int, float] = {}
    for event_id in all_events:
        try:
            pairs = edge_index.xs(event_id, level="target_concept_code_int")
            tbc = (
                pairs["n_target_before_code"].fillna(0.0)
                if "n_target_before_code" in pairs
                else 0.0
            )
            cbt = (
                pairs["n_code_before_target"].fillna(0.0)
                if "n_code_before_target" in pairs
                else 0.0
            )
            scores[event_id] = float((tbc - cbt).sum() if hasattr(tbc, "sum") else 0.0)
        except KeyError:
            scores[event_id] = 0.0
    sorted_scores = pd.Series(scores).sort_values(ascending=False)
    return [
        int(event) for event in sorted_scores.index if int(event) != int(outcome)
    ] + [int(outcome)]


def _coerce_global_temporal_scores(
    global_temporal_order: Union[pd.Series, pd.DataFrame, Dict[int, Any]],
) -> pd.Series:
    """Normalize indexed global temporal-order values to a numeric Series."""
    if isinstance(global_temporal_order, pd.Series):
        scores = global_temporal_order.copy()
    elif isinstance(global_temporal_order, pd.DataFrame):
        required = {"concept_code_int", "temporal_order"}
        missing = required - set(global_temporal_order.columns)
        if missing:
            raise KeyError(
                "Global temporal-order DataFrame is missing columns: "
                f"{sorted(missing)}"
            )
        frame = global_temporal_order[
            ["concept_code_int", "temporal_order"]
        ].drop_duplicates("concept_code_int", keep="first")
        scores = frame.set_index("concept_code_int")["temporal_order"]
    elif isinstance(global_temporal_order, dict):
        scores = pd.Series(global_temporal_order, dtype=float)
    else:
        raise TypeError(
            "global_temporal_order must be a Series, DataFrame, or dictionary."
        )

    numeric_index = pd.to_numeric(pd.Series(scores.index), errors="coerce")
    numeric_values = pd.to_numeric(pd.Series(scores.to_numpy()), errors="coerce")
    frame = pd.DataFrame(
        {
            "concept_code_int": numeric_index.to_numpy(),
            "temporal_order": numeric_values.to_numpy(),
        }
    ).dropna()
    frame["concept_code_int"] = frame["concept_code_int"].astype(int)
    frame = frame.drop_duplicates("concept_code_int", keep="first")
    result = frame.set_index("concept_code_int")["temporal_order"].astype(float)
    result.name = "temporal_order"
    return result


def _compute_global_temporal_order(
    event_ints: Iterable[int],
    outcome: int,
    global_temporal_order: Union[pd.Series, pd.DataFrame, Dict[int, Any]],
    *,
    strict: bool = True,
) -> List[int]:
    """Order concepts by the indexed global score and put the outcome last."""
    events = sorted({int(x) for x in event_ints} | {int(outcome)})
    source = _coerce_global_temporal_scores(global_temporal_order)
    scores = source.reindex(events)
    missing_ids = [
        int(x)
        for x in scores.index[scores.isna()].tolist()
        if int(x) != int(outcome)
    ]
    if missing_ids and strict:
        raise ValueError(
            "Global temporal order is missing "
            f"{len(missing_ids)} selected concept(s). Examples: {missing_ids[:10]}"
        )
    scores = scores.fillna(0.0).astype(float)
    order_frame = pd.DataFrame(
        {
            "concept_code_int": scores.index.astype(int),
            "temporal_order": scores.to_numpy(dtype=float),
        }
    ).sort_values(
        ["temporal_order", "concept_code_int"],
        ascending=[False, True],
        kind="mergesort",
    )
    return [
        int(x)
        for x in order_frame["concept_code_int"].tolist()
        if int(x) != int(outcome)
    ] + [int(outcome)]


def _build_outcome_feature_records(
    nodes: List[int],
    outcome: int,
    edge_index: pd.DataFrame,
    *,
    sample_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    records: List[Dict[str, Any]] = []
    total_log_or = pd.Series(np.nan, index=nodes, dtype=float)
    for predictor in nodes:
        key = (int(outcome), int(predictor))
        if key not in edge_index.index:
            continue
        row = edge_index.loc[key]
        a, b, c, d = _row_to_abcd(row)
        count_log_or, count_direction = _v23_log_or_value_and_direction(
            a, b, c, d, sample_type=sample_type
        )
        count_or = _finite_or_from_log_or(count_log_or) if count_log_or is not None else None
        is_edge = not (a > 0 and b > 0 and c > 0 and d > 0)

        stored_or = _finite_optional_float(row.get("total_effects"))
        stored_log_or = _finite_optional_float(row.get("total_effects_log"))
        stored_abs_log_or = _finite_optional_float(row.get("total_effects_abs"))
        stored_or_valid = stored_or is not None and stored_or > 0.0
        stored_log_valid = stored_log_or is not None

        # A valid stored total effect is the DB source of truth; counts remain
        # the compatibility fallback. Direct effects are never read here.
        direction = _parse_stored_direction(row.get("direction"))
        if direction is None:
            direction = int(count_direction)
        if stored_log_valid:
            resolved_log_or = float(stored_log_or)
            if stored_or_valid:
                or_value = float(stored_or)
            else:
                max_log = _math.log(np.finfo(float).max)
                min_log = _math.log(np.nextafter(0.0, 1.0))
                or_value = float(
                    _math.exp(min(max(resolved_log_or, min_log), max_log))
                )
            used_db_total_effect = True
        elif stored_or_valid:
            or_value = float(stored_or)
            resolved_log_or = float(_math.log(or_value))
            used_db_total_effect = True
        elif count_log_or is not None and np.isfinite(count_log_or):
            or_value = float(count_or)
            resolved_log_or = float(count_log_or)
            used_db_total_effect = False
        else:
            or_value = None
            resolved_log_or = np.nan
            used_db_total_effect = False

        # Preserve the database total effect as the source value. Only the
        # small net-count scores depend on phi, not on this OR magnitude.
        scoring_or = (
            np.nan
            if str(sample_type).lower() == "small" and is_edge
            else float(or_value) if or_value is not None else np.nan
        )
        if np.isfinite(resolved_log_or):
            total_log_or.loc[predictor] = float(resolved_log_or)
        abs_log_value = (
            float(stored_abs_log_or)
            if used_db_total_effect
            and stored_abs_log_or is not None
            and stored_abs_log_or >= 0.0
            else abs(float(resolved_log_or))
            if np.isfinite(resolved_log_or)
            else None
        )
        # Phi (stored when valid, else from counts) and the Pearson
        # chi-square N * phi^2 used by the net-count dead zone.
        phi_value = _phi_from_edge_row(row)
        n_total = float(a + b + c + d)
        chi_square = (
            float(n_total * phi_value**2)
            if n_total > 0 and np.isfinite(phi_value)
            else np.nan
        )
        records.append(
            {
                "predictor_int": int(predictor),
                "a": int(a),
                "b": int(b),
                "c": int(c),
                "d": int(d),
                "or_val": scoring_or,
                "direction": int(direction),
                "abs_log_or": abs_log_value,
                "is_edge": bool(is_edge),
                "phi": float(phi_value),
                "chi_square": chi_square,
            }
        )
    return pd.DataFrame(records), total_log_or


def run_demi(
    df_in: Union[str, pd.DataFrame],
    *,
    force_outcome: int,
    calibration_table: Optional[Union[str, pd.DataFrame]] = None,
    target_id: Optional[Any] = None,
    pref_override: Optional[float] = None,
    noisy_or_zero_threshold: float = NOISY_OR_ZERO_THRESHOLD,
    global_temporal_order: Optional[
        Union[pd.Series, pd.DataFrame, Dict[int, Any]]
    ] = None,
    require_global_temporal_order: bool = True,
    pairwise_block_reader: Optional[
        Callable[[List[int], List[int]], pd.DataFrame]
    ] = None,
    pairwise_stream_chunk_size: int = MAX_SQL_IN_CHUNK_SIZE,
    chi_square_critical: float = CHI_SQUARE_CRITICAL_DEFAULT,
    net_count_all_targets: bool = False,
    large_phi_filter: bool = True,
) -> Dict[str, Any]:
    """Build only the scoring state required by ``DEMIService``.

    Small targets (and every target when ``net_count_all_targets`` is set)
    use the net-count rule: each present predictor adds sign(phi) when its
    chi-square N * phi^2 exceeds ``chi_square_critical`` and 0 otherwise.
    Large targets otherwise keep the v23 noisy-OR direct-effect scoring;
    with ``large_phi_filter`` (default) only predictors whose chi-square
    exceeds ``chi_square_critical`` enter the noisy-OR step and the score.
    """
    if not np.isfinite(chi_square_critical) or chi_square_critical < 0:
        raise ValueError("chi_square_critical must be finite and nonnegative.")
    outcome = int(force_outcome)
    target_key = str(target_id if target_id is not None else outcome)
    df = _prepare_demi_dataframe_v23(df_in)

    outcome_rows = df[df["target_concept_code_int"] == outcome]
    (
        _,
        n_outcome_positive,
        n_outcome_negative,
    ) = _compute_outcome_prevalence(outcome_rows, pref_override=pref_override)

    n_cases_for_selection = (
        int(n_outcome_positive) if n_outcome_positive is not None else 0
    )
    sample_type = "small" if n_cases_for_selection < SMALL_THRESHOLD else "large"
    use_net_count = sample_type == "small" or bool(net_count_all_targets)
    scoring_rule = (
        SCORING_RULE_NET_COUNT if use_net_count else SCORING_RULE_DIRECT_EFFECTS
    )

    if sample_type == "large":
        df = _apply_large_rollup_exclusion(df, outcome=outcome)

    edge_index = _build_edge_index(df)
    candidate_ids = sorted(set(int(x) for x in df.loc[df["target_concept_code_int"] == outcome, "concept_code_int"].dropna()) - {outcome})
    feature_records, total_log_or = _build_outcome_feature_records(
        candidate_ids, outcome, edge_index, sample_type=sample_type)
    selected_features = select_features(feature_records, sample_type="small" if use_net_count else sample_type)
    n_dead_zone_removed = 0
    if not use_net_count and large_phi_filter and not selected_features.empty:
        chi = pd.to_numeric(selected_features["chi_square"], errors="coerce").to_numpy(dtype=float)
        passes = np.isfinite(chi) & (chi > float(chi_square_critical))
        n_dead_zone_removed = int((~passes).sum())
        selected_features = selected_features[passes].reset_index(drop=True)
    selected_set = set(int(x) for x in selected_features["predictor_int"]) if not selected_features.empty else set()
    event_ids = selected_set | {outcome}
    if use_net_count or not selected_set:
        temporal_order = sorted(selected_set) + [outcome]
    elif global_temporal_order is not None:
        temporal_order = _compute_global_temporal_order(event_ids, outcome, global_temporal_order, strict=require_global_temporal_order)
    else:
        if require_global_temporal_order:
            raise ValueError("Indexed global temporal-order values are required.")
        local_order = _compute_temporal_order(df, edge_index, outcome)
        temporal_order = [x for x in local_order if x in selected_set] + [outcome]
    nodes = temporal_order[:-1]

    selected_features = (
        selected_features.set_index("predictor_int", drop=False)
        .reindex(nodes)
        .dropna(subset=["predictor_int"])
        .reset_index(drop=True)
        if not selected_features.empty
        else selected_features
    )

    if not use_net_count:
        if pairwise_block_reader is not None:
            direct_log_or_z1, direct_log_or_z0 = (
                _run_noisy_or_direct_effects_streaming(
                    nodes,
                    temporal_order,
                    outcome,
                    edge_index,
                    total_log_or,
                    pairwise_block_reader,
                    chunk_size=pairwise_stream_chunk_size,
                    zero_threshold=noisy_or_zero_threshold,
                )
            )
        else:
            direct_log_or_z1, direct_log_or_z0 = _run_noisy_or_direct_effects(
                nodes,
                temporal_order,
                outcome,
                edge_index,
                total_log_or,
                zero_threshold=noisy_or_zero_threshold,
            )
    else:
        direct_log_or_z1 = pd.Series(np.nan, index=nodes, dtype=float)
        direct_log_or_z0 = pd.Series(np.nan, index=nodes, dtype=float)

    if not selected_features.empty:
        selected_features["direct_log_or_z1"] = (
            selected_features["predictor_int"]
            .map(direct_log_or_z1.to_dict())
            .astype(float)
        )
        selected_features["direct_log_or_z0"] = (
            selected_features["predictor_int"]
            .map(direct_log_or_z0.to_dict())
            .astype(float)
        )
        selected_features["direct_log_or_delta"] = selected_features["direct_log_or_z1"] - selected_features["direct_log_or_z0"]
        selected_features["direct_or_z1"] = _finite_or_array_from_log_or(selected_features["direct_log_or_z1"])
        selected_features["scoring_rule"] = scoring_rule
        selected_features["chi_square_critical"] = float(chi_square_critical)
        selected_features["sample_type"] = sample_type
        if use_net_count:
            selected_features["net_count_weight"] = _net_count_weights(
                selected_features["phi"].to_numpy(),
                selected_features["chi_square"].to_numpy(),
                chi_square_critical=chi_square_critical,
            )

    def _predictor_series(column: str) -> pd.Series:
        if selected_features.empty or column not in selected_features.columns:
            return pd.Series(np.nan, index=nodes, dtype=float)
        return pd.Series(
            pd.to_numeric(selected_features[column], errors="coerce").to_numpy(
                dtype=float
            ),
            index=selected_features["predictor_int"].astype(int).to_numpy(),
            dtype=float,
        )

    phi_by_node = _predictor_series("phi")
    chi_square_by_node = _predictor_series("chi_square")
    net_count_weight = (
        _predictor_series("net_count_weight")
        if use_net_count
        else pd.Series(dtype=float)
    )

    p_marginal = (
        float(n_outcome_positive) / float(n_outcome_positive + n_outcome_negative)
        if (
            n_outcome_positive is not None
            and n_outcome_negative is not None
            and n_outcome_positive + n_outcome_negative > 0
        )
        else 0.5
    )
    p_marginal = float(pref_override) if pref_override is not None else p_marginal
    p_marginal = min(max(p_marginal, 1e-12), 1.0 - 1e-12)
    prior_odds = float(p_marginal / (1.0 - p_marginal))
    prior_log_odds = float(np.log(prior_odds))
    probability_method = (
        PROBABILITY_METHOD_CALIBRATION if use_net_count
        else PROBABILITY_METHOD_PRIOR_SCORE
    )
    calibration = read_calibration(target_key, calibration_table) if use_net_count else None

    def predict_score(patient_events: Any, measured_features: Any = None) -> Any:
        return predict_score_only(
            selected_features, patient_events, measured_features
        )

    def predict_proba(patient_events: Any, measured_features: Any = None) -> Any:
        return predict_patient(
            selected_features, patient_events, calibration, measured_features,
            prior_log_odds=prior_log_odds, scoring_rule=scoring_rule,
        )

    return {
        "version": "DEMI_v23",
        "updated_date": "2026-10-15",
        "sample_type": sample_type,
        "scoring_rule": scoring_rule,
        "direct_effect_method": scoring_rule,
        "chi_square_critical": (
            float(chi_square_critical)
            if use_net_count or large_phi_filter
            else None
        ),
        "large_phi_filter": bool(large_phi_filter and not use_net_count),
        "n_dead_zone_removed": n_dead_zone_removed,
        "pairwise_processing_mode": (
            "not_required_net_count" if use_net_count
            else "streamed_db" if pairwise_block_reader
            else "materialized"
        ),
        "selected_features": selected_features,
        "effect_df": selected_features.copy(),
        "net_count_weight": net_count_weight,
        "phi": phi_by_node,
        "chi_square": chi_square_by_node,
        "n_net_count_votes": (
            int((net_count_weight != 0).sum()) if use_net_count else None
        ),
        "direct_log_or_z1": direct_log_or_z1,
        "direct_log_or_z0": direct_log_or_z0,
        "direct_or_z1": pd.Series(_finite_or_array_from_log_or(direct_log_or_z1), index=direct_log_or_z1.index),
        "direct_log_or_delta": direct_log_or_z1 - direct_log_or_z0,
        "total_log_or": total_log_or.reindex(nodes),
        "prior_log_odds": prior_log_odds,
        "prior_odds": prior_odds,
        "probability_method": probability_method,
        "prior_source": "pref_override" if pref_override is not None else "target_counts",
        "prior_log_odds_applied": not use_net_count,
        "calibration_prior": float(calibration["prior"]) if calibration is not None else None,
        "n_outcome_positive": n_outcome_positive,
        "n_outcome_negative": n_outcome_negative,
        "calibration_status": (
            "not_used_weighted_logistic" if not use_net_count
            else "read_from_table" if calibration is not None else "missing"
        ),
        "baseline_probability": p_marginal,
        "predict_score": predict_score,
        "predict_proba": predict_proba,
    }

__all__ = [
    "run_demi",
    "predict_patient",
    "predict_score_only",
    "select_features",
    "compute_or_v23_scalar",
    "compute_or_vectorized",
    "v23_or_value_and_direction",
    "read_calibration",
    "three_anchor_transform",
]
