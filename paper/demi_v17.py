"""DEMI version 17 as a library: the algorithm version evaluated in Alemi et al.
(2026, doi:10.1016/j.jbi.2026.105111).

Kept in paper/ so the article's computation stays runnable; the current
algorithm is demi.py in the repository root. paper/demi_as_reviewed.py is the
same v17 algorithm in the form that accompanied the manuscript through review.

Requires NumPy and pandas. Set DEMI_DB_PATH to an authorized SQLite database
(not distributed). Pass db_path=None to run on an in-memory pairwise table, as in
examples/synthetic_example.py.

Usage:
    import demi_v17 as demi
    edges = demi.fetch_demi_subgraph_by_ints(db_path, event_ids)
    result = demi.run_demi(edges, force_outcome=outcome_id, db_path=db_path)
    probability = result["predict_proba"]({predictor_id: 1})

Copyright license: see the accompanying LICENSE (code only)."""

import os
import sqlite3
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Exported symbols
__all__ = [
    "SNOMEDHierarchy",
    "compute_or_v17_scalar",
    "compute_or_vectorized",
    "make_correction_factor_row",
    "build_correction_factor_table_for_target",
    "get_snomed_ancestor_exclusion_mask",
    "connect_sqlite_read_only",
    "fetch_demi_correction_factors_from_db",
    "fetch_temporal_order_from_db",
    "fetch_code_group_blocklist_from_db",
    "get_code_group_ancestor_exclusion_mask",
    "select_top_predictors",
    "analyze_causal_sequence_py",
    "run_demi",
    "fetch_demi_subgraph_for_events",
    "fetch_demi_subgraph_by_ints",
    "fetch_demi_subgraph_two_sided_chunked",
]


# CONFIG: set these environment variables to point at an authorized database and a
# SNOMED CT release. None of these files is distributed with the code.
DEMI_DB_PATH    = os.getenv("DEMI_DB_PATH") or None
EDGE_TABLE      = os.getenv("DEMI_EDGE_TABLE",   "demi_counts_published")
CONCEPT_TABLE   = os.getenv("DEMI_CONCEPT_TABLE", "concept_names")
CORRECTION_FACTOR_TABLE = os.getenv("DEMI_CORRECTION_FACTOR_TABLE", "demi_correction_factors")
CODE_GROUP_TABLE        = os.getenv("DEMI_CODE_GROUP_TABLE",        "code_group_mapping")
TEMPORAL_ORDER_TABLE    = os.getenv("DEMI_TEMPORAL_ORDER_TABLE",    "demi_temporal_order_published")

# Full RF2 relationship file — used by SNOMEDHierarchy for blocklist + subtype enumeration
SNOMED_REL_PATH = os.getenv("SNOMED_REL_PATH") or None

# Accepts either the Full file (sct2_Relationship_Full_*) or a Snapshot file
# (sct2_Relationship_Snapshot_*) — build_snomed_ancestor_sets handles both.
SNOMED_SNAPSHOT_PATH = os.getenv("SNOMED_SNAPSHOT_PATH") or None

TOP_K           = int(os.getenv("DEMI_TOP_K", "500"))

# Optional CSV for stored correction-factor audit values.
# Stored values do not replace the current-model correction factor.
# Columns: concept_code_int, demi_log_lambda_net, demi_beta0_corrected
CORRECTION_FACTOR_TABLE_PATH = os.getenv("DEMI_CORRECTION_FACTOR_TABLE_PATH", "")

# Backward-compatible alias for older notebooks.
DEMI_CORRECTION_FACTORS_PATH = CORRECTION_FACTOR_TABLE_PATH

# SQLite typically allows at most 999 bound parameters per statement.
# We use SQLITE_PARAM_LIMIT as a conservative ceiling across environments.
# Two-sided queries use the list twice, so the safe per-list ceiling is SQLITE_PARAM_LIMIT // 2.
SQLITE_PARAM_LIMIT = 900


def connect_sqlite_read_only(db_path: str) -> sqlite3.Connection:
    """Open an existing SQLite database in read-only mode."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DEMI database not found: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def fetch_demi_correction_factors_from_db(
    db_path: str,
    concept_code_int: int,
    *,
    correction_factor_table: str = CORRECTION_FACTOR_TABLE,
) -> pd.DataFrame:
    """Fetch the one-row correction-factor table for an outcome from SQLite."""
    q = f"""
        SELECT concept_code_int, demi_log_lambda_net, demi_beta0_corrected
        FROM {correction_factor_table}
        WHERE concept_code_int = ?
        LIMIT 1
    """
    with connect_sqlite_read_only(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name = ? COLLATE NOCASE",
            (correction_factor_table,),
        ).fetchone()
        if exists is None:
            raise KeyError(f"Optional audit table not found: {correction_factor_table}")
        out = pd.read_sql_query(q, conn, params=[int(concept_code_int)])
    if out.empty:
        raise KeyError(
            f"No row found in {correction_factor_table} for "
            f"concept_code_int={int(concept_code_int)}."
        )
    return out


def fetch_temporal_order_from_db(
    db_path: str,
    concept_code_ints: Set[int],
    *,
    temporal_order_table: str = TEMPORAL_ORDER_TABLE,
    chunk_size: int = SQLITE_PARAM_LIMIT,
) -> pd.Series:
    """Fetch global temporal-order scores from demi_temporal_order_published."""
    codes = sorted({int(x) for x in concept_code_ints if pd.notna(x)})
    if not codes:
        return pd.Series(dtype=float)

    frames: List[pd.DataFrame] = []
    with connect_sqlite_read_only(db_path) as conn:
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            ph = ",".join(["?"] * len(chunk))
            q = f"""
                SELECT concept_code_int, temporal_order
                FROM {temporal_order_table}
                WHERE concept_code_int IN ({ph})
            """
            frames.append(pd.read_sql_query(q, conn, params=chunk))

    if not frames:
        return pd.Series(dtype=float)

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return pd.Series(dtype=float)

    out["concept_code_int"] = pd.to_numeric(out["concept_code_int"], errors="coerce")
    out["temporal_order"] = pd.to_numeric(out["temporal_order"], errors="coerce")
    out = out.dropna(subset=["concept_code_int", "temporal_order"])
    if out.empty:
        return pd.Series(dtype=float)

    out["concept_code_int"] = out["concept_code_int"].astype(int)
    out = out.sort_values(["concept_code_int"]).drop_duplicates("concept_code_int")

    return pd.Series(
        out["temporal_order"].astype(float).values,
        index=out["concept_code_int"].astype(int),
        dtype=float,
    )


def _compute_local_temporal_scores(
    all_events: List[int],
    edge_index: pd.DataFrame,
) -> Dict[int, float]:
    """Recompute temporal-order scores from the local subgraph as a fallback."""
    scores: Dict[int, float] = {}
    for zi in all_events:
        try:
            pairs = edge_index.xs(zi, level="target_concept_code_int")
            tbc = (
                pairs["n_target_before_code"].fillna(0.0)
                if "n_target_before_code" in pairs else 0.0
            )
            cbt = (
                pairs["n_code_before_target"].fillna(0.0)
                if "n_code_before_target" in pairs else 0.0
            )
            scores[int(zi)] = float((tbc - cbt).sum() if hasattr(tbc, "sum") else 0.0)
        except KeyError:
            scores[int(zi)] = 0.0
    return scores


def _get_temporal_scores(
    all_events: List[int],
    edge_index: pd.DataFrame,
    *,
    db_path: Optional[str] = DEMI_DB_PATH,
    temporal_order_table_name: str = TEMPORAL_ORDER_TABLE,
    use_db_temporal_order: bool = True,
) -> Dict[int, float]:
    """Return temporal scores, preferring the DB-published table when available."""
    if not use_db_temporal_order or db_path is None:
        return _compute_local_temporal_scores(all_events, edge_index)

    try:
        db_scores = fetch_temporal_order_from_db(
            db_path,
            set(all_events),
            temporal_order_table=temporal_order_table_name,
        )
    except Exception as e:
        print(
            f"[WARN] Could not read {temporal_order_table_name}: {e}. "
            "Falling back to local temporal-order recomputation."
        )
        return _compute_local_temporal_scores(all_events, edge_index)

    local_scores: Optional[Dict[int, float]] = None
    scores: Dict[int, float] = {}
    missing: List[int] = []
    for ev in all_events:
        ev_int = int(ev)
        if ev_int in db_scores.index and np.isfinite(float(db_scores.loc[ev_int])):
            scores[ev_int] = float(db_scores.loc[ev_int])
        else:
            missing.append(ev_int)

    if missing:
        local_scores = _compute_local_temporal_scores(all_events, edge_index)
        for ev_int in missing:
            scores[ev_int] = float(local_scores.get(ev_int, 0.0))
        print(
            f"[WARN] {len(missing):,} concepts were missing from "
            f"{temporal_order_table_name}; used local fallback scores for them."
        )

    return scores


def fetch_code_group_blocklist_from_db(
    db_path: str,
    target_concept_code_int: int,
    *,
    code_group_table: str = CODE_GROUP_TABLE,
    include_ancestors: bool = True,
    include_descendants: bool = True,
) -> Set[int]:
    """Return target self + mapped ancestors/descendants from code_group_mapping."""
    target = int(target_concept_code_int)
    block: Set[int] = {target}
    with connect_sqlite_read_only(db_path) as conn:
        if include_ancestors:
            q_anc = f"""
                SELECT ancestor_concept_code_int AS concept_code_int
                FROM {code_group_table}
                WHERE descendant_concept_code_int = ?
            """
            anc = pd.read_sql_query(q_anc, conn, params=[target])
            block.update(anc["concept_code_int"].dropna().astype(int).tolist())
        if include_descendants:
            q_desc = f"""
                SELECT descendant_concept_code_int AS concept_code_int
                FROM {code_group_table}
                WHERE ancestor_concept_code_int = ?
            """
            desc = pd.read_sql_query(q_desc, conn, params=[target])
            block.update(desc["concept_code_int"].dropna().astype(int).tolist())
    return block


def _fetch_code_group_pairs_for_present_codes(
    db_path: str,
    present_codes: Set[int],
    *,
    code_group_table: str = CODE_GROUP_TABLE,
    chunk_size: int = SQLITE_PARAM_LIMIT,
) -> pd.DataFrame:
    """Fetch mapping rows where the ancestor is in the current candidate set."""
    codes = sorted({int(c) for c in present_codes if pd.notna(c)})
    if not codes:
        return pd.DataFrame(columns=["descendant_concept_code_int", "ancestor_concept_code_int"])
    frames: List[pd.DataFrame] = []
    with connect_sqlite_read_only(db_path) as conn:
        for i in range(0, len(codes), chunk_size):
            chunk = codes[i:i + chunk_size]
            ph = ",".join(["?"] * len(chunk))
            q = f"""
                SELECT descendant_concept_code_int, ancestor_concept_code_int
                FROM {code_group_table}
                WHERE ancestor_concept_code_int IN ({ph})
            """
            frames.append(pd.read_sql_query(q, conn, params=chunk))
    if not frames:
        return pd.DataFrame(columns=["descendant_concept_code_int", "ancestor_concept_code_int"])
    pairs = pd.concat(frames, ignore_index=True).drop_duplicates()
    if pairs.empty:
        return pairs
    pairs["descendant_concept_code_int"] = pd.to_numeric(
        pairs["descendant_concept_code_int"], errors="coerce"
    ).astype("Int64")
    pairs["ancestor_concept_code_int"] = pd.to_numeric(
        pairs["ancestor_concept_code_int"], errors="coerce"
    ).astype("Int64")
    return pairs[pairs["descendant_concept_code_int"].isin(codes)].copy()


def get_code_group_ancestor_exclusion_mask(
    concept_codes: pd.Series,
    db_path: str,
    *,
    code_group_table: str = CODE_GROUP_TABLE,
    keep_codes: Optional[Set[int]] = None,
) -> pd.Series:
    """Mask rollup/ancestor concepts using the DB code_group_mapping table."""
    present = set(concept_codes.dropna().astype(int).unique().tolist())
    if not present:
        return pd.Series(False, index=concept_codes.index)
    keep = {int(x) for x in (keep_codes or set())}
    pairs = _fetch_code_group_pairs_for_present_codes(
        db_path, present, code_group_table=code_group_table
    )
    if pairs.empty:
        return pd.Series(False, index=concept_codes.index)
    ancestors_to_exclude = set(
        pairs.loc[
            pairs["descendant_concept_code_int"] != pairs["ancestor_concept_code_int"],
            "ancestor_concept_code_int",
        ].dropna().astype(int).tolist()
    )
    ancestors_to_exclude -= keep
    if not ancestors_to_exclude:
        return pd.Series(False, index=concept_codes.index)
    return concept_codes.astype("Int64").isin(ancestors_to_exclude)


# SNOMED HIERARCHY  — single consolidated class
class SNOMEDHierarchy:
    """Loads a SNOMED CT Full RF2 relationship file once and exposes all
    hierarchy operations as instance methods."""

    _IS_A = "116680003"
    _CHAR = {
        "inferred": "900000000000011006",
        "stated":   "900000000000010007",
    }

    def __init__(self, rel_full_path: str, characteristic: str = "inferred") -> None:
        self._path = rel_full_path
        self._char = characteristic
        self._p2c: Dict[str, Set[str]] = defaultdict(set)
        self._c2p: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    def _load(self) -> None:
        print(f"[SNOMED] Loading from {self._path} ...")
        cols = [
            "id", "effectiveTime", "active", "moduleId",
            "sourceId", "destinationId", "relationshipGroup",
            "typeId", "characteristicTypeId", "modifierId",
        ]
        df = pd.read_csv(self._path, sep="\t", dtype=str, usecols=cols)
        df = df[
            (df["typeId"] == self._IS_A) &
            (df["characteristicTypeId"] == self._CHAR[self._char])
        ].copy()
        df["_t"] = df["effectiveTime"].astype(int)
        snap = df.loc[df.groupby("id")["_t"].idxmax()]
        snap = snap[snap["active"] == "1"][["sourceId", "destinationId"]]
        for child, parent in zip(snap["sourceId"], snap["destinationId"]):
            self._p2c[parent].add(child)
            self._c2p[child].add(parent)
        n = sum(len(v) for v in self._p2c.values())
        print(f"[SNOMED] Loaded {n:,} IS-A relationships.")

    # ── Static helpers ───────────────────────────────────────────────────────
    @staticmethod
    def extract_snomed_id(dx_code: str) -> Optional[str]:
        """'dx_SNOMED_363354003' → '363354003'. Non-SNOMED → None."""
        p = "dx_SNOMED_"
        return dx_code[len(p):] if dx_code.startswith(p) else None

    @staticmethod
    def make_dx_snomed(concept_id: str) -> str:
        return f"dx_SNOMED_{concept_id}"

    # ── Traversal ────────────────────────────────────────────────────────────
    def find_descendants(self, concept_id: str) -> Set[str]:
        """All IS-A descendants via BFS."""
        out: Set[str] = set()
        q = deque([concept_id])
        while q:
            cur = q.popleft()
            for kid in self._p2c.get(cur, ()):
                if kid not in out:
                    out.add(kid)
                    q.append(kid)
        return out

    def find_ancestors(self, concept_id: str) -> Set[str]:
        """All IS-A ancestors via BFS."""
        out: Set[str] = set()
        visited = {concept_id}
        q = deque([concept_id])
        while q:
            cur = q.popleft()
            for par in self._c2p.get(cur, ()):
                if par not in visited:
                    visited.add(par)
                    out.add(par)
                    q.append(par)
        return out

    def find_parents(self, concept_id: str) -> Set[str]:
        """Direct parents only."""
        return set(self._c2p.get(concept_id, ()))

    def find_siblings(self, concept_id: str) -> Set[str]:
        """Children of the same parents, excluding concept_id itself."""
        sibs: Set[str] = set()
        for p in self._c2p.get(concept_id, set()):
            sibs |= self._p2c.get(p, set())
        sibs.discard(concept_id)
        return sibs

    # ── Main public methods ──────────────────────────────────────────────────
    def enumerate_subtypes(
        self,
        root_code: str,
        min_children: int = 0,
        max_depth: Optional[int] = None,
    ) -> List[str]:
        """Return dx_SNOMED_* codes for root and all its descendants."""
        root_id = self.extract_snomed_id(root_code)
        if not root_id:
            return [root_code]

        result: List[str] = []
        q: deque = deque([(root_id, 0)])
        visited = {root_id}
        while q:
            cur, depth = q.popleft()
            children = self._p2c.get(cur, set())
            if len(children) >= min_children:
                result.append(self.make_dx_snomed(cur))
            if max_depth is None or depth < max_depth:
                for child in children:
                    if child not in visited:
                        visited.add(child)
                        q.append((child, depth + 1))

        print(f"[SNOMED] enumerate_subtypes({root_code}): {len(result)} found.")
        return result

    def compute_blocklist(
        self,
        target_code: str,
        verbose: bool = False,
    ) -> Tuple[Set[str], Dict[str, Any]]:
        """Build the set of dx_SNOMED_* codes to exclude as predictors for
        a given target outcome. Blocks: self + parents + all ancestors +
        all descendants."""
        tid = self.extract_snomed_id(target_code)
        if not tid:
            return {target_code}, {"error": "Not a SNOMED concept"}

        parents     = self.find_parents(tid)
        ancestors   = self.find_ancestors(tid)
        descendants = self.find_descendants(tid)

        block_ids = {tid} | parents | ancestors | descendants
        blocklist = {self.make_dx_snomed(cid) for cid in block_ids}
        blocklist.add(target_code)

        diag = {
            "target": target_code, "target_id": tid,
            "total_blocked": len(blocklist),
            "breakdown": {
                "self": 1, "parents": len(parents),
                "ancestors": len(ancestors), "descendants": len(descendants),
            },
        }
        if verbose:
            print(f"[BLOCKLIST] {target_code}  total={diag['total_blocked']}")
            for k, v in diag["breakdown"].items():
                print(f"    {k}: {v}")
        return blocklist, diag

    def run_blocklist_diagnostics(
        self,
        targets: List[str],
        save_dir: str = "./diagnostics",
    ) -> pd.DataFrame:
        """Run compute_blocklist for all targets, print summary stats,
        and save blocklist_summary.csv."""
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("BLOCKLIST DIAGNOSTIC ANALYSIS")
        print("=" * 80)
        rows = []
        for idx, T in enumerate(targets, 1):
            print(f"[{idx}/{len(targets)}] {T}")
            _, diag = self.compute_blocklist(T, verbose=False)
            if "error" in diag:
                print(f"  ⚠  {diag['error']}"); continue
            bd = diag["breakdown"]
            rows.append({
                "target": T, "snomed_id": diag["target_id"],
                "total_blocked": diag["total_blocked"],
                "parents": bd["parents"],
                "ancestors": bd["ancestors"], "descendants": bd["descendants"],
            })
            print(f"  blocked={diag['total_blocked']}  "
                  f"anc={bd['ancestors']}  desc={bd['descendants']}")

        df = pd.DataFrame(rows)
        csv_path = os.path.join(save_dir, "blocklist_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[SAVED] {csv_path}")
        if not df.empty:
            print(f"Blocklist size — mean={df['total_blocked'].mean():.1f}  "
                  f"min={df['total_blocked'].min()}  max={df['total_blocked'].max()}")
        return df


# Private shared math  — v17 OR formula (v16 eleven-case OR carried forward)

def compute_or_v17_scalar(
    n_exposed_outcome: float,
    n_exposed_no_outcome: float,
    n_unexposed_outcome: float,
    n_unexposed_no_outcome: float,
) -> float:
    """DEMI v17 adjusted OR for a single 2×2 table."""
    a = n_exposed_outcome
    b = n_exposed_no_outcome
    c = n_unexposed_outcome
    d = n_unexposed_no_outcome

    if any(np.isnan(x) for x in [a, b, c, d]):
        return np.nan

    # Logically undefined: no cases at all
    if a == 0 and c == 0:
        return np.nan

    THRESH = 30.0
    n_exposed_total  = a + b
    n_outcome_total  = a + c
    both_large = (n_exposed_total >= THRESH) and (n_outcome_total >= THRESH)

    # ── a=0 (any sample size) ─────────────────────────────────────────────────
    if a == 0:
        if b == 0:
            return float(1.0 / (c + 1))
        return float(1.0 / min(b, c))

    # Remaining zero-cell cases do not depend on sample size.
    if b == 0 and c == 0:
        return float(a + 1)
    if b == 0:
        return float(n_outcome_total / (b + 1))
    if c == 0:
        return float(n_exposed_total / (c + 1))

    if both_large:
        return float((a * d) / (b * c))

    # a>0, b>0, c>0, one or both small — d* cross-ratio selection
    raw = (a * d) / (b * c)
    OR1 = n_exposed_total / c
    OR2 = n_outcome_total  / b
    if raw >= 1.0:
        return float(max(OR1, OR2))
    else:
        return float(min(OR1, OR2))


def _compute_or_vectorized(
    outcome_edges_df: pd.DataFrame,
    *,
    # Legacy parameters — accepted, ignored
    is_diagnosis: Optional[np.ndarray] = None,
    is_outcome_target: bool = False,
    se_thr: float = 1.0,
    min_cell_thr: int = 5,
    cap_thr: int = 30,
) -> np.ndarray:
    """Vectorized DEMI v17 OR. Drop-in replacement for v16."""
    n_exposed_outcome      = outcome_edges_df.get(
        "n_code_target",       pd.Series(np.nan, index=outcome_edges_df.index)
    ).to_numpy(float)
    n_exposed_no_outcome   = outcome_edges_df.get(
        "n_code_no_target",    pd.Series(np.nan, index=outcome_edges_df.index)
    ).to_numpy(float)
    n_unexposed_outcome    = outcome_edges_df.get(
        "n_target_no_code",    pd.Series(np.nan, index=outcome_edges_df.index)
    ).to_numpy(float)
    n_unexposed_no_outcome = outcome_edges_df.get(
        "n_no_code_no_target", pd.Series(np.nan, index=outcome_edges_df.index)
    ).to_numpy(float)

    a = n_exposed_outcome
    b = n_exposed_no_outcome
    c = n_unexposed_outcome
    d = n_unexposed_no_outcome

    THRESH = 30.0
    n_exposed_total = a + b
    n_outcome_total = a + c
    odds_ratios = np.full(len(a), np.nan, dtype=float)

    any_nan = np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(d)
    valid   = ~any_nan

    # Logically undefined: a=0, c=0 → remain nan
    defined = valid & ~((a == 0) & (c == 0))

    both_large = defined & (n_exposed_total >= THRESH) & (n_outcome_total >= THRESH)
    small      = defined & ~both_large

    # ── a=0: 1/min(b,c) or 1/(c+1) ──────────────────────────────────────────
    a0    = defined & (a == 0)
    a0_b0 = a0 & (b == 0)
    a0_ok = a0 & (b > 0)
    odds_ratios[a0_b0] = 1.0 / (c[a0_b0] + 1)
    min_bc = np.minimum(b, c)
    safe_a0 = a0_ok & (min_bc > 0)
    odds_ratios[safe_a0] = 1.0 / min_bc[safe_a0]

    # Handle the same zero-cell cases once for both sample-size groups.
    positive_a = defined & (a > 0)
    m = positive_a & (b == 0) & (c == 0)
    odds_ratios[m] = a[m] + 1.0
    m = positive_a & (b == 0) & (c > 0)
    odds_ratios[m] = n_outcome_total[m] / (b[m] + 1)
    m = positive_a & (b > 0) & (c == 0)
    odds_ratios[m] = n_exposed_total[m] / (c[m] + 1)
    m = both_large & (a > 0) & (b > 0) & (c > 0)
    odds_ratios[m] = (a[m] * d[m]) / (b[m] * c[m])
    m = small & (a > 0) & (b > 0) & (c > 0)
    raw = (a[m] * d[m]) / (b[m] * c[m])
    OR1 = n_exposed_total[m] / c[m]
    OR2 = n_outcome_total[m]  / b[m]
    odds_ratios[m] = np.where(raw >= 1.0, np.maximum(OR1, OR2), np.minimum(OR1, OR2))

    return odds_ratios


def compute_or_vectorized(
    df: pd.DataFrame,
    se_thr: float = 1.0,
    min_cell_thr: int = 5,
    cap_thr: int = 30,
    is_diagnosis: Union[bool, np.ndarray] = True,
    is_outcome_target: Union[bool, np.ndarray] = False,
) -> np.ndarray:
    """Public wrapper for the v17 adjusted OR computation used by DEMI."""
    return _compute_or_vectorized(df)


# Prevalence / prior-odds helper

def _compute_outcome_prevalence(
    outcome_rows: pd.DataFrame,
    pref_override: Optional[float] = None,
) -> tuple:
    """Compute baseline prevalence, prior odds, and outcome-positive count."""
    n_y1 = None
    n_y0 = None

    if len(outcome_rows) > 0:
        d = outcome_rows.copy()

        # Convert available count columns to numeric
        for col in [
            "n_target", "n_no_target",
            "n_code_target", "n_code_no_target",
            "n_target_no_code", "n_no_code_no_target",
        ]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        # Preferred path: use DB marginal columns directly
        if "n_target" in d.columns and "n_no_target" in d.columns:
            marg = d[["n_target", "n_no_target"]].dropna().copy()

            if not marg.empty:
                # Group so we can choose a real observed denominator pair.
                # Primary sort: largest n_target, because EPV is based on n(Y=1).
                # Tie-breakers: largest total N, then most rows.
                group = (
                    marg.groupby(["n_target", "n_no_target"])
                    .size()
                    .reset_index(name="n_rows")
                )
                group["N_total"] = group["n_target"] + group["n_no_target"]

                group = group.sort_values(
                    ["n_target", "N_total", "n_rows"],
                    ascending=[False, False, False],
                )

                chosen = group.iloc[0]
                n_y1 = float(chosen["n_target"])
                n_y0 = float(chosen["n_no_target"])

        # Fallback path for older tables without n_target/n_no_target
        if n_y1 is None or n_y0 is None:
            required = [
                "n_code_target",
                "n_code_no_target",
                "n_target_no_code",
                "n_no_code_no_target",
            ]
            if all(c in d.columns for c in required):
                d["y1_cells"] = d["n_code_target"].fillna(0) + d["n_target_no_code"].fillna(0)
                d["y0_cells"] = d["n_code_no_target"].fillna(0) + d["n_no_code_no_target"].fillna(0)
                d["N_total"] = d["y1_cells"] + d["y0_cells"]

                group = (
                    d.groupby(["y1_cells", "y0_cells", "N_total"])
                    .size()
                    .reset_index(name="n_rows")
                )

                group = group.sort_values(
                    ["y1_cells", "N_total", "n_rows"],
                    ascending=[False, False, False],
                )

                chosen = group.iloc[0]
                n_y1 = float(chosen["y1_cells"])
                n_y0 = float(chosen["y0_cells"])

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

# Backward-compatibility alias
_compute_prior_odds = _compute_outcome_prevalence


# Step 3 helpers — positive/protective partition and backward recursion

def _compute_predictor_prevalence(
    nodes: List[int],
    outcome: int,
    edge_index: pd.DataFrame,
) -> pd.Series:
    """Use each predictor-outcome pair's own denominator, never a reverse row."""
    prevalence: Dict[int, float] = {}
    count_columns = (
        "n_code_target", "n_code_no_target",
        "n_target_no_code", "n_no_code_no_target",
    )
    for predictor in nodes:
        key = (int(outcome), int(predictor))
        if key not in edge_index.index:
            raise KeyError(f"Missing predictor-outcome counts for {key}.")
        row = edge_index.loc[key]
        if all(column in row.index for column in count_columns):
            cells = np.array([row[column] for column in count_columns], dtype=float)
            if not np.isfinite(cells).all() or (cells < 0).any():
                raise ValueError(f"Invalid predictor-outcome counts for {key}.")
            numerator = float(cells[0] + cells[1])
            denominator = float(cells.sum())
        else:
            # Older inputs may provide only marginals from this same pair.
            numerator = float(row.get("n_code", np.nan))
            denominator = (
                float(row.get("n_target", np.nan))
                + float(row.get("n_no_target", np.nan))
            )
        if (not np.isfinite(numerator) or not np.isfinite(denominator)
                or denominator <= 0 or numerator < 0 or numerator > denominator):
            raise ValueError(f"Invalid predictor-outcome prevalence for {key}.")
        prevalence[int(predictor)] = numerator / denominator
    return pd.Series(prevalence, dtype=float)


def _partition_by_effect_direction(
    nodes: List[int],
    total_log_or: pd.Series,
) -> Tuple[List[int], List[int]]:
    """Partition nodes by sign of total_log_or."""
    positive_predictors = [
        predictor for predictor in nodes
        if np.isfinite(total_log_or.loc[predictor]) and total_log_or.loc[predictor] > 0.0
    ]
    protective_predictors = [
        predictor for predictor in nodes
        if np.isfinite(total_log_or.loc[predictor]) and total_log_or.loc[predictor] < 0.0
    ]
    return positive_predictors, protective_predictors


def _run_backward_recursion(
    group: List[int],
    total_log_or: pd.Series,
    weights: Dict[int, pd.Series],
    sign: int,  # +1 for positive_predictors, -1 for protective_predictors
    predictor_prevalence: Optional[pd.Series] = None,
) -> pd.Series:
    """Subtract downstream indirect effects within one sign group.

    D_k = T_k - sum((weight_kj - prevalence_j) * D_j), using nonzero
    downstream weights. Clip D_k to >= 0 for positive predictors and <= 0
    for protective predictors. Process nodes from last to first."""
    if not group:
        return pd.Series(dtype=float)

    direct_log_or = pd.Series(np.nan, index=group, dtype=float)
    # Initialise last variable: no same-sign downstream mediator → D = T
    direct_log_or.iloc[-1] = total_log_or.loc[group[-1]]

    for i in range(len(group) - 2, -1, -1):
        predictor       = group[i]
        total_log_or_k  = float(total_log_or.loc[predictor])
        predictor_weights = weights.get(predictor, pd.Series(dtype=float))

        # Sum weight_excess_{k,j} · direct_log_or_{j} over ALL downstream
        # same-sign nodes
        indirect = 0.0
        for downstream_predictor in group[i + 1:]:
            weight_kj = float(predictor_weights.get(downstream_predictor, 0.0))
            # Only apply excess correction when a real k→j edge exists
            # (weight > 0). If weight_{k,j} = 0 (no edge), excess is also 0.
            if weight_kj != 0.0 and predictor_prevalence is not None:
                pj        = float(predictor_prevalence.get(downstream_predictor, 0.0))
                weight_kj = weight_kj - pj          # signed; negatives retained
            if weight_kj != 0.0:
                indirect += weight_kj * float(direct_log_or.loc[downstream_predictor])

        raw_d = total_log_or_k - indirect
        direct_log_or.loc[predictor] = (
            max(0.0, raw_d) if sign == 1 else min(0.0, raw_d)
        )

    return direct_log_or


def select_top_predictors(
    total_log_or: pd.Series,
    n_cases: int,
    *,
    epv: int = 5,
    user_cap: int = 500,
    min_a: int = 0,
    a_series: Optional[pd.Series] = None,
) -> pd.Index:
    """EPV-balanced dual-tail variable selection for DEMI v17."""
    T = total_log_or.dropna()
    T = T[T != 0.0]

    # Step 1 — EPV cap
    N_max  = min(max(n_cases // epv, 1), user_cap)
    n_each = N_max // 2

    # Step 2 — Balanced dual-tail selection
    harmful    = T[T > 0].sort_values(ascending=False)
    protective = T[T < 0].sort_values(ascending=True)

    n_harm = min(n_each, len(harmful))
    n_prot = min(n_each, len(protective))

    # Fill spare slots from the other pool if one is exhausted
    spare = N_max - n_harm - n_prot
    if spare > 0:
        if n_harm < n_each:
            n_prot = min(len(protective), n_prot + spare)
        elif n_prot < n_each:
            n_harm = min(len(harmful), n_harm + spare)

    selected_harm = harmful.index[:n_harm]
    selected_prot = protective.index[:n_prot]
    result = selected_harm.append(selected_prot)
    print(f"[SELECT] N_max={N_max}  harm={n_harm}  prot={n_prot}  total={len(result)}")
    return result


def build_snomed_ancestor_sets(
    relationship_path: str,
    concept_ids: Optional[Set[int]] = None,
) -> Dict[int, Set[int]]:
    """Build transitive ancestor sets from a SNOMED CT RF2 relationship file."""
    ISA = 116680003

    rel = pd.read_csv(
        relationship_path,
        sep="\t",
        usecols=["id", "effectiveTime", "active", "typeId", "sourceId", "destinationId"],
        dtype=str,
    )
    rel["_t"] = rel["effectiveTime"].astype(int)
    rel = rel.loc[rel.groupby("id")["_t"].idxmax()].copy()
    rel = rel[(rel["active"] == "1") & (rel["typeId"] == str(ISA))]
    rel["sourceId"]      = rel["sourceId"].astype(int)
    rel["destinationId"] = rel["destinationId"].astype(int)

    if concept_ids is not None:
        seeds: Set[int] = set(concept_ids)
        frontier: Set[int] = set(seeds)
        all_relevant: Set[int] = set()
        while frontier:
            parents_of_frontier = set(
                rel.loc[rel["sourceId"].isin(frontier), "destinationId"].unique()
            )
            new_nodes = parents_of_frontier - all_relevant - seeds
            all_relevant.update(frontier)
            frontier = new_nodes
        keep = all_relevant | seeds
        rel = rel[rel["sourceId"].isin(keep) | rel["destinationId"].isin(keep)]

    parents_map: Dict[int, Set[int]] = defaultdict(set)
    for _, row in rel.iterrows():
        parents_map[int(row["sourceId"])].add(int(row["destinationId"]))

    all_concepts: Set[int] = set(parents_map.keys()) | {
        p for ps in parents_map.values() for p in ps
    }

    ancestors: Dict[int, Set[int]] = {
        c: set(parents_map.get(c, set())) for c in all_concepts
    }
    changed = True
    while changed:
        changed = False
        for c in all_concepts:
            new_anc: Set[int] = set()
            for p in list(ancestors[c]):
                new_anc.update(ancestors.get(p, set()))
            before = len(ancestors[c])
            ancestors[c].update(new_anc)
            if len(ancestors[c]) > before:
                changed = True

    return ancestors


def get_snomed_ancestor_exclusion_mask(
    concept_codes: pd.Series,
    concept_code_strs: Optional[pd.Series],
    relationship_path: str,
) -> pd.Series:
    """Return a boolean mask of rows to EXCLUDE because the concept is a
    SNOMED transitive ancestor of another concept already present in the data."""
    import warnings

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
            "Falling back to name-pattern rollup filter.",
            RuntimeWarning,
        )
        return pd.Series(False, index=concept_codes.index)

    to_exclude: Set[int] = set()
    for code_a in snomed_int_codes:
        for code_b in snomed_int_codes:
            if code_a != code_b and code_a in ancestors.get(code_b, set()):
                to_exclude.add(code_a)
                break

    return concept_codes.isin(to_exclude) & is_snomed


# Concept-code column helper
def _get_concept_code_str_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return the best available string concept-code column for SNOMED filters."""
    for col in ("concept_code", "concept_code_str", "standard_concept_code"):
        if col in df.columns:
            return df[col].astype(str)
    return None


# demi_correction_factors table helpers
# The demi_correction_factors table has three columns:
#   concept_code_int      — target outcome concept
#   demi_log_lambda_net   — log(correction_factor); correction_factor =
#                           Π_k [ p_k · exp(D_k) + (1 - p_k) ]
#   demi_beta0_corrected  — corrected intercept β_0 = log(prior_odds)
#                           − demi_log_lambda_net

_DEMI_CF_REQUIRED_COLS = [
    "concept_code_int",
    "demi_log_lambda_net",
    "demi_beta0_corrected",
]


def _lookup_demi_correction_factors(
    demi_correction_factors: Union[str, pd.DataFrame],
    *,
    concept_code_int: int,
) -> Tuple[float, float]:
    """Look up demi_log_lambda_net and demi_beta0_corrected from the
    demi_correction_factors table using concept_code_int as the key."""
    if isinstance(demi_correction_factors, str):
        df = pd.read_csv(demi_correction_factors)
    else:
        df = demi_correction_factors.copy()

    missing = [c for c in _DEMI_CF_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"demi_correction_factors table missing required columns: {missing}. "
            f"Expected: {_DEMI_CF_REQUIRED_COLS}"
        )

    df["concept_code_int"] = pd.to_numeric(df["concept_code_int"], errors="coerce").astype("Int64")
    mask = df["concept_code_int"] == int(concept_code_int)
    hits = df.loc[mask]

    if hits.empty:
        raise KeyError(
            f"No row found in demi_correction_factors for "
            f"concept_code_int={concept_code_int}."
        )

    row                = hits.iloc[0]
    log_lambda_net     = float(row["demi_log_lambda_net"])
    beta0_corrected    = float(row["demi_beta0_corrected"])

    if not np.isfinite(log_lambda_net):
        raise ValueError(
            f"demi_log_lambda_net is not finite ({log_lambda_net}) "
            f"for concept_code_int={concept_code_int}."
        )
    if not np.isfinite(beta0_corrected):
        raise ValueError(
            f"demi_beta0_corrected is not finite ({beta0_corrected}) "
            f"for concept_code_int={concept_code_int}."
        )

    return log_lambda_net, beta0_corrected


def make_correction_factor_row(
    *,
    concept_code_int: int,
    demi_log_lambda_net: float,
    demi_beta0_corrected: float,
) -> Dict[str, Any]:
    """Create one demi_correction_factors table row."""
    return {
        "concept_code_int":     int(concept_code_int),
        "demi_log_lambda_net":  float(demi_log_lambda_net),
        "demi_beta0_corrected": float(demi_beta0_corrected),
    }


def _compute_log_correction_factor(
    predictor_prevalence: pd.Series,
    direct_log_or: pd.Series,
) -> float:
    """Return log(Lambda) = sum(log(p_k * exp(D_k) + 1 - p_k)).

    Evaluate on the log scale to avoid overflow."""
    log_cf = 0.0
    common = [k for k in direct_log_or.index if k in predictor_prevalence.index]
    for k in common:
        d_k = float(direct_log_or.loc[k])
        p_k = float(predictor_prevalence.loc[k])
        if not np.isfinite(d_k) or p_k < 0.0 or p_k > 1.0:
            continue
        if p_k == 0.0:
            continue
        if p_k == 1.0:
            log_cf += d_k
            continue
        log_cf += float(np.logaddexp(np.log(p_k) + d_k, np.log1p(-p_k)))
    return float(log_cf)


def build_correction_factor_table_for_target(
    df_in: Union[str, pd.DataFrame],
    *,
    force_outcome: int,
    lambda_min_count: int = 15,
    epv: int = 5,
    user_cap: int = 500,
    min_a: int = 0,
    snomed_relationship_path: Optional[str] = None,
    db_path: Optional[str] = DEMI_DB_PATH,
    temporal_order_table_name: str = TEMPORAL_ORDER_TABLE,
    use_db_temporal_order: bool = True,
    pref_override: Optional[float] = None,
    n_missing_value: int = 0,
    fix_suppression: bool = True,
    se_thr: float = 1.0,
    min_cell_thr: int = 5,
    cap_thr: int = 30,
    is_diagnosis_set: Optional[Set[int]] = None,
    is_outcome_target: bool = False,
) -> pd.DataFrame:
    """Build a one-row correction-factor table using the legacy helper path.
    
    This helper recomputes total effects from counts and retains its original
    filtering behavior. Pairwise prevalences use the shared helper. It is not
    used by the main analysis, which
    reads precalculated total effects and computes its own correction factor.
    Returns concept_code_int, demi_log_lambda_net, demi_beta0_corrected."""
    df = pd.read_csv(df_in) if isinstance(df_in, str) else df_in.copy()

    for col in ["target_concept_code_int", "concept_code_int"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in [
        "n_code_target", "n_code", "n_target", "n_no_target",
        "n_code_before_target", "n_target_before_code",
        "n_code_no_target", "n_target_no_code", "n_no_code_no_target",
        "total_effects", "total_effects_log", "total_effects_abs",
        "total_effects_rank_abs", "total_effects_rank_ascending",
        "total_effects_rank_descending",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outcome = int(force_outcome)

    name_col: Optional[str] = None
    for candidate in ("concept_name", "standard_concept_name"):
        if candidate in df.columns:
            name_col = candidate
            break

    if name_col is not None:
        qa_mask = (
            df[name_col].str.startswith("Question:", na=False)
            | df[name_col].str.contains("Answer:", na=False, regex=False)
        )
        df = df[~qa_mask].copy()

    if snomed_relationship_path is not None:
        code_str_col = _get_concept_code_str_series(df)
        snomed_exclude = get_snomed_ancestor_exclusion_mask(
            df["concept_code_int"],
            code_str_col,
            snomed_relationship_path,
        )
        snomed_exclude = snomed_exclude & (df["concept_code_int"] != outcome)
        df = df[~snomed_exclude].copy()

    if name_col is not None:
        name_col_cur = name_col if name_col in df.columns else None
        if name_col_cur is not None:
            name = df[name_col_cur].fillna("")
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
            if snomed_relationship_path is not None:
                code_str_col = _get_concept_code_str_series(df)
                if code_str_col is not None:
                    already_handled = code_str_col.str.contains("SNOMED", na=False, regex=False)
                    rollup_mask = rollup_mask & ~already_handled
            df = df[~rollup_mask].copy()

    edge_cols = [
        "n_target_before_code", "n_code_before_target",
        "n_code_target", "n_code", "n_target", "n_no_target",
        "n_code_no_target", "n_target_no_code", "n_no_code_no_target",
    ]
    present    = [c for c in edge_cols if c in df.columns]
    edge_index = df.groupby(
        ["target_concept_code_int", "concept_code_int"], as_index=True
    )[present].sum()

    outcome_rows = df[df["target_concept_code_int"] == outcome]
    _, _, n_outcome_positive, n_outcome_negative = _compute_outcome_prevalence(
        outcome_rows, pref_override=pref_override
    )

    all_events = sorted(
        set(int(x) for x in df["target_concept_code_int"].dropna().unique())
        | set(int(x) for x in df["concept_code_int"].dropna().unique())
    )
    scores = _get_temporal_scores(
        all_events,
        edge_index,
        db_path=db_path,
        temporal_order_table_name=temporal_order_table_name,
        use_db_temporal_order=use_db_temporal_order,
    )

    sorted_scores  = pd.Series(scores).sort_values(ascending=False)
    temporal_order = [int(ev) for ev in sorted_scores.index if ev != outcome] + [outcome]
    nodes          = temporal_order[:-1]

    e_rows, e_keys = [], []
    for predictor in nodes:
        key = (outcome, int(predictor))
        if key in edge_index.index:
            e_rows.append(edge_index.loc[key])
            e_keys.append(predictor)

    total_log_or = pd.Series(0.0, index=nodes, dtype=float)

    if e_rows:
        outcome_edges_df = pd.DataFrame(e_rows, index=e_keys)
        or_vals = _compute_or_vectorized(
            outcome_edges_df, se_thr=se_thr, min_cell_thr=min_cell_thr, cap_thr=cap_thr,
        )
        for predictor, ov in zip(e_keys, or_vals):
            if np.isfinite(ov) and ov > 0:
                total_log_or.loc[predictor] = float(np.log(ov))

    selected_idx = select_top_predictors(
        total_log_or,
        n_cases=int(n_outcome_positive) if n_outcome_positive is not None else 1,
        epv=epv,
        user_cap=user_cap,
    )
    selected_set = set(selected_idx.tolist())
    nodes = [n for n in nodes if n in selected_set]
    pos_by_event = {ev: i for i, ev in enumerate(temporal_order)}

    predictor_prevalence_series = _compute_predictor_prevalence(
        nodes, outcome, edge_index
    )

    weights: Dict[int, pd.Series] = {}
    for predictor in nodes:
        pos_k      = pos_by_event[predictor]
        downstream = [int(ev) for ev in temporal_order[pos_k + 1 : -1]]
        predictor_weights: Dict[int, float] = {}
        for downstream_predictor in downstream:
            key = (int(downstream_predictor), int(predictor))
            if key not in edge_index.index:
                predictor_weights[downstream_predictor] = 0.0
                continue
            row  = edge_index.loc[key]
            n_kj = float(row.get("n_code_target", 0.0))
            n_k  = float(row.get("n_code", 0.0))
            if n_k == 0.0:
                n_k = (
                    float(row.get("n_code_target", 0.0))
                    + float(row.get("n_code_no_target", 0.0))
                )
            predictor_weights[downstream_predictor] = (
                0.0 if n_k < lambda_min_count
                else float(np.clip(n_kj / n_k if n_k > 0 else 0.0, 0.0, 1.0))
            )
        weights[predictor] = pd.Series(predictor_weights, dtype=float)

    positive_predictors, protective_predictors = _partition_by_effect_direction(
        nodes, total_log_or
    )

    direct_log_or_plus  = _run_backward_recursion(
        positive_predictors, total_log_or, weights,
        sign=+1, predictor_prevalence=predictor_prevalence_series
    )
    direct_log_or_minus = _run_backward_recursion(
        protective_predictors, total_log_or, weights,
        sign=-1, predictor_prevalence=predictor_prevalence_series
    )

    direct_log_or = pd.Series(0.0, index=nodes, dtype=float)
    for k, v in direct_log_or_plus.items():
        direct_log_or.loc[k] = v if np.isfinite(v) else float(total_log_or.loc[k])
    for k, v in direct_log_or_minus.items():
        direct_log_or.loc[k] = v if np.isfinite(v) else float(total_log_or.loc[k])

    log_lambda_net = _compute_log_correction_factor(
        predictor_prevalence_series, direct_log_or
    )
    p_marginal = (
        float(n_outcome_positive) / float(n_outcome_positive + n_outcome_negative)
        if (n_outcome_positive and n_outcome_negative
            and n_outcome_positive + n_outcome_negative > 0)
        else 0.5
    )
    beta_0_prior     = float(np.log(p_marginal / (1.0 - p_marginal)))
    beta0_corrected  = beta_0_prior - log_lambda_net

    row = make_correction_factor_row(
        concept_code_int=outcome,
        demi_log_lambda_net=log_lambda_net,
        demi_beta0_corrected=beta0_corrected,
    )
    return pd.DataFrame([row])


def analyze_causal_sequence_py(
    df_in: Union[str, pd.DataFrame],
    *,
    force_outcome: int,
    lambda_min_count: int = 15,
    # EPV-balanced selection parameters
    epv: int = 5,
    user_cap: int = 500,
    min_a: int = 0,
    snomed_relationship_path: Optional[str] = None,
    db_path: Optional[str] = DEMI_DB_PATH,
    use_code_group_mapping: bool = True,
    code_group_table: str = CODE_GROUP_TABLE,
    correction_factor_table_name: str = CORRECTION_FACTOR_TABLE,
    temporal_order_table_name: str = TEMPORAL_ORDER_TABLE,
    use_db_temporal_order: bool = True,
    # Compatibility parameters
    fix_suppression: bool = True,
    se_thr: float = 1.0,
    min_cell_thr: int = 5,
    cap_thr: int = 30,
    pref_override: Optional[float] = None,
    correction_factor_table: Optional[Union[str, pd.DataFrame]] = None,
    n_missing_value: int = 0,
    debug_total_effect: bool = False,
    is_diagnosis_set: Optional[Set[int]] = None,
    is_outcome_target: bool = False,
) -> Dict[str, Any]:
    """Select predictors and compute v17 direct effects and probabilities.
    
    force_outcome is the integer target ID. Input must include precalculated
    total_effects_log and the available pairwise count columns. EPV and user_cap
    limit predictor selection. Database hierarchy and temporal-order tables are
    used when enabled; RF2 and local ordering provide the existing fallbacks.
    
    The correction factor and intercept are calculated from this model. Optional
    stored correction factors are checked only for audit comparison. pref_override
    sets the baseline probability; otherwise it is inferred from outcome counts.
    
    Returns effects, ordering, selection diagnostics, coef_df, and predict_proba.
    Compatibility arguments and historical output names are retained."""
    # ── Load / validate ──────────────────────────────────────────────────────
    df = pd.read_csv(df_in) if isinstance(df_in, str) else df_in.copy()

    for col in ["target_concept_code_int", "concept_code_int"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in [
        "n_code_target", "n_code", "n_target", "n_no_target",
        "n_code_before_target", "n_target_before_code",
        "n_code_no_target", "n_target_no_code", "n_no_code_no_target",
        "total_effects", "total_effects_log", "total_effects_abs",
        "total_effects_rank_abs", "total_effects_rank_ascending",
        "total_effects_rank_descending",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    outcome = int(force_outcome)

    # ── Resolve concept name column ───────────────────────────────────────────
    _name_col: Optional[str] = None
    for _candidate in ("concept_name", "standard_concept_name"):
        if _candidate in df.columns:
            _name_col = _candidate
            break

    # Exclude questionnaire prompts and answers.
    if _name_col is not None:
        qa_mask = (
            df[_name_col].str.startswith("Question:", na=False) |
            df[_name_col].str.contains("Answer:", na=False, regex=False)
        )
        n_qa = int(df.loc[qa_mask, "concept_code_int"].nunique())
        df = df[~qa_mask].copy()
    else:
        n_qa = 0

    n_rollup_snomed  = 0
    n_rollup_pattern = 0

    # Tier 1 — DB-backed hierarchy/rollup exclusion.
    # Prefer code_group_mapping because the updated DEMI DB already stores
    # descendant_concept_code_int -> ancestor_concept_code_int mappings.
    if use_code_group_mapping and db_path is not None:
        try:
            group_exclude = get_code_group_ancestor_exclusion_mask(
                df["concept_code_int"],
                db_path,
                code_group_table=code_group_table,
                keep_codes={outcome},
            )
            n_rollup_snomed = int(group_exclude.sum())
            df = df[~group_exclude].copy()
        except Exception as e:
            print(f"[WARN] code_group_mapping exclusion failed: {e}")
    elif snomed_relationship_path is not None:
        # Legacy fallback for old DBs or offline testing without code_group_mapping.
        code_str_col = _get_concept_code_str_series(df)
        snomed_exclude = get_snomed_ancestor_exclusion_mask(
            df["concept_code_int"],
            code_str_col,
            snomed_relationship_path,
        )
        snomed_exclude = snomed_exclude & (df["concept_code_int"] != outcome)
        n_rollup_snomed = int(snomed_exclude.sum())
        df = df[~snomed_exclude].copy()

    # Tier 2 + 3 — Name-pattern fallback
    if _name_col is not None:
        _name_col_cur = _name_col if _name_col in df.columns else None
        if _name_col_cur is not None:
            name = df[_name_col_cur].fillna("")
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
    present    = [c for c in edge_cols if c in df.columns]
    edge_index = df.groupby(
        ["target_concept_code_int", "concept_code_int"], as_index=True
    )[present].sum()

    # Precalculated total effects from demi_counts_published.
    # Do NOT sum these values across duplicate rows; they are pair-level values.
    effect_index: Optional[pd.DataFrame] = None
    effect_cols = [c for c in ["total_effects", "total_effects_log"] if c in df.columns]
    if "total_effects_log" in effect_cols:
        effect_index = (
            df.dropna(subset=["total_effects_log"])
              .sort_values(["target_concept_code_int", "concept_code_int"])
              .drop_duplicates(["target_concept_code_int", "concept_code_int"])
              .set_index(["target_concept_code_int", "concept_code_int"])[effect_cols]
        )

    outcome_rows = df[df["target_concept_code_int"] == outcome]
    (outcome_prevalence,
     _,
     n_outcome_positive,
     n_outcome_negative) = _compute_outcome_prevalence(outcome_rows, pref_override=pref_override)

    # ── Step 1: Temporal ordering ─────────────────────────────────────────────
    # Preferred DB-backed behavior: read global precomputed temporal scores from
    # demi_temporal_order_published. Fallback: recompute from the local subgraph.
    all_events = sorted(
        set(int(x) for x in df["target_concept_code_int"].dropna().unique()) |
        set(int(x) for x in df["concept_code_int"].dropna().unique())
    )
    scores = _get_temporal_scores(
        all_events,
        edge_index,
        db_path=db_path,
        temporal_order_table_name=temporal_order_table_name,
        use_db_temporal_order=use_db_temporal_order,
    )

    sorted_scores  = pd.Series(scores).sort_values(ascending=False)
    temporal_order = [int(ev) for ev in sorted_scores.index if ev != outcome] + [outcome]
    nodes          = temporal_order[:-1]

    # ── Step 2: Total effects total_log_or_{k,Y} ─────────────────────────────
    # Updated DB behavior: read total_effects_log directly from
    # demi_counts_published instead of recomputing OR from 2×2 counts.
    total_or     = pd.Series(1.0, index=nodes, dtype=float)
    total_log_or = pd.Series(0.0, index=nodes, dtype=float)

    if effect_index is None or effect_index.empty:
        raise KeyError(
            "Missing precalculated total_effects_log. The updated DEMI v17 "
            "expects df_in to come from demi_counts_published with the "
            "total_effects_log column."
        )

    for predictor in nodes:
        key = (outcome, int(predictor))
        if key not in effect_index.index:
            continue
        row_eff = effect_index.loc[key]
        # If duplicates somehow remain after indexing, keep the first pair-level row.
        if isinstance(row_eff, pd.DataFrame):
            row_eff = row_eff.iloc[0]
        tlog = float(row_eff.get("total_effects_log", np.nan))
        if np.isfinite(tlog):
            total_log_or.loc[predictor] = tlog
            if "total_effects" in row_eff.index and np.isfinite(float(row_eff.get("total_effects", np.nan))):
                total_or.loc[predictor] = float(row_eff.get("total_effects"))
            else:
                total_or.loc[predictor] = float(np.exp(np.clip(tlog, -700, 700)))

    if debug_total_effect:
        nonzero = int((total_log_or != 0.0).sum())
        print(f"[DEBUG] Loaded precalculated total_effects_log for {nonzero:,} predictors.")

    # ── Step 2b: EPV-balanced variable selection ──────────────────────────────
    selected_idx = select_top_predictors(
        total_log_or,
        n_cases=int(n_outcome_positive) if n_outcome_positive is not None else 1,
        epv=epv,
        user_cap=user_cap,
    )
    selected_set = set(selected_idx.tolist())
    nodes = [n for n in nodes if n in selected_set]
    pos_by_event = {ev: i for i, ev in enumerate(temporal_order)}

    # ── Step 3: Mediation weights and marginal prevalences ────────────────────
    predictor_prevalence_series = _compute_predictor_prevalence(
        nodes, outcome, edge_index
    )

    weights: Dict[int, pd.Series] = {}
    for predictor in nodes:
        pos_k      = pos_by_event[predictor]
        downstream = [int(ev) for ev in temporal_order[pos_k + 1 : -1]]
        predictor_weights_dict: Dict[int, float] = {}
        for downstream_predictor in downstream:
            key = (int(downstream_predictor), int(predictor))
            if key not in edge_index.index:
                predictor_weights_dict[downstream_predictor] = 0.0
                continue
            row  = edge_index.loc[key]
            n_kj = float(row.get("n_code_target", 0.0))
            n_k  = float(row.get("n_code", 0.0))
            if n_k == 0.0:
                n_k = (float(row.get("n_code_target", 0.0))
                       + float(row.get("n_code_no_target", 0.0)))
            predictor_weights_dict[downstream_predictor] = (
                0.0 if n_k < lambda_min_count
                else float(np.clip(n_kj / n_k if n_k > 0 else 0.0, 0.0, 1.0))
            )
        weights[predictor] = pd.Series(predictor_weights_dict, dtype=float)

    # ── Step 3: Partition and backward recursion ──────────────────────────────
    positive_predictors, protective_predictors = _partition_by_effect_direction(
        nodes, total_log_or
    )

    direct_log_or_plus  = _run_backward_recursion(
        positive_predictors, total_log_or, weights,
        sign=+1, predictor_prevalence=predictor_prevalence_series
    )
    direct_log_or_minus = _run_backward_recursion(
        protective_predictors, total_log_or, weights,
        sign=-1, predictor_prevalence=predictor_prevalence_series
    )

    direct_log_or = pd.Series(0.0, index=nodes, dtype=float)
    for k, v in direct_log_or_plus.items():
        direct_log_or.loc[k] = v if np.isfinite(v) else float(total_log_or.loc[k])
    for k, v in direct_log_or_minus.items():
        direct_log_or.loc[k] = v if np.isfinite(v) else float(total_log_or.loc[k])

    # ── Step 4: Current-model correction factor and adjusted intercept ───────
    # Equation 15 requires Λ to be calculated over the predictors and D_k
    # values used by this model.  A correction factor keyed only by outcome can
    # be stale when selection, temporal order, or direct effects change.
    p_marginal = float(outcome_prevalence)
    if not np.isfinite(p_marginal) or not (0.0 < p_marginal < 1.0):
        p_marginal = 0.5
    prior_odds   = p_marginal / (1.0 - p_marginal)
    beta_0_prior = float(np.log(prior_odds))

    demi_log_lambda_net = _compute_log_correction_factor(
        predictor_prevalence_series.reindex(direct_log_or.index).fillna(0.0),
        direct_log_or,
    )
    beta_0 = beta_0_prior - demi_log_lambda_net
    correction_factor = float(np.exp(demi_log_lambda_net))

    stored_demi_log_lambda_net = np.nan
    stored_beta_0 = np.nan
    if correction_factor_table is None and db_path is not None:
        try:
            correction_factor_table = fetch_demi_correction_factors_from_db(
                db_path,
                outcome,
                correction_factor_table=correction_factor_table_name,
            )
        except (KeyError, sqlite3.Error):
            correction_factor_table = None
    if correction_factor_table is not None:
        stored_demi_log_lambda_net, stored_beta_0 = _lookup_demi_correction_factors(
            correction_factor_table,
            concept_code_int=outcome,
        )

    # ── Assemble coefficient DataFrame ───────────────────────────────────────
    beta_vals   = direct_log_or.astype(float)
    odds_ratios = np.exp(beta_vals)

    coef_df = pd.DataFrame({
        "predictor_int": list(beta_vals.index) + ["(intercept)"],
        "predictor":     list(beta_vals.index) + ["(intercept)"],
        "beta":          list(beta_vals.values) + [beta_0],
        "odds_ratio":    list(odds_ratios.values) + [float(np.exp(beta_0))],
        "group": (
            [
                "K+" if p in positive_predictors
                else "K-" if p in protective_predictors
                else "excluded"
                for p in beta_vals.index
            ]
            + ["intercept"]
        ),
    })

    predictors = list(beta_vals.index)
    beta_vec   = beta_vals.values

    def predict_proba(patient_events):
        """Predict outcome probability."""
        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))

        if isinstance(patient_events, pd.DataFrame):
            return sigmoid(
                beta_0
                + patient_events.reindex(
                    columns=predictors, fill_value=0.0
                ).astype(float).to_numpy() @ beta_vec
            )
        if isinstance(patient_events, (dict, pd.Series)):
            feature_vector = np.array(
                [float(patient_events.get(p, 0.0)) for p in predictors]
            )
            return float(sigmoid(beta_0 + feature_vector @ beta_vec))
        feature_vector = np.asarray(patient_events, dtype=float)
        if feature_vector.ndim == 1:
            return float(sigmoid(beta_0 + feature_vector @ beta_vec))
        if feature_vector.ndim == 2:
            return sigmoid(beta_0 + feature_vector @ beta_vec)
        raise ValueError("patient_events must be 1D or 2D array-like")

    baseline_prob = p_marginal

    k_to_total = pd.DataFrame({
        "target_concept_code_int": outcome,
        "concept_code_int":        [int(p) for p in nodes],
        "total":                   total_or.reindex(nodes).values,
        "total_log_or":            total_log_or.reindex(nodes).values,
        "direct_log_or":           direct_log_or.reindex(nodes).values,
        "group": [
            "K+" if p in positive_predictors
            else "K-" if p in protective_predictors
            else "excluded"
            for p in nodes
        ],
    })

    return {
        "sorted_scores":            sorted_scores,
        "temporal_order":           temporal_order,
        "positive_predictors":      positive_predictors,
        "protective_predictors":    protective_predictors,
        "total_or":                 total_or,
        "total_log_or":             total_log_or,
        "direct_log_or":            direct_log_or,
        "weights":                  weights,
        "predictor_prevalence":     predictor_prevalence_series,
        "coef_df":                  coef_df,
        "beta_0":                   beta_0,
        "beta_0_prior":             beta_0_prior,
        "correction_factor":        correction_factor,
        "demi_log_lambda_net":      demi_log_lambda_net,
        "stored_demi_log_lambda_net": stored_demi_log_lambda_net,
        "stored_beta_0":            stored_beta_0,
        "prior_odds":               prior_odds,
        "baseline_probability":     baseline_prob,
        "n_missing_value":          int(n_missing_value),
        "n_factors_or_gt1":         int(len(positive_predictors)),
        "n_factors_or_lt1":         int(len(protective_predictors)),
        "outcome_prevalence":       outcome_prevalence,
        "n_outcome_positive":       n_outcome_positive,
        "n_outcome_negative":       n_outcome_negative,
        "n_qa_excluded":            n_qa,
        "n_rollup_excluded":        n_rollup,
        "n_rollup_snomed":          n_rollup_snomed,
        "n_rollup_pattern":         n_rollup_pattern,
        "beta":                     pd.Series(beta_vec, index=predictors, dtype=float),
        "logit_predictors":         predictors,
        "predict_proba":            predict_proba,
        "k_to_total":               k_to_total,
    }


def run_demi(
    df_in: Union[str, pd.DataFrame],
    *,
    force_outcome: int,
    lambda_min_count: int = 15,
    epv: int = 5,
    user_cap: int = 500,
    min_a: int = 0,
    snomed_relationship_path: Optional[str] = None,
    db_path: Optional[str] = DEMI_DB_PATH,
    use_code_group_mapping: bool = True,
    code_group_table: str = CODE_GROUP_TABLE,
    correction_factor_table_name: str = CORRECTION_FACTOR_TABLE,
    temporal_order_table_name: str = TEMPORAL_ORDER_TABLE,
    use_db_temporal_order: bool = True,
    # Compatibility parameters
    fix_suppression: bool = True,
    se_thr: float = 1.0,
    min_cell_thr: int = 5,
    cap_thr: int = 30,
    pref_override: Optional[float] = None,
    correction_factor_table: Optional[Union[str, pd.DataFrame]] = None,
    n_missing_value: int = 0,
    debug_total_effect: bool = False,
    is_diagnosis_set: Optional[Set[int]] = None,
    is_outcome_target: bool = False,
) -> Dict[str, Any]:
    """Run v17 selection, decomposition, and prediction setup.
    
    Arguments are forwarded to analyze_causal_sequence_py. An optional legacy
    CSV correction-factor override is used only for the stored-value audit."""
    if correction_factor_table is None and CORRECTION_FACTOR_TABLE_PATH:
        # Optional stored-value audit override.
        correction_factor_table = CORRECTION_FACTOR_TABLE_PATH

    return analyze_causal_sequence_py(
        df_in,
        force_outcome=force_outcome,
        lambda_min_count=lambda_min_count,
        epv=epv,
        user_cap=user_cap,
        min_a=min_a,
        snomed_relationship_path=snomed_relationship_path,
        db_path=db_path,
        use_code_group_mapping=use_code_group_mapping,
        code_group_table=code_group_table,
        correction_factor_table_name=correction_factor_table_name,
        temporal_order_table_name=temporal_order_table_name,
        use_db_temporal_order=use_db_temporal_order,
        fix_suppression=fix_suppression,
        se_thr=se_thr, min_cell_thr=min_cell_thr, cap_thr=cap_thr,
        pref_override=pref_override,
        correction_factor_table=correction_factor_table,
        n_missing_value=n_missing_value,
        debug_total_effect=debug_total_effect,
        is_diagnosis_set=is_diagnosis_set,
        is_outcome_target=is_outcome_target,
    )


# DB helpers
def _fetch_k_to_T(conn, outcome_code: str) -> pd.DataFrame:
    """One-sided fetch: rows where target matches the SNOMED string outcome code.
    Cheap — used for ranking only (no subgraph)."""
    q = f"""
      SELECT m.*
      FROM {EDGE_TABLE} m
      JOIN {CONCEPT_TABLE} tcn ON m.target_concept_code_int = tcn.concept_code_int
      WHERE tcn.concept_code = ?
    """
    return pd.read_sql_query(q, conn, params=[outcome_code])


def _fetch_subgraph_by_targets(conn, events_list_int: List[int]) -> pd.DataFrame:
    """One-sided fetch: rows where target_concept_code_int ∈ events_list_int.
    Call after EPV-balanced selection so the list is bounded before the query."""
    if not events_list_int:
        return pd.DataFrame()
    if len(events_list_int) > SQLITE_PARAM_LIMIT:
        raise ValueError(
            f"events_list_int has {len(events_list_int)} items, exceeding "
            f"safe SQLite parameter limit ({SQLITE_PARAM_LIMIT}). Reduce TOP_K."
        )
    ph = ",".join(["?"] * len(events_list_int))
    return pd.read_sql_query(
        f"SELECT * FROM {EDGE_TABLE} WHERE target_concept_code_int IN ({ph})",
        conn, params=[int(e) for e in events_list_int],
    )


def fetch_demi_subgraph_for_events(
    db_path: str,
    events: List[str],
    edge_table: str = EDGE_TABLE,
    concept_table: str = CONCEPT_TABLE,
) -> pd.DataFrame:
    """Two-sided DEMI subgraph fetch using SNOMED string codes."""
    if not events:
        raise ValueError("events list cannot be empty")
    events = sorted(set(str(e) for e in events))
    if len(events) > SQLITE_PARAM_LIMIT // 2:
        raise ValueError(
            f"Too many events for a two-sided subgraph query: "
            f"{len(events)} > {SQLITE_PARAM_LIMIT // 2}. "
            "Use fetch_demi_subgraph_two_sided_chunked(...) or reduce the event set."
        )

    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        ph = ",".join(["?"] * len(events))
        lookup_q = (
            f"SELECT concept_code, concept_code_int FROM {concept_table} "
            f"WHERE concept_code IN ({ph})"
        )
        lookup = pd.read_sql_query(lookup_q, conn, params=events)
        if lookup.empty:
            raise ValueError(
                f"None of the provided concept codes were found in {concept_table}. "
                f"First few: {events[:5]}"
            )
        found = set(lookup["concept_code"].astype(str))
        missing = [c for c in events if c not in found]
        if missing:
            print(f"[WARN] {len(missing)} codes not found in {concept_table}; "
                  f"ignoring. Example: {missing[:5]}")
        event_ints = [int(x) for x in lookup["concept_code_int"].dropna().astype(int).tolist()]
    return fetch_demi_subgraph_by_ints(
        db_path,
        event_ints,
        edge_table=edge_table,
        concept_table=concept_table,
    )


def fetch_demi_subgraph_by_ints(
    db_path: str,
    event_ints: List[int],
    edge_table: str = EDGE_TABLE,
    concept_table: str = CONCEPT_TABLE,
) -> pd.DataFrame:
    """Two-sided fetch: rows where BOTH target AND code are in the event set."""
    if not event_ints:
        raise ValueError("event_ints cannot be empty.")

    event_ints = sorted(set(int(e) for e in event_ints))

    if len(event_ints) * 2 > SQLITE_PARAM_LIMIT:
        raise ValueError(
            f"Two-sided query needs {len(event_ints) * 2} parameters for "
            f"{len(event_ints)} events, exceeding safe SQLite limit "
            f"({SQLITE_PARAM_LIMIT}). Use _fetch_subgraph_by_targets() instead."
        )

    uri = f"file:{db_path}?mode=ro"
    ph  = ",".join(["?"] * len(event_ints))
    query = f"""
        SELECT
            e.*,
            tcn.concept_code AS target_concept_code,
            ccn.concept_code AS concept_code_str
        FROM {edge_table} e
        LEFT JOIN {concept_table} tcn
            ON e.target_concept_code_int = tcn.concept_code_int
        LEFT JOIN {concept_table} ccn
            ON e.concept_code_int = ccn.concept_code_int
        WHERE e.target_concept_code_int IN ({ph})
          AND e.concept_code_int         IN ({ph})
    """
    try:
        with sqlite3.connect(uri, uri=True) as conn:
            df = pd.read_sql_query(query, conn, params=event_ints + event_ints)
    except sqlite3.Error as e:
        raise RuntimeError(f"[fetch_demi_subgraph_by_ints] DB error: {e}") from e

    if df.empty:
        raise RuntimeError(
            f"[fetch_demi_subgraph_by_ints] No edges found for "
            f"{len(event_ints)} event IDs."
        )

    print(f"[FETCH] {len(df):,} edges | {len(event_ints)} events | two-sided filter.")
    return df


def fetch_demi_subgraph_two_sided_chunked(
    conn: "sqlite3.Connection",
    event_ints: List[int],
    edge_table: str = EDGE_TABLE,
    concept_table: str = CONCEPT_TABLE,
    chunk_size: Optional[int] = None,
    include_names: bool = False,
) -> pd.DataFrame:
    """Fetch the induced DEMI subgraph on event_ints using two-sided filters,
    while safely handling SQLite's parameter limit via chunking."""
    if not event_ints:
        return pd.DataFrame()

    events = sorted({int(e) for e in event_ints})
    if chunk_size is None:
        chunk_size = max(1, SQLITE_PARAM_LIMIT // 2)

    chunks = [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]

    def _query_two_sided(t_chunk: List[int], c_chunk: List[int]) -> pd.DataFrame:
        t_ph = ",".join(["?"] * len(t_chunk))
        c_ph = ",".join(["?"] * len(c_chunk))

        if include_names:
            q = f"""
                SELECT
                    e.*,
                    tcn.concept_code AS target_concept_code,
                    ccn.concept_code AS concept_code_str
                FROM {edge_table} e
                LEFT JOIN {concept_table} tcn
                    ON e.target_concept_code_int = tcn.concept_code_int
                LEFT JOIN {concept_table} ccn
                    ON e.concept_code_int = ccn.concept_code_int
                WHERE e.target_concept_code_int IN ({t_ph})
                  AND e.concept_code_int        IN ({c_ph})
            """
        else:
            q = f"""
                SELECT *
                FROM {edge_table}
                WHERE target_concept_code_int IN ({t_ph})
                  AND concept_code_int        IN ({c_ph})
            """
        return pd.read_sql_query(q, conn, params=list(t_chunk) + list(c_chunk))

    # Fast path
    if len(events) <= chunk_size:
        df = _query_two_sided(events, events)
        return df.drop_duplicates()

    # Chunked path
    frames: List[pd.DataFrame] = []
    for t_chunk in chunks:
        for c_chunk in chunks:
            frames.append(_query_two_sided(t_chunk, c_chunk))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates()


