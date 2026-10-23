# DEMI: Direct Effects Multiplicative Inference

DEMI predicts a binary clinical outcome from pairwise relationships among clinical concepts: co-occurrence counts and temporal-precedence counts aggregated over a population. It does not need patient-level records or a complete feature vector. The method orders a patient's features by population-level precedence, decomposes each feature's total effect on the outcome into direct and indirect (mediated) parts by working backward through that order, and aggregates the direct effects multiplicatively into odds of the outcome.

This repository is the code companion to:

> Alemi F, Lin Y, Elyazori H, Franzuela Cardenas V, Ramezani N, Lybarger K. Direct Effects Multiplicative Inference (DEMI): causal prediction from pairwise clinical relationships, applied to antidepressant response. *Journal of Biomedical Informatics*. 2026;105111. https://doi.org/10.1016/j.jbi.2026.105111

## What is here

| Path | What it is |
|---|---|
| `demi.py` | The DEMI algorithm as a Python library (NumPy and pandas): the odds-ratio rules of Table 2, the backward recursion for direct effects, EPV-balanced predictor selection, the calibration correction factor, and `run_demi`, which returns a `predict_proba` function. Includes helpers for reading an authorized SQLite knowledgebase, which is not distributed. |
| `examples/synthetic_example.py` | The paper's synthetic example (Section 3.2) run end to end on an in-memory pairwise table. Needs no database. Prints every intermediate value next to the published one. **Run this first.** |
| `examples/worked_example.ipynb` | The paper's sertraline example (anxiety, major depressive disorder, citalopram non-response) with saved outputs that reproduce the published values. Rerunning it requires authorized access to the study knowledgebase. Its last section reports a known difference in feature ordering between the library's global temporal scores and the article's Equation 8; see `paper/REPRODUCIBILITY.md`. |
| `paper/demi_as_reviewed.py` | The algorithm code that accompanied the manuscript through peer review (it was part of the supplementary Appendix and was removed before publication so that it is distributed only here, under this license). The code of record for the article's results, kept unchanged for provenance; use `demi.py` for new work. |
| `paper/REPRODUCIBILITY.md` | What the article reports, which file produced it, what cannot be rerun from this repository, and the known differences between the two code files. |
| `LICENSE`, `NOTICE`, `CITATION.cff`, `requirements.txt` | License terms, attribution, citation metadata, and Python dependencies. |

## What is not here

- **The knowledgebase.** The pairwise relationship tables were derived from the NIH All of Us Research Program under its data use agreement and are not distributed with this release. The All of Us data themselves are available only to approved researchers through the Researcher Workbench. We are preparing a reduced version of the knowledgebase that can be shared under All of Us dissemination rules (aggregate counts only, no count below 20) and will post it here, or a pointer to it, when it is cleared. Until then, derived tables may be shared on reasonable request where the data use agreement permits; contact the corresponding author. (Status as of October 2026.)
- **The Table 3 experiment harness.** The case-control sampling, SAFE/EPV feature selection, split-sample evaluation, AUC computation, and logistic regression comparison ran inside the All of Us Researcher Workbench and are not available outside it. See `paper/REPRODUCIBILITY.md`.
- **SNOMED CT release files.** The library's hierarchy helpers read a SNOMED CT relationship file that you must obtain under your own SNOMED CT license.

## Checking your installation

Requires Python 3.9 or later with NumPy and pandas (`requirements.txt`). Tested with Python 3.10 (NumPy 2.2, pandas 2.3) and Python 3.13 (NumPy 2.5, pandas 3.0).

```
pip install -r requirements.txt
python examples/synthetic_example.py
```

The script builds the 1,000-patient synthetic knowledgebase from the paper and checks the temporal order, odds ratios (2.786 and 2.299), total effects (1.024 and 0.832), excess probability (0.075), direct effects (0.962 and 0.832), correction factor (2.288), posterior odds (0.657), and predicted probability (0.397) against the published values. Every line should read `PASS`.

## Using the library

```python
import pandas as pd
import demi

edges = pd.DataFrame(...)   # pairwise rows: see examples/synthetic_example.py for the columns
result = demi.run_demi(edges, force_outcome=outcome_id, db_path=None,
                       use_code_group_mapping=False, use_db_temporal_order=False)
probability = result["predict_proba"]({feature_id: 1, ...})
```

### Input format

Each row of `edges` describes one ordered pair of concepts, a target and a predictor ("code"), with counts of unique patients:

| Column | Meaning |
|---|---|
| `target_concept_code_int`, `concept_code_int` | integer IDs of the target and the predictor |
| `n_code_target` | predictor present and target present (a) |
| `n_code_no_target` | predictor present, target absent (b) |
| `n_target_no_code` | predictor absent, target present (c) |
| `n_no_code_no_target` | neither present (d) |
| `n_code`, `n_target`, `n_no_target` | marginals: predictor present; target present; target absent |
| `n_code_before_target`, `n_target_before_code` | precedence counts among patients with both: predictor first; target first |
| `total_effects`, `total_effects_log` | adjusted odds ratio (Table 2 rules) and its natural log, required on rows whose target is the outcome; `compute_or_v17_scalar(a, b, c, d)` computes it |

Rows whose target is the outcome supply total effects and prevalences. Rows whose target is a downstream feature supply the mediation weight P(downstream = 1 | predictor = 1) and the precedence counts for temporal ordering. `examples/synthetic_example.py` builds a complete table by hand.

With `db_path` set to an authorized knowledgebase, `run_demi` also reads published temporal scores and the concept hierarchy from the database; `examples/worked_example.ipynb` shows that mode.

## Versions

Versions are marked with git tags, listed in `CHANGELOG.md`. The tag `v17` is the algorithm version evaluated in the article and is the one to cite for reproducing the paper; later tags may change the library. Internal version numbers (v17, v23, ...) follow the lab's development history.

## License

The code is available for non-commercial use under the George Mason University Noncommercial Use License; the full text is in `LICENSE`, with the scope of what it covers stated at the top. It is a copyright license and does not grant patent or trademark rights. Commercial use requires a separate license from George Mason University: Office of Technology Transfer, ott@gmu.edu.

The license covers the code only. It does not cover All of Us data, the derived pairwise tables, SNOMED CT content, or any outputs saved in the notebook.

## Patent notice

The DEMI method is the subject of a pending U.S. patent application: "Medical Artificial General Intelligence (MAGI) System," U.S. Patent Application No. 19/253,342, published as US 2026/0066114 A1. Assignee: George Mason University.

## Funding

Patient-Centered Outcomes Research Institute (PCORI), award ME-2024C1-36732. The views expressed are those of the authors and do not necessarily represent the views of PCORI, its Board of Governors, or its Methodology Committee. We thank the participants of the All of Us Research Program and the NIH All of Us Research Program for access to the data used to construct the knowledgebase.

## Contact

Questions about the code or the article: Yili Lin (corresponding author), ylin26@gmu.edu, or open an issue in this repository.
