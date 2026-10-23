# DEMI: Direct Effects Multiplicative Inference

DEMI predicts a binary clinical outcome from pairwise relationships among clinical concepts: co-occurrence counts and temporal-precedence counts aggregated over a population. It does not need patient-level records or a complete feature vector. The method orders a patient's features by population-level precedence, decomposes each feature's total effect on the outcome into direct and indirect (mediated) parts by working backward through that order, and aggregates the direct effects multiplicatively into odds of the outcome.

This repository is the code companion to:

> Alemi F, Lin Y, Elyazori H, Franzuela Cardenas V, Ramezani N, Lybarger K. Direct Effects Multiplicative Inference (DEMI): causal prediction from pairwise clinical relationships, applied to antidepressant response. *Journal of Biomedical Informatics*. 2026;105111. https://doi.org/10.1016/j.jbi.2026.105111

## Two versions, and which is which

The article evaluated **version 17**. The lab has kept developing the method, and the current version is **23**, which scores differently (phi-based handling of zero-cell tables, noisy-OR direct effects, chi-square feature selection, and a calibration step in place of the correction factor; see `CHANGELOG.md`). This repository carries both:

- `demi.py` in the root is the **current algorithm (v23)**.
- `paper/` holds the **article's algorithm (v17)**: the library, the code that went through peer review, the worked example, and the reproducibility notes. The git tag `v17` also freezes the whole repository as it stood for the article.

To reproduce the paper, use `paper/`. To use DEMI, use `demi.py`.

## What is here

| Path | What it is |
|---|---|
| `demi.py` | The current DEMI algorithm (version 23) as a Python library (NumPy and pandas): odds-ratio rules with phi-based handling of zero cells, noisy-OR direct effects with present, absent, and missing states, feature selection, and `run_demi`, which returns `predict_proba` and `predict_score` functions. No database code. In the lab's deployed system this logic runs inside a service that supplies global temporal scores, hierarchy filtering, and a calibration table from a database; here those are optional inputs, and without them the module orders features locally and reports probabilities as prior log-odds plus score. |
| `examples/synthetic_example.py` | The paper's synthetic example (Section 3.2) on an in-memory pairwise table. Checks the article's algorithm against every published value, then runs the current algorithm on the same table and prints the two side by side. Needs no database. **Run this first.** |
| `paper/demi_v17.py` | The article's algorithm (version 17) as a library. |
| `paper/demi_as_reviewed.py` | The version 17 code that accompanied the manuscript through peer review (it was part of the supplementary Appendix and was removed before publication so that it is distributed only here, under this license). The code of record for the article's results, kept unchanged for provenance. |
| `paper/worked_example.ipynb` | The paper's sertraline example (anxiety, major depressive disorder, citalopram non-response) run with `paper/demi_v17.py`, with saved outputs that reproduce the published values. Rerunning it requires authorized access to the study knowledgebase. Its last section reports a known difference in feature ordering between the library's global temporal scores and the article's Equation 8; see `paper/REPRODUCIBILITY.md`. |
| `paper/REPRODUCIBILITY.md` | What the article reports, which file produced it, what cannot be rerun from this repository, and the known differences between the code files. |
| `CHANGELOG.md`, `LICENSE`, `NOTICE`, `CITATION.cff`, `requirements.txt` | Version history, license terms, attribution, citation metadata, and Python dependencies. |

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

The script builds the 1,000-patient synthetic knowledgebase from the paper and checks the article's algorithm against the published values: temporal order, odds ratios (2.786 and 2.299), total effects (1.024 and 0.832), excess probability (0.075), direct effects (0.962 and 0.832), correction factor (2.288), posterior odds (0.657), and predicted probability (0.397). Every line in that block should read `PASS`. It then prints the current algorithm's values on the same table; those are expected to differ (for example, probability 0.615 instead of 0.397), and the table shows by how much.

## Using the library

```python
import pandas as pd
import demi

edges = pd.DataFrame(...)   # pairwise rows: see examples/synthetic_example.py for the columns
result = demi.run_demi(edges, force_outcome=outcome_id, require_global_temporal_order=False)
probability = result["predict_proba"]({feature_id: 1, ...})
```

`run_demi` also accepts `global_temporal_order` (precomputed temporal scores), `calibration_table` (the three-anchor calibration used in the deployed system), and `pairwise_block_reader` (a callback that streams pairwise rows for large targets). Without `calibration_table`, probabilities are prior log-odds plus score; with it, the calibrated probability the deployed system returns.

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
| `total_effects`, `total_effects_log` | adjusted odds ratio and its natural log on rows whose target is the outcome; `compute_or_v23_scalar(a, b, c, d)` computes it for the current version, `compute_or_v17_scalar` in `paper/demi_v17.py` for the article's |

Rows whose target is the outcome supply total effects and prevalences. Rows whose target is a downstream feature supply the mediation weight and the precedence counts for temporal ordering. `examples/synthetic_example.py` builds a complete table by hand. The input format is the same for both versions.

## Versions

Versions are git tags, listed in `CHANGELOG.md` with what changed. `v17` is the version evaluated in the article and the one to cite for reproducing it; `v23` is the current version. Internal version numbers follow the lab's development history, and not every internal version is released here.

## License

The code is available for non-commercial use under the George Mason University Noncommercial Use License; the full text is in `LICENSE`, with the scope of what it covers stated at the top. It is a copyright license and does not grant patent or trademark rights. Commercial use requires a separate license from George Mason University: Office of Technology Transfer, ott@gmu.edu.

The license covers the code only. It does not cover All of Us data, the derived pairwise tables, SNOMED CT content, or any outputs saved in the notebook.

## Patent notice

The DEMI method is the subject of a pending U.S. patent application: "Medical Artificial General Intelligence (MAGI) System," U.S. Patent Application No. 19/253,342, published as US 2026/0066114 A1. Assignee: George Mason University.

## Funding

Patient-Centered Outcomes Research Institute (PCORI), award ME-2024C1-36732. The views expressed are those of the authors and do not necessarily represent the views of PCORI, its Board of Governors, or its Methodology Committee. We thank the participants of the All of Us Research Program and the NIH All of Us Research Program for access to the data used to construct the knowledgebase.

## Contact

Questions about the code or the article: Yili Lin (corresponding author), ylin26@gmu.edu, or open an issue in this repository.
