# DEMI: Direct Effects Multiplicative Inference

DEMI predicts a binary clinical outcome from pairwise relationships among clinical concepts: co-occurrence counts and temporal-precedence counts aggregated over a population. It does not need patient-level records or a complete feature vector. The method orders a patient's features by population-level precedence, decomposes each feature's total effect on the outcome into direct and indirect (mediated) parts by working backward through that order, and aggregates the direct effects multiplicatively into odds of the outcome.

This repository is the code companion to:

> Alemi F, Lin Y, Elyazori H, Franzuela Cardenas V, Ramezani N, Lybarger K. Direct Effects Multiplicative Inference (DEMI): causal prediction from pairwise clinical relationships, applied to antidepressant response. *Journal of Biomedical Informatics*. 2026;105111. https://doi.org/10.1016/j.jbi.2026.105111

A preprint is on SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6537875

## What is here

| Path | What it is |
|---|---|
| `demi_v17.py` | The DEMI algorithm as a Python library (NumPy and pandas): the odds-ratio rules of Table 2, the backward recursion for direct effects, EPV-balanced predictor selection, the calibration correction factor, and `run_demi`, which returns a `predict_proba` function. Includes helpers for reading an authorized SQLite knowledgebase, which is not distributed. |
| `examples/synthetic_example.py` | The paper's synthetic example (Section 3.2) run end to end on an in-memory pairwise table. Needs no database. Prints every intermediate value next to the published one. **Run this first.** |
| `examples/worked_example.ipynb` | The paper's sertraline example (anxiety, major depressive disorder, citalopram non-response) with saved outputs that reproduce the published values. Rerunning it requires authorized access to the study knowledgebase. |
| `paper/appendix_code_v17.py` | The algorithm code exactly as submitted to the journal in the supplementary Appendix: the code of record for the article's results. Kept for provenance; use `demi_v17.py` for new work. |
| `paper/REPRODUCIBILITY.md` | What the article reports, which file produced it, what cannot be rerun from this repository, and the known differences between the two code files. |
| `LICENSE`, `NOTICE`, `CITATION.cff` | License terms, attribution, and citation metadata. |

## What is not here

- **The knowledgebase.** The pairwise relationship tables were derived from the NIH All of Us Research Program under its data use agreement and are not distributed. The All of Us data themselves are available only to approved researchers through the Researcher Workbench. Derived tables may be shared on reasonable request if the data use agreement permits; contact the corresponding author.
- **The Table 3 experiment harness.** The case-control sampling, SAFE/EPV feature selection, split-sample evaluation, AUC computation, and logistic regression comparison ran inside the All of Us Researcher Workbench and are not available outside it. See `paper/REPRODUCIBILITY.md`.
- **SNOMED CT release files.** The library's hierarchy helpers read a SNOMED CT relationship file that you must obtain under your own SNOMED CT license.

## Checking your installation

```
pip install numpy pandas
python examples/synthetic_example.py
```

The script builds the 1,000-patient synthetic knowledgebase from the paper and checks the temporal order, odds ratios (2.786 and 2.299), total effects (1.024 and 0.832), excess probability (0.075), direct effects (0.962 and 0.832), correction factor (2.288), posterior odds (0.657), and predicted probability (0.397) against the published values. Every line should read `PASS`.

## Using the library

```python
import pandas as pd
import demi_v17 as demi

edges = pd.DataFrame(...)   # pairwise rows: see examples/synthetic_example.py for the columns
result = demi.run_demi(edges, force_outcome=outcome_id, db_path=None,
                       use_code_group_mapping=False, use_db_temporal_order=False)
probability = result["predict_proba"]({feature_id: 1, ...})
```

Each row of `edges` describes one ordered pair (target concept, predictor concept) with the four 2 x 2 counts (`n_code_target`, `n_code_no_target`, `n_target_no_code`, `n_no_code_no_target`), the marginals, the two precedence counts, and `total_effects_log` for predictor-outcome rows. With `db_path` set to an authorized knowledgebase, `run_demi` also reads published temporal scores and the concept hierarchy from the database; `examples/worked_example.ipynb` shows that mode.

## License

The code is available for non-commercial use under the George Mason University Noncommercial Use License; the full text is in `LICENSE`, with the scope of what it covers stated at the top. It is a copyright license and does not grant patent or trademark rights. Commercial use requires a separate license from George Mason University: Office of Technology Transfer, ott@gmu.edu.

The license covers the code only. It does not cover All of Us data, the derived pairwise tables, SNOMED CT content, or any outputs saved in the notebook.

## Patent notice

The DEMI method is the subject of a pending U.S. patent application: "Medical Artificial General Intelligence (MAGI) System," U.S. Patent Application No. 19/253,342, published as US 2026/0066114 A1. Assignee: George Mason University.

## Funding

Patient-Centered Outcomes Research Institute (PCORI), award ME-2024C1-36732. The views expressed are those of the authors and do not necessarily represent the views of PCORI, its Board of Governors, or its Methodology Committee. We thank the participants of the All of Us Research Program and the NIH All of Us Research Program for access to the data used to construct the knowledgebase.

## Contact

Yili Lin (corresponding author), ylin26@gmu.edu
