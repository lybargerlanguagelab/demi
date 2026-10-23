# Reproducibility notes

What the article reports, which file in this repository produced it, and what cannot be rerun from here. Everything in this note concerns the article's algorithm, version 17, which lives in this `paper/` directory. The current algorithm in the repository root (`demi.py`, version 23) scores differently and is not expected to reproduce the article's numbers; `CHANGELOG.md` lists the differences and `examples/synthetic_example.py` shows them side by side.

## Two copies of the article's algorithm

`paper/demi_as_reviewed.py` is the DEMI v17 code that accompanied the manuscript through peer review as part of its supplementary Appendix, unchanged apart from the header comment. The code was removed from the supplement before publication so that it is distributed only here, under the repository license; the published Appendix contains the assumptions and the factorization proof only. This file is the code of record: the algorithm that produced the Table 3 results and the worked examples in the article.

`paper/demi_v17.py` is the same algorithm as a library (it was the root `demi.py` at tag `v17`). The odds-ratio rules (Table 2), the sign-group partition, and the backward recursion for direct effects are functionally identical to the reviewed code. The library adds a SQLite knowledgebase layer, a SNOMED CT hierarchy helper, and a `run_demi` entry point, and it differs from the reviewed code in three ways that can change intermediate values on real data:

1. **Source of the temporal order.** The reviewed code orders features from the precedence counts in the rows it is given (the both-occurred terms of Equation 8: "target before code" minus "code before target"). The library prefers precomputed global temporal scores from the knowledgebase (`demi_temporal_order_published`), computed over all concepts, and falls back to the local computation when no database is supplied. The two can order features differently. In the sertraline example, the article's Equation 8 scores (all four terms, as in the notebook's step 3) and the local precedence-only scores both give the order anxiety, citalopram non-response, MDD, while the library's global scores give anxiety, MDD, citalopram non-response. The predicted probability is identical (0.1837) because the order within each sign group is unchanged, and the notebook checks this explicitly.
2. **Total effects.** The library reads precalculated `total_effects_log` values from the knowledgebase rows rather than recomputing them from the 2 x 2 counts. The notebook recomputes them from the counts and stops if the two disagree.
3. **Predictor prevalence.** The library computes each predictor's marginal prevalence from that predictor-outcome pair's own denominator, never from a reversed row.

Both files reproduce the article's synthetic example exactly (see below), so either can be used to verify the computation.

## What reproduces, and how

| Article result | File | Needs | Status |
|---|---|---|---|
| Synthetic example, Section 3.2 (T = 1.024 and 0.832, D = 0.962 and 0.832, Lambda = 2.288, probability 0.397) | `examples/synthetic_example.py` with `paper/demi_v17.py` | nothing beyond NumPy and pandas | reproduces exactly; the reviewed code gives the same values on the same input |
| Sertraline worked example, Section 3.2 (scores 90,886 / -7,698 / -83,188; D = 0.376, 0.269, -0.631; Lambda = 1.126; posterior odds 0.225; probability 0.184) | `paper/worked_example.ipynb` | authorized study knowledgebase | saved outputs in the notebook reproduce the published values; rerunning requires the database |
| Table 3 (AUC for 15 antidepressant prediction tasks, DEMI versus logistic regression) | `paper/demi_as_reviewed.py` for the DEMI side | the All of Us Researcher Workbench and the experiment harness | not reproducible from this repository; see below |

## What is not available

The Table 3 experiments ran inside the NIH All of Us Researcher Workbench on the full knowledgebase (430 million pairwise relationships from 358,082 participants). The harness around the algorithm (sampling 1,000 responders and 4,000 non-responders per task, the modified SAFE/EPV feature selection, 2-fold cross-validation for logistic regression, 50% split-sample evaluation for DEMI, AUC computation, and the paired comparison) was workspace code that did not survive a workspace migration. We have chosen not to reconstruct it from memory. The algorithm it called is `paper/demi_as_reviewed.py`, and the knowledgebase cannot leave the All of Us environment.

Anyone with All of Us access who rebuilds the knowledgebase will not get identical pairwise counts, because the All of Us dataset is versioned and participant counts change between releases. Results in the article should be compared at the level of the reported ranges and the conclusion (no significant difference in AUC between DEMI and logistic regression), not cell by cell.

## If your numbers differ

Run `examples/synthetic_example.py` first. If the version 17 block passes, the installation computes the article's algorithm correctly, and any difference on your own data comes from the data or from the temporal-order source (point 1 above). To force the reviewed code's behavior in the library, pass `db_path=None` and `use_db_temporal_order=False` to `run_demi`, which uses local precedence counts.
