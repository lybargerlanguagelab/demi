# Changelog

Versions are git tags on this repository. Internal version numbers follow the lab's development history; not every internal version is released here.

## v23 (2026-10-23)

The current algorithm, as used in the lab's deployed system, extracted as a standalone library (`demi.py`). Same input format as v17. Changes relative to v17:

- **Odds ratios.** Tables with no zero cells keep the article's rules (raw odds ratio for large samples, the Table 2 cross-ratio rule for small ones), so the synthetic example gives 2.786 and 2.299 under both versions. Tables with a zero cell no longer use the Table 2 add-one formulas: in large samples the log odds ratio is derived from the phi coefficient (`compute_or_v23_scalar`), and in small samples a zero-cell table contributes only its direction. "Small" is now decided per target (fewer than 53 outcome-positive cases) rather than per feature (a + b or a + c below 30).
- **Direct effects.** The backward recursion that subtracted mediated effects within each sign group is replaced by a noisy-OR formulation with three states per predictor (present, absent, missing), giving a log odds ratio for presence (D1) and for absence (D0) rather than one direct effect. On the synthetic example, Z1's direct effect is 1.024 (v23) versus 0.962 (v17), since the noisy-OR step removes much less of the mediated path.
- **Feature selection.** The SAFE/EPV cap is gone; predictors enter when the chi-square of their 2 x 2 table exceeds the 1-df critical value at alpha = 0.05. Targets with fewer than 53 positive cases use a net-count rule (each present predictor votes by the sign of its phi) instead of direct effects.
- **Probability.** The correction factor Lambda (Equation 15) is replaced by a per-target three-anchor calibration table used in the deployed system. Without a calibration table the library reports prior log-odds plus score. On the synthetic example the predicted probability is 0.615 (v23, uncalibrated) versus 0.397 (v17).

What stayed the same: the pairwise input, temporal ordering by precedence scores (global scores from the knowledgebase when supplied, local otherwise), and the sign-group structure.

The article's algorithm remains in `paper/` and at tag `v17`. Results in the article were produced with v17; v23 has not been evaluated in a publication yet.

## v17 (2026-10-23)

The algorithm version evaluated in Alemi et al., *Journal of Biomedical Informatics*, 2026, doi:10.1016/j.jbi.2026.105111. Contents: `demi.py` (library), `paper/demi_as_reviewed.py` (the code that accompanied the manuscript through review, unchanged), the synthetic and sertraline worked examples, license, and reproducibility notes. Both code files reproduce the article's synthetic example exactly. In later versions the v17 library lives at `paper/demi_v17.py`.
