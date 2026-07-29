# Validation matrix

StressFold is tested against datasets chosen to make failure visible, not only against examples that make the tool look successful. The checks below validate numerical behavior and interface guardrails. They do not validate a user's split policy, perturbation assumptions, or deployment claim.

## Automated cases

| Case | Expected behavior | Why it matters |
| --- | --- | --- |
| Strong linear classification | Held-out AUROC above 0.95 and clear separation from shuffled-label fits | Detects an ordinary recoverable signal |
| Independent binary labels | Held-out AUROC remains near 0.5 and normalized stress rankings are withheld | A no-signal dataset must not receive a confident robustness ranking |
| Strong linear regression | Held-out R² above 0.95; measurement noise lowers retained skill | Checks regression direction, loss normalization, and stress response |
| XOR classification | A regularized linear model stays near chance while nearest neighbors exceed 0.85 AUROC | Demonstrates that the audit describes the supplied model, not an intrinsic property of the dataset |
| Interaction regression | Linear and nearest-neighbor baselines diverge sharply | Exposes model dependence outside additive structure |
| Imbalance and missing cells | Results remain finite; rare-class and imputation warnings are emitted | Exercises stratification and train-fold-only median imputation |
| One-row minority class | Browser audit is rejected before fitting | Repeated holdout evidence would otherwise be unusable |
| Constant regression target | Browser audit is rejected before fitting | R² and retained-skill normalization are undefined without target variation |
| Duplicate predictor patterns | A neutral review warning is shown without automatically blocking interpretation | Repeated patterns can be ordinary for discrete data and are not proof of leakage |
| Repeated entity identifiers | The generalization headline is blocked and an entity-aware split is requested | Random row splits can place one entity in both training and audit data |
| Near-perfect target proxy | The proxy is named and the generalization finding is blocked | A high held-out score is not credible when a predictor is target-derived or unavailable at prediction time |
| Exact null ties | The finite-sample midrank is exactly 50 | Degenerate shuffled scores must not look like evidence against the null |
| Edit after row 2,000 | The source fingerprint changes | Provenance covers the complete uploaded table rather than a prefix |
| Repeated run with one seed | Curves, summaries, and scores reproduce exactly | Makes browser results inspectable and rerunnable |
| Quoted CSV and missing cells | Parsing round-trips values correctly | Protects the upload boundary |
| More than 5,000 uploaded rows | The browser rejects the table and directs the user to Python | Prevents undisclosed prefix-only analysis |

The Python suite separately checks stressor invariants, paired summaries, report generation, CLI artifacts, seed recording, classification and regression behavior, and deterministic paper artifacts.

## Run the checks

```bash
python -m pytest -q
npm test
npm run lint
npx tsc --noEmit
```

At the current revision these commands execute 25 Python tests and 17 web tests. The web command includes a production build before the behavioral tests.

The paper figures and tables have an independent deterministic reproduction path:

```bash
python scripts/reproduce_paper.py --paper-dir paper
```

## Deliberate non-claims

The current validation does not establish support for multiclass targets, grouped or longitudinal dependence, rolling-origin evaluation, arbitrary categorical predictors in the browser lab, causal conclusions, calibration, fairness, privacy, or future-domain performance. The pooled Python null-exceedance rate and the browser null midrank are descriptive diagnostics, not permutation p-values. Inferential permutation testing requires coherent dataset-level permutations and one prespecified statistic for each complete workflow rerun.
