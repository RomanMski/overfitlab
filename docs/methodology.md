# Methodology

StressFold is a protocol for asking narrower questions than “is this model overfit?” It separates clean generalization, response to a declared perturbation, stability of the fitting procedure, and a falsification null. None is treated as a certificate.

## Unit under test

The unit under test is the complete estimator passed to `audit`: preprocessing, feature selection, calibration, hyperparameter choices, and final model. Any learned operation performed before the split sits outside the audit and can leak information.

For repeat $b$, the split policy produces a training fold $T_b$ and audit fold $H_b$. StressFold fits a fresh estimator $f_b = A(T_b)$ and evaluates the requested metric on both folds. Binary tasks use stratified shuffle splits when feasible; regression uses shuffled holdouts. The current engine assumes exchangeable rows.

## Evidence layers

### Generalization

The clean gap is computed within each repeat. For a loss metric,

$$
G_b = L(f_b, H_b) - L(f_b, T_b).
$$

For a score where higher is better, StressFold reverses the subtraction. In every output, positive `gap` or `degradation` means worse. The distribution over repeats shows sensitivity to the sampled split; it does not make the overlapping holdouts independent observations.

### Prediction robustness

A fixed clean model is evaluated on a perturbed audit fold. For operator $q_\lambda$ at severity $\lambda$,

$$
R_{b,\lambda} = L(f_b, q_\lambda(H_b)) - L(f_b, H_b).
$$

The paired comparison removes split-to-split level differences from the perturbation contrast. It remains conditional on the operator. A feature-noise curve is evidence about that feature-noise mechanism, not a universal statement about distribution shift.

### Training stability

Label noise and training fraction alter the training fold, refit the estimator, and evaluate it on the same clean audit fold. These paths measure sensitivity of the learning procedure to corrupted targets or reduced sample size. They are not fixed-model robustness tests.

### Falsification

The permutation null destroys the training target association, refits the complete estimator, and scores the paired clean audit fold. For $M$ valid null fits, StressFold reports

$$
p_{+1} = \frac{1 + \sum_{m=1}^{M}\mathbb{1}(S_m \text{ matches or exceeds } S_{\mathrm{obs}})}{M + 1}.
$$

“Exceeds” respects metric direction. The result tests predictive structure under label exchangeability. It is not a probability that the model is overfit, and it is only valid when the full model-selection procedure represented by the claim is repeated inside the null workflow.

## Operators

| Operator | Partition changed | Model | Severity meaning |
| --- | --- | --- | --- |
| Feature noise | Audit features | Fixed | Gaussian standard deviation in training-fold robust-scale units |
| Missingness | Audit features | Fixed | Fraction of currently observed selected cells masked uniformly without replacement |
| Label noise, binary | Training target | Refit | Fraction of labels flipped, rounded to the nearest attainable row count |
| Label noise, regression | Training target | Refit | Gaussian standard deviation in training-target robust-scale units |
| Training fraction | Training rows | Refit | Fraction retained; binary targets are stratified when feasible |
| Permutation null | Training target | Refit | Random permutation index, not a severity |

Feature scales use `IQR / 1.349`, then scaled MAD, sample standard deviation, and finally a documented unit fallback if earlier estimates are zero or undefined. Existing feature missingness is preserved during feature-noise injection. Missingness sampling excludes already missing cells. Zero severity is an exact identity operation.

Operators are calibrated within the training fold. This prevents the audit fold from determining perturbation scale, but it does not make an implausible operator realistic.

## Metrics and aggregation

Default binary metrics are ROC AUC, log loss, and balanced accuracy. Available binary metrics also include accuracy and Brier score. Regression defaults are RMSE, MAE, and $R^2$.

`records_frame()` retains every metric evaluation with its split, operation, and model seeds. `summary_frame()` groups by experiment, evidence type, evaluation, level, and metric. `mc_low` and `mc_high` are the requested empirical quantiles across Monte Carlo repeats. They are deliberately not named confidence intervals: repeated holdouts overlap, and the interval does not include every source of sampling, selection, or deployment uncertainty.

Scenario failures are recorded rather than silently discarded. Inspect `result.errors` and the per-group `n` before interpreting a curve.

## Valid uses

StressFold is useful for comparing complete estimators under one fixed protocol, finding abrupt sensitivity boundaries, checking whether apparent skill survives a simple null, and producing reproducible cases for deeper investigation.

Before using results for a decision:

1. choose a split policy that matches the sampling and deployment structure;
2. define perturbations from credible measurement or domain mechanisms;
3. keep a final untouched evaluation set if the audit guides model selection;
4. inspect paired records rather than only aggregate curves; and
5. report assumptions, failures, and uncertainty with the result.

The accompanying [`paper/main.tex`](../paper/main.tex) develops the estimands and controlled counterexamples in greater detail.
