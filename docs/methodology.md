# Methodology

StressFold is a protocol for asking narrower questions than “is this model overfit?” It separates clean generalization, response to a declared perturbation, stability of the fitting procedure, and a falsification null. None is treated as a certificate.

## Unit under test

The unit under test is the complete estimator passed to `audit`: preprocessing, feature selection, calibration, hyperparameter choices, and final model. Any learned operation performed before the split sits outside the audit and can leak information.

For repeat $b$, the split policy produces a training fold $T_b$ and audit fold $H_b$. StressFold fits a fresh estimator $f_b = A(T_b)$ and evaluates the requested metric on both folds. Binary tasks use stratified shuffle splits when feasible, and regression uses shuffled holdouts. The current engine assumes exchangeable rows.

## Evidence layers

### Generalization

The clean gap is computed within each repeat. For a loss metric,

$$
G_b = L(f_b, H_b) - L(f_b, T_b).
$$

For a score where higher is better, StressFold reverses the subtraction. In every output, positive `gap` or `degradation` means worse. The distribution over repeats shows sensitivity to the sampled split, and it does not turn overlapping holdouts into independent observations.

### Prediction robustness

A fixed clean model is evaluated on a perturbed audit fold. For operator $q_\lambda$ at severity $\lambda$,

$$
R_{b,\lambda} = L(f_b, q_\lambda(H_b)) - L(f_b, H_b).
$$

The paired comparison removes split-to-split level differences from the perturbation contrast. It remains conditional on the operator. A feature-noise curve is evidence about that feature-noise mechanism, not a universal statement about distribution shift.

### Training stability

Label noise and training fraction alter the training fold, refit the estimator, and evaluate it on the same clean audit fold. These paths measure sensitivity of the learning procedure to corrupted targets or reduced sample size. They are not fixed-model robustness tests.

### Falsification

The package permutation control destroys the training target association, refits the complete estimator, and scores the paired clean audit fold. For valid comparison $(b,m)$ at repeated holdout $b$ and shuffled refit $m$, define

$$
e_{bm}=\mathbb{1}(S^{\pi}_{bm}\text{ matches or exceeds }S_b),
\qquad
r_{\mathrm{pool}}=\frac{\sum_{b,m}e_{bm}}{\sum_b M_b}.
$$

“Exceeds” respects metric direction. StressFold reports $r_{\mathrm{pool}}$ as a descriptive paired null-exceedance rate together with the number of holdouts and null fits per holdout. Repeated holdouts overlap, so this pooled rate is not a permutation p-value and must not be given a significance interpretation.

A valid inferential permutation test needs one coherent dataset-level permutation for each $m$, a rerun of the complete repeated-holdout and selection workflow, and one prespecified aggregate statistic $T_m$ per permutation. The methods paper defines that broader reference protocol. The current package does not claim to implement its p-value.

## Operators

| Operator | Partition changed | Model | Severity meaning |
| --- | --- | --- | --- |
| Feature noise | Audit features | Fixed | Gaussian standard deviation in training-fold robust-scale units |
| Missingness | Audit features | Fixed | Fraction of currently observed selected cells masked uniformly without replacement |
| Label noise, binary | Training target | Refit | Fraction of labels flipped, rounded to the nearest attainable row count |
| Label noise, regression | Training target | Refit | Gaussian standard deviation in training-target robust-scale units |
| Training fraction | Training rows | Refit | Fraction retained, with binary targets stratified when feasible |
| Permutation null | Training target | Refit | Random permutation index, not a severity |

Feature scales use `IQR / 1.349`, then scaled MAD, sample standard deviation, and finally a documented unit fallback if earlier estimates are zero or undefined. Existing feature missingness is preserved during feature-noise injection. Missingness sampling excludes already missing cells. Zero severity is an exact identity operation.

Operators are calibrated within the training fold. This prevents the audit fold from determining perturbation scale, but it does not make an implausible operator realistic.

## Metrics and aggregation

Default binary metrics are ROC AUC, log loss, and balanced accuracy. Available binary metrics also include accuracy and Brier score. Regression defaults are RMSE, MAE, and $R^2$.

`records_frame()` retains every metric evaluation with its split, operation, and model seeds. `summary_frame()` groups by experiment, evidence type, evaluation, level, and metric. `mc_low` and `mc_high` are the requested empirical quantiles across Monte Carlo repeats. They are deliberately not named confidence intervals: repeated holdouts overlap, and the interval does not include every source of sampling, selection, or deployment uncertainty.

Scenario failures are recorded rather than silently discarded. Inspect `result.errors` and the per-group `n` before interpreting a curve.

## Auditing the search rather than the estimator

`audit()` treats one fitted estimator as the unit under test. When that estimator was chosen by a hyperparameter search, the unit under test is the search, and `audit_search()` audits it. Three quantities are reported and none is combined with the others.

**Selection optimism.** The complete search is refitted inside each outer holdout, so the search performs its own inner resampling on training rows only. The score it reports to itself is compared against the score its chosen configuration earns on the untouched outer fold:

$$
O_b = S^{\text{inner}}_b - S^{\text{outer}}_b .
$$

Positive $O_b$ means the search flattered itself. This is the paper's selection-optimism estimand written for a scorer where higher is better, so the sign convention matches the loss-scale definition rather than the arithmetic.

**Selection-aware permutation null.** For permutation $m$ the dataset target is shuffled once, the entire candidate search is rerun, and that search's own best score $S^{\pi}_m$ becomes the single statistic for that permutation:

$$
p = \frac{1 + \sum_{m=1}^{M} \mathbb{1}\left(S^{\pi}_m \ge S^{\text{reported}}\right)}{M + 1}.
$$

Because selection is repeated inside every permutation, the null describes a search of this size rather than one fixed configuration. Holding the winner fixed and permuting around it instead inflates the false-positive rate severely, which the paper measures at 77.3% against a nominal 5%.

Two properties of this p-value are worth stating plainly. Its smallest attainable value is $1/(M+1)$, so 30 permutations cannot report anything below 0.032. And a low value says the search found structure the shuffled null does not reach, which is not the same as saying the structure is causal, leak-free or stable.

**Winner stability.** The search is rerun on copies of the table perturbed at graded noise levels, and the audit records how often the reported configuration wins again. Noise scales are calibrated on the supplied table rather than on a training fold, because the question is whether the search settles in the same place when the data moves, not how well the winner generalizes.

Instability here constrains what you may claim about the configuration, not about the score. Many configurations are often close to equivalent, so a dataset with real signal can return a different winner on every run while its score stays sound. Report the score, and stop reporting the settings as optimal.

The audit refits the supplied search roughly `outer_repeats + permutation_repeats + noise_repeats * (levels - 1)` times, so cost scales with the search rather than with the data. `SearchAuditConfig.quick()` exists for a first pass.

## Valid uses

StressFold is useful for comparing complete estimators under one fixed protocol, finding abrupt sensitivity boundaries, checking whether apparent skill survives a simple null, auditing whether a tuned score survives its own selection, and producing reproducible cases for deeper investigation.

Before using results for a decision:

1. Choose a split policy that matches the sampling and deployment structure.
2. Define perturbations from credible measurement or domain mechanisms.
3. Keep a final untouched evaluation set if the audit guides model selection.
4. Inspect the paired records, not only the aggregate curves.
5. Report assumptions, failures, and uncertainty alongside the result.

The accompanying [`paper/main.tex`](../paper/main.tex) develops the estimands and controlled counterexamples in greater detail.
