# Browser lab

The browser lab is a local, bounded implementation of the StressFold idea. It is intended for protocol exploration, quick comparisons, and inspectable dataset variants. The Python package is the extensible interface for research audits.

## Current scope

The lab:

- accepts CSV files up to 5 MB and 5,000 data rows, rejecting larger tables rather than silently truncating them;
- supports binary classification and numeric-target regression;
- audits numeric predictors and reports ignored non-numeric or identifier-like columns;
- offers a regularized linear/logistic reference model and a high-capacity nearest-neighbor comparison;
- runs paired clean splits, fixed-model feature noise and missingness, label-noise refits, training-size refits, and a quick permutation null; and
- exports a self-contained HTML report and results JSON.

Audit controls expose the target, task, estimator, repeat count, audit share, and seed. A deterministic sample dataset is included so the complete interface can be exercised without supplying data.

## Local data path

The selected file is read with the browser file API and analyzed in the page. The current implementation has no dataset-upload endpoint. Hosting the page can still expose ordinary request metadata to its host, and downloaded reports or variants can contain sensitive information. Review exports before sharing them.

## Variant generator

The export panel can produce one feature-noise, label-noise, missingness, or empirical-bootstrap CSV together with a JSON manifest. The manifest identifies the source fingerprint, operator, severity, and seed.

These variants are stress instruments. They do not estimate a new population, increase the effective sample size, repair class imbalance, or create an independent test set. Bootstrap export is a row-resampling convenience and is not part of the audit response curves.

## Browser and Python interfaces

| Capability | Browser lab | Python package |
| --- | --- | --- |
| Estimators | Two built-in references | Any compatible estimator or pipeline |
| Predictors | Numeric | Numeric, categorical, or mixed when the supplied pipeline supports them |
| Input size | 5 MB and 5,000-row guardrails | Limited by the local Python process |
| Protocol control | Bounded UI presets and controls | Explicit `AuditConfig` and `StressSuite` |
| Artifacts | HTML, JSON, one-off variants | HTML, JSON, pandas tables, full retained variant set |
| Implementation | TypeScript, in browser | Python, NumPy/pandas/scikit-learn |

The implementations share concepts and defaults where practical, but they are not designed for bit-for-bit equivalence. Use the browser lab to learn the shape of a result or identify a follow-up question. Use the Python package, retained environment, and run-level records for publication or decision support.

## Run locally

Node.js 22.13 or newer is required:

```bash
npm ci
npm run dev
```

Run the browser build and its deterministic checks with:

```bash
npm test
```
