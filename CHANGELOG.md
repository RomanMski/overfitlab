# Changelog

All notable changes to StressFold will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning once its public interfaces stabilize.

## [Unreleased]

### Planned

- Broader split policies and domain-informed stress operators
- Public benchmark cases with known failure modes
- An HTML report for the search audit, which currently emits JSON and pandas frames only

## [0.3.0] - 2026-07-30

### Added

- `audit_search()`, `SearchAuditConfig` and `SearchAuditResult`, which audit the hyperparameter search that produced a model rather than the fitted model alone
- Selection optimism measurement, implementing the paper's `O_b` by wrapping a complete search in an outer holdout it never sees
- A selection-aware permutation null that reruns the entire candidate search inside every permutation, which is the requirement the paper states for a valid permutation p-value
- Winner stability measurement, which reruns the search on jittered copies of the table and records whether the selected configuration survives
- `examples/python/search_optimism.py`, which runs one 24-candidate search over real signal and over pure noise and shows that shuffled-target searches can beat the score reported on noise

### Changed

- The paper no longer describes full-workflow permutation inference as unimplemented, because the search audit now satisfies its three stated requirements. Generator auditing and group- and time-aware splitting remain unimplemented and are still labelled as such.
- The package version has a single source in `stressfold._version`, replacing three separately hardcoded copies in the package, the JSON artifact and the HTML footer

## [0.2.0] - 2026-07-30

### Added

- Interactive measurement-noise explainer with fixed clean positions, a fixed decision boundary, and live prediction-change counts
- Explicit contribution, non-claim, and current-software boundary statements in the paper and project overview

### Changed

- Replaced the pooled repeat-by-permutation `plus_one_p` field with a descriptive paired null-exceedance rate, so overlapping holdouts are no longer presented as independent permutation evidence
- Added tie-adjusted browser null ranks, minimum class-count and non-constant-target checks, and full-dataset provenance hashing
- Bumped the exported JSON schema to `1.1` for the corrected permutation-summary fields

### Fixed

- Withhold normalized stressor rankings when the clean baseline does not reliably outperform its constant reference
- Block interpretation for repeated entity identifiers and near-perfect target proxies, while keeping repeated predictor patterns as a neutral review note
- Reject browser uploads above 5,000 rows instead of silently analyzing only the first 5,000

## [0.1.0] - 2026-07-29

### Added

- Repeated holdout audits for tabular binary classification and regression
- Clean train/audit generalization gaps with direction-normalized degradation
- Fixed-model feature-noise and missingness response curves
- Refit label-noise and training-fraction stability paths
- Label-permutation nulls with plus-one Monte Carlo p-values
- Stable named seeds, source fingerprints, structured JSON, and self-contained HTML reports
- Optional CSV stress-variant export with row-level provenance manifests
- Local browser lab for bounded numeric-data audits and inspectable variant generation
- Reproducible LaTeX paper figures and tables
