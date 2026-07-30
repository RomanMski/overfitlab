# Changelog

All notable changes to StressFold will be documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning once its public interfaces stabilize.

## [Unreleased]

### Planned

- Broader split policies and domain-informed stress operators
- Public benchmark cases with known failure modes

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
