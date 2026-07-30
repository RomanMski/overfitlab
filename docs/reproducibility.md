# Reproducibility and artifacts

StressFold records enough protocol state to rerun an audit and diagnose many accidental changes. Exact reproduction still requires the source data, code revision, dependency environment, and estimator definition.

## Randomness

`AuditConfig.random_state` is the root of a stable named seed tree. Child seeds are derived from semantic paths such as split index, operator, severity, model fit, and permutation index. This has two useful properties:

- adding a new operator does not shift the random streams already assigned to other operators
- every recorded score can be traced back to a split seed, an operation seed, and a model seed

StressFold fills `random_state=None` parameters on scikit-learn-compatible estimators and nested pipeline steps. Explicit estimator seeds are respected. Estimators that use hidden, external, multithreaded, or hardware-dependent randomness can still prevent exact replay.

## Data fingerprint

The JSON result contains a 128-bit BLAKE2b fingerprint over feature names, dtypes, row order, feature values, and target values after input normalization. It is intended to detect changed audit inputs. It is not a cryptographic commitment for adversarial settings, a privacy guarantee, or a replacement for versioned source data.

## Artifact set

### `report.html`

A self-contained human-readable report with inline styles and figures. It includes the scope statement, generalization table, permutation table, stress curves, errors, estimator representation, fingerprint, and count of recorded seeds. It can be opened without a running StressFold service.

### `results.json`

The machine-readable audit object includes:

- schema and package versions
- creation time and interpretation scope
- `AuditConfig` and `StressSuite` values
- input dimensions and fingerprint
- estimator representation
- the complete named seed ledger
- scenario errors
- stress, generalization, and permutation summaries
- run-level records, when retained
- variant manifest entries, when exported

Non-finite numeric values are serialized as `null`, not non-standard JSON tokens.

### Variant CSVs and `manifest.json`

Set `AuditConfig(store_variants=True)` before `audit()` and call `export_variants(...)`. Each CSV begins with `__row_index__`, retains the feature columns, and appends the target. If the target name collides with a feature, StressFold uses `__stressfold_target__`.

The manifest records source fingerprint, filename, repeat, operator, level, partition, seed, original row indices, and operator-specific metadata. Exported variants may reveal source values and should inherit the source dataset’s access policy.

## Recommended experiment record

Retain these together:

```text
experiment/
├── results.json
├── report.html
├── estimator-and-feature-code/
├── environment lock or package list
├── source-data version or governed identifier
└── variants/                  # only when required
    ├── manifest.json
    └── *.csv
```

Also record the repository revision, operating system, Python version, BLAS/runtime details for strict numerical comparisons, and any model-selection decisions made after viewing an audit.

## Reproduction checklist

1. Verify the source fingerprint and row order.
2. Recreate the complete preprocessing and estimator pipeline.
3. Use the serialized config, suite, and root seed.
4. Use the same StressFold and dependency versions.
5. Compare run-level records and seed ledgers before comparing rounded summaries.
6. Treat small floating-point differences separately from changed splits or operator draws.

The technical paper has an independent fixed-seed reproduction path described in [`paper/README.md`](../paper/README.md). Its controlled experiments do not call the package API, which makes them a useful cross-check rather than a circular demonstration.
