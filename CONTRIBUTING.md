# Contributing to OverfitLab

OverfitLab is small enough that a focused pull request can materially improve it. Contributions are welcome in four areas: statistical method, Python implementation, browser instrumentation, and reproducible examples.

## Before changing the method

Open an issue for a new estimand, stress operator, split policy, or inferential claim. State:

1. The question the change answers.
2. The unit under test and the data assumptions.
3. The failure mode it should reveal.
4. The expected behavior at zero severity and as severity increases.
5. How leakage and randomness will be controlled.
6. A reference or controlled experiment that could falsify the implementation.

A new chart or summary should not collapse generalization, robustness, stability, and falsification into a single score.

## Development setup

Python work requires Python 3.10 or newer:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Browser work requires Node.js 22.13 or newer:

```bash
npm ci
npm test
```

To regenerate the technical paper artifacts:

```bash
python scripts/reproduce_paper.py
```

Compile `paper/main.tex` with `latexmk` only after the numerical artifacts have been regenerated. Generated values should never be edited by hand.

## Reproducibility requirements

Changes that involve randomness should:

- draw randomness from the named seed tree instead of shared global state
- leave existing random streams stable when unrelated operators are added
- make zero severity an exact identity operation where applicable
- record operator parameters and provenance in machine-readable output
- calibrate feature or target scales from training data only
- include deterministic tests plus at least one statistical-behavior test

Tests should verify claims, not only snapshots. A useful statistical test specifies a data-generating process, an expected qualitative result, a fixed seed, and enough tolerance to stay stable across supported dependency versions.

## Pull requests

Keep the scope narrow and explain:

- what scientific or user-facing problem is addressed
- which public schema or interpretation changes
- how the result was tested
- what remains outside the claim

Run the relevant Python and browser checks before requesting review. Update `CHANGELOG.md` for user-visible behavior and documentation when a claim, default, or output field changes.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are licensed under the repository’s [MIT License](LICENSE).
