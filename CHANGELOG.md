# Changelog

All notable changes to OverfitLab are documented here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed, and it breaks callers

- `path_stress` no longer takes `scheme` and always permutes. Its percentile, p-value and structure dependence are only meaningful when the marginal distribution is held fixed, and the bootstrap schemes do not hold it fixed, so asking for one returned numbers labelled as though it had. `generate_datasets` and `write_datasets` still take `scheme` because they score nothing and now say in the manifest what they produced.

### Fixed

- Generated CSVs were rounded on the way out, `.10g` in Python and `toPrecision(10)` in the browser, so the files broke the exactness the tool claims for them. Both now write the shortest text that reads back as the identical double, negative zero included.
- The percentile counted synthetic scores strictly below the real one. A strategy that ignores ordering scores identically on every path, differing only in the last bits of a sum taken in a different order, so buy and hold reported 23 on one real series and 11 on another when the answer is 50. Ties are split within a tolerance.
- The browser verdict compared the overfitting result object against a number rather than its `pbo` field, which is always false, so trials that passed both checks were told the selection was unreliable.
- The manifest described every scheme as a reordering, which is false for both bootstraps.
- The site served its own committed copy of the paper and it had drifted. It is generated at build time now and untracked.

### Added

- A permutation p-value at every block length, and `shuffled_p_value()`. Structure dependence is a ratio and says how wide the gap is, not how often chance closes it, and on real data the two disagree in both directions.
- `scripts/real_data.py` and `docs/real-data.md`, the sweep on five liquid ETFs since 2000.
- Transaction costs through `cost_bps` on `path_stress`, charged against turnover for strategies that report their positions.

### Known limits

- Costs are a flat rate on turnover, which ignores that cost rises with size and worsens in the volatile stretches a strategy tends to trade. The two selection statistics report on whatever returns they are given, gross or net.
- Structure dependence is a ratio of two Sharpes, so it is unstable near zero and carries no confidence interval. Read it next to the p-value.
- The real data run is four fixed strategies on five instruments and no multiple testing correction is applied to those p-values.

## [1.0.0] - 2026-07-30

Rewritten. Earlier releases were a validation package for tabular machine
learning, which assumed rows were independent of one another. That assumption
does not hold for a price series, so the statistics, the interactive site and
the documentation were all replaced rather than adapted.

### Added

- `deflated_sharpe_ratio`, which asks whether a result clears what the best of N trials reaches when nothing has an edge
- `probability_of_backtest_overfitting`, the combinatorially symmetric cross-validation estimate of how often an in-sample winner underperforms out of sample
- `path_stress`, which rebuilds a price history hundreds of times and reruns a strategy on each, sweeping the block size so market structure is destroyed by degrees
- `stationary_bootstrap`, `moving_block_bootstrap` and `iid_bootstrap` for generating those histories
- Three interactive explainers on the site, covering overfitting itself, how searching manufactures a winner, and what a strategy does on markets that never happened

### Removed

- The tabular audit engine, its stress operators, its HTML report and its methods paper
- The scikit-learn dependency, which nothing imports now

### Known limits

- Both selection statistics assume trials are roughly exchangeable, so adaptive searches that stop early on poor candidates will be flattered
- The probability of backtest overfitting is blind to time order and cannot see a regime change
- Resampling preserves volatility clustering only within blocks
- Transaction costs are not modelled anywhere
