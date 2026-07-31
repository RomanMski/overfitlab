# Changelog

All notable changes to OverfitLab are documented here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
