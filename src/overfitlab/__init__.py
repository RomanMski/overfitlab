"""Measure how much of a backtest result came from searching, not from skill."""

from ._version import __version__
from .deflated import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .pbo import PBOResult, probability_of_backtest_overfitting
from .synthetic import (
    PathStressResult,
    block_permutation,
    generate_datasets,
    iid_bootstrap,
    moving_block_bootstrap,
    path_stress,
    stationary_bootstrap,
    write_datasets,
)

__all__ = [
    "DeflatedSharpeResult",
    "PBOResult",
    "PathStressResult",
    "block_permutation",
    "deflated_sharpe_ratio",
    "expected_maximum_sharpe",
    "probabilistic_sharpe_ratio",
    "generate_datasets",
    "iid_bootstrap",
    "moving_block_bootstrap",
    "path_stress",
    "probability_of_backtest_overfitting",
    "stationary_bootstrap",
    "write_datasets",
    "sharpe_ratio",
]
