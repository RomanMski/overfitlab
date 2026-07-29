"""Validated configuration objects for StressFold audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


_TASK_ALIASES = {
    "binary": "binary_classification",
    "classification": "binary_classification",
    "binary_classification": "binary_classification",
    "regression": "regression",
}


def normalize_task(task: str) -> str:
    """Return the canonical task name or raise a useful error."""

    try:
        return _TASK_ALIASES[str(task).strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(_TASK_ALIASES))
        raise ValueError(
            f"Unsupported task {task!r}; choose one of: {choices}"
        ) from exc


def _levels(
    values: Iterable[float],
    *,
    name: str,
    lower: float,
    upper: float | None,
    include: float,
) -> tuple[float, ...]:
    parsed = {float(value) for value in values}
    parsed.add(float(include))
    if not parsed:
        raise ValueError(f"{name} must contain at least one level")
    for value in parsed:
        if not math.isfinite(value):
            raise ValueError(f"{name} levels must be finite; got {value}")
        if value < lower or (upper is not None and value > upper):
            bound = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise ValueError(f"{name} levels must be in {bound}; got {value}")
    return tuple(sorted(parsed))


@dataclass(frozen=True, slots=True)
class AuditConfig:
    """Controls the repeated holdout protocol.

    ``random_state`` is the root of a stable, named seed tree. Adding a new
    stressor does not change seeds already assigned to other stressors.
    """

    task: str
    metrics: tuple[str, ...] | None = None
    repeats: int = 10
    test_size: float = 0.25
    random_state: int = 0
    interval: float = 0.90
    positive_label: Any | None = None
    store_variants: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_task(self.task))
        if self.metrics is not None:
            metrics = tuple(
                dict.fromkeys(str(metric).strip().lower() for metric in self.metrics)
            )
            if not metrics or any(not metric for metric in metrics):
                raise ValueError("metrics must contain non-empty metric names")
            object.__setattr__(self, "metrics", metrics)
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must lie strictly between 0 and 1")
        if not 0.0 < self.interval < 1.0:
            raise ValueError("interval must lie strictly between 0 and 1")
        if (
            isinstance(self.random_state, bool)
            or not isinstance(self.random_state, int)
            or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StressSuite:
    """A set of controlled perturbation paths and a permutation null."""

    feature_noise: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
    label_noise: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    missingness: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)
    train_fraction: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    permutation_repeats: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_noise",
            _levels(
                self.feature_noise,
                name="feature_noise",
                lower=0.0,
                upper=None,
                include=0.0,
            ),
        )
        object.__setattr__(
            self,
            "label_noise",
            _levels(
                self.label_noise, name="label_noise", lower=0.0, upper=1.0, include=0.0
            ),
        )
        object.__setattr__(
            self,
            "missingness",
            _levels(
                self.missingness, name="missingness", lower=0.0, upper=1.0, include=0.0
            ),
        )
        object.__setattr__(
            self,
            "train_fraction",
            _levels(
                self.train_fraction,
                name="train_fraction",
                lower=0.0,
                upper=1.0,
                include=1.0,
            ),
        )
        if self.train_fraction[0] <= 0.0:
            raise ValueError("train_fraction levels must be greater than 0")
        if self.permutation_repeats < 0:
            raise ValueError("permutation_repeats cannot be negative")

    @classmethod
    def standard(cls, **overrides: Any) -> "StressSuite":
        return cls(**overrides)

    @classmethod
    def quick(cls, **overrides: Any) -> "StressSuite":
        values: dict[str, Any] = {
            "feature_noise": (0.0, 0.25, 0.5),
            "label_noise": (0.0, 0.1, 0.2),
            "missingness": (0.0, 0.1, 0.2),
            "train_fraction": (0.5, 1.0),
            "permutation_repeats": 5,
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
