"""Deterministic, fold-aware perturbations for tabular stress tests.

The functions in this module never mutate their inputs.  Any statistic used to
calibrate a perturbation is either supplied explicitly as a training fold or is
computed from the object being perturbed when that object *is* the training
fold.  Every operation returns JSON-serialisable metadata sufficient to replay
the perturbation from its seed and configuration.

Mixed numeric/categorical data are supported through :class:`pandas.DataFrame`.
Two-dimensional NumPy arrays are also accepted; object arrays are inspected
column by column so genuinely numeric object columns can still be identified.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numbers
from typing import Any, Generic, Hashable, Literal, Sequence, TypeVar

import numpy as np
import pandas as pd


T = TypeVar("T")
Task = Literal["binary", "regression"]
Tabular = pd.DataFrame | np.ndarray
Target = pd.Series | pd.DataFrame | np.ndarray


@dataclass(frozen=True)
class StressResult(Generic[T]):
    """A copied perturbation result and its JSON-serialisable provenance."""

    data: T
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        # Fail close if a future metadata addition accidentally contains a
        # NumPy scalar, Timestamp, or another value that report JSON cannot use.
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(
                "stress metadata must be strictly JSON-serialisable"
            ) from exc

    def __iter__(self):
        """Allow ``data, metadata = result`` without hiding named attributes."""

        yield self.data
        yield self.metadata


@dataclass(frozen=True)
class _FrameSpec:
    kind: Literal["dataframe", "ndarray"]


@dataclass(frozen=True)
class _TargetSpec:
    kind: Literal["series", "dataframe", "ndarray_1d", "ndarray_2d"]
    name: Hashable | None
    dtype: np.dtype[Any] | None


def _validate_seed(random_state: int) -> int:
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state, (int, np.integer)
    ):
        raise TypeError("random_state must be a non-negative integer")
    seed = int(random_state)
    if seed < 0:
        raise ValueError("random_state must be a non-negative integer")
    return seed


def _validate_level(
    value: float,
    *,
    name: str,
    lower: float = 0.0,
    upper: float | None = None,
    include_lower: bool = True,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    below = result < lower if include_lower else result <= lower
    if below or (upper is not None and result > upper):
        left = "[" if include_lower else "("
        right = "]" if upper is not None else ", infinity)"
        bound = (
            f"{left}{lower}, {upper}{right}"
            if upper is not None
            else f"{left}{lower}{right}"
        )
        raise ValueError(f"{name} must be in {bound}")
    return result


def _normalise_task(task: str) -> Task:
    if not isinstance(task, str):
        raise TypeError("task must be 'binary' or 'regression'")
    normalised = task.strip().lower().replace("-", "_")
    aliases: dict[str, Task] = {
        "binary": "binary",
        "classification": "binary",
        "binary_classification": "binary",
        "regression": "regression",
    }
    try:
        return aliases[normalised]
    except KeyError as exc:
        raise ValueError("task must be binary classification or regression") from exc


def _as_frame(X: Tabular, *, name: str) -> tuple[pd.DataFrame, _FrameSpec]:
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError(f"{name} must have unique column labels")
        frame = X.copy(deep=True)
        spec = _FrameSpec("dataframe")
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        frame = pd.DataFrame(X.copy())
        spec = _FrameSpec("ndarray")
    else:
        raise TypeError(f"{name} must be a pandas DataFrame or NumPy array")

    if frame.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column")
    return frame, spec


def _restore_frame(frame: pd.DataFrame, spec: _FrameSpec) -> Tabular:
    if spec.kind == "dataframe":
        return frame.copy(deep=True)
    return frame.to_numpy(copy=True)


def _as_target(y: Target, *, name: str) -> tuple[pd.Series, _TargetSpec]:
    if isinstance(y, pd.Series):
        series = y.copy(deep=True)
        spec = _TargetSpec("series", y.name, None)
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"{name} must have exactly one column")
        series = y.iloc[:, 0].copy(deep=True)
        spec = _TargetSpec("dataframe", y.columns[0], None)
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            series = pd.Series(y.copy())
            spec = _TargetSpec("ndarray_1d", None, y.dtype)
        elif y.ndim == 2 and y.shape[1] == 1:
            series = pd.Series(y[:, 0].copy())
            spec = _TargetSpec("ndarray_2d", None, y.dtype)
        else:
            raise ValueError(f"{name} must be one-dimensional or a single-column array")
    else:
        raise TypeError(f"{name} must be a pandas Series/DataFrame or NumPy array")

    if len(series) == 0:
        raise ValueError(f"{name} must contain at least one row")
    return series, spec


def _restore_target(
    series: pd.Series, spec: _TargetSpec, *, preserve_numpy_dtype: bool
) -> Target:
    if spec.kind == "series":
        result = series.copy(deep=True)
        result.name = spec.name
        return result
    if spec.kind == "dataframe":
        return series.copy(deep=True).to_frame(name=spec.name)

    values = series.to_numpy(copy=True)
    if preserve_numpy_dtype and spec.dtype is not None:
        values = values.astype(spec.dtype, copy=False)
    if spec.kind == "ndarray_2d":
        return values.reshape(-1, 1)
    return values


def _json_value(value: Any) -> str | int | float | bool | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)
    if isinstance(value, str):
        return value
    return str(value)


def _is_numeric_series(series: pd.Series) -> bool:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_complex_dtype(dtype):
        return False
    if pd.api.types.is_numeric_dtype(dtype):
        return True

    # A mixed NumPy array becomes an object DataFrame.  Recover numeric columns
    # without treating numeric-looking strings or booleans as measurements.
    non_missing = series[~series.isna()]
    if len(non_missing) == 0:
        return False
    return bool(
        non_missing.map(
            lambda value: (
                isinstance(value, numbers.Number)
                and not isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            )
        ).all()
    )


def _resolve_columns(
    frame: pd.DataFrame,
    columns: Sequence[Hashable] | None,
    *,
    numeric_only: bool,
    operation: str,
) -> list[Hashable]:
    if columns is None:
        selected = [
            column
            for column in frame.columns
            if not numeric_only or _is_numeric_series(frame[column])
        ]
    else:
        if isinstance(columns, (str, bytes)):
            requested: list[Hashable] = [columns]
        else:
            requested = list(columns)
        if not requested:
            raise ValueError(f"{operation} columns cannot be empty")
        if len(set(requested)) != len(requested):
            raise ValueError(f"{operation} columns cannot contain duplicates")
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            raise KeyError(f"unknown {operation} columns: {missing!r}")
        selected = requested

    if numeric_only:
        non_numeric = [
            column for column in selected if not _is_numeric_series(frame[column])
        ]
        if non_numeric:
            raise TypeError(
                f"{operation} only supports numeric, non-boolean columns; "
                f"received {non_numeric!r}"
            )
    return selected


def _numeric_values(series: pd.Series, *, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if np.isinf(values).any():
        raise ValueError(f"{name} cannot contain positive or negative infinity")
    return values


def _scale_details(values: np.ndarray) -> tuple[float, str, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)

    if count:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        iqr_scale = float((q75 - q25) / 1.349)
        if math.isfinite(iqr_scale) and iqr_scale > 0.0:
            return iqr_scale, "iqr", count

        median = float(np.median(finite))
        mad_scale = float(np.median(np.abs(finite - median)) * 1.482602218505602)
        if math.isfinite(mad_scale) and mad_scale > 0.0:
            return mad_scale, "mad", count

        with np.errstate(over="ignore", invalid="ignore"):
            std_scale = float(np.std(finite, ddof=1)) if count > 1 else 0.0
        if math.isfinite(std_scale) and std_scale > 0.0:
            return std_scale, "std", count

    # Constant and entirely missing training columns still receive a defined,
    # explicit perturbation unit rather than silently becoming no-ops.
    return 1.0, "unit_fallback", count


def robust_scale(values: pd.Series | np.ndarray) -> float:
    """Return a robust one-dimensional scale with deterministic fallbacks.

    The primary estimate is ``IQR / 1.349``.  A scaled MAD, sample standard
    deviation, and finally ``1.0`` are used when the preceding estimate is zero
    or undefined.  Missing values are ignored and infinities are rejected.
    """

    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, np.ndarray) and values.ndim == 1:
        series = pd.Series(values)
    else:
        raise TypeError("values must be a pandas Series or one-dimensional NumPy array")
    array = _numeric_values(series, name="values")
    return _scale_details(array)[0]


def robust_scales(
    X_train: Tabular, columns: Sequence[Hashable] | None = None
) -> pd.Series:
    """Compute per-column robust scales from a training fold only."""

    frame, _ = _as_frame(X_train, name="X_train")
    if len(frame) == 0:
        raise ValueError("X_train must contain at least one row")
    selected = _resolve_columns(
        frame,
        columns,
        numeric_only=True,
        operation="feature noise",
    )
    scales = [
        _scale_details(_numeric_values(frame[column], name=f"X_train[{column!r}]"))[0]
        for column in selected
    ]
    return pd.Series(scales, index=pd.Index(selected), dtype=float, name="robust_scale")


def _align_training_frame(
    frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    *,
    frame_spec: _FrameSpec,
    train_spec: _FrameSpec,
) -> pd.DataFrame:
    if frame.shape[1] != train_frame.shape[1]:
        raise ValueError(
            "X and X_train must contain the same number of feature columns"
        )

    if frame_spec.kind == "dataframe" and train_spec.kind == "dataframe":
        missing = [
            column for column in frame.columns if column not in train_frame.columns
        ]
        extra = [
            column for column in train_frame.columns if column not in frame.columns
        ]
        if missing or extra:
            raise ValueError("X and X_train must have the same feature column labels")
        return train_frame.loc[:, frame.columns].copy(deep=True)

    aligned = train_frame.copy(deep=True)
    aligned.columns = frame.columns
    return aligned


def inject_feature_noise(
    X: Tabular,
    level: float,
    *,
    X_train: Tabular,
    columns: Sequence[Hashable] | None = None,
    random_state: int = 0,
) -> StressResult[Tabular]:
    """Add Gaussian noise to numeric columns using training-fold scales.

    ``level`` is measured in robust standard-deviation units.  Existing missing
    values are preserved, categorical and boolean columns are untouched, and
    the returned container matches the input container kind.
    """

    noise_level = _validate_level(level, name="level")
    seed = _validate_seed(random_state)
    frame, frame_spec = _as_frame(X, name="X")
    train_frame, train_spec = _as_frame(X_train, name="X_train")
    if len(train_frame) == 0:
        raise ValueError("X_train must contain at least one row")
    train_frame = _align_training_frame(
        frame, train_frame, frame_spec=frame_spec, train_spec=train_spec
    )

    selected = _resolve_columns(
        train_frame,
        columns,
        numeric_only=True,
        operation="feature noise",
    )
    # The evaluation fold must also be numeric in every selected column.
    non_numeric_input = [
        column for column in selected if not _is_numeric_series(frame[column])
    ]
    if non_numeric_input:
        raise TypeError(
            "X columns selected for feature noise are not numeric: "
            f"{non_numeric_input!r}"
        )

    rng = np.random.default_rng(seed)
    output = frame.copy(deep=True)
    column_details: list[dict[str, Any]] = []
    eligible_total = 0
    changed_total = 0

    for column in selected:
        train_values = _numeric_values(train_frame[column], name=f"X_train[{column!r}]")
        values = _numeric_values(frame[column], name=f"X[{column!r}]")
        scale, method, training_count = _scale_details(train_values)
        eligible = np.isfinite(values)
        eligible_count = int(eligible.sum())
        changed_count = 0
        perturbed = values.copy()
        if noise_level > 0.0 and eligible_count:
            draws = rng.normal(loc=0.0, scale=noise_level * scale, size=eligible_count)
            before = perturbed[eligible].copy()
            perturbed[eligible] = before + draws
            changed_count = int(np.count_nonzero(perturbed[eligible] != before))

        # At level zero preserve the original dtype as well as all values.  A
        # numeric assignment would otherwise upcast integer extension columns
        # even though no perturbation was requested.
        if noise_level > 0.0:
            output[column] = perturbed
        eligible_total += eligible_count
        changed_total += changed_count
        column_details.append(
            {
                "column": _json_value(column),
                "scale": float(scale),
                "scale_method": method,
                "training_non_missing": training_count,
                "eligible_cells": eligible_count,
                "changed_cells": changed_count,
            }
        )

    metadata: dict[str, Any] = {
        "stressor": "feature_noise",
        "level": noise_level,
        "random_state": seed,
        "distribution": "normal",
        "scale_source": "training_fold",
        "training_rows": int(len(train_frame)),
        "input_rows": int(len(frame)),
        "numeric_columns": [_json_value(column) for column in selected],
        "column_details": column_details,
        "eligible_cells": eligible_total,
        "changed_cells": changed_total,
    }
    return StressResult(_restore_frame(output, frame_spec), metadata)


def _validate_no_missing_target(series: pd.Series, *, name: str) -> None:
    if bool(series.isna().any()):
        raise ValueError(f"{name} cannot contain missing target values")


def _binary_classes(series: pd.Series, *, name: str) -> list[Any]:
    _validate_no_missing_target(series, name=name)
    classes = list(pd.unique(series))
    if len(classes) != 2:
        raise ValueError(
            f"{name} must contain exactly two classes; found {len(classes)}"
        )
    return classes


def _class_count_records(
    series: pd.Series, classes: Sequence[Any]
) -> list[dict[str, Any]]:
    return [
        {
            "label": _json_value(label),
            "count": int((series == label).sum()),
        }
        for label in classes
    ]


def inject_label_noise(
    y: Target,
    level: float,
    *,
    task: str,
    y_train: Target | None = None,
    random_state: int = 0,
) -> StressResult[Target]:
    """Perturb binary labels or regression targets deterministically.

    For binary classification, ``level`` is the exact (up to nearest-integer
    rounding) fraction of rows whose labels are flipped.  For regression it is
    the Gaussian noise standard deviation in robust training-target units.
    ``y_train`` may be supplied when perturbing a fold other than the training
    fold; otherwise ``y`` itself is treated as the training fold.
    """

    task_name = _normalise_task(task)
    upper = 1.0 if task_name == "binary" else None
    noise_level = _validate_level(level, name="level", upper=upper)
    seed = _validate_seed(random_state)
    series, spec = _as_target(y, name="y")
    reference, _ = _as_target(y if y_train is None else y_train, name="y_train")
    rng = np.random.default_rng(seed)

    if task_name == "binary":
        classes = _binary_classes(reference, name="y_train")
        _validate_no_missing_target(series, name="y")
        unknown = [value for value in pd.unique(series) if value not in classes]
        if unknown:
            raise ValueError(f"y contains labels absent from y_train: {unknown!r}")

        count = int(math.floor(noise_level * len(series) + 0.5))
        selected = (
            np.sort(rng.choice(len(series), size=count, replace=False)).astype(int)
            if count
            else np.empty(0, dtype=int)
        )
        output = series.copy(deep=True)
        if count:
            flipped = [
                classes[1] if value == classes[0] else classes[0]
                for value in output.iloc[selected].tolist()
            ]
            output.iloc[selected] = flipped

        metadata: dict[str, Any] = {
            "stressor": "label_noise",
            "task": "binary",
            "level": noise_level,
            "random_state": seed,
            "mechanism": "binary_flip",
            "scale_source": None,
            "training_rows": int(len(reference)),
            "input_rows": int(len(series)),
            "classes": [_json_value(label) for label in classes],
            "class_counts_before": _class_count_records(series, classes),
            "class_counts_after": _class_count_records(output, classes),
            "changed_rows": count,
            "selected_positions": selected.tolist(),
        }
        return StressResult(
            _restore_target(output, spec, preserve_numpy_dtype=True), metadata
        )

    _validate_no_missing_target(series, name="y")
    _validate_no_missing_target(reference, name="y_train")
    values = _numeric_values(series, name="y")
    reference_values = _numeric_values(reference, name="y_train")
    scale, method, training_count = _scale_details(reference_values)
    output_values = values.copy()
    if noise_level > 0.0:
        output_values += rng.normal(
            loc=0.0, scale=noise_level * scale, size=len(output_values)
        )
    changed = int(np.count_nonzero(output_values != values))
    output = (
        series.copy(deep=True)
        if noise_level == 0.0
        else pd.Series(output_values, index=series.index, name=series.name)
    )

    metadata = {
        "stressor": "label_noise",
        "task": "regression",
        "level": noise_level,
        "random_state": seed,
        "mechanism": "gaussian_additive",
        "scale_source": "training_fold"
        if y_train is not None
        else "input_training_fold",
        "scale": float(scale),
        "scale_method": method,
        "training_non_missing": training_count,
        "training_rows": int(len(reference)),
        "input_rows": int(len(series)),
        "changed_rows": changed,
    }
    return StressResult(
        _restore_target(
            output,
            spec,
            preserve_numpy_dtype=noise_level == 0.0,
        ),
        metadata,
    )


def inject_missingness(
    X: Tabular,
    level: float,
    *,
    columns: Sequence[Hashable] | None = None,
    random_state: int = 0,
) -> StressResult[Tabular]:
    """Set an exact fraction of currently observed cells to missing.

    Sampling is uniform over eligible cells.  Existing missing values do not
    count toward ``level`` and entirely-null columns are always skipped.
    """

    missing_level = _validate_level(level, name="level", upper=1.0)
    seed = _validate_seed(random_state)
    frame, spec = _as_frame(X, name="X")
    selected = _resolve_columns(
        frame,
        columns,
        numeric_only=False,
        operation="missingness",
    )

    skipped = [column for column in selected if bool(frame[column].isna().all())]
    active = [column for column in selected if column not in skipped]
    eligible_mask = np.zeros(frame.shape, dtype=bool)
    for column in active:
        column_position = int(frame.columns.get_loc(column))
        eligible_mask[:, column_position] = ~frame[column].isna().to_numpy()

    eligible_positions = np.argwhere(eligible_mask)
    eligible_count = int(len(eligible_positions))
    missing_count = int(math.floor(missing_level * eligible_count + 0.5))
    rng = np.random.default_rng(seed)
    selected_positions = (
        np.sort(rng.choice(eligible_count, size=missing_count, replace=False))
        if missing_count
        else np.empty(0, dtype=int)
    )
    injection_mask = np.zeros(frame.shape, dtype=bool)
    if missing_count:
        cells = eligible_positions[selected_positions]
        injection_mask[cells[:, 0], cells[:, 1]] = True

    output = frame.mask(injection_mask)
    column_details: list[dict[str, Any]] = []
    for column in selected:
        column_position = int(frame.columns.get_loc(column))
        column_details.append(
            {
                "column": _json_value(column),
                "skipped_all_null": column in skipped,
                "eligible_cells": int(eligible_mask[:, column_position].sum()),
                "injected_missing_cells": int(injection_mask[:, column_position].sum()),
                "preexisting_missing_cells": int(frame[column].isna().sum()),
            }
        )

    injected_cells = [
        {
            "row_position": int(row_position),
            "column": _json_value(frame.columns[int(column_position)]),
        }
        for row_position, column_position in np.argwhere(injection_mask)
    ]
    metadata: dict[str, Any] = {
        "stressor": "missingness",
        "level": missing_level,
        "random_state": seed,
        "mechanism": "uniform_observed_cell_masking",
        "input_rows": int(len(frame)),
        "selected_columns": [_json_value(column) for column in selected],
        "skipped_all_null_columns": [_json_value(column) for column in skipped],
        "eligible_cells": eligible_count,
        "injected_missing_cells": missing_count,
        "column_details": column_details,
        "injected_cells": injected_cells,
    }
    return StressResult(_restore_frame(output, spec), metadata)


def _take_rows(value: Tabular | Target, positions: np.ndarray):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.iloc[positions].copy(deep=True)
    return np.take(value, positions, axis=0).copy()


def _validate_paired_indexes(X: Tabular, y: Target) -> None:
    if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
        if not X.index.equals(y.index):
            raise ValueError("X and y pandas indexes must match exactly")


def subsample_training(
    X: Tabular,
    y: Target,
    fraction: float,
    *,
    task: str,
    random_state: int = 0,
) -> StressResult[tuple[Tabular, Target]]:
    """Return a paired training subset, stratified for binary targets when feasible."""

    sample_fraction = _validate_level(
        fraction,
        name="fraction",
        lower=0.0,
        upper=1.0,
        include_lower=False,
    )
    task_name = _normalise_task(task)
    seed = _validate_seed(random_state)
    frame, _ = _as_frame(X, name="X")
    series, _ = _as_target(y, name="y")
    if len(frame) == 0:
        raise ValueError("X and y must contain at least one row")
    if len(frame) != len(series):
        raise ValueError("X and y must contain the same number of rows")
    _validate_paired_indexes(X, y)
    _validate_no_missing_target(series, name="y")

    total = len(frame)
    sample_size = max(1, min(total, int(math.floor(sample_fraction * total + 0.5))))
    rng = np.random.default_rng(seed)
    stratified = False
    stratification_reason: str | None = None
    classes: list[Any] | None = None

    if task_name == "binary":
        classes = _binary_classes(series, name="y")
    else:
        # Subsampling does not alter values, but rejecting non-numeric or
        # infinite regression targets here keeps the stressor contract honest
        # and prevents delayed estimator failures that look like audit errors.
        _numeric_values(series, name="y")

    if sample_size == total:
        positions = np.arange(total, dtype=int)
        stratified = task_name == "binary"
        stratification_reason = "full_training_fold"
    elif task_name == "binary" and sample_size >= 2 and classes is not None:
        first_positions = np.flatnonzero((series == classes[0]).to_numpy())
        second_positions = np.flatnonzero((series == classes[1]).to_numpy())

        # For two classes, clipping the nearest proportional allocation to this
        # interval guarantees capacity and at least one observation per class.
        lower_first = max(1, sample_size - len(second_positions))
        upper_first = min(len(first_positions), sample_size - 1)
        ideal_first = sample_size * len(first_positions) / total
        first_count = min(
            upper_first,
            max(lower_first, int(math.floor(ideal_first + 0.5))),
        )
        second_count = sample_size - first_count
        first_sample = rng.choice(first_positions, size=first_count, replace=False)
        second_sample = rng.choice(second_positions, size=second_count, replace=False)
        positions = np.sort(np.concatenate([first_sample, second_sample])).astype(int)
        stratified = True
        stratification_reason = "proportional_binary_allocation"
    else:
        positions = np.sort(rng.choice(total, size=sample_size, replace=False)).astype(
            int
        )
        if task_name == "binary":
            stratification_reason = "sample_too_small_for_both_classes"
        else:
            stratification_reason = "not_applicable_to_regression"

    X_sample = _take_rows(X, positions)
    y_sample = _take_rows(y, positions)
    selected_index = (
        [_json_value(value) for value in X.index[positions].tolist()]
        if isinstance(X, pd.DataFrame)
        else positions.tolist()
    )

    metadata: dict[str, Any] = {
        "stressor": "train_fraction",
        "task": task_name,
        "fraction": sample_fraction,
        "random_state": seed,
        "input_rows": int(total),
        "selected_rows": int(sample_size),
        "selected_positions": positions.tolist(),
        "selected_index": selected_index,
        "preserved_input_order": True,
        "stratified": stratified,
        "stratification_reason": stratification_reason,
    }
    if classes is not None:
        sampled_series, _ = _as_target(y_sample, name="sampled y")
        metadata.update(
            {
                "classes": [_json_value(label) for label in classes],
                "class_counts_before": _class_count_records(series, classes),
                "class_counts_after": _class_count_records(sampled_series, classes),
            }
        )

    return StressResult((X_sample, y_sample), metadata)


@dataclass(frozen=True)
class FeatureNoise:
    """Configuration wrapper for :func:`inject_feature_noise`."""

    level: float
    columns: tuple[Hashable, ...] | None = None

    def __post_init__(self) -> None:
        _validate_level(self.level, name="level")
        if self.columns is not None:
            normalised = (
                (self.columns,)
                if isinstance(self.columns, (str, bytes))
                else tuple(self.columns)
            )
            object.__setattr__(self, "columns", normalised)

    def apply(
        self,
        X: Tabular,
        *,
        X_train: Tabular,
        random_state: int = 0,
    ) -> StressResult[Tabular]:
        return inject_feature_noise(
            X,
            self.level,
            X_train=X_train,
            columns=self.columns,
            random_state=random_state,
        )


@dataclass(frozen=True)
class LabelNoise:
    """Configuration wrapper for :func:`inject_label_noise`."""

    level: float
    task: str

    def __post_init__(self) -> None:
        task = _normalise_task(self.task)
        _validate_level(
            self.level,
            name="level",
            upper=1.0 if task == "binary" else None,
        )
        object.__setattr__(self, "task", task)

    def apply(
        self,
        y: Target,
        *,
        y_train: Target | None = None,
        random_state: int = 0,
    ) -> StressResult[Target]:
        return inject_label_noise(
            y,
            self.level,
            task=self.task,
            y_train=y_train,
            random_state=random_state,
        )


@dataclass(frozen=True)
class Missingness:
    """Configuration wrapper for :func:`inject_missingness`."""

    level: float
    columns: tuple[Hashable, ...] | None = None

    def __post_init__(self) -> None:
        _validate_level(self.level, name="level", upper=1.0)
        if self.columns is not None:
            normalised = (
                (self.columns,)
                if isinstance(self.columns, (str, bytes))
                else tuple(self.columns)
            )
            object.__setattr__(self, "columns", normalised)

    def apply(self, X: Tabular, *, random_state: int = 0) -> StressResult[Tabular]:
        return inject_missingness(
            X,
            self.level,
            columns=self.columns,
            random_state=random_state,
        )


@dataclass(frozen=True)
class TrainFraction:
    """Configuration wrapper for :func:`subsample_training`."""

    fraction: float
    task: str

    def __post_init__(self) -> None:
        _validate_level(
            self.fraction,
            name="fraction",
            lower=0.0,
            upper=1.0,
            include_lower=False,
        )
        object.__setattr__(self, "task", _normalise_task(self.task))

    def apply(
        self,
        X: Tabular,
        y: Target,
        *,
        random_state: int = 0,
    ) -> StressResult[tuple[Tabular, Target]]:
        return subsample_training(
            X,
            y,
            self.fraction,
            task=self.task,
            random_state=random_state,
        )


__all__ = [
    "FeatureNoise",
    "LabelNoise",
    "Missingness",
    "StressResult",
    "TrainFraction",
    "inject_feature_noise",
    "inject_label_noise",
    "inject_missingness",
    "robust_scale",
    "robust_scales",
    "subsample_training",
]
