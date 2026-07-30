"""Dependency-light, self-contained HTML reporting."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

from .metrics import METRICS

if TYPE_CHECKING:
    from .results import AuditResult


def _number(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if not np.isfinite(numeric):
        return "n/a"
    if numeric == 0:
        return "0"
    if abs(numeric) >= 1_000 or abs(numeric) < 0.001:
        return f"{numeric:.3e}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _table(frame: pd.DataFrame, columns: Iterable[tuple[str, str]]) -> str:
    specs = list(columns)
    if frame.empty:
        return (
            '<p class="empty">No successful runs were available for this section.</p>'
        )
    header = "".join(f"<th>{escape(label)}</th>" for _, label in specs)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for key, _ in specs:
            value = row.get(key, "")
            display = (
                _number(value)
                if isinstance(value, (int, float, np.number))
                else escape(str(value))
            )
            cells.append(f"<td>{display}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _curve_svg(frame: pd.DataFrame, metric: str, experiment: str) -> str:
    data = frame[
        (frame["metric"] == metric)
        & (frame["experiment"] == experiment)
        & (frame["evaluation"] == "test")
    ].sort_values("level")
    if data.empty:
        return ""
    width, height = 560, 250
    left, right, top, bottom = 55, 18, 18, 42
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = data["level"].to_numpy(float)
    ys = data["median"].to_numpy(float)
    lows = data["mc_low"].to_numpy(float)
    highs = data["mc_high"].to_numpy(float)
    finite = np.concatenate(
        [ys[np.isfinite(ys)], lows[np.isfinite(lows)], highs[np.isfinite(highs)]]
    )
    if not len(finite):
        return ""
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(finite.min()), float(finite.max())
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    pad = (y_max - y_min) * 0.12 or max(abs(y_max) * 0.12, 0.1)
    y_min, y_max = y_min - pad, y_max + pad

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    line = " ".join(
        f"{px(x):.1f},{py(y):.1f}"
        for x, y in zip(xs, ys, strict=True)
        if np.isfinite(y)
    )
    upper = [(px(x), py(y)) for x, y in zip(xs, highs, strict=True) if np.isfinite(y)]
    lower = [(px(x), py(y)) for x, y in zip(xs, lows, strict=True) if np.isfinite(y)][
        ::-1
    ]
    band = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
    points = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5"><title>level {x:g}: {_number(y)}</title></circle>'
        for x, y in zip(xs, ys, strict=True)
        if np.isfinite(y)
    )
    y_ticks = "".join(
        f'<line x1="{left}" y1="{py(value):.1f}" x2="{width - right}" y2="{py(value):.1f}" class="grid"/>'
        f'<text x="{left - 8}" y="{py(value) + 4:.1f}" text-anchor="end">{_number(value)}</text>'
        for value in np.linspace(y_min, y_max, 4)
    )
    x_ticks = "".join(
        f'<text x="{px(value):.1f}" y="{height - 17}" text-anchor="middle">{value:g}</text>'
        for value in xs
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(experiment)} {escape(metric)} curve">'
        f'{y_ticks}<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>'
        f'<polygon points="{band}" class="band"/><polyline points="{line}" class="curve"/>{points}{x_ticks}'
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 2}" text-anchor="middle" class="axis-label">severity / fraction</text>'
        "</svg>"
    )


def render_html(result: "AuditResult") -> str:
    summary = result.summary_frame()
    gaps = result.generalization_summary()
    permutations = result.permutation_summary()
    stress = summary[
        (summary["experiment"] != "baseline") & (summary["evaluation"] == "test")
    ].copy()
    if not stress.empty:
        stress["metric"] = stress["metric"].map(lambda value: METRICS[value].label)
    gap_table = _table(
        gaps,
        [
            ("metric", "Metric"),
            ("n", "Runs"),
            ("median_gap", "Median gap"),
            ("mc_low", "MC low"),
            ("mc_high", "MC high"),
        ],
    )
    stress_table = _table(
        stress,
        [
            ("experiment", "Experiment"),
            ("level", "Level"),
            ("metric", "Metric"),
            ("median", "Median"),
            ("median_degradation", "Degradation"),
            ("mc_low", "MC low"),
            ("mc_high", "MC high"),
        ],
    )
    permutation_table = _table(
        permutations,
        [
            ("metric", "Metric"),
            ("n", "Null fits"),
            ("repeat_count", "Holdouts"),
            ("null_mean", "Null mean"),
            ("observed_mean", "Observed mean"),
            ("pooled_paired_exceedance_rate", "Paired exceedance rate"),
        ],
    )
    charts = []
    chart_data = result.summary_frame()
    for experiment in ("feature_noise", "label_noise", "missingness", "train_fraction"):
        for metric in result.config.metrics or (
            ("roc_auc", "log_loss", "balanced_accuracy")
            if result.config.task == "binary_classification"
            else ("rmse", "mae", "r2")
        ):
            svg = _curve_svg(chart_data, metric, experiment)
            if svg:
                charts.append(
                    f'<article class="chart"><div><span class="eyebrow">{escape(experiment.replace("_", " "))}</span>'
                    f"<h3>{escape(METRICS[metric].label)}</h3></div>{svg}</article>"
                )
    errors = ""
    if result.errors:
        items = "".join(
            f"<li><strong>{escape(str(item['experiment']))} · level {escape(str(item['level']))}</strong>"
            f"<span>{escape(str(item['error_type']))}: {escape(str(item['message']))}</span></li>"
            for item in result.errors
        )
        errors = f'<section><div class="section-head"><span>Diagnostics</span><h2>Incomplete scenarios</h2></div><ul class="errors">{items}</ul></section>'
    interval_percent = int(round(result.config.interval * 100))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StressFold audit</title>
<style>
:root{{--ink:#17201c;--muted:#64706a;--paper:#f5f3ed;--panel:#fffdfa;--rule:#d9ddd8;--green:#176b4d;--mint:#b9e2cf;--amber:#a75b16}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1160px;margin:auto;padding:58px 32px 90px}} header{{border-top:5px solid var(--ink);padding-top:28px;display:grid;grid-template-columns:1.5fr 1fr;gap:48px}}
h1{{font:600 clamp(42px,7vw,78px)/.95 Georgia,serif;letter-spacing:-.045em;margin:8px 0 22px}} h2{{font:500 31px/1.1 Georgia,serif;margin:6px 0}} h3{{margin:3px 0 0;font-size:18px}}
.lede{{font-size:18px;max-width:680px;color:#39433e}} .scope{{align-self:end;border-left:2px solid var(--green);padding:2px 0 2px 22px;color:var(--muted)}}
.eyebrow,.section-head>span{{color:var(--green);font-size:11px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule);margin:44px 0}} .stat{{background:var(--panel);padding:22px}} .stat b{{display:block;font:500 28px Georgia,serif}} .stat span{{color:var(--muted);font-size:12px}}
section{{margin-top:60px}} .section-head{{display:flex;align-items:end;justify-content:space-between;border-bottom:1px solid var(--rule);padding-bottom:14px;margin-bottom:20px}} .note{{color:var(--muted);max-width:700px}}
.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .chart{{background:var(--panel);border:1px solid var(--rule);padding:20px}} svg{{width:100%;height:auto;margin-top:10px}} svg text{{font-size:11px;fill:var(--muted)}} .grid{{stroke:#e4e5e0;stroke-width:1}} .axis{{stroke:#808a84;stroke-width:1}} .curve{{fill:none;stroke:var(--green);stroke-width:2.5}} .band{{fill:var(--mint);opacity:.48}} circle{{fill:var(--green)}} .axis-label{{font-size:10px}}
.table-wrap{{overflow:auto;border:1px solid var(--rule);background:var(--panel)}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}} th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid #e8e9e5;white-space:nowrap}} th{{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);background:#f8f7f2}} th:first-child,td:first-child{{text-align:left}}
.errors{{padding:0;list-style:none}} .errors li{{display:grid;grid-template-columns:240px 1fr;border-bottom:1px solid var(--rule);padding:12px 0}} .errors span{{color:var(--amber)}} details{{border-top:1px solid var(--rule);padding:16px 0}} code{{font:12px ui-monospace,SFMono-Regular,Consolas,monospace}} footer{{margin-top:70px;padding-top:18px;border-top:1px solid var(--rule);color:var(--muted);font-size:12px}} .empty{{color:var(--muted);font-style:italic}}
@media(max-width:760px){{header{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}.charts{{grid-template-columns:1fr}}main{{padding:35px 18px 70px}}}}
</style></head><body><main>
<header><div><span class="eyebrow">Generalization stress report</span><h1>StressFold</h1><p class="lede">Controlled perturbations, refit experiments, and null checks for one tabular learning procedure.</p></div><p class="scope">This audit estimates performance under the stated resampling and perturbation protocol. It does not prove the absence of overfitting, causal validity, or robustness to arbitrary future shift.</p></header>
<div class="stats"><div class="stat"><b>{result.n_samples:,}</b><span>rows</span></div><div class="stat"><b>{result.n_features:,}</b><span>features</span></div><div class="stat"><b>{result.config.repeats}</b><span>paired holdouts</span></div><div class="stat"><b>{len(result.variants):,}</b><span>captured variants</span></div></div>
<section><div class="section-head"><div><span>Generalization evidence</span><h2>Clean train-test gap</h2></div></div><p class="note">Positive gap means test performance was worse than training performance, after respecting each metric’s direction.</p>{gap_table}</section>
<section><div class="section-head"><div><span>Response paths</span><h2>Performance across severity</h2></div></div><p class="note">Lines show medians, and shaded envelopes span the central {interval_percent}% of repeated runs. These are Monte Carlo variability intervals, not classical confidence intervals.</p><div class="charts">{"".join(charts)}</div></section>
<section><div class="section-head"><div><span>Paired effects</span><h2>Stress-test summary</h2></div></div>{stress_table}</section>
<section><div class="section-head"><div><span>Falsification</span><h2>Permutation null</h2></div></div><p class="note">The paired exceedance rate is the descriptive fraction of shuffled-label fits that matched or exceeded their clean result on the same holdout. Repeated holdouts overlap, so this pooled rate is not a permutation p-value or an estimate of future error.</p>{permutation_table}</section>
{errors}
<section><div class="section-head"><div><span>Provenance</span><h2>Reproduction record</h2></div></div><details><summary>Protocol and data fingerprint</summary><p><code>{escape(result.data_fingerprint)}</code></p><p><code>{escape(result.estimator)}</code></p><p>{len(result.seeds):,} named seeds recorded in the JSON artifact.</p></details></section>
<footer>Generated by StressFold 0.2.0 · {escape(result.created_at)}</footer>
</main></body></html>"""
