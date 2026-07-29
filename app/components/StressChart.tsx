"use client";

import { useId, useState } from "react";
import type { StressCurve } from "../lib/analysis";

interface StressChartProps {
  curves: StressCurve[];
}

type CurvePoint = StressCurve["points"][number];

interface Selection {
  signature: string;
  curveId: StressCurve["id"];
  pointIndex: number;
}

const OPERATOR_COPY: Record<
  StressCurve["id"],
  {
    question: string;
    method: string;
    axis: string;
    caveat: string;
  }
> = {
  "feature-noise": {
    question: "What if measurements become less precise?",
    method: "Gaussian noise is added to each numeric evaluation input using its training-fold spread. The fitted model is held fixed.",
    axis: "Added noise per feature",
    caveat: "This measures prediction robustness, not overfitting by itself.",
  },
  "label-noise": {
    question: "What if some training labels are wrong?",
    method: "Binary labels flip at the chosen rate. Numeric targets receive scaled Gaussian noise. The model is refitted.",
    axis: "Training labels corrupted",
    caveat: "A steep fall suggests a fragile fitting process; it is not an overfitting verdict on its own.",
  },
  missingness: {
    question: "What if evaluation inputs arrive incomplete?",
    method: "A share of evaluation cells is masked, then filled with the training-fold median. The fitted model is held fixed.",
    axis: "Evaluation cells masked",
    caveat: "This measures tolerance to missing inputs, not overfitting by itself.",
  },
  "train-size": {
    question: "How dependent is the fit on having all training rows?",
    method: "A share of training rows is removed. Preprocessing and the model are refitted on the remaining rows.",
    axis: "Training rows removed",
    caveat: "A steep fall indicates sensitivity to data volume. Read it alongside the clean gap between training and audit loss.",
  },
};

const VIEWBOX = { width: 760, height: 350, left: 76, right: 28, top: 28, bottom: 66 };

export function StressChart({ curves }: StressChartProps) {
  const titleId = useId();
  const descriptionId = useId();
  const [selection, setSelection] = useState<Selection | null>(null);
  const signature = curves
    .map((curve) =>
      `${curve.id}:${curve.points
        .map((point) => `${point.level},${point.median},${point.low},${point.high}`)
        .join(";")}`,
    )
    .join("|");

  if (curves.length === 0) {
    return (
      <div className="chart-shell">
        <p className="chart-note" role="status" style={{ paddingTop: 14 }}>
          No stress-response values were returned for this audit.
        </p>
      </div>
    );
  }

  const requestedCurve =
    selection?.signature === signature
      ? curves.find((curve) => curve.id === selection.curveId)
      : undefined;
  const activeCurve = requestedCurve ?? curves[0];
  const points = activeCurve.points
    .filter(
      (point) =>
        Number.isFinite(point.level) &&
        Number.isFinite(point.median) &&
        Number.isFinite(point.low) &&
        Number.isFinite(point.high),
    )
    .sort((left, right) => left.level - right.level);

  if (points.length === 0) {
    return (
      <div className="chart-shell">
        <OperatorPicker
          curves={curves}
          activeId={activeCurve.id}
          onSelect={(curve) =>
            setSelection({ signature, curveId: curve.id, pointIndex: Math.max(0, curve.points.length - 1) })
          }
        />
        <p className="chart-note" role="status" style={{ paddingTop: 14 }}>
          {activeCurve.shortLabel} did not return finite values for this audit.
        </p>
      </div>
    );
  }

  const requestedIndex = selection?.signature === signature ? selection.pointIndex : points.length - 1;
  const selectedIndex = Math.min(Math.max(0, requestedIndex), points.length - 1);
  const selectedPoint = points[selectedIndex];
  const copy = OPERATOR_COPY[activeCurve.id];
  const geometry = chartGeometry(points);
  const bandPath = [
    ...points.map((point) => `${geometry.x(point.level)},${geometry.y(point.high)}`),
    ...[...points].reverse().map((point) => `${geometry.x(point.level)},${geometry.y(point.low)}`),
  ].join(" ");
  const linePath = points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${geometry.x(point.level)} ${geometry.y(point.median)}`)
    .join(" ");

  const chooseCurve = (curve: StressCurve) => {
    const validPointCount = curve.points.filter(
      (point) =>
        Number.isFinite(point.level) &&
        Number.isFinite(point.median) &&
        Number.isFinite(point.low) &&
        Number.isFinite(point.high),
    ).length;
    setSelection({ signature, curveId: curve.id, pointIndex: Math.max(0, validPointCount - 1) });
  };

  const choosePoint = (pointIndex: number) => {
    setSelection({ signature, curveId: activeCurve.id, pointIndex });
  };

  return (
    <div className="chart-shell">
      <OperatorPicker curves={curves} activeId={activeCurve.id} onSelect={chooseCurve} />

      <div className="chart-visual" style={{ padding: "16px 14px 4px" }}>
        <div style={{ margin: "0 0 10px" }}>
          <strong style={{ display: "block", fontSize: 13, lineHeight: 1.35 }}>{copy.question}</strong>
          <span style={{ color: "var(--muted)", fontSize: 10, lineHeight: 1.45 }}>{copy.method}</span>
        </div>

        <svg
          viewBox={`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`}
          role="img"
          aria-labelledby={`${titleId} ${descriptionId}`}
          style={{ display: "block", width: "100%", height: "auto", overflow: "visible" }}
        >
          <title id={titleId}>{activeCurve.label} retained-skill response</title>
          <desc id={descriptionId}>
            Selectable median retained-skill points with an empirical fifth-to-ninety-fifth-percentile
            band across repeated paired splits. The horizontal axis uses the operator’s actual severity.
          </desc>

          {geometry.yTicks.map((tick) => (
            <g key={`y-${tick}`} aria-hidden="true">
              <line
                x1={VIEWBOX.left}
                x2={VIEWBOX.width - VIEWBOX.right}
                y1={geometry.y(tick)}
                y2={geometry.y(tick)}
                stroke={approximately(tick, 1) ? "#8c8d87" : "#dedbd0"}
                strokeDasharray={approximately(tick, 1) ? "4 5" : undefined}
                vectorEffect="non-scaling-stroke"
              />
              <text
                x={VIEWBOX.left - 12}
                y={geometry.y(tick) + 4}
                textAnchor="end"
                fill="#6f716b"
                fontSize="14"
                fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              >
                {formatRetained(tick)}
              </text>
            </g>
          ))}

          {points.map((point) => (
            <line
              key={`x-${point.level}`}
              x1={geometry.x(point.level)}
              x2={geometry.x(point.level)}
              y1={VIEWBOX.top}
              y2={VIEWBOX.height - VIEWBOX.bottom}
              stroke="#ebe8de"
              vectorEffect="non-scaling-stroke"
              aria-hidden="true"
            />
          ))}

          <line
            x1={VIEWBOX.left}
            x2={VIEWBOX.left}
            y1={VIEWBOX.top}
            y2={VIEWBOX.height - VIEWBOX.bottom}
            stroke="#8c8d87"
            vectorEffect="non-scaling-stroke"
            aria-hidden="true"
          />
          <line
            x1={VIEWBOX.left}
            x2={VIEWBOX.width - VIEWBOX.right}
            y1={VIEWBOX.height - VIEWBOX.bottom}
            y2={VIEWBOX.height - VIEWBOX.bottom}
            stroke="#8c8d87"
            vectorEffect="non-scaling-stroke"
            aria-hidden="true"
          />

          <polygon points={bandPath} fill={activeCurve.stroke} opacity="0.13" aria-hidden="true" />
          <path
            d={linePath}
            fill="none"
            stroke={activeCurve.stroke}
            strokeWidth="2.6"
            strokeDasharray={activeCurve.dash.length ? activeCurve.dash.join(" ") : undefined}
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
            aria-hidden="true"
          />

          <line
            x1={geometry.x(selectedPoint.level)}
            x2={geometry.x(selectedPoint.level)}
            y1={geometry.y(selectedPoint.low)}
            y2={geometry.y(selectedPoint.high)}
            stroke={activeCurve.stroke}
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
            aria-hidden="true"
          />

          {points.map((point, pointIndex) => {
            const selected = pointIndex === selectedIndex;
            const label = `${formatSeverity(activeCurve.id, point.level)}: median retained skill ${formatRetained(point.median)}, fifth-to-ninety-fifth percentile ${formatRetained(point.low)} to ${formatRetained(point.high)}`;
            return (
              <g
                key={`${activeCurve.id}-${point.level}`}
                role="button"
                tabIndex={0}
                aria-label={label}
                aria-pressed={selected}
                onClick={() => choosePoint(pointIndex)}
                onFocus={() => choosePoint(pointIndex)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    choosePoint(pointIndex);
                  }
                }}
                style={{ cursor: "pointer" }}
              >
                <circle
                  cx={geometry.x(point.level)}
                  cy={geometry.y(point.median)}
                  r="14"
                  fill="transparent"
                />
                {selected && (
                  <circle
                    cx={geometry.x(point.level)}
                    cy={geometry.y(point.median)}
                    r="8"
                    fill="#fbfaf5"
                    stroke={activeCurve.stroke}
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                <circle
                  cx={geometry.x(point.level)}
                  cy={geometry.y(point.median)}
                  r={selected ? 3.7 : 3.2}
                  fill={selected ? activeCurve.stroke : "#fbfaf5"}
                  stroke={activeCurve.stroke}
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              </g>
            );
          })}

          {points.map((point) => (
            <text
              key={`label-${point.level}`}
              x={geometry.x(point.level)}
              y={VIEWBOX.height - VIEWBOX.bottom + 22}
              textAnchor="middle"
              fill="#6f716b"
              fontSize="14"
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              aria-hidden="true"
            >
              {formatSeverity(activeCurve.id, point.level, true)}
            </text>
          ))}

          <text
            x={(VIEWBOX.left + VIEWBOX.width - VIEWBOX.right) / 2}
            y={VIEWBOX.height - 12}
            textAnchor="middle"
            fill="#6f716b"
            fontSize="12"
            fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
            letterSpacing="0.04em"
            aria-hidden="true"
          >
            {copy.axis.toUpperCase()}
          </text>
          <text
            x="15"
            y={(VIEWBOX.top + VIEWBOX.height - VIEWBOX.bottom) / 2}
            textAnchor="middle"
            fill="#6f716b"
            fontSize="12"
            fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
            letterSpacing="0.04em"
            transform={`rotate(-90 15 ${(VIEWBOX.top + VIEWBOX.height - VIEWBOX.bottom) / 2})`}
            aria-hidden="true"
          >
            PERFORMANCE RETAINED
          </text>
        </svg>
      </div>

      <div
        className="chart-reading"
        role="status"
        aria-live="polite"
        style={{
          margin: "0 14px 14px",
          padding: "12px 14px",
          borderLeft: `3px solid ${activeCurve.stroke}`,
          background: "var(--card-bright)",
        }}
      >
        <span
          style={{
            display: "block",
            marginBottom: 5,
            color: "var(--muted)",
            fontFamily: "var(--font-geist-mono), monospace",
            fontSize: 9,
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          Reading / {formatSeverity(activeCurve.id, selectedPoint.level)}
        </span>
        <strong style={{ display: "block", fontSize: 13, lineHeight: 1.4 }}>
          {plainReading(selectedPoint)}
        </strong>
        <p style={{ margin: "5px 0 0", color: "var(--muted)", fontSize: 10, lineHeight: 1.5 }}>
          {selectedPoint.level === 0
            ? "The zero-severity point is the clean reference and is fixed at 100%."
            : `Across repeated paired splits, the middle 90% of observed outcomes ran from ${formatRetained(selectedPoint.low)} to ${formatRetained(selectedPoint.high)}.`}{" "}
          {copy.caveat}
        </p>
      </div>

      <div className="summary-table-wrap chart-data" style={{ margin: "0 14px 14px" }}>
        <table className="summary-table">
          <caption
            style={{
              padding: "10px 12px",
              color: "var(--ink-soft)",
              fontSize: 10,
              fontWeight: 600,
              textAlign: "left",
            }}
          >
            Exact values for {activeCurve.shortLabel}
          </caption>
          <thead>
            <tr>
              <th scope="col">Severity</th>
              <th scope="col">Median retained</th>
              <th scope="col">5th to 95th percentile</th>
              <th scope="col">Compared with clean</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, pointIndex) => {
              const selected = pointIndex === selectedIndex;
              return (
                <tr key={`row-${point.level}`} style={{ background: selected ? "var(--teal-open)" : undefined }}>
                  <td>
                    <button
                      type="button"
                      aria-pressed={selected}
                      onClick={() => choosePoint(pointIndex)}
                      style={{
                        border: 0,
                        padding: 0,
                        background: "transparent",
                        color: selected ? "var(--teal-dark)" : "inherit",
                        font: "inherit",
                        fontWeight: selected ? 700 : 500,
                        textDecoration: "underline",
                        textUnderlineOffset: 3,
                        cursor: "pointer",
                      }}
                    >
                      {formatSeverity(activeCurve.id, point.level)}
                    </button>
                  </td>
                  <td>{formatRetained(point.median)}</td>
                  <td>{formatRetained(point.low)} to {formatRetained(point.high)}</td>
                  <td>{comparisonLabel(point.median)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="chart-note">
        Performance retained compares stressed loss with clean loss. The scale is normalized by the clean model&apos;s advantage over a constant predictor, with a 1% floor when that advantage is very small. At 100%, stressed and clean loss match. Values below 0% or above 100% mean the loss changed by more than the normalization margin. The shaded band is the empirical 5th to 95th percentile across repeated paired splits. It is not generated extra data.
      </p>
    </div>
  );
}

function OperatorPicker({
  curves,
  activeId,
  onSelect,
}: {
  curves: StressCurve[];
  activeId: StressCurve["id"];
  onSelect: (curve: StressCurve) => void;
}) {
  return (
    <div
      className="chart-legend chart-operator-picker"
      role="group"
      aria-label="Choose a stress operator"
      style={{ borderTop: 0, borderBottom: "1px solid var(--line)" }}
    >
      {curves.map((curve) => {
        const active = curve.id === activeId;
        return (
          <button
            className="legend-item chart-operator-button"
            key={curve.id}
            type="button"
            aria-pressed={active}
            onClick={() => onSelect(curve)}
            style={{
              minHeight: 32,
              padding: "6px 9px",
              border: `1px solid ${active ? curve.stroke : "var(--line)"}`,
              background: active ? "var(--card-bright)" : "transparent",
              color: active ? "var(--ink)" : "var(--muted)",
              fontWeight: active ? 700 : 500,
              cursor: "pointer",
            }}
          >
            <span className="legend-line" style={{ backgroundColor: curve.stroke }} aria-hidden="true" />
            <span>{curve.shortLabel}</span>
          </button>
        );
      })}
    </div>
  );
}

function chartGeometry(points: CurvePoint[]) {
  const minimumLevel = Math.min(...points.map((point) => point.level));
  const maximumLevel = Math.max(...points.map((point) => point.level));
  const levelSpan = Math.max(1e-9, maximumLevel - minimumLevel);
  const observedMinimum = Math.min(...points.flatMap((point) => [point.low, point.median, point.high]));
  const observedMaximum = Math.max(...points.flatMap((point) => [point.low, point.median, point.high]));
  const paddedMinimum = Math.min(-0.05, observedMinimum - 0.05);
  const paddedMaximum = Math.max(1.05, observedMaximum + 0.05);
  const step = niceStep((paddedMaximum - paddedMinimum) / 5);
  const minimumSkill = Math.floor(paddedMinimum / step) * step;
  const maximumSkill = Math.ceil(paddedMaximum / step) * step;
  const yTicks: number[] = [];
  for (let tick = minimumSkill; tick <= maximumSkill + step / 2; tick += step) {
    yTicks.push(Number(tick.toFixed(8)));
  }

  const plotWidth = VIEWBOX.width - VIEWBOX.left - VIEWBOX.right;
  const plotHeight = VIEWBOX.height - VIEWBOX.top - VIEWBOX.bottom;
  return {
    yTicks,
    x: (level: number) => VIEWBOX.left + ((level - minimumLevel) / levelSpan) * plotWidth,
    y: (skill: number) =>
      VIEWBOX.top + ((maximumSkill - skill) / Math.max(1e-9, maximumSkill - minimumSkill)) * plotHeight,
  };
}

function niceStep(value: number) {
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(value, 1e-9)));
  const normalized = value / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

function formatSeverity(id: StressCurve["id"], level: number, compact = false) {
  if (id === "feature-noise") {
    const digits = approximately(level, 0) ? 0 : 2;
    return `${level.toFixed(digits)}${compact ? "×" : " × typical training spread"}`;
  }
  return `${Math.round(level * 100)}%`;
}

function formatRetained(value: number) {
  return `${Math.round(value * 100)}%`;
}

function comparisonLabel(value: number) {
  if (approximately(value, 1)) return "Clean reference";
  if (value > 1) return `${Math.round((value - 1) * 100)}% above clean`;
  if (value >= 0) return `${Math.round((1 - value) * 100)}% of normalized margin lost`;
  return "More than one normalization unit lost";
}

function plainReading(point: CurvePoint) {
  if (point.level === 0) return "This is the clean, unstressed reference: performance retained is fixed at 100%.";
  if (point.median >= 0.9) return `The typical run retained ${formatRetained(point.median)} of the normalized clean margin. Degradation is small at this severity.`;
  if (point.median >= 0.5) return `The typical run retained ${formatRetained(point.median)} of the normalized clean margin. The loss is noticeable, but more than half remains.`;
  if (point.median >= 0) return `The typical run retained only ${formatRetained(point.median)} of the normalized clean margin. This is a material failure boundary.`;
  return "The typical stressed run lost more than the full normalization margin at this severity.";
}

function approximately(left: number, right: number) {
  return Math.abs(left - right) < 1e-8;
}
