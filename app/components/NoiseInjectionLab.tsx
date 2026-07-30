"use client";

import { useId, useState } from "react";

type ClassLabel = 0 | 1;

interface NoisePoint {
  x: number;
  y: number;
  label: ClassLabel;
  epsilonX: number;
  epsilonY: number;
}

const POINTS: NoisePoint[] = [
  { x: 0.2, y: 0.66, label: 0, epsilonX: 0.4, epsilonY: 0.2 },
  { x: 0.28, y: 0.69, label: 0, epsilonX: 0.5, epsilonY: 0.8 },
  { x: 0.35, y: 0.58, label: 0, epsilonX: -0.4, epsilonY: 0.9 },
  { x: 0.42, y: 0.62, label: 0, epsilonX: 0.7, epsilonY: 0.4 },
  { x: 0.48, y: 0.52, label: 0, epsilonX: -0.6, epsilonY: 0.2 },
  { x: 0.56, y: 0.5, label: 0, epsilonX: 0.2, epsilonY: 0.7 },
  { x: 0.64, y: 0.42, label: 0, epsilonX: 0.8, epsilonY: 0.2 },
  { x: 0.7, y: 0.38, label: 0, epsilonX: -0.2, epsilonY: 0.6 },
  { x: 0.78, y: 0.28, label: 0, epsilonX: 0.4, epsilonY: 0.9 },
  { x: 0.82, y: 0.25, label: 0, epsilonX: -0.7, epsilonY: 0.4 },
  { x: 0.18, y: 0.9, label: 1, epsilonX: -0.5, epsilonY: -0.8 },
  { x: 0.26, y: 0.86, label: 1, epsilonX: 0.2, epsilonY: -0.9 },
  { x: 0.34, y: 0.78, label: 1, epsilonX: -0.8, epsilonY: -0.1 },
  { x: 0.4, y: 0.73, label: 1, epsilonX: -0.3, epsilonY: -0.7 },
  { x: 0.48, y: 0.68, label: 1, epsilonX: 0.5, epsilonY: -0.6 },
  { x: 0.56, y: 0.62, label: 1, epsilonX: -0.5, epsilonY: -0.6 },
  { x: 0.62, y: 0.58, label: 1, epsilonX: 0.1, epsilonY: -0.7 },
  { x: 0.7, y: 0.51, label: 1, epsilonX: -0.7, epsilonY: -0.3 },
  { x: 0.76, y: 0.47, label: 1, epsilonX: 0.6, epsilonY: -0.9 },
  { x: 0.82, y: 0.41, label: 1, epsilonX: -0.4, epsilonY: -0.4 },
];

const NOISE_SCALE = 0.18;
const VIEWBOX = { width: 720, height: 390, left: 62, right: 28, top: 30, bottom: 54 };

function prediction(x: number, y: number): ClassLabel {
  return y + 0.75 * x >= 0.98 ? 1 : 0;
}

function toPlotX(value: number) {
  return VIEWBOX.left + value * (VIEWBOX.width - VIEWBOX.left - VIEWBOX.right);
}

function toPlotY(value: number) {
  return VIEWBOX.top + (1 - value) * (VIEWBOX.height - VIEWBOX.top - VIEWBOX.bottom);
}

export function NoiseInjectionLab() {
  const [lambda, setLambda] = useState(0.3);
  const sliderId = useId();
  const titleId = useId();
  const descriptionId = useId();

  const displacedPoints = POINTS.map((point) => {
    const x = point.x + lambda * NOISE_SCALE * point.epsilonX;
    const y = point.y + lambda * NOISE_SCALE * point.epsilonY;
    return {
      ...point,
      displacedX: x,
      displacedY: y,
      predictionChanged: prediction(point.x, point.y) !== prediction(x, y),
    };
  });
  const changedCount = displacedPoints.filter((point) => point.predictionChanged).length;
  const meanDisplacement =
    displacedPoints.reduce(
      (sum, point) =>
        sum + Math.hypot(point.displacedX - point.x, point.displacedY - point.y),
      0,
    ) / displacedPoints.length;
  const boundaryStart = { x: 0.04, y: 0.98 - 0.75 * 0.04 };
  const boundaryEnd = { x: 0.96, y: 0.98 - 0.75 * 0.96 };

  return (
    <section className="sf-noise-lab" aria-labelledby={`${titleId}-heading`}>
      <header className="sf-noise-lab__header">
        <div>
          <p className="sf-noise-lab__eyebrow">Measurement-noise injection</p>
          <h3 id={`${titleId}-heading`}>Move the measurements, not the answers</h3>
        </div>
        <p>
          Each point receives one fixed noise direction. The slider only scales that same direction,
          while the class label and fitted decision boundary stay fixed.
        </p>
      </header>

      <div className="sf-noise-lab__control">
        <label htmlFor={sliderId}>
          Noise multiplier
          <output htmlFor={sliderId}>&lambda; = {lambda.toFixed(2)}</output>
        </label>
        <input
          id={sliderId}
          type="range"
          min="0"
          max="0.8"
          step="0.01"
          value={lambda}
          aria-valuetext={`${lambda.toFixed(2)} times the fixed noise scale`}
          onChange={(event) => setLambda(Number(event.target.value))}
        />
        <div className="sf-noise-lab__range-labels" aria-hidden="true">
          <span>0 / clean</span>
          <span>0.8 / stronger noise</span>
        </div>
      </div>

      <div className="sf-noise-lab__formula">
        <code>
          x<sub>i</sub>(&lambda;) = x<sub>i</sub> + &lambda; s &epsilon;<sub>i</sub>
        </code>
        <span>
          Here s = {NOISE_SCALE.toFixed(2)} is the feature scale and each &epsilon;<sub>i</sub> is fixed
          for the demonstration.
        </span>
      </div>

      <svg
        className="sf-noise-lab__plot"
        viewBox={`0 0 ${VIEWBOX.width} ${VIEWBOX.height}`}
        role="img"
        aria-labelledby={`${titleId} ${descriptionId}`}
      >
        <title id={titleId}>Effect of measurement noise on a fixed linear classifier</title>
        <desc id={descriptionId}>
          Twenty labelled points move away from their faint clean positions as lambda increases. At
          lambda {lambda.toFixed(2)}, {changedCount} of 20 fixed-model predictions differ from their
          clean predictions. Labels do not change.
        </desc>

        <line
          className="sf-noise-lab__axis"
          x1={VIEWBOX.left}
          x2={VIEWBOX.left}
          y1={VIEWBOX.top}
          y2={VIEWBOX.height - VIEWBOX.bottom}
          aria-hidden="true"
        />
        <line
          className="sf-noise-lab__axis"
          x1={VIEWBOX.left}
          x2={VIEWBOX.width - VIEWBOX.right}
          y1={VIEWBOX.height - VIEWBOX.bottom}
          y2={VIEWBOX.height - VIEWBOX.bottom}
          aria-hidden="true"
        />
        <text
          className="sf-noise-lab__axis-label"
          x={(VIEWBOX.left + VIEWBOX.width - VIEWBOX.right) / 2}
          y={VIEWBOX.height - 12}
          textAnchor="middle"
          aria-hidden="true"
        >
          feature x1
        </text>
        <text
          className="sf-noise-lab__axis-label"
          x="17"
          y={(VIEWBOX.top + VIEWBOX.height - VIEWBOX.bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 17 ${(VIEWBOX.top + VIEWBOX.height - VIEWBOX.bottom) / 2})`}
          aria-hidden="true"
        >
          feature x2
        </text>

        <line
          className="sf-noise-lab__boundary"
          x1={toPlotX(boundaryStart.x)}
          y1={toPlotY(boundaryStart.y)}
          x2={toPlotX(boundaryEnd.x)}
          y2={toPlotY(boundaryEnd.y)}
          aria-hidden="true"
        />
        <text
          className="sf-noise-lab__boundary-label"
          x={toPlotX(0.53)}
          y={toPlotY(0.98 - 0.75 * 0.53) - 10}
          textAnchor="middle"
          aria-hidden="true"
        >
          fixed rule: x2 + 0.75x1 = 0.98
        </text>

        {displacedPoints.map((point, index) => (
          <g key={index}>
            <line
              className="sf-noise-lab__displacement"
              x1={toPlotX(point.x)}
              y1={toPlotY(point.y)}
              x2={toPlotX(point.displacedX)}
              y2={toPlotY(point.displacedY)}
              aria-hidden="true"
            />

            {point.label === 0 ? (
              <circle
                className="sf-noise-lab__ghost sf-noise-lab__point--zero"
                cx={toPlotX(point.x)}
                cy={toPlotY(point.y)}
                r="6.5"
                aria-hidden="true"
              />
            ) : (
              <rect
                className="sf-noise-lab__ghost sf-noise-lab__point--one"
                x={toPlotX(point.x) - 5}
                y={toPlotY(point.y) - 5}
                width="10"
                height="10"
                transform={`rotate(45 ${toPlotX(point.x)} ${toPlotY(point.y)})`}
                aria-hidden="true"
              />
            )}

            {point.predictionChanged && (
              <circle
                className="sf-noise-lab__changed-ring"
                cx={toPlotX(point.displacedX)}
                cy={toPlotY(point.displacedY)}
                r="9"
                aria-hidden="true"
              />
            )}

            {point.label === 0 ? (
              <circle
                className="sf-noise-lab__point sf-noise-lab__point--zero"
                cx={toPlotX(point.displacedX)}
                cy={toPlotY(point.displacedY)}
                r="5"
                aria-hidden="true"
              />
            ) : (
              <rect
                className="sf-noise-lab__point sf-noise-lab__point--one"
                x={toPlotX(point.displacedX) - 4}
                y={toPlotY(point.displacedY) - 4}
                width="8"
                height="8"
                transform={`rotate(45 ${toPlotX(point.displacedX)} ${toPlotY(point.displacedY)})`}
                aria-hidden="true"
              />
            )}
          </g>
        ))}
      </svg>

      <div className="sf-noise-lab__legend" aria-label="Plot legend">
        <span>
          <i className="sf-noise-lab__legend-mark sf-noise-lab__point--zero" aria-hidden="true" />
          Label 0
        </span>
        <span>
          <i
            className="sf-noise-lab__legend-mark sf-noise-lab__legend-mark--diamond sf-noise-lab__point--one"
            aria-hidden="true"
          />
          Label 1
        </span>
        <span>
          <i className="sf-noise-lab__legend-ghost" aria-hidden="true" />
          Clean position
        </span>
        <span>
          <i className="sf-noise-lab__legend-ring" aria-hidden="true" />
          Prediction changed
        </span>
      </div>

      <div className="sf-noise-lab__readout" role="status" aria-live="polite">
        <span>
          Predictions changed <strong>{changedCount} / {POINTS.length}</strong>
        </span>
        <span>
          Mean displacement <strong>{meanDisplacement.toFixed(3)}</strong>
        </span>
      </div>

      <p className="sf-noise-lab__note">
        This isolates sensitivity to the chosen measurement error. It does not establish whether the
        model generalizes better or worse.
      </p>
    </section>
  );
}
