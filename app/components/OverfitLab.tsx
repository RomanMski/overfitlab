"use client";

import { useMemo, useState } from "react";

import {
  polynomialFit,
  rootMeanSquaredError,
  sampleOverfitData,
  trueFunction,
} from "../lib/quant";

const WIDTH = 720;
const HEIGHT = 380;
const PAD = { left: 46, right: 18, top: 20, bottom: 40 };
const X_MIN = -3.2;
const X_MAX = 3.2;
const Y_MIN = -2.1;
const Y_MAX = 2.1;

const px = (x: number) =>
  PAD.left + ((x - X_MIN) / (X_MAX - X_MIN)) * (WIDTH - PAD.left - PAD.right);
const py = (y: number) =>
  PAD.top + ((Y_MAX - y) / (Y_MAX - Y_MIN)) * (HEIGHT - PAD.top - PAD.bottom);

function path(points: [number, number][]): string {
  return points
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${px(x).toFixed(1)},${py(Math.max(Math.min(y, Y_MAX + 2), Y_MIN - 2)).toFixed(1)}`)
    .join(" ");
}

export function OverfitLab() {
  const [degree, setDegree] = useState(3);
  const [seed, setSeed] = useState(7);
  const data = useMemo(() => sampleOverfitData(seed), [seed]);

  const view = useMemo(() => {
    const fit = polynomialFit(data.trainX, data.trainY, degree);
    const grid: [number, number][] = [];
    for (let i = 0; i <= 400; i += 1) {
      const x = X_MIN + ((X_MAX - X_MIN) * i) / 400;
      grid.push([x, fit(x)]);
    }
    const truth: [number, number][] = grid.map(([x]) => [x, trueFunction(x)]);
    return {
      curve: path(grid),
      truth: path(truth),
      trainError: rootMeanSquaredError(data.trainX, data.trainY, fit),
      testError: rootMeanSquaredError(data.testX, data.testY, fit),
    };
  }, [data, degree]);

  const verdict =
    degree <= 2
      ? "Too stiff. The line cannot bend enough to follow the real shape, so it is wrong nearly everywhere."
      : degree <= 6
        ? "About right. The curve follows the underlying shape and ignores the scatter around it."
        : "Overfitting. The curve is now bending to reach individual dots, and it is wrong in the gaps between them.";

  return (
    <div className="ol">
      <style>{`
        .ol{display:grid;gap:18px}
        .ol-figure{background:var(--panel,#fffdfa);border:1px solid var(--rule,#d9ddd8);border-radius:14px;padding:8px}
        .ol-figure svg{display:block;width:100%;height:auto}
        .ol-controls{display:grid;gap:14px}
        .ol-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
        .ol-row label{min-width:150px;font-weight:600}
        .ol-row input[type=range]{flex:1;min-width:200px}
        .ol-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
        .ol-stat{border:1px solid var(--rule,#d9ddd8);border-radius:12px;padding:12px 14px;background:var(--panel,#fffdfa)}
        .ol-stat span{display:block;font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#64706a)}
        .ol-stat strong{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums;margin-top:4px}
        .ol-verdict{border-left:3px solid var(--green,#176b4d);padding:2px 0 2px 14px}
        .ol-dot{fill:#2f5d7e}
        .ol-fit{fill:none;stroke:#a75b16;stroke-width:2.4}
        .ol-truth{fill:none;stroke:#64706a;stroke-width:1.6;stroke-dasharray:6 5}
        .ol-axis{stroke:#20272e;stroke-width:.9}
        .ol-grid{stroke:#d9ddd8;stroke-width:.6}
      `}</style>

      <div className="ol-figure">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
             aria-label={`Polynomial of degree ${degree} fitted to 22 noisy points`}>
          {[-2, -1, 0, 1, 2].map((y) => (
            <line key={y} className="ol-grid" x1={PAD.left} x2={WIDTH - PAD.right}
                  y1={py(y)} y2={py(y)} />
          ))}
          <line className="ol-axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={HEIGHT - PAD.bottom} />
          <line className="ol-axis" x1={PAD.left} x2={WIDTH - PAD.right}
                y1={py(0)} y2={py(0)} />
          <path className="ol-truth" d={view.truth} />
          <path className="ol-fit" d={view.curve} />
          {data.trainX.map((x, index) => (
            <circle key={index} className="ol-dot" cx={px(x)} cy={py(data.trainY[index])} r={4.2} />
          ))}
          <text x={PAD.left} y={HEIGHT - 12} fontSize="12" fill="#64706a">
            dots are the data you have, dashed grey is the process that produced them
          </text>
        </svg>
      </div>

      <div className="ol-controls">
        <div className="ol-row">
          <label htmlFor="ol-degree">Model flexibility</label>
          <input id="ol-degree" type="range" min={1} max={15} step={1} value={degree}
                 onChange={(event) => setDegree(Number(event.target.value))} />
          <strong style={{ minWidth: 84, fontVariantNumeric: "tabular-nums" }}>
            degree {degree}
          </strong>
        </div>
        <div className="ol-row">
          <label htmlFor="ol-seed">Redraw the data</label>
          <input id="ol-seed" type="range" min={1} max={40} step={1} value={seed}
                 onChange={(event) => setSeed(Number(event.target.value))} />
          <strong style={{ minWidth: 84, fontVariantNumeric: "tabular-nums" }}>
            sample {seed}
          </strong>
        </div>
      </div>

      <div className="ol-readouts">
        <div className="ol-stat">
          <span>Error on the dots it was given</span>
          <strong>{view.trainError.toFixed(3)}</strong>
        </div>
        <div className="ol-stat">
          <span>Error on fresh data it never saw</span>
          <strong>{view.testError.toFixed(3)}</strong>
        </div>
      </div>

      <p className="ol-verdict">{verdict}</p>

      <p>
        Drag flexibility to the right and the first number keeps falling, because a
        bendier curve can always pass closer to the dots in front of it. Watch the
        second number instead. It falls, bottoms out, and then climbs, and the gap
        between the two is the overfitting. Every drop in the first number past that
        point is bought by fitting noise that will not repeat.
      </p>
    </div>
  );
}
