"use client";

import { useMemo, useState } from "react";

import { generateNoiseTrials, searchSnapshot } from "../lib/quant";

const WIDTH = 720;
const HEIGHT = 340;
const PAD = { left: 52, right: 18, top: 18, bottom: 38 };
const PERIODS = 750;
const POOL = 500;
const PERIODS_PER_YEAR = 252;

export function SearchLab() {
  const [nTrials, setNTrials] = useState(1);
  const trials = useMemo(() => generateNoiseTrials(20260730, POOL, PERIODS), []);
  const snapshot = useMemo(
    () => searchSnapshot(trials, nTrials, PERIODS_PER_YEAR),
    [trials, nTrials],
  );

  const shown = trials.slice(0, nTrials);
  const bounds = useMemo(() => {
    let lo = 0;
    let hi = 0;
    for (const trial of trials) {
      for (const value of trial.equity) {
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
    }
    return { lo, hi };
  }, [trials]);

  const px = (i: number) =>
    PAD.left + (i / (PERIODS - 1)) * (WIDTH - PAD.left - PAD.right);
  const py = (v: number) =>
    PAD.top +
    ((bounds.hi - v) / (bounds.hi - bounds.lo || 1)) * (HEIGHT - PAD.top - PAD.bottom);

  const line = (equity: number[]) =>
    equity
      .map((v, i) => (i % 3 === 0 || i === equity.length - 1
        ? `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`
        : ""))
      .filter(Boolean)
      .join(" ");

  return (
    <div className="sl">
      <style>{`
        .sl{display:grid;gap:18px}
        .sl-figure{background:var(--panel,#fffdfa);border:1px solid var(--rule,#d9ddd8);border-radius:14px;padding:8px}
        .sl-figure svg{display:block;width:100%;height:auto}
        .sl-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
        .sl-row label{min-width:150px;font-weight:600}
        .sl-row input[type=range]{flex:1;min-width:220px}
        .sl-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
        .sl-stat{border:1px solid var(--rule,#d9ddd8);border-radius:12px;padding:12px 14px;background:var(--panel,#fffdfa)}
        .sl-stat span{display:block;font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#64706a)}
        .sl-stat strong{display:block;font-size:1.5rem;font-variant-numeric:tabular-nums;margin-top:4px}
        .sl-warn{border-left:3px solid #a75b16;padding:2px 0 2px 14px}
        .sl-other{fill:none;stroke:#c9cfc9;stroke-width:1}
        .sl-best{fill:none;stroke:#176b4d;stroke-width:2.6}
        .sl-axis{stroke:#20272e;stroke-width:.9}
      `}</style>

      <div className="sl-figure">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
             aria-label={`Best of ${nTrials} equity curves drawn from pure noise`}>
          <line className="sl-axis" x1={PAD.left} x2={WIDTH - PAD.right}
                y1={py(0)} y2={py(0)} />
          {shown.map((trial, index) =>
            index === snapshot.bestIndex ? null : (
              <path key={index} className="sl-other" d={line(trial.equity)} />
            ),
          )}
          <path className="sl-best" d={line(shown[snapshot.bestIndex].equity)} />
          <text x={PAD.left} y={HEIGHT - 12} fontSize="12" fill="#64706a">
            {nTrials === 1
              ? "one strategy, drawn from noise"
              : `${nTrials} strategies, all drawn from noise, best one in green`}
          </text>
        </svg>
      </div>

      <div className="sl-row">
        <label htmlFor="sl-trials">Configurations you tried</label>
        <input id="sl-trials" type="range" min={1} max={POOL} step={1} value={nTrials}
               onChange={(event) => setNTrials(Number(event.target.value))} />
        <strong style={{ minWidth: 84, fontVariantNumeric: "tabular-nums" }}>
          {nTrials}
        </strong>
      </div>

      <div className="sl-readouts">
        <div className="sl-stat">
          <span>Best Sharpe found</span>
          <strong>{snapshot.bestAnnualised.toFixed(2)}</strong>
        </div>
        <div className="sl-stat">
          <span>What luck alone reaches</span>
          <strong>
            {(snapshot.expectedBest * Math.sqrt(PERIODS_PER_YEAR)).toFixed(2)}
          </strong>
        </div>
        <div className="sl-stat">
          <span>Left over after luck</span>
          <strong>
            {(snapshot.excess * Math.sqrt(PERIODS_PER_YEAR)).toFixed(2)}
          </strong>
        </div>
      </div>

      <p className="sl-warn">
        Nothing on this screen has any edge. Every curve is a random walk with a
        true Sharpe of exactly zero.
      </p>

      <p>
        Drag the slider. The best curve gets better and its Sharpe climbs past what
        most people would call investable, only because you are looking at more of
        them. The middle number is what the best of that many coin flips reaches
        anyway. The third is what is left, and it stays near zero however far you
        drag, because there was never anything there. A positive number there is
        not proof of an edge. It only means selection does not explain it.
      </p>

      <p>
        That is the problem with a backtest that reports only its winner. The number
        is real. What it measures is how hard you looked.
      </p>
    </div>
  );
}
