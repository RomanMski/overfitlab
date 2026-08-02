"use client";

import { useMemo, useState } from "react";

import { cumulative, makeRng, normalDraws, sharpe } from "../lib/quant";

const WIDTH = 700;
const HEIGHT = 240;
const PAD = { left: 46, right: 16, top: 16, bottom: 30 };
const PERIODS = 756;
const PERIODS_PER_YEAR = 252;

export function SharpeLab() {
  const [driftBp, setDriftBp] = useState(4);
  const [volBp, setVolBp] = useState(100);

  const series = useMemo(() => {
    const rng = makeRng(4242);
    const shocks = normalDraws(rng, PERIODS);
    return shocks.map((z) => z * (volBp / 10000) + driftBp / 10000);
  }, [driftBp, volBp]);

  const equity = useMemo(() => cumulative(series), [series]);
  const perPeriod = sharpe(series);
  const annualised = perPeriod * Math.sqrt(PERIODS_PER_YEAR);

  const lo = Math.min(...equity, 0);
  const hi = Math.max(...equity, 0);
  const px = (i: number) =>
    PAD.left + (i / (PERIODS - 1)) * (WIDTH - PAD.left - PAD.right);
  const py = (v: number) =>
    PAD.top + ((hi - v) / (hi - lo || 1)) * (HEIGHT - PAD.top - PAD.bottom);
  const path = equity
    .map((v, i) => (i % 2 === 0 || i === equity.length - 1
      ? `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`
      : ""))
    .filter(Boolean)
    .join(" ");

  return (
    <div className="sh">
      <style>{`
        .sh{display:grid;gap:16px}
        .sh-figure{background:var(--panel,#fffdfa);border:1px solid var(--rule,#d9ddd8);border-radius:14px;padding:8px}
        .sh-figure svg{display:block;width:100%;height:auto}
        .sh-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
        .sh-row label{min-width:210px;font-weight:600}
        .sh-row input[type=range]{flex:1;min-width:190px}
        .sh-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
        .sh-stat{border:1px solid var(--rule,#d9ddd8);border-radius:12px;padding:12px 14px;background:var(--panel,#fffdfa)}
        .sh-stat span{display:block;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#64706a)}
        .sh-stat strong{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums;margin-top:4px}
        .sh-line{fill:none;stroke:#2f5d7e;stroke-width:2.2}
        .sh-axis{stroke:#20272e;stroke-width:.9}
      `}</style>

      <div className="sh-figure">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
             aria-label="Equity curve for the chosen drift and volatility">
          <line className="sh-axis" x1={PAD.left} x2={WIDTH - PAD.right}
                y1={py(0)} y2={py(0)} />
          <path className="sh-line" d={path} />
        </svg>
      </div>

      <div className="sh-row">
        <label htmlFor="sh-drift">Average daily return, basis points</label>
        <input id="sh-drift" type="range" min={-10} max={20} step={1} value={driftBp}
               onChange={(event) => setDriftBp(Number(event.target.value))} />
        <strong style={{ minWidth: 56, fontVariantNumeric: "tabular-nums" }}>{driftBp}</strong>
      </div>

      <div className="sh-row">
        <label htmlFor="sh-vol">Daily volatility, basis points</label>
        <input id="sh-vol" type="range" min={20} max={300} step={10} value={volBp}
               onChange={(event) => setVolBp(Number(event.target.value))} />
        <strong style={{ minWidth: 56, fontVariantNumeric: "tabular-nums" }}>{volBp}</strong>
      </div>

      <div className="sh-readouts">
        <div className="sh-stat">
          <span>Per day</span>
          <strong>{perPeriod.toFixed(4)}</strong>
        </div>
        <div className="sh-stat">
          <span>Annualised</span>
          <strong>{annualised.toFixed(2)}</strong>
        </div>
      </div>

      <p>
        The Sharpe ratio is average return divided by the standard deviation of
        those returns. Nothing more. Raise the drift and it climbs, raise the
        volatility and it falls, and the equity curve above changes shape while
        the arithmetic stays the same.
      </p>

      <p>
        Annualising multiplies the per period figure by the square root of the
        number of periods in a year, 252 for daily data. That step assumes
        returns are independent across days. When they are not, which is the
        normal case for anything with momentum or volatility clustering, the
        square root rule is wrong and the annualised number is misstated. Lo
        (2002) gives the correction. This library does not apply it, which is
        worth knowing when you read any annualised figure it reports.
      </p>

      <p>
        Two other things the ratio cannot see. It treats an upside surprise and
        a downside surprise identically, because standard deviation is
        symmetric, so a strategy that grinds out small gains and occasionally
        loses everything can post a high Sharpe right up until it does. And it
        says nothing about how the return was obtained, so leverage, illiquidity
        and a short volatility position all pass through it unremarked.
      </p>
    </div>
  );
}
