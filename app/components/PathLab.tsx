"use client";

import { useMemo, useState } from "react";

import {
  autocorrelatedMarket,
  blockPermutation,
  buyAndHold,
  cumulative,
  makeRng,
  momentumStrategy,
  pathStress,
} from "../lib/quant";

const WIDTH = 720;
const HEIGHT = 300;
const PAD = { left: 52, right: 18, top: 18, bottom: 34 };
const PERIODS = 900;
const BLOCKS = [1, 5, 20, 60];

export function PathLab() {
  const [blockIndex, setBlockIndex] = useState(0);
  const [useMomentum, setUseMomentum] = useState(true);

  const market = useMemo(() => autocorrelatedMarket(20260731, PERIODS, 0.22), []);
  const strategy = useMomentum ? momentumStrategy : buyAndHold;

  const stress = useMemo(
    () => pathStress(strategy, market, BLOCKS, 120, 5),
    [market, strategy],
  );

  const block = BLOCKS[blockIndex];
  const level = stress.levels[blockIndex];

  const shown = useMemo(() => {
    const rng = makeRng(1000 + block);
    return Array.from({ length: 24 }, () =>
      cumulative(blockPermutation(market, block, rng)),
    );
  }, [market, block]);

  const real = useMemo(() => cumulative(market), [market]);

  const bounds = useMemo(() => {
    let lo = 0;
    let hi = 0;
    for (const path of [...shown, real]) {
      for (const value of path) {
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
    }
    return { lo, hi };
  }, [shown, real]);

  const px = (i: number) =>
    PAD.left + (i / (PERIODS - 1)) * (WIDTH - PAD.left - PAD.right);
  const py = (v: number) =>
    PAD.top +
    ((bounds.hi - v) / (bounds.hi - bounds.lo || 1)) * (HEIGHT - PAD.top - PAD.bottom);
  const line = (equity: number[]) =>
    equity
      .map((v, i) =>
        i % 4 === 0 || i === equity.length - 1
          ? `${i === 0 ? "M" : "L"}${px(i).toFixed(1)},${py(v).toFixed(1)}`
          : "",
      )
      .filter(Boolean)
      .join(" ");

  const keeps =
    block === 1
      ? "Order only. Every observation appears exactly once, so the mean, the spread and the fat tails are identical to the real series. Only the arrangement changes, which is why this is not a noise series."
      : `Runs of ${block} periods stay intact and only their order is shuffled.`;

  return (
    <div className="pl">
      <style>{`
        .pl{display:grid;gap:18px}
        .pl-figure{background:var(--panel,#fffdfa);border:1px solid var(--rule,#d9ddd8);border-radius:14px;padding:8px}
        .pl-figure svg{display:block;width:100%;height:auto}
        .pl-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
        .pl-row label{min-width:170px;font-weight:600}
        .pl-row input[type=range]{flex:1;min-width:200px}
        .pl-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
        .pl-stat{border:1px solid var(--rule,#d9ddd8);border-radius:12px;padding:12px 14px;background:var(--panel,#fffdfa)}
        .pl-stat span{display:block;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#64706a)}
        .pl-stat strong{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums;margin-top:4px}
        .pl-note{border-left:3px solid var(--green,#176b4d);padding:2px 0 2px 14px}
        .pl-synth{fill:none;stroke:#c9cfc9;stroke-width:1}
        .pl-real{fill:none;stroke:#2f5d7e;stroke-width:2.6}
        .pl-axis{stroke:#20272e;stroke-width:.9}
        .pl-toggle{display:flex;gap:8px;flex-wrap:wrap}
        .pl-toggle button{border:1px solid var(--rule,#d9ddd8);background:var(--panel,#fffdfa);border-radius:999px;padding:7px 15px;cursor:pointer;font:inherit}
        .pl-toggle button[aria-pressed=true]{background:#20272e;color:#fff;border-color:#20272e}
      `}</style>

      <div className="pl-figure">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
             aria-label={`Real market against 24 synthetic markets at block size ${block}`}>
          <line className="pl-axis" x1={PAD.left} x2={WIDTH - PAD.right}
                y1={py(0)} y2={py(0)} />
          {shown.map((path, index) => (
            <path key={index} className="pl-synth" d={line(path)} />
          ))}
          <path className="pl-real" d={line(real)} />
          <text x={PAD.left} y={HEIGHT - 10} fontSize="12" fill="#64706a">
            blue is the market that happened, grey are markets that could have
          </text>
        </svg>
      </div>

      <div className="pl-toggle" role="group" aria-label="Strategy">
        <button type="button" aria-pressed={useMomentum}
                onClick={() => setUseMomentum(true)}>
          Trend follower
        </button>
        <button type="button" aria-pressed={!useMomentum}
                onClick={() => setUseMomentum(false)}>
          Buy and hold
        </button>
      </div>

      <div className="pl-row">
        <label htmlFor="pl-block">Structure kept</label>
        <input id="pl-block" type="range" min={0} max={BLOCKS.length - 1} step={1}
               value={blockIndex}
               onChange={(event) => setBlockIndex(Number(event.target.value))} />
        <strong style={{ minWidth: 96, fontVariantNumeric: "tabular-nums" }}>
          block {block}
        </strong>
      </div>

      <p className="pl-note">{keeps}</p>

      <div className="pl-readouts">
        <div className="pl-stat">
          <span>On the real market</span>
          <strong>{stress.observedAnnualised.toFixed(2)}</strong>
        </div>
        <div className="pl-stat">
          <span>Median on synthetic</span>
          <strong>{level.medianAnnualised.toFixed(2)}</strong>
        </div>
        <div className="pl-stat">
          <span>Real result beats</span>
          <strong>{level.percentile.toFixed(0)}%</strong>
        </div>
      </div>

      <p>
        {useMomentum
          ? "The trend follower makes its money from the market's tendency to keep going. Drag to block 1 and that tendency is gone. The strategy collapses to about zero, and the real result beats nearly all the synthetic markets. Widen the blocks and the persistence comes back, so the synthetic markets support the strategy too and the real result stops looking special. That gradient is the evidence. It shows the result needs the ordering, which is consistent with a timing edge and also with volatility targeting or a lookback bug, so it narrows the question rather than closing it."
          : "Buy and hold makes its money from drift, not timing. Reordering returns cannot change their mean or their spread. So the synthetic markets pay almost exactly what the real one did, at every block size. The flat line across the slider is the tell. Nothing here depends on what order things happened in."}
      </p>
    </div>
  );
}
