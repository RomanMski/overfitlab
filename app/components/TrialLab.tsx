"use client";

import { useMemo, useRef, useState } from "react";

import {
  deflatedSharpe,
  generateNoiseTrials,
  probabilityOfBacktestOverfitting,
  sharpe,
} from "../lib/quant";

interface Loaded {
  name: string;
  columns: string[];
  trials: number[][];
  skipped: number;
}

/** Parse a CSV of period returns, one column per configuration tried. */
function parseTrials(name: string, text: string): Loaded {
  const rows = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/[,;\t]/).map((cell) => cell.trim()));
  if (rows.length < 3) throw new Error("that file has fewer than three rows");

  const first = rows[0];
  const headerLooksNumeric = first.every((cell) => cell !== "" && Number.isFinite(Number(cell)));
  const columns = headerLooksNumeric
    ? first.map((_, index) => `column ${index + 1}`)
    : first;
  const body = headerLooksNumeric ? rows : rows.slice(1);

  const width = columns.length;
  const trials: number[][] = Array.from({ length: width }, () => []);
  let skipped = 0;
  for (const row of body) {
    if (row.length !== width) {
      skipped += 1;
      continue;
    }
    const values = row.map(Number);
    if (!values.every(Number.isFinite)) {
      skipped += 1;
      continue;
    }
    values.forEach((value, index) => trials[index].push(value));
  }

  if (trials[0].length < 20) {
    throw new Error("that file has fewer than 20 usable rows of returns");
  }
  if (width < 2) {
    throw new Error("only one configuration found, so there is no selection to measure");
  }
  return { name, columns, trials, skipped };
}

function exampleTrials(): Loaded {
  const generated = generateNoiseTrials(20260801, 60, 750);
  return {
    name: "example, 60 configurations with no edge",
    columns: generated.map((_, index) => `config_${index + 1}`),
    trials: generated.map((trial) => trial.returns),
    skipped: 0,
  };
}

export function TrialLab() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const report = useMemo(() => {
    if (!loaded) return null;
    const splits = loaded.trials[0].length >= 200 ? 10 : 8;
    return {
      deflated: deflatedSharpe(loaded.trials, 252),
      pbo: probabilityOfBacktestOverfitting(loaded.trials, splits),
      best: Math.max(...loaded.trials.map(sharpe)),
    };
  }, [loaded]);

  const load = (name: string, text: string) => {
    try {
      setLoaded(parseTrials(name, text));
      setError(null);
    } catch (problem) {
      setLoaded(null);
      setError(problem instanceof Error ? problem.message : "could not read that file");
    }
  };

  const verdict = (() => {
    if (!report) return null;
    const survives = report.deflated.deflated >= 0.95;
    const stable = report.pbo <= 0.5;
    if (survives && stable) {
      return "Selection alone does not explain this result. That is the strongest thing these statistics can say, and it is not the same as an edge.";
    }
    if (!survives && !stable) {
      return "The best result does not clear what your number of trials reaches by luck, and the winner usually underperforms out of sample. Both statistics point the same way.";
    }
    if (!survives) {
      return "The best result does not clear the bar set by how many configurations you tried.";
    }
    return "The result clears the luck bar, but the in-sample winner lands below the median out of sample more often than not, so the selection itself is unreliable.";
  })();

  return (
    <div className="tl">
      <style>{`
        .tl{display:grid;gap:18px}
        .tl-drop{border:2px dashed var(--rule,#d9ddd8);border-radius:14px;padding:26px;text-align:center;background:var(--panel,#fffdfa)}
        .tl-drop p{margin:0 0 10px}
        .tl-actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
        .tl-actions button{border:1px solid var(--rule,#d9ddd8);background:var(--panel,#fffdfa);border-radius:999px;padding:9px 18px;cursor:pointer;font:inherit}
        .tl-actions button.primary{background:#20272e;color:#fff;border-color:#20272e}
        .tl-readouts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
        .tl-stat{border:1px solid var(--rule,#d9ddd8);border-radius:12px;padding:12px 14px;background:var(--panel,#fffdfa)}
        .tl-stat span{display:block;font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted,#64706a)}
        .tl-stat strong{display:block;font-size:1.45rem;font-variant-numeric:tabular-nums;margin-top:4px}
        .tl-verdict{border-left:3px solid var(--green,#176b4d);padding:2px 0 2px 14px}
        .tl-error{border-left:3px solid #a75b16;padding:2px 0 2px 14px;color:#a75b16}
      `}</style>

      <div
        className="tl-drop"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const file = event.dataTransfer.files?.[0];
          if (file) void file.text().then((text) => load(file.name, text));
        }}
      >
        <p>
          Drop a CSV of period returns here, one column per configuration you
          backtested. It is read in this page and never uploaded.
        </p>
        <div className="tl-actions">
          <button type="button" className="primary" onClick={() => inputRef.current?.click()}>
            Choose a CSV
          </button>
          <button type="button" onClick={() => { setLoaded(exampleTrials()); setError(null); }}>
            Use the example instead
          </button>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv,text/plain"
          style={{ display: "none" }}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void file.text().then((text) => load(file.name, text));
            event.currentTarget.value = "";
          }}
        />
      </div>

      {error ? <p className="tl-error">{error}</p> : null}

      {loaded && report ? (
        <>
          <p>
            <strong>{loaded.name}</strong>, {loaded.trials.length} configurations
            over {loaded.trials[0].length} periods
            {loaded.skipped > 0 ? `, ${loaded.skipped} unusable rows skipped` : ""}.
          </p>

          <div className="tl-readouts">
            <div className="tl-stat">
              <span>Best Sharpe you found</span>
              <strong>{report.deflated.observedAnnualised.toFixed(2)}</strong>
            </div>
            <div className="tl-stat">
              <span>What luck alone reaches</span>
              <strong>{report.deflated.expectedMaxAnnualised.toFixed(2)}</strong>
            </div>
            <div className="tl-stat">
              <span>Deflated Sharpe</span>
              <strong>{report.deflated.deflated.toFixed(3)}</strong>
            </div>
            <div className="tl-stat">
              <span>Overfitting probability</span>
              <strong>{report.pbo.pbo.toFixed(2)}</strong>
            </div>
          </div>

          <p className="tl-verdict">{verdict}</p>

          <p>
            The deflated Sharpe is a probability, and 0.95 is the usual bar. The
            overfitting probability is the fraction of {report.pbo.nCombinations} ways of
            splitting your history in half where the in-sample winner then landed
            below the median of its peers. Above 0.5 means your selection was worse
            than choosing at random.
          </p>
        </>
      ) : null}
    </div>
  );
}
