"use client";

import { useMemo, useRef, useState } from "react";

import { autocorrelatedMarket, blockPermutation, makeRng } from "../lib/quant";

const BLOCKS = [1, 5, 20, 60];

interface Series {
  name: string;
  values: number[];
}

/** Read one numeric column out of a CSV. First column that parses is used. */
function parseSeries(name: string, text: string): Series {
  const rows = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => line.split(/[,;\t]/).map((cell) => cell.trim()));

  const width = Math.max(...rows.map((row) => row.length));
  let best: number[] = [];
  for (let column = 0; column < width; column += 1) {
    const values: number[] = [];
    for (const row of rows) {
      const value = Number(row[column]);
      if (row[column] !== undefined && row[column] !== "" && Number.isFinite(value)) {
        values.push(value);
      }
    }
    if (values.length > best.length) best = values;
  }
  if (best.length < 30) {
    throw new Error("could not find a numeric column with at least 30 rows");
  }

  // Prices arrive far more often than returns. Convert if the column never
  // goes negative and its typical step is small relative to its level, which
  // is what a price series looks like and a return series does not.
  const negatives = best.filter((value) => value < 0).length;
  const mean = best.reduce((a, b) => a + b, 0) / best.length;
  if (negatives === 0 && Math.abs(mean) > 1.5) {
    const returns: number[] = [];
    for (let i = 1; i < best.length; i += 1) {
      if (best[i - 1] !== 0) returns.push(best[i] / best[i - 1] - 1);
    }
    return { name: `${name}, converted from prices`, values: returns };
  }
  return { name, values: best };
}

function toCsv(paths: number[][]): string {
  const header = paths.map((_, index) => `path_${index + 1}`).join(",");
  const lines = [header];
  for (let row = 0; row < paths[0].length; row += 1) {
    lines.push(paths.map((path) => path[row].toPrecision(10)).join(","));
  }
  return lines.join("\n");
}

function download(name: string, text: string): void {
  const blob = new Blob([text], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function GeneratorLab() {
  const [series, setSeries] = useState<Series | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nPaths, setNPaths] = useState(50);
  const inputRef = useRef<HTMLInputElement>(null);

  const example = useMemo(
    () => ({ name: "example series, 900 periods", values: autocorrelatedMarket(7, 900, 0.2) }),
    [],
  );
  const active = series ?? example;

  const generate = (block: number) => {
    const rng = makeRng(20260801 + block);
    return Array.from({ length: nPaths }, () =>
      blockPermutation(active.values, block, rng),
    );
  };

  const load = (name: string, text: string) => {
    try {
      setSeries(parseSeries(name, text));
      setError(null);
    } catch (problem) {
      setSeries(null);
      setError(problem instanceof Error ? problem.message : "could not read that file");
    }
  };

  const downloadAll = () => {
    for (const block of BLOCKS) {
      download(`block-${String(block).padStart(3, "0")}.csv`, toCsv(generate(block)));
    }
    const manifest = {
      source: active.name,
      source_periods: active.values.length,
      paths_per_block: nPaths,
      block_sizes: BLOCKS,
      scheme: "block permutation, sampled without replacement",
      note:
        "Every observation in the source appears exactly once in every " +
        "generated series. The mean, the variance, the skew and the extremes " +
        "are identical to the source and only the order changes.",
    };
    download("manifest.json", JSON.stringify(manifest, null, 2));
  };

  return (
    <div className="gl">
      <style>{`
        .gl{display:grid;gap:18px}
        .gl-drop{border:2px dashed var(--rule,#d9ddd8);border-radius:14px;padding:24px;text-align:center;background:var(--panel,#fffdfa)}
        .gl-actions{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:10px}
        .gl-actions button{border:1px solid var(--rule,#d9ddd8);background:var(--panel,#fffdfa);border-radius:999px;padding:9px 18px;cursor:pointer;font:inherit}
        .gl-actions button.primary{background:#20272e;color:#fff;border-color:#20272e}
        .gl-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
        .gl-row label{min-width:150px;font-weight:600}
        .gl-row input[type=range]{flex:1;min-width:200px}
        .gl-table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
        .gl-table th,.gl-table td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule,#d9ddd8)}
        .gl-table th{font-size:.74rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted,#64706a);font-weight:600}
        .gl-error{border-left:3px solid #a75b16;padding:2px 0 2px 14px;color:#a75b16}
        .gl-note{border-left:3px solid var(--green,#176b4d);padding:2px 0 2px 14px}
      `}</style>

      <div
        className="gl-drop"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const file = event.dataTransfer.files?.[0];
          if (file) void file.text().then((text) => load(file.name, text));
        }}
      >
        <p style={{ margin: 0 }}>
          Drop a CSV holding a price or return column. It is read in this page
          and never uploaded.
        </p>
        <div className="gl-actions">
          <button type="button" className="primary" onClick={() => inputRef.current?.click()}>
            Choose a CSV
          </button>
          <button type="button" onClick={() => { setSeries(null); setError(null); }}>
            Use the example series
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

      {error ? <p className="gl-error">{error}</p> : null}

      <p>
        Using <strong>{active.name}</strong>, {active.values.length} periods.
      </p>

      <div className="gl-row">
        <label htmlFor="gl-paths">Series per block length</label>
        <input id="gl-paths" type="range" min={10} max={300} step={10} value={nPaths}
               onChange={(event) => setNPaths(Number(event.target.value))} />
        <strong style={{ minWidth: 64 }}>{nPaths}</strong>
      </div>

      <table className="gl-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Keeps</th>
            <th>What it answers</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>block-001.csv</td>
            <td>order destroyed, every value kept</td>
            <td>does the model need the ordering at all</td>
          </tr>
          <tr>
            <td>block-005.csv</td>
            <td>runs of five</td>
            <td>does it rely on structure shorter than a week</td>
          </tr>
          <tr>
            <td>block-020.csv</td>
            <td>runs of twenty</td>
            <td>does it rely on structure inside a month</td>
          </tr>
          <tr>
            <td>block-060.csv</td>
            <td>runs of sixty</td>
            <td>a near copy, as a control</td>
          </tr>
        </tbody>
      </table>

      <div className="gl-actions" style={{ justifyContent: "flex-start" }}>
        <button type="button" className="primary" onClick={downloadAll}>
          Generate and download {BLOCKS.length * nPaths} series
        </button>
      </div>

      <p className="gl-note">
        Every observation in your source appears exactly once in every generated
        series, so the mean, the variance, the skew and the extremes are
        identical and only the order changes. These are reorderings rather than
        new observations, so they cannot tell your model about behaviour your
        data never contained. What they can do is
        show you how much of its result depended on the particular order your
        history happened to arrive in.
      </p>

      <p>
        Run your model on each block length the same way you ran it on your real
        data, then compare. A model whose performance holds up at block 1 is not
        using time structure, whatever it claims. One that collapses there and
        recovers as the blocks lengthen is reading the ordering, though that
        could equally be volatility targeting or a lookback bug rather than a
        genuine edge.
      </p>
    </div>
  );
}
