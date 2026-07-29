"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  generateStressVariant,
  inferTarget,
  inferTask,
  makeSampleDataset,
  runAudit,
  type AuditResult,
  type AuditSettings,
  type DataTable,
  type ModelKind,
  type TaskType,
} from "../lib/analysis";
import { downloadText, parseCsv, tableToCsv } from "../lib/csv";
import { buildHtmlReport } from "../lib/report";
import { StressChart } from "./StressChart";

const SAMPLE_TABLE = makeSampleDataset();
const SAMPLE_TARGET = inferTarget(SAMPLE_TABLE);
const DEFAULT_SETTINGS: AuditSettings = {
  target: SAMPLE_TARGET,
  task: inferTask(SAMPLE_TABLE, SAMPLE_TARGET),
  model: "regularized",
  repeats: 12,
  testSize: 0.25,
  seed: 1729,
};

type VariantFamily = "feature-noise" | "label-noise" | "missingness" | "bootstrap";

const VARIANT_LEVELS: Record<VariantFamily, number[]> = {
  "feature-noise": [0.1, 0.25, 0.5],
  "label-noise": [0.05, 0.1, 0.2],
  missingness: [0.05, 0.15, 0.3],
  bootstrap: [1],
};

export function StressFoldApp() {
  const [table, setTable] = useState<DataTable>(SAMPLE_TABLE);
  const [target, setTarget] = useState(DEFAULT_SETTINGS.target);
  const [task, setTask] = useState<TaskType>(DEFAULT_SETTINGS.task);
  const [model, setModel] = useState<ModelKind>(DEFAULT_SETTINGS.model);
  const [repeats, setRepeats] = useState(DEFAULT_SETTINGS.repeats);
  const [testSize, setTestSize] = useState(DEFAULT_SETTINGS.testSize);
  const [seed, setSeed] = useState(DEFAULT_SETTINGS.seed);
  const [result, setResult] = useState<AuditResult | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("Preparing protocol");
  const [error, setError] = useState<string | null>(null);
  const [variantFamily, setVariantFamily] = useState<VariantFamily>("feature-noise");
  const [variantLevel, setVariantLevel] = useState(0.25);
  const initialRunStarted = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const settings = useMemo<AuditSettings>(
    () => ({ target, task, model, repeats, testSize, seed }),
    [target, task, model, repeats, testSize, seed],
  );

  useEffect(() => {
    if (initialRunStarted.current) return;
    initialRunStarted.current = true;
    void executeAudit(SAMPLE_TABLE, DEFAULT_SETTINGS);
    // The first run deliberately uses the locked sample protocol once.
  }, []);

  async function executeAudit(data: DataTable, auditSettings: AuditSettings) {
    setRunning(true);
    setProgress(0);
    setError(null);
    try {
      const nextResult = await runAudit(data, auditSettings, (fraction, label) => {
        setProgress(fraction);
        setProgressLabel(label);
      });
      setResult(nextResult);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The audit could not be completed.");
    } finally {
      setRunning(false);
      setProgress(1);
    }
  }

  async function handleFile(file: File) {
    if (file.size > 5 * 1024 * 1024) {
      setError("Keep browser-lab CSV files below 5 MB. The Python package handles larger audits.");
      return;
    }
    try {
      const parsed = parseCsv(await file.text(), file.name);
      const nextTarget = inferTarget(parsed);
      setTable(parsed);
      setTarget(nextTarget);
      setTask(inferTask(parsed, nextTarget));
      setResult(null);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The CSV could not be read.");
    }
  }

  function restoreSample() {
    setTable(SAMPLE_TABLE);
    setTarget(DEFAULT_SETTINGS.target);
    setTask(DEFAULT_SETTINGS.task);
    setModel(DEFAULT_SETTINGS.model);
    setRepeats(DEFAULT_SETTINGS.repeats);
    setTestSize(DEFAULT_SETTINGS.testSize);
    setSeed(DEFAULT_SETTINGS.seed);
    setError(null);
    void executeAudit(SAMPLE_TABLE, DEFAULT_SETTINGS);
  }

  function applyPreset(preset: "quick" | "audit") {
    setRepeats(preset === "quick" ? 8 : 24);
    setModel(preset === "quick" ? "regularized" : model);
  }

  function handleFamilyChange(family: VariantFamily) {
    setVariantFamily(family);
    setVariantLevel(VARIANT_LEVELS[family][Math.min(1, VARIANT_LEVELS[family].length - 1)]);
  }

  function downloadVariant(kind: "csv" | "manifest") {
    const generated = generateStressVariant(table, target, variantFamily, variantLevel, seed);
    if (kind === "csv") {
      downloadText(tableToCsv(generated.table), generated.table.name, "text/csv;charset=utf-8");
    } else {
      const filename = generated.table.name.replace(/\.csv$/i, ".manifest.json");
      downloadText(JSON.stringify(generated.manifest, null, 2), filename, "application/json;charset=utf-8");
    }
  }

  const resultHash = result?.protocol.sourceHash.replace("fnv1a-", "") ?? "pending";

  return (
    <div className="site-frame">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="StressFold home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>StressFold</span>
        </a>
        <nav className="primary-nav" aria-label="Primary navigation">
          <a href="#lab">Browser lab</a>
          <a href="#method">Method</a>
          <a href="#paper">Paper</a>
        </nav>
        <div className="header-status"><span /> v0.1 research preview</div>
      </header>

      <main id="top">
        <section className="hero section-boundary">
          <div className="hero-copy">
            <div className="kicker"><span>Generalization stress testing</span><span>Tabular models</span></div>
            <h1>Find where a model<br />stops generalizing.</h1>
            <p className="hero-lede">
              StressFold traces model performance across controlled perturbations, repeated refits, and label-permutation nulls. Every split, seed, and generated variant is recorded.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#lab">Open browser lab <span aria-hidden="true">→</span></a>
              <a className="button button-quiet" href="#method">Read the protocol</a>
            </div>
            <div className="trust-row" aria-label="Product properties">
              <span>Runs in your browser</span>
              <span>Paired Monte Carlo</span>
              <span>Reproducible exports</span>
            </div>
          </div>
          <div className="protocol-figure" aria-label="StressFold protocol from observed data to an evidence profile">
            <div className="figure-label">Protocol / 0.1</div>
            <div className="protocol-nodes">
              <div className="protocol-node active"><b>01</b><span>Split</span><small>Repeated holdout</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>02</b><span>Fit</span><small>Train-fold only</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>03</b><span>Stress</span><small>Perturb or refit</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>04</b><span>Compare</span><small>Paired loss</small></div>
            </div>
            <div className="figure-output">
              <div>
                <span>Evidence profile</span>
                <strong>G · R(λ) · P<sub>null</sub></strong>
              </div>
              <div className="mini-bars" aria-hidden="true">
                <i style={{ height: "86%" }} /><i style={{ height: "70%" }} /><i style={{ height: "44%" }} /><i style={{ height: "22%" }} />
              </div>
            </div>
            <p>One instrument, three distinct claims: generalization, robustness, and falsification.</p>
          </div>
        </section>

        <section className="definition-strip" aria-label="What StressFold measures">
          <div><b>G</b><span>Train-audit gap</span><small>Generalization</small></div>
          <div><b>R(λ)</b><span>Retained skill curve</span><small>Robustness</small></div>
          <div><b>V</b><span>Across-split spread</span><small>Instability</small></div>
          <div><b>P<sub>null</sub></b><span>Permutation percentile</span><small>Falsification</small></div>
        </section>

        <section className="lab-section" id="lab">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Local browser instrument</div>
              <h2>Run a bounded audit before writing model code.</h2>
            </div>
            <p>Upload a CSV or use the reproducible sample. Files stay on this device; the browser lab supports binary classification and regression over numeric predictors.</p>
          </div>

          <div className="lab-workspace">
            <aside className="control-panel" aria-label="Audit setup">
              <div className="panel-heading">
                <div><span>Setup</span><strong>Audit protocol</strong></div>
                <span className="step-counter">01 / 03</span>
              </div>

              <div className="dataset-card">
                <div className="file-glyph" aria-hidden="true"><span>CSV</span></div>
                <div className="dataset-meta">
                  <strong title={table.name}>{table.name}</strong>
                  <span>{table.rows.length.toLocaleString()} rows · {table.headers.length} columns</span>
                </div>
                <button className="icon-button" type="button" onClick={() => fileInputRef.current?.click()} aria-label="Choose another CSV">↗</button>
                <input
                  ref={fileInputRef}
                  className="visually-hidden"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleFile(file);
                    event.currentTarget.value = "";
                  }}
                />
              </div>
              <div className="dataset-actions">
                <button type="button" onClick={() => fileInputRef.current?.click()}>Upload CSV</button>
                <button type="button" onClick={restoreSample}>Restore sample</button>
              </div>

              <div className="control-group">
                <label htmlFor="target-column">Target column</label>
                <select
                  id="target-column"
                  value={target}
                  onChange={(event) => {
                    setTarget(event.target.value);
                    setTask(inferTask(table, event.target.value));
                    setResult(null);
                  }}
                >
                  {table.headers.map((header) => <option key={header} value={header}>{header}</option>)}
                </select>
              </div>

              <fieldset className="control-group">
                <legend>Task</legend>
                <div className="segmented-control">
                  <button className={task === "classification" ? "selected" : ""} type="button" onClick={() => setTask("classification")}>Binary</button>
                  <button className={task === "regression" ? "selected" : ""} type="button" onClick={() => setTask("regression")}>Regression</button>
                </div>
              </fieldset>

              <div className="control-group">
                <label htmlFor="model-kind">Estimator</label>
                <select id="model-kind" value={model} onChange={(event) => setModel(event.target.value as ModelKind)}>
                  <option value="regularized">Regularized linear / logistic</option>
                  <option value="nearest-neighbor">High-capacity nearest neighbor</option>
                </select>
                <small>The Python API accepts any scikit-learn compatible pipeline.</small>
              </div>

              <div className="preset-row" aria-label="Run preset">
                <button type="button" onClick={() => applyPreset("quick")}>Quick · 8</button>
                <button type="button" onClick={() => applyPreset("audit")}>Audit · 24</button>
              </div>

              <div className="inline-controls">
                <div className="control-group compact">
                  <label htmlFor="repeat-count">Repeats</label>
                  <input id="repeat-count" type="number" min="4" max="40" step="1" value={repeats} onChange={(event) => setRepeats(Number(event.target.value))} />
                </div>
                <div className="control-group compact">
                  <label htmlFor="test-size">Audit share</label>
                  <select id="test-size" value={testSize} onChange={(event) => setTestSize(Number(event.target.value))}>
                    <option value="0.2">20%</option><option value="0.25">25%</option><option value="0.3">30%</option><option value="0.35">35%</option>
                  </select>
                </div>
                <div className="control-group compact">
                  <label htmlFor="seed">Seed</label>
                  <input id="seed" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </div>
              </div>

              <div className="operator-list" aria-label="Enabled stress operators">
                <div><span className="operator-dot teal" /><b>Feature noise</b><small>fixed model</small></div>
                <div><span className="operator-dot amber" /><b>Label noise</b><small>refit</small></div>
                <div><span className="operator-dot violet" /><b>Missingness</b><small>fixed model</small></div>
                <div><span className="operator-dot gold" /><b>Train size</b><small>refit</small></div>
              </div>

              <button
                className="run-button"
                type="button"
                disabled={running}
                onClick={() => void executeAudit(table, settings)}
              >
                <span>{running ? "Running paired splits" : "Run stress audit"}</span>
                <b aria-hidden="true">{running ? `${Math.round(progress * 100)}%` : "→"}</b>
              </button>
              {running && <div className="progress-block" role="status"><i style={{ width: `${progress * 100}%` }} /><span>{progressLabel}</span></div>}
              {error && <div className="error-message" role="alert">{error}</div>}
              <p className="privacy-note"><span aria-hidden="true">◇</span> Local computation. No dataset leaves the browser.</p>
            </aside>

            <section className="result-panel" aria-live="polite" aria-busy={running}>
              <div className="result-toolbar">
                <div>
                  <span className="result-state"><i className={result ? "ready" : ""} /> {result ? "Protocol complete" : running ? "Protocol running" : "Awaiting audit"}</span>
                  <strong>{result ? `${result.dataset.rows.toLocaleString()} rows / ${result.dataset.features.length} features` : "Run the sample or upload data"}</strong>
                </div>
                <div className="hash-label">RUN / {resultHash.toUpperCase()}</div>
              </div>

              {result ? (
                <>
                  <div className="metric-grid">
                    <Metric label={`Median ${result.baseline.scoreLabel}`} value={formatMetric(result.baseline.score)} note="clean audit splits" />
                    <Metric label="Train-audit gap" value={formatMetric(result.baseline.gap)} note={result.baseline.lossLabel.toLowerCase()} />
                    <Metric label="Split variability" value={formatMetric(result.baseline.splitSpread)} note="5th-95th span" />
                    <Metric label="Permutation null" value={`${Math.round(result.permutation.percentile)}th`} note={`${result.permutation.runs} quick refits`} />
                  </div>

                  <div className="result-section chart-section">
                    <div className="result-section-heading">
                      <div><span>Stress response</span><h3>Retained predictive skill</h3></div>
                      <div className="method-chip">median + 90% MC band</div>
                    </div>
                    <StressChart curves={result.curves} />
                  </div>

                  <div className="result-section">
                    <div className="result-section-heading">
                      <div><span>Operator summary</span><h3>Failure boundaries</h3></div>
                      <div className="method-chip">paired by split</div>
                    </div>
                    <div className="summary-table-wrap">
                      <table className="summary-table">
                        <thead><tr><th>Operator</th><th>Experiment</th><th>Degradation area</th><th>First-step loss</th><th>50% skill boundary</th></tr></thead>
                        <tbody>
                          {result.summaries.map((summary) => (
                            <tr key={summary.id}>
                              <td><span className={`operator-dot ${operatorColor(summary.id)}`} />{summary.label}</td>
                              <td>{summary.mode}</td>
                              <td>{summary.degradationArea.toFixed(3)}</td>
                              <td>{summary.firstStepLoss.toFixed(3)}</td>
                              <td>{summary.halfSkillAt}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="finding-grid">
                    {result.findings.map((finding) => (
                      <article className={`finding-card ${finding.status}`} key={finding.kind}>
                        <div><span>{finding.eyebrow}</span><i /></div>
                        <h3>{finding.title}</h3>
                        <p>{finding.detail}</p>
                      </article>
                    ))}
                  </div>

                  {result.warnings.length > 0 && (
                    <details className="run-notes">
                      <summary>{result.warnings.length} protocol note{result.warnings.length === 1 ? "" : "s"}</summary>
                      <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                    </details>
                  )}

                  <div className="export-panel">
                    <div className="export-heading">
                      <div><span>Reproducible artifact</span><h3>Generate a stress dataset</h3></div>
                      <p>Exports include a source fingerprint, operator, severity, and seed. They are probes, not synthetic ground truth.</p>
                    </div>
                    <div className="export-controls">
                      <select aria-label="Variant family" value={variantFamily} onChange={(event) => handleFamilyChange(event.target.value as VariantFamily)}>
                        <option value="feature-noise">Feature noise</option>
                        <option value="label-noise">Label noise</option>
                        <option value="missingness">Missingness</option>
                        <option value="bootstrap">Empirical bootstrap</option>
                      </select>
                      <select aria-label="Variant severity" value={variantLevel} disabled={variantFamily === "bootstrap"} onChange={(event) => setVariantLevel(Number(event.target.value))}>
                        {VARIANT_LEVELS[variantFamily].map((level) => <option key={level} value={level}>{variantFamily === "bootstrap" ? "row resample" : `${Math.round(level * 100)}% severity`}</option>)}
                      </select>
                      <button type="button" onClick={() => downloadVariant("csv")}>Download CSV</button>
                      <button type="button" onClick={() => downloadVariant("manifest")}>Manifest</button>
                    </div>
                    <div className="report-actions">
                      <button type="button" onClick={() => downloadText(buildHtmlReport(result), `stressfold_${resultHash}.html`, "text/html;charset=utf-8")}>Self-contained HTML report</button>
                      <button type="button" onClick={() => downloadText(JSON.stringify(result, null, 2), `stressfold_${resultHash}.json`, "application/json;charset=utf-8")}>Results JSON</button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty-result">
                  <div className="empty-orbit" aria-hidden="true"><i /><i /><i /></div>
                  <h3>{running ? "Estimating response curves" : "No audit result yet"}</h3>
                  <p>{running ? progressLabel : "Choose a target and run the protocol. The output will separate clean-split generalization, stress robustness, and falsification evidence."}</p>
                </div>
              )}
            </section>
          </div>
        </section>

        <section className="method-section section-boundary" id="method">
          <div className="section-heading compact-heading">
            <div><div className="eyebrow">Method, not magic</div><h2>Noise sensitivity is not proof of overfitting.</h2></div>
            <p>StressFold keeps different estimands separate. A model can be brittle without being overfit; a flexible model can fit random labels and still generalize on real signal.</p>
          </div>
          <div className="method-grid">
            <article>
              <div className="method-number">01</div>
              <span>Generalization</span>
              <h3>Measure the clean split gap.</h3>
              <code>G<sub>b</sub> = L<sub>audit,b</sub> - L<sub>train,b</sub></code>
              <p>Repeated splits expose optimism and refit variability. Preprocessing is learned inside each training fold.</p>
            </article>
            <article>
              <div className="method-number">02</div>
              <span>Robustness</span>
              <h3>Trace the full severity path.</h3>
              <code>R<sub>j</sub>(λ) = retained skill</code>
              <p>Feature noise and missingness hold the model fixed. Label corruption and train-size tests refit it.</p>
            </article>
            <article>
              <div className="method-number">03</div>
              <span>Falsification</span>
              <h3>Try to break the signal.</h3>
              <code>p = (1 + exceedances) / (B + 1)</code>
              <p>Permutation nulls and leakage checks ask whether apparent performance survives a deliberately hostile control.</p>
            </article>
          </div>

          <div className="scope-grid">
            <div className="scope-card">
              <div className="eyebrow">Why no diffusion in v0.1</div>
              <h3>A generator adds another fitted model to validate.</h3>
              <p>Diffusion can create plausible tabular replicas, but it also introduces fidelity, mode-collapse, copying, and target-leakage risks. The first release uses transparent stress operators and empirical resampling. Conditional copulas and TabDDPM remain optional research backends only after independent fidelity and authenticity checks.</p>
            </div>
            <div className="scope-card dark">
              <div className="eyebrow">Claim boundary</div>
              <h3>Passing the suite means surviving this audit.</h3>
              <p>It does not establish the absence of overfitting, arbitrary-shift robustness, causal validity, fairness, or privacy. Those require their own designs and source data.</p>
              <ul><li>i.i.d. tabular v1</li><li>binary + regression</li><li>group/time splits in Python roadmap</li></ul>
            </div>
          </div>
        </section>

        <section className="paper-section" id="paper">
          <div className="paper-index">SF / 01</div>
          <div className="paper-copy">
            <div className="eyebrow">Technical note</div>
            <h2>Perturbation-response profiling for tabular model generalization</h2>
            <p>The accompanying paper formalizes the protocol, stress operators, paired Monte Carlo summaries, null controls, and known-data-generating-process benchmarks. Every figure is reproduced from a fixed script and configuration.</p>
            <div className="paper-actions">
              <a className="button button-primary" href="/paper/stressfold.pdf" target="_blank" rel="noreferrer">Read the paper</a>
              <a className="button button-quiet" href="/paper/stressfold.tex" target="_blank" rel="noreferrer">LaTeX source</a>
            </div>
          </div>
          <div className="citation-block">
            <span>Suggested citation</span>
            <code>StressFold contributors (2026). StressFold: generalization stress tests for tabular models.</code>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="brand footer-brand"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span>StressFold</span></div>
        <p>Generalization stress tests for tabular models.</p>
        <div><a href="#method">Method</a><a href="#paper">Paper</a><a href="#top">Back to top</a></div>
      </footer>
    </div>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function formatMetric(value: number) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(Math.abs(value) < 0.1 ? 3 : 2);
}

function operatorColor(id: string) {
  if (id === "feature-noise") return "teal";
  if (id === "label-noise") return "amber";
  if (id === "missingness") return "violet";
  return "gold";
}
