"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  generateStressVariant,
  inferTarget,
  inferTask,
  makeSampleDataset,
  runAudit,
  type AuditFinding,
  type AuditResult,
  type AuditSettings,
  type DataTable,
  type ModelKind,
  type TaskType,
} from "../lib/analysis";
import { downloadText, parseCsv, tableToCsv } from "../lib/csv";
import { buildHtmlReport } from "../lib/report";
import { ConceptExplainers } from "./ConceptExplainers";
import { FormulaGuide } from "./FormulaGuide";
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

interface TargetInfo {
  valid: boolean;
  task: TaskType;
  description: string;
}

function describeTarget(table: DataTable, header: string): TargetInfo {
  const values = table.rows
    .map((row) => row[header])
    .filter((value): value is string | number => value !== null && value !== "");
  const unique = new Set(values.map(String));
  const numericShare = values.length
    ? values.filter((value) => Number.isFinite(Number(value))).length / values.length
    : 0;
  const looksLikeIdentifier =
    /(^|_)(id|uuid|key|index)($|_)/i.test(header) && unique.size / Math.max(values.length, 1) > 0.9;

  if (values.length < 40) {
    return { valid: false, task: "regression", description: "too few filled rows" };
  }
  if (unique.size < 2) {
    return { valid: false, task: "regression", description: "only one value" };
  }
  if (looksLikeIdentifier) {
    return { valid: false, task: "regression", description: "looks like an identifier" };
  }
  if (unique.size === 2) {
    return { valid: true, task: "classification", description: "two possible outcomes" };
  }
  if (numericShare >= 0.95) {
    return { valid: true, task: "regression", description: "number to predict" };
  }
  return {
    valid: false,
    task: "classification",
    description: "browser demo needs two classes or a number",
  };
}

function chooseSupportedTarget(table: DataTable) {
  const preferred = inferTarget(table);
  if (preferred && describeTarget(table, preferred).valid) return preferred;
  return [...table.headers].reverse().find((header) => describeTarget(table, header).valid) ?? "";
}

function makeRunKey(tableVersion: number, settings: AuditSettings) {
  return JSON.stringify([tableVersion, ...Object.values(settings)]);
}

export function StressFoldApp() {
  const [table, setTable] = useState<DataTable>(SAMPLE_TABLE);
  const [tableVersion, setTableVersion] = useState(0);
  const [target, setTarget] = useState(DEFAULT_SETTINGS.target);
  const [task, setTask] = useState<TaskType>(DEFAULT_SETTINGS.task);
  const [model, setModel] = useState<ModelKind>(DEFAULT_SETTINGS.model);
  const [repeats, setRepeats] = useState(DEFAULT_SETTINGS.repeats);
  const [testSize, setTestSize] = useState(DEFAULT_SETTINGS.testSize);
  const [seed, setSeed] = useState(DEFAULT_SETTINGS.seed);
  const [result, setResult] = useState<AuditResult | null>(null);
  const [lastRunKey, setLastRunKey] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressLabel, setProgressLabel] = useState("Preparing the worked example");
  const [error, setError] = useState<string | null>(null);
  const [variantFamily, setVariantFamily] = useState<VariantFamily>("feature-noise");
  const [variantLevel, setVariantLevel] = useState(0.25);
  const initialRunStarted = useRef(false);
  const latestRun = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const settings = useMemo<AuditSettings>(
    () => ({ target, task, model, repeats, testSize, seed }),
    [target, task, model, repeats, testSize, seed],
  );
  const currentRunKey = useMemo(
    () => makeRunKey(tableVersion, settings),
    [tableVersion, settings],
  );
  const changesNotApplied = Boolean(result && lastRunKey !== currentRunKey);
  const targetOptions = useMemo(
    () => table.headers.map((header) => ({ header, ...describeTarget(table, header) })),
    [table],
  );
  const previewHeaders = useMemo(() => {
    const first = table.headers.filter((header) => header !== target).slice(0, 3);
    return [...first, target].filter(Boolean);
  }, [table, target]);

  useEffect(() => {
    if (initialRunStarted.current) return;
    initialRunStarted.current = true;
    void executeAudit(SAMPLE_TABLE, DEFAULT_SETTINGS, makeRunKey(0, DEFAULT_SETTINGS));
    // This is the one automatic run: a complete example should be visible without setup.
  }, []);

  async function executeAudit(data: DataTable, auditSettings: AuditSettings, auditKey: string) {
    const runNumber = latestRun.current + 1;
    latestRun.current = runNumber;
    setRunning(true);
    setProgress(0);
    setProgressLabel("Preparing repeated unseen-data tests");
    setError(null);
    try {
      const nextResult = await runAudit(data, auditSettings, (fraction, label) => {
        if (latestRun.current !== runNumber) return;
        setProgress(fraction);
        setProgressLabel(label.replace("Repeated split", "Unseen-data repeat"));
      });
      if (latestRun.current !== runNumber) return;
      setResult(nextResult);
      setLastRunKey(auditKey);
    } catch (caught) {
      if (latestRun.current !== runNumber) return;
      setError(caught instanceof Error ? caught.message : "The test could not be completed.");
    } finally {
      if (latestRun.current === runNumber) {
        setRunning(false);
        setProgress(1);
      }
    }
  }

  async function handleFile(file: File) {
    if (file.size > 5 * 1024 * 1024) {
      setError("This browser demo accepts CSV files up to 5 MB. Use the Python package for larger data.");
      return;
    }
    try {
      const parsed = parseCsv(await file.text(), file.name);
      const nextTarget = chooseSupportedTarget(parsed);
      if (!nextTarget) {
        throw new Error(
          "Choose a CSV with at least one usable outcome: either two categories or a numeric column with 40 filled rows.",
        );
      }
      setTable(parsed);
      setTableVersion((version) => version + 1);
      setTarget(nextTarget);
      setTask(describeTarget(parsed, nextTarget).task);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The CSV could not be read.");
    }
  }

  function restoreSample() {
    const nextVersion = tableVersion + 1;
    setTable(SAMPLE_TABLE);
    setTableVersion(nextVersion);
    setTarget(DEFAULT_SETTINGS.target);
    setTask(DEFAULT_SETTINGS.task);
    setModel(DEFAULT_SETTINGS.model);
    setRepeats(DEFAULT_SETTINGS.repeats);
    setTestSize(DEFAULT_SETTINGS.testSize);
    setSeed(DEFAULT_SETTINGS.seed);
    setError(null);
    void executeAudit(
      SAMPLE_TABLE,
      DEFAULT_SETTINGS,
      makeRunKey(nextVersion, DEFAULT_SETTINGS),
    );
  }

  function applyPreset(preset: "quick" | "careful") {
    setRepeats(preset === "quick" ? 8 : 24);
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

  const resultHash = result?.protocol.sourceHash.replace("fnv1a-", "") ?? "working";
  const weakest = result
    ? [...result.summaries].sort((left, right) => right.degradationArea - left.degradationArea)[0]
    : null;
  const headline = result && weakest ? buildResultHeadline(result, weakest.label) : "";
  const visibleWarnings = result?.warnings.filter(
    (warning) => !warning.startsWith("Identifier columns excluded from modeling:"),
  ) ?? [];

  return (
    <div className="site-frame">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="StressFold home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>StressFold</span>
        </a>
        <nav className="primary-nav" aria-label="Primary navigation">
          <a href="#lab">Use the lab</a>
          <a href="#python">Use your model</a>
          <a href="#math">Mathematics</a>
          <a href="#paper">Paper</a>
        </nav>
        <div className="header-status"><span /> v0.2.0 alpha</div>
      </header>

      <main id="top">
        <section className="hero section-boundary">
          <div className="hero-copy">
            <div className="kicker"><span>Generalization stress tests</span><span>Tabular models</span></div>
            <h1>Test a model beyond its training data.</h1>
            <p className="hero-lede">
              StressFold measures the gap between training and unseen data. It then checks what happens under measurement noise, missing values, wrong labels, and smaller training sets.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#lab">Open the browser lab <span aria-hidden="true">→</span></a>
              <a className="button button-quiet" href="#python">Use the Python package</a>
            </div>
            <div className="trust-row" aria-label="Product properties">
              <span>Use a CSV in the browser</span>
              <span>Bring your model in Python</span>
              <span>Repeatable results</span>
            </div>
          </div>
          <div className="protocol-figure" aria-label="A four-step model stress test">
            <div className="figure-label">StressFold protocol</div>
            <div className="protocol-nodes">
              <div className="protocol-node active"><b>01</b><span>Split</span><small>Make a train and audit split</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>02</b><span>Fit</span><small>Use training rows only</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>03</b><span>Stress</span><small>Apply one perturbation</small></div>
              <div className="protocol-link"><i /><i /><i /></div>
              <div className="protocol-node"><b>04</b><span>Repeat</span><small>Use many new folds</small></div>
            </div>
            <div className="figure-output">
              <div><span>The output</span><strong>A profile of where performance fails</strong></div>
              <div className="mini-bars" aria-hidden="true">
                <i style={{ height: "86%" }} /><i style={{ height: "74%" }} /><i style={{ height: "48%" }} /><i style={{ height: "25%" }} />
              </div>
            </div>
            <p>Repeated train and audit splits under controlled stress. This is what the name refers to.</p>
          </div>
        </section>

        <section className="lab-section" id="lab">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Interactive browser lab</div>
              <h2>Run a generalization and robustness audit.</h2>
            </div>
            <p>
              The browser demonstrates the protocol with a transparent baseline model. The Python package runs related repeated-holdout diagnostics on your own scikit-learn compatible pipeline.
            </p>
          </div>

          <div className="lab-workspace">
            <aside className="control-panel" id="audit-controls" aria-label="Test setup">
              <div className="panel-heading">
                <div><span>Dataset and model</span><strong>Set up a test</strong></div>
                <span className="step-counter">Local only</span>
              </div>

              <fieldset className="setup-fields" disabled={running}>
                <legend className="visually-hidden">Dataset and model settings</legend>
                <div className="dataset-card">
                  <div className="file-glyph" aria-hidden="true"><span>CSV</span></div>
                  <div className="dataset-meta">
                    <strong title={table.name}>{table.name}</strong>
                    <span>{table.rows.length.toLocaleString()} rows, {table.headers.length} columns</span>
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
                    <button type="button" onClick={() => fileInputRef.current?.click()}>Test a baseline on my CSV</button>
                  <button type="button" onClick={restoreSample}>Restore example</button>
                </div>

                <details className="dataset-preview">
                  <summary>Preview the first five rows</summary>
                  <div className="dataset-preview__scroll">
                    <table>
                      <thead><tr>{previewHeaders.map((header) => <th key={header}>{header}</th>)}</tr></thead>
                      <tbody>
                        {table.rows.slice(0, 5).map((row, index) => (
                          <tr key={index}>{previewHeaders.map((header) => <td key={header}>{String(row[header] ?? "missing")}</td>)}</tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>

                <div className="control-group">
                  <label htmlFor="target-column">What should the model predict?</label>
                  <select
                    id="target-column"
                    value={target}
                    onChange={(event) => {
                      const nextTarget = event.target.value;
                      setTarget(nextTarget);
                      setTask(describeTarget(table, nextTarget).task);
                    }}
                  >
                    {targetOptions.map((option) => (
                      <option key={option.header} value={option.header} disabled={!option.valid}>
                        {option.header} ({option.description})
                      </option>
                    ))}
                  </select>
                  <small>
                    Detected automatically: {task === "classification" ? "a two-outcome prediction" : "a number prediction"}.
                  </small>
                </div>

                <div className="control-group">
                  <label htmlFor="model-kind">Which baseline model?</label>
                  <select id="model-kind" value={model} onChange={(event) => setModel(event.target.value as ModelKind)}>
                    <option value="regularized">Steady baseline (regularized linear model)</option>
                    <option value="nearest-neighbor">Flexible baseline (nearest neighbors)</option>
                  </select>
                  <small>This choice tests a built-in baseline, not a model already running elsewhere.</small>
                </div>

                <div className="preset-row" aria-label="Repeat preset">
                  <button type="button" onClick={() => applyPreset("quick")}>Quick, 8 repeats</button>
                  <button type="button" onClick={() => applyPreset("careful")}>Careful, 24 repeats</button>
                </div>

                <details className="advanced-controls">
                  <summary>Advanced repeat settings</summary>
                  <div className="inline-controls">
                    <div className="control-group compact">
                      <label htmlFor="repeat-count">Repeats</label>
                      <input id="repeat-count" type="number" min="4" max="40" step="1" value={repeats} onChange={(event) => setRepeats(Number(event.target.value))} />
                    </div>
                    <div className="control-group compact">
                      <label htmlFor="test-size">Rows kept unseen</label>
                      <select id="test-size" value={testSize} onChange={(event) => setTestSize(Number(event.target.value))}>
                        <option value="0.2">20%</option><option value="0.25">25%</option><option value="0.3">30%</option><option value="0.35">35%</option>
                      </select>
                    </div>
                    <div className="control-group compact">
                      <label htmlFor="seed">Random seed</label>
                      <input id="seed" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                    </div>
                  </div>
                </details>
              </fieldset>

              <button
                className="run-button"
                type="button"
                disabled={running}
                onClick={() => void executeAudit(table, settings, currentRunKey)}
              >
                <span>{running ? "Running repeated audit splits" : changesNotApplied ? "Run updated test" : result ? "Run this test again" : "Run the sample audit"}</span>
                <b aria-hidden="true">{running ? `${Math.round(progress * 100)}%` : "→"}</b>
              </button>
              {running && <div className="progress-block" role="status"><i style={{ width: `${progress * 100}%` }} /><span>{progressLabel}</span></div>}
              {error && <div className="error-message" role="alert"><strong>We could not run that setup.</strong><br />{error}</div>}
              <p className="privacy-note"><span aria-hidden="true">◇</span> The CSV never leaves this browser.</p>
            </aside>

            <section className="result-panel" aria-live="polite" aria-busy={running}>
              {result && changesNotApplied && (
                <div className="changes-banner" role="status">
                  <div><strong>{running ? "Running your updated choices" : "Changes not applied yet"}</strong><span>The visible result still describes {result.dataset.target} in {result.dataset.name}.</span></div>
                  {!running && <button type="button" onClick={() => void executeAudit(table, settings, currentRunKey)}>Run updated test</button>}
                </div>
              )}

              <div className="result-toolbar">
                <div>
                  <span className="result-state"><i className={result ? "ready" : ""} /> {result ? "Audit result" : running ? "Running the audit" : "Audit queued"}</span>
                  <strong>{result ? `${result.dataset.rows.toLocaleString()} usable rows, predicting ${result.dataset.target}` : "Preparing the sample churn audit"}</strong>
                </div>
                <div className="hash-label">RUN / {resultHash.toUpperCase()}</div>
              </div>

              {result ? (
                <>
                  <div className="result-answer">
                    <span>Result summary</span>
                    <h3>{headline}</h3>
                    <p>
                      The model was fitted without seeing the audit rows. Every stress below changes one thing at a time, then compares the result with the same clean split.
                    </p>
                  </div>

                  {visibleWarnings.length > 0 && (
                    <aside className="audit-warning-strip" aria-label="Checks required before interpretation">
                      <strong>Check before interpreting</strong>
                      <ul>{visibleWarnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
                    </aside>
                  )}

                  <div className="run-context" aria-label="Settings used for the visible result">
                    <span>Target <strong>{result.dataset.target}</strong></span>
                    <span>Browser model <strong>{modelName(result.protocol.model)}</strong></span>
                    <span>Repeats <strong>{result.protocol.repeats}</strong></span>
                    <span>Unseen rows <strong>{Math.round(result.protocol.testSize * 100)}%</strong></span>
                    <span>Seed <strong>{result.protocol.seed}</strong></span>
                  </div>

                  <div className="metric-grid">
                    <Metric
                      label="Performance on unseen rows"
                      value={formatMetric(result.baseline.score)}
                      note={result.baseline.scoreLabel === "AUROC" ? "AUROC: 1 is perfect; 0.5 is chance" : "R²: 1 is perfect; 0 matches predicting the average"}
                    />
                    <Metric
                      label="Extra error on unseen rows"
                      value={formatMetric(result.baseline.gap)}
                      note={`${result.baseline.lossLabel}: median of paired unseen minus training gaps`}
                    />
                    <Metric
                      label="Dependence on the split"
                      value={formatMetric(result.baseline.splitSpread)}
                      note="5th to 95th percentile spread; smaller is steadier"
                    />
                    <Metric
                      label="Real score versus shuffled median"
                      value={`${formatMetric(result.permutation.observed)} vs ${formatMetric(result.permutation.nullMedian)}`}
                      note={`${result.baseline.scoreLabel}; ${result.permutation.runs} shuffled refits; tie-adjusted rank ${Math.round(result.permutation.percentile)} / 100`}
                    />
                  </div>

                  <div className="result-section chart-section">
                    <div className="result-section-heading">
                      <div><span>Stress response</span><h3>Performance retained under each stress</h3></div>
                      <div className="method-chip">median, 5th to 95th percentile</div>
                    </div>
                    <StressChart curves={result.curves} />
                  </div>

                  <div className="finding-grid" aria-label="Three audit conclusions">
                    {result.findings.map((finding) => (
                      <article className={`finding-card ${finding.status}`} key={finding.kind}>
                        <div><span>{plainFindingLabel(finding.kind)}</span><i /></div>
                        <h3>{finding.title}</h3>
                        <p>{plainFindingDetail(finding, result)}</p>
                      </article>
                    ))}
                  </div>

                  <details className="advanced-results">
                    <summary>Advanced boundaries and protocol notes</summary>
                    <p>These summaries are useful for comparing runs. They are not combined into one made-up “robustness score.”</p>
                    <div className="summary-table-wrap">
                      <table className="summary-table">
                        <thead><tr><th>Stress test</th><th>What changes</th><th>Normalized curve area</th><th>First tested step loss</th><th>First tested level at or below 50%</th></tr></thead>
                        <tbody>
                          {result.summaries.map((summary) => (
                            <tr key={summary.id}>
                              <td>{summary.label}</td><td>{summary.mode}</td><td>{result.baseline.retainedSkillReliable ? summary.degradationArea.toFixed(3) : "not interpretable"}</td><td>{result.baseline.retainedSkillReliable ? summary.firstStepLoss.toFixed(3) : "not interpretable"}</td><td>{result.baseline.retainedSkillReliable ? summary.halfSkillAt : "not interpretable"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {result.warnings.length > 0 && <ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
                  </details>

                  <div className="export-panel">
                    <div className="export-heading">
                      <div><span>Export a stress case</span><h3>Create a perturbed CSV</h3></div>
                      <p>Use the exported CSV to challenge your own model. It is a test case. It is not extra training data or new real world evidence.</p>
                    </div>
                    <div className="export-controls">
                      <select aria-label="Damage type" value={variantFamily} onChange={(event) => handleFamilyChange(event.target.value as VariantFamily)}>
                        <option value="feature-noise">Add Gaussian feature noise</option>
                        <option value="label-noise">Corrupt training targets</option>
                        <option value="missingness">Mask input cells</option>
                        <option value="bootstrap">Resample rows with replacement</option>
                      </select>
                      <select aria-label="Damage severity" value={variantLevel} disabled={variantFamily === "bootstrap"} onChange={(event) => setVariantLevel(Number(event.target.value))}>
                        {VARIANT_LEVELS[variantFamily].map((level) => <option key={level} value={level}>{variantFamily === "bootstrap" ? "row resample" : `${Math.round(level * 100)}% severity`}</option>)}
                      </select>
                      <button type="button" onClick={() => downloadVariant("csv")}>Download test CSV</button>
                      <button type="button" onClick={() => downloadVariant("manifest")}>Download recipe</button>
                    </div>
                    <div className="report-actions">
                      <button type="button" onClick={() => downloadText(buildHtmlReport(result), `stressfold_${resultHash}.html`, "text/html;charset=utf-8")}>Download report</button>
                      <button type="button" onClick={() => downloadText(JSON.stringify(result, null, 2), `stressfold_${resultHash}.json`, "application/json;charset=utf-8")}>Download result JSON</button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="empty-result">
                  <div className="empty-orbit" aria-hidden="true"><i /><i /><i /></div>
                  <h3>{error ? "This setup needs one fix" : "Running the sample audit"}</h3>
                  <p>{error ?? "StressFold is fitting the baseline model, reserving unseen rows, applying four controlled stresses, and repeating the split."}</p>
                </div>
              )}
            </section>
          </div>
        </section>

        <section className="method-section section-boundary" id="python">
          <div className="section-heading compact-heading">
            <div><div className="eyebrow">Python library</div><h2>Run the protocol on your actual pipeline.</h2></div>
            <p>The browser uses transparent baselines so every step is easy to inspect. The Python package accepts the estimator and preprocessing you actually use.</p>
          </div>
          <div className="scope-grid">
            <div className="scope-card">
              <div className="eyebrow">Browser lab</div>
              <h3>Inspect the method on a built-in baseline.</h3>
              <p>Upload a CSV, choose a target, and explore every split, stress curve, and shuffled-label comparison locally. This is a protocol demonstration, not a silent claim about a deployed model.</p>
            </div>
            <div className="scope-card dark">
              <div className="eyebrow">Python package</div>
              <h3>Audit the model and preprocessing you actually use.</h3>
              <p>Pass a scikit-learn compatible pipeline to StressFold. Preprocessing, fitting, and every refit stay inside the appropriate training fold.</p>
              <ul><li>binary classification</li><li>regression</li><li>custom pipelines</li></ul>
            </div>
          </div>
        </section>

        <FormulaGuide />

        <ConceptExplainers />

        <section className="paper-section" id="paper">
          <div className="paper-index">SF / 02</div>
          <div className="paper-copy">
            <div className="eyebrow">Technical note</div>
            <h2>Perturbation-response profiling for tabular model generalization</h2>
            <p>The paper turns established validation ideas into one reproducible protocol and uses controlled counterexamples to show why their outputs must stay separate. It does not claim a new theorem. The browser lab implements a narrower four-stressor subset; the Mathematics section above gives its exact formulas.</p>
            <div className="paper-actions">
              <a className="button button-primary" href="/paper/stressfold.pdf" target="_blank" rel="noreferrer">Open PDF preview</a>
              <a className="button button-quiet" href="/paper/stressfold.tex" target="_blank" rel="noreferrer">LaTeX source (.tex)</a>
            </div>
          </div>
          <div className="citation-block">
            <span>Suggested citation</span>
            <code>StressFold contributors (2026). StressFold: generalization stress tests for tabular models. Version 0.2.0.</code>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="brand footer-brand"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span>StressFold</span></div>
        <p>Controlled model stress tests, with the limits left visible.</p>
        <div><a href="#lab">Lab</a><a href="#python">Python</a><a href="#math">Mathematics</a><a href="#learn">Visual notes</a><a href="#paper">Paper</a><a href="#top">Back to top</a></div>
      </footer>
    </div>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function formatMetric(value: number) {
  if (!Number.isFinite(value)) return "not available";
  return value.toFixed(Math.abs(value) < 0.1 ? 3 : 2);
}

function modelName(model: ModelKind) {
  return model === "regularized" ? "steady linear baseline" : "flexible neighbor baseline";
}

function buildResultHeadline(result: AuditResult, weakestLabel: string) {
  const leakageWarning = result.warnings.some((warning) =>
    warning.includes("Random row splits can place the same entity") || warning.includes("nearly determines the target"),
  );
  if (leakageWarning) {
    return `${result.baseline.scoreLabel} on unseen rows is ${formatMetric(result.baseline.score)}, but possible split leakage must be checked before interpreting it.`;
  }
  if (!result.baseline.retainedSkillReliable) {
    return `${result.baseline.scoreLabel} on unseen rows is ${formatMetric(result.baseline.score)}. The baseline does not reliably beat a constant predictor, so the normalized stress ranking is withheld.`;
  }
  return `${result.baseline.scoreLabel} on unseen rows is ${formatMetric(result.baseline.score)}. Across the four tested grids, ${weakestLabel.toLowerCase()} causes the largest overall decline.`;
}

function plainFindingLabel(kind: AuditFinding["kind"]) {
  if (kind === "generalization") return "Training versus unseen loss";
  if (kind === "robustness") return "Largest curve decline";
  return "Real labels versus shuffled labels";
}

function plainFindingDetail(finding: AuditFinding, result: AuditResult) {
  if (finding.kind === "generalization") {
    if (finding.title === "Check possible split leakage first") return finding.detail;
    return `Within each repeat, unseen loss minus training loss was computed. The median paired gap was ${formatMetric(result.baseline.gap)}. For context, the separate medians were ${formatMetric(result.baseline.trainLoss)} for training and ${formatMetric(result.baseline.auditLoss)} for unseen rows.`;
  }
  if (finding.kind === "robustness") {
    if (!result.baseline.retainedSkillReliable) return finding.detail;
    const weakest = [...result.summaries].sort((left, right) => right.degradationArea - left.degradationArea)[0];
    return `${weakest.label} has the largest normalized trapezoid area on its tested grid. Open the curve above to inspect the actual severities. This does not claim that unlike stress levels are equally realistic.`;
  }
  return `The real score has a tie-adjusted null rank of ${Math.round(result.permutation.percentile)} on a 0 to 100 scale. Lower null scores count fully, exact ties count one half, and this is not a p-value.`;
}
