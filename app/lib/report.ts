import type { AuditResult } from "./analysis";

export function buildHtmlReport(result: AuditResult): string {
  const retainedSkillReliable = result.baseline.retainedSkillReliable;
  const rows = result.summaries
    .map(
      (summary) => `
        <tr>
          <td>${escapeHtml(summary.label)}</td>
          <td>${escapeHtml(summary.mode)}</td>
          <td>${retainedSkillReliable ? summary.degradationArea.toFixed(3) : "not interpretable"}</td>
          <td>${retainedSkillReliable ? summary.firstStepLoss.toFixed(3) : "not interpretable"}</td>
          <td>${retainedSkillReliable ? escapeHtml(summary.halfSkillAt) : "not interpretable"}</td>
        </tr>`,
    )
    .join("");
  const findings = result.findings
    .map(
      (finding) => `
        <article class="finding ${finding.status}">
          <div class="eyebrow">${escapeHtml(finding.eyebrow)}</div>
          <h3>${escapeHtml(finding.title)}</h3>
          <p>${escapeHtml(finding.detail)}</p>
        </article>`,
    )
    .join("");
  const warnings = result.warnings.length
    ? `<section><h2>Run notes</h2><ul>${result.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul></section>`
    : "";

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StressFold audit - ${escapeHtml(result.dataset.name)}</title>
  <style>
    :root{--paper:#f5f2e9;--card:#fffefa;--ink:#172427;--muted:#626966;--line:#d6d2c5;--teal:#1f5c63;--amber:#b56a28;--violet:#7c5a8d}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}.page{max-width:1080px;margin:0 auto;padding:54px 28px 80px}.eyebrow{font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}h1{font-size:44px;line-height:1.04;letter-spacing:-.04em;margin:12px 0 14px}h2{font-size:22px;margin:44px 0 14px}h3{font-size:17px;margin:8px 0}.lede{max-width:760px;color:var(--muted);font-size:18px}.meta{display:flex;flex-wrap:wrap;gap:10px;margin:24px 0 34px}.tag{border:1px solid var(--line);background:var(--card);padding:7px 10px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.metric{background:var(--card);border:1px solid var(--line);padding:20px}.metric strong{display:block;font:30px/1 ui-monospace,SFMono-Regular,Menlo,monospace;margin:8px 0}.metric span{color:var(--muted);font-size:12px}.findings{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.finding{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--teal);padding:20px}.finding.watch{border-top-color:var(--amber)}.finding.warning{border-top-color:var(--violet)}.finding p{color:var(--muted);margin-bottom:0}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line)}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line)}th{font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}td:not(:first-child){font-family:ui-monospace,SFMono-Regular,Menlo,monospace}footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:12px}@media(max-width:760px){.metrics,.findings{grid-template-columns:1fr 1fr}h1{font-size:36px}}@media print{body{background:white}.page{padding:24px}.metric,.finding,table{break-inside:avoid}}
  </style>
</head>
<body><main class="page">
  <div class="eyebrow">StressFold / generalization stress audit</div>
  <h1>${escapeHtml(result.dataset.name)}</h1>
  <p class="lede">Controlled perturbation-response profiling with repeated paired splits. The report separates generalization, robustness, and falsification evidence; it does not claim to prove the absence or presence of overfitting.</p>
  <div class="meta">
    <span class="tag">${result.dataset.rows} rows</span>
    <span class="tag">${result.dataset.features.length} numeric features</span>
    <span class="tag">target: ${escapeHtml(result.dataset.target)}</span>
    <span class="tag">${result.protocol.repeats} repeated splits</span>
    <span class="tag">seed ${result.protocol.seed}</span>
  </div>
  <section class="metrics">
    <div class="metric"><span>Median ${result.baseline.scoreLabel}</span><strong>${result.baseline.score.toFixed(3)}</strong><span>Clean audit folds</span></div>
    <div class="metric"><span>Train-audit loss gap</span><strong>${result.baseline.gap.toFixed(3)}</strong><span>${result.baseline.lossLabel}</span></div>
    <div class="metric"><span>Split variability</span><strong>${result.baseline.splitSpread.toFixed(3)}</strong><span>5th-95th score span</span></div>
    <div class="metric"><span>Null midrank</span><strong>${result.permutation.percentile.toFixed(0)} / 100</strong><span>${result.permutation.runs} quick null refits; descriptive, not a p-value</span></div>
  </section>
  <h2>Interpretation</h2><section class="findings">${findings}</section>
  <h2>Stress-response summaries</h2>
  ${retainedSkillReliable ? "" : '<p class="lede">Normalized curve rankings are withheld because the clean baseline did not reliably beat its constant reference. Inspect raw losses and model specification before comparing stressors.</p>'}
  <table><thead><tr><th>Operator</th><th>Experiment</th><th>Degradation area</th><th>First-step loss</th><th>50% skill boundary</th></tr></thead><tbody>${rows}</tbody></table>
  ${warnings}
  <footer>Source ${escapeHtml(result.protocol.sourceHash)}. Generated ${escapeHtml(result.protocol.generatedAt)} with ${escapeHtml(result.protocol.browserEngine)}. Monte Carlo intervals describe variation between runs under this protocol; they are not classical confidence intervals.</footer>
</main></body></html>`;
}

function escapeHtml(value: string) {
  return value.replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[character] ?? character);
}
