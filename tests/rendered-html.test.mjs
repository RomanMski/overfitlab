import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const repositoryRoot = new URL("../", import.meta.url);

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the StressFold tool before its method reference", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>StressFold \| Generalization stress tests for tabular models<\/title>/i);
  assert.match(html, /Test a model beyond its training data/);
  assert.match(html, /Run the sample audit/);
  assert.match(html, /Run the protocol on your actual pipeline/);
  assert.match(html, /Read the audit equations term by term/);
  assert.match(html, /id="math"/);
  assert.match(html, /Stress operators and curve summaries/);
  assert.match(html, /Normalized trapezoid area/);
  assert.match(html, /Browser losses/);
  assert.match(html, /Browser scores/);
  assert.match(html, /Training loss versus unseen loss/);
  assert.match(html, /Move the measurements, not the answers/);
  assert.match(html, /x<sub>i<\/sub>\(\u03bb\) = x<sub>i<\/sub>/);
  assert.match(html, /Predictions changed/);
  assert.match(html, /Test a baseline on my CSV/);
  assert.match(html, /browser lab implements a narrower four-stressor subset/i);
  assert.match(html, /does not claim a new theorem/i);
  assert.match(html, /Open PDF preview/);
  assert.match(html, /LaTeX source \(\.tex\)/);
  assert.match(html, />Limit</);
  assert.ok(
    html.indexOf("Run a generalization and robustness audit") <
      html.indexOf("Read the audit equations term by term"),
    "the working lab should appear before the mathematics reference",
  );
  assert.ok(
    html.indexOf("Run the protocol on your actual pipeline") <
      html.indexOf("Explore how each test behaves"),
    "the Python tool should be explained before the interactive lessons",
  );
  assert.doesNotMatch(html, /—/);
  assert.doesNotMatch(html, /No audit result yet/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("removes all temporary starter-preview infrastructure", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /StressFoldApp/);
  assert.match(layout, /StressFold \| Generalization stress tests/);
  assert.doesNotMatch(page, /codex-preview|SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await access(new URL("../app/components/StressFoldApp.tsx", import.meta.url));
  await access(new URL("../app/lib/analysis.ts", import.meta.url));
  await access(new URL(".openai/hosting.json", repositoryRoot));
});
