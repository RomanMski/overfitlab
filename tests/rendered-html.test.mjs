import assert from "node:assert/strict";
import test from "node:test";

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

test("server-renders all four labs and the package section", async () => {
  const response = await render();
  assert.equal(response.status, 200);

  const html = await response.text();

  // The tool view is what renders on load. Concepts sit behind a tab, so their
  // headings are present in the markup but the tool must come first.
  assert.match(html, /How much of your backtest is the search\?/);
  assert.match(html, /Make versions of your history to test against/);
  assert.match(html, /Or hand over your trial results/);
  assert.match(html, /Or from Python/);
  assert.match(html, /id="generate"/);
  assert.match(html, /write_datasets/);
  assert.match(html, /Drop a CSV holding a price or return column/);
  assert.match(html, /Generate and download/);

  // Both tabs must reach the browser.
  assert.match(html, /The tool<\/button>/);
  assert.match(html, /Concepts<\/button>/);

  assert.ok(
    html.indexOf("Make versions of your history") < html.indexOf("Or from Python"),
    "the generator should come before the package section",
  );

  assert.doesNotMatch(html, /—/);

  // Nothing from the tabular project should survive in the visible copy.
  // Style and script blocks are stripped first, because "tabular-nums" is a
  // legitimate CSS font-variant and would otherwise match forever.
  const copy = html
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ");
  for (const stale of ["scikit-learn", "holdout", "estimator", "train-audit"]) {
    assert.ok(
      !copy.toLowerCase().includes(stale),
      `the rendered page still mentions "${stale}" from the tabular project`,
    );
  }
});
