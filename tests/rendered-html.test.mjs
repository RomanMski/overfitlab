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

test("server-renders both labs and the package section", async () => {
  const response = await render();
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /How much of your backtest is the search\?/);
  assert.match(html, /What overfitting actually is/);
  assert.match(html, /How searching manufactures a strategy/);
  assert.match(html, /Measure it on your own trials/);
  assert.match(html, /id="overfit"/);
  assert.match(html, /id="search"/);
  assert.match(html, /probability_of_backtest_overfitting/);
  assert.match(html, /Nothing on this screen has any edge/);

  // The explainer must reach the browser with a fitted curve already drawn,
  // rather than an empty frame that only fills in after hydration.
  assert.match(html, /Model flexibility/);
  assert.match(html, /Configurations you tried/);

  assert.ok(
    html.indexOf("What overfitting actually is") <
      html.indexOf("How searching manufactures a strategy"),
    "the plain explanation should come before the backtesting one",
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
