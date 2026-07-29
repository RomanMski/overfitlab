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

test("server-renders the StressFold research instrument", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>StressFold - Generalization stress tests for tabular models<\/title>/i);
  assert.match(html, /Does your model still work when the data gets slightly worse/);
  assert.match(html, /Build the worked example/);
  assert.match(html, /A stress profile is evidence, not a certificate/);
  assert.match(html, /Five ideas, no statistical shorthand required/);
  assert.match(html, /one roll of the dice/i);
  assert.match(html, /What it cannot prove/);
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
  assert.match(layout, /StressFold - Generalization stress tests/);
  assert.doesNotMatch(page, /codex-preview|SkeletonPreview/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
  await access(new URL("../app/components/StressFoldApp.tsx", import.meta.url));
  await access(new URL("../app/lib/analysis.ts", import.meta.url));
  await access(new URL(".openai/hosting.json", repositoryRoot));
});
