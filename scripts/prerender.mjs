/**
 * Render the page once at build time and write a static site into dist/client.
 *
 * The labs all run in the browser, so nothing here needs a server at runtime.
 * This reuses the same worker entry the tests exercise, so the HTML that ships
 * is the HTML the render test asserts against.
 *
 * Asset URLs are rewritten from absolute to relative because GitHub Pages
 * serves the project from a /<repo>/ subpath, where /assets/... would 404.
 */
import { writeFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("../", import.meta.url);
const clientDir = new URL("dist/client/", root);

const workerUrl = new URL("dist/server/index.js", root);
const { default: worker } = await import(workerUrl.href);

const response = await worker.fetch(
  new Request("http://localhost/", { headers: { accept: "text/html" } }),
  { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
  { waitUntil() {}, passThroughOnException() {} },
);

if (response.status !== 200) {
  throw new Error(`prerender got HTTP ${response.status}`);
}

let html = await response.text();
const before = html;
html = html
  .replace(/(href|src)="\/(assets\/)/g, '$1="./$2')
  .replace(/(href|src)="\/(favicon\.svg|og\.png)"/g, '$1="./$2"');

if (html === before) {
  throw new Error("no absolute asset URLs were rewritten, check the output shape");
}
if (/(href|src)="\/[a-z]/.test(html)) {
  throw new Error("an absolute URL survived the rewrite and would 404 on Pages");
}

writeFileSync(new URL("index.html", clientDir), html, "utf8");
// GitHub Pages runs Jekyll by default, which drops files beginning with _.
writeFileSync(new URL(".nojekyll", clientDir), "", "utf8");
// A single page app with hash anchors: send unknown paths to the same document.
writeFileSync(new URL("404.html", clientDir), html, "utf8");

console.log(`prerendered ${html.length} bytes into dist/client/index.html`);
