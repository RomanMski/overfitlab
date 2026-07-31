/**
 * Render the page once at build time and write a static site into dist/client.
 *
 * The labs all run in the browser, so nothing here needs a server at runtime.
 * This reuses the same worker entry the tests exercise, so the HTML that ships
 * is the HTML the render test asserts against.
 */
import { writeFileSync } from "node:fs";

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

const base = process.env.SITE_BASE_PATH || "/";
let html = await response.text();

// Vite puts everything it emits under the base, with one exception: the font
// plugin writes /assets/_vinext_fonts/... preload links directly and ignores
// the base entirely. On Windows those come out as file:// paths, so this only
// shows up on Linux, which is how it reached CI unnoticed. Patch that prefix
// specifically rather than rewriting URLs in general.
if (base !== "/") {
  html = html.replace(/(href|src)="\/assets\//g, `$1="${base}assets/`);
  // The same plugin also writes an inline @font-face block, where the paths
  // sit in url(...) rather than in an attribute.
  html = html.replace(/url\(\/assets\//g, `url(${base}assets/`);
}

if (base !== "/" && /url\(\/assets\//.test(html)) {
  throw new Error("a font url() escaped the base and the page would load fallback type");
}

// Then assert, rather than trust, that nothing absolute escaped the base. A
// page whose scripts 404 still renders and still looks correct, right up until
// someone touches a control and nothing happens, which is how this shipped
// broken the first time.
const stray = [...html.matchAll(/(?:href|src)="(\/[^"]*)"/g)]
  .map((match) => match[1])
  .filter((url) => !url.startsWith(base));

if (stray.length > 0) {
  const list = stray.join("\n  ");
  throw new Error(`these URLs sit outside ${base} and would 404 once live:\n  ${list}`);
}

writeFileSync(new URL("index.html", clientDir), html, "utf8");
// GitHub Pages runs Jekyll by default, which drops paths beginning with _.
writeFileSync(new URL(".nojekyll", clientDir), "", "utf8");
// One page with hash anchors, so unknown paths should land on the same document.
writeFileSync(new URL("404.html", clientDir), html, "utf8");

console.log(`prerendered ${html.length} bytes under base ${base}`);
