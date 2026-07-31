/**
 * Render the page once at build time and write a static site into dist/client.
 *
 * The labs all run in the browser, so nothing here needs a server at runtime.
 * This reuses the same worker entry the tests exercise, so the HTML that ships
 * is the HTML the render test asserts against.
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";

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

// The font plugin writes /assets/_vinext_fonts/... into whichever built file
// carries the @font-face rule, and it ignores the base. Which file that is
// differs by platform: on Windows the fonts inline as file:// paths and this
// never appears, on Linux it lands in the emitted CSS. So sweep every built
// text file rather than guessing which one. Missing this only costs the custom
// typeface, but a 404 on a deployed site is still a defect.
if (base !== "/") {
  const needle = "/assets/_vinext_fonts/";
  let patched = 0;

  const sweep = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const child = new URL(`${entry.name}${entry.isDirectory() ? "/" : ""}`, dir);
      if (entry.isDirectory()) {
        sweep(child);
      } else if (/\.(css|js|html)$/.test(entry.name)) {
        const body = readFileSync(child, "utf8");
        if (body.includes(needle)) {
          writeFileSync(child, body.replaceAll(needle, `${base}assets/_vinext_fonts/`), "utf8");
          patched += 1;
        }
      }
    }
  };

  sweep(clientDir);
  console.log(`patched font paths in ${patched} built file(s)`);
}

writeFileSync(new URL("index.html", clientDir), html, "utf8");
// GitHub Pages runs Jekyll by default, which drops paths beginning with _.
writeFileSync(new URL(".nojekyll", clientDir), "", "utf8");
// One page with hash anchors, so unknown paths should land on the same document.
writeFileSync(new URL("404.html", clientDir), html, "utf8");

console.log(`prerendered ${html.length} bytes under base ${base}`);
