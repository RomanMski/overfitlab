#!/usr/bin/env node
// Put the paper where the site can serve it.
//
// Only files under public/ are served, so the site needs a copy of the paper.
// That copy used to be committed, and it drifted. The site was serving a build
// from before the real data section existed while the tracked source had moved
// on, and nothing noticed. So it is generated at build time instead and is not
// in version control, which makes drift impossible rather than detectable.
//
// Runs automatically as the prebuild step. Node rather than Python so that a
// plain npm build needs nothing extra.

import { copyFile, mkdir, access } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const PAIRS = [
  ["paper/main.pdf", "public/paper/overfitlab.pdf"],
  ["paper/main.tex", "public/paper/overfitlab.tex"],
];

let missing = 0;
for (const [from, to] of PAIRS) {
  const source = resolve(root, from);
  const destination = resolve(root, to);
  try {
    await access(source);
  } catch {
    // The PDF is built in CI, so a fresh clone that has never built it should
    // warn rather than fail and take the whole site build down with it.
    console.warn(`sync_paper: ${from} is missing, skipping`);
    missing += 1;
    continue;
  }
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
  console.log(`sync_paper: ${from} -> ${to}`);
}

if (missing === PAIRS.length) {
  console.warn("sync_paper: nothing copied, the paper link will 404");
}
