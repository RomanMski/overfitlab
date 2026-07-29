import assert from "node:assert/strict";
import test from "node:test";
import {
  generateStressVariant,
  inferTarget,
  inferTask,
  makeSampleDataset,
  runAudit,
} from "../app/lib/analysis";
import { parseCsv, tableToCsv } from "../app/lib/csv";

test("sample audit is deterministic and returns all evidence families", async () => {
  const table = makeSampleDataset();
  const target = inferTarget(table);
  assert.equal(target, "churn");
  assert.equal(inferTask(table, target), "classification");

  const settings = {
    target,
    task: "classification" as const,
    model: "regularized" as const,
    repeats: 4,
    testSize: 0.25,
    seed: 1729,
  };
  const first = await runAudit(table, settings);
  const second = await runAudit(table, settings);

  assert.equal(first.protocol.sourceHash, second.protocol.sourceHash);
  assert.equal(first.baseline.score, second.baseline.score);
  assert.deepEqual(first.curves, second.curves);
  assert.equal(first.curves.length, 4);
  assert.deepEqual(first.findings.map((finding) => finding.kind), [
    "generalization",
    "robustness",
    "falsification",
  ]);
  assert.ok(first.baseline.score >= 0 && first.baseline.score <= 1);
});

test("stress variant preserves target values under feature noise", () => {
  const table = makeSampleDataset();
  const generated = generateStressVariant(table, "churn", "feature-noise", 0.25, 42);
  assert.equal(generated.table.rows.length, table.rows.length);
  assert.deepEqual(
    generated.table.rows.map((row) => row.churn),
    table.rows.map((row) => row.churn),
  );
  assert.equal(generated.manifest.seed, 42);
  assert.equal(generated.manifest.stressor, "feature-noise");
  assert.notEqual(generated.table.rows[0].age, table.rows[0].age);
});

test("CSV parser round-trips quoted values and missing cells", () => {
  const source = 'name,value,target\n"A, one",1,yes\nB,,no\n';
  const parsed = parseCsv(source, "quoted.csv");
  assert.equal(parsed.rows[0].name, "A, one");
  assert.equal(parsed.rows[1].value, null);
  const reparsed = parseCsv(tableToCsv(parsed), "quoted-again.csv");
  assert.deepEqual(reparsed.rows, parsed.rows);
});
