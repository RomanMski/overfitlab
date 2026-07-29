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

test("task inference reserves classification for binary targets", () => {
  const stringBinary = parseCsv(
    "feature,target\n1,retain\n2,churn\n3,retain\n4,churn\n",
    "string-binary.csv",
  );
  const numericBinary = parseCsv(
    "feature,target\n1,0\n2,1\n3,0\n4,1\n",
    "numeric-binary.csv",
  );
  const numericOutcome = parseCsv(
    "feature,target\n1,10\n2,20\n3,30\n4,40\n",
    "numeric-outcome.csv",
  );

  assert.equal(inferTask(stringBinary, "target"), "classification");
  assert.equal(inferTask(numericBinary, "target"), "classification");
  assert.equal(inferTask(numericOutcome, "target"), "regression");
});

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
  assert.equal(first.protocol.browserEngine, "StressFold browser protocol 0.2");
  for (const curve of first.curves) {
    assert.equal(curve.points[0].level, 0);
    assert.equal(curve.points[0].median, 1);
    assert.ok(curve.points.every((point) =>
      [point.median, point.low, point.high].every(Number.isFinite)
    ));
  }
  assert.deepEqual(first.findings.map((finding) => finding.kind), [
    "generalization",
    "robustness",
    "falsification",
  ]);
  assert.ok(first.baseline.score >= 0 && first.baseline.score <= 1);
});

test("every viable sample target completes a four-repeat audit", async () => {
  const table = makeSampleDataset();
  const viableTargets = table.headers.filter((target) => {
    const values = table.rows
      .map((row) => row[target])
      .filter((value): value is string | number => value !== null && value !== "");
    const uniqueValues = new Set(values.map(String));
    const isBinary = uniqueValues.size === 2;
    const isNumeric = values.length > 0 && values.every((value) => Number.isFinite(Number(value)));
    return isBinary || isNumeric;
  });

  assert.deepEqual(viableTargets, [
    "age",
    "tenure_months",
    "weekly_sessions",
    "median_latency_ms",
    "support_tickets_90d",
    "monthly_price",
    "churn",
  ]);

  for (const [index, target] of viableTargets.entries()) {
    const task = inferTask(table, target);
    const result = await runAudit(table, {
      target,
      task,
      model: "regularized",
      repeats: 4,
      testSize: 0.25,
      seed: 1729 + index,
    });

    assert.equal(result.dataset.target, target);
    assert.equal(result.dataset.task, task);
    assert.equal(result.protocol.repeats, 4);
    assert.equal(result.curves.length, 4);
    assert.ok(Number.isFinite(result.baseline.score), `${target} should produce a finite baseline score`);
  }
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
