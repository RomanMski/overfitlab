import assert from "node:assert/strict";
import test from "node:test";
import { correctedNullRank, runAudit, type DataTable } from "../app/lib/analysis";
import { buildHtmlReport } from "../app/lib/report";

function rng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normal(random: () => number) {
  const left = Math.max(random(), 1e-12);
  const right = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(left)) * Math.cos(2 * Math.PI * right);
}

function table(name: string, rows: DataTable["rows"]): DataTable {
  return { name, headers: Object.keys(rows[0]), rows };
}

function assertFiniteAudit(result: Awaited<ReturnType<typeof runAudit>>) {
  assert.ok(Number.isFinite(result.baseline.score));
  assert.ok(Number.isFinite(result.baseline.trainLoss));
  assert.ok(Number.isFinite(result.baseline.auditLoss));
  assert.ok(Number.isFinite(result.baseline.referenceLoss));
  assert.equal(typeof result.baseline.retainedSkillReliable, "boolean");
  assert.equal(result.curves.length, 4);
  for (const curve of result.curves) {
    assert.equal(curve.points.length, 4);
    assert.ok(curve.points.every((point) =>
      [point.median, point.low, point.high].every(Number.isFinite),
    ));
  }
}

test("browser audit separates strong signal from an independent-label null", async () => {
  const signalRandom = rng(11);
  const signalRows = Array.from({ length: 360 }, (_, index) => {
    const x1 = normal(signalRandom);
    const x2 = normal(signalRandom);
    const margin = 2.4 * x1 - 1.7 * x2 + 0.35 * normal(signalRandom);
    return { row_id: `signal-${index}`, x1, x2, target: margin > 0 ? "yes" : "no" };
  });
  const nullRandom = rng(29);
  const nullRows = Array.from({ length: 360 }, (_, index) => ({
    row_id: `null-${index}`,
    x1: normal(nullRandom),
    x2: normal(nullRandom),
    target: nullRandom() > 0.5 ? "yes" : "no",
  }));

  const settings = {
    target: "target",
    task: "classification" as const,
    model: "regularized" as const,
    repeats: 12,
    testSize: 0.25,
    seed: 404,
  };
  const signal = await runAudit(table("strong-signal.csv", signalRows), settings);
  const nullResult = await runAudit(table("independent-labels.csv", nullRows), settings);

  assertFiniteAudit(signal);
  assertFiniteAudit(nullResult);
  assert.ok(signal.baseline.score > 0.95);
  assert.ok(nullResult.baseline.score > 0.35 && nullResult.baseline.score < 0.65);
  assert.ok(signal.baseline.score - nullResult.baseline.score > 0.35);
  assert.ok(signal.permutation.observed > signal.permutation.nullMedian + 0.25);
  assert.equal(signal.baseline.retainedSkillReliable, true);
  assert.equal(nullResult.baseline.retainedSkillReliable, false);
  const nullReport = buildHtmlReport(nullResult);
  assert.match(nullReport, /Normalized curve rankings are withheld/);
  assert.match(nullReport, /not interpretable/);
  assert.match(nullReport, /descriptive, not a p-value/);
});

test("browser regression audit reports degradation under measurement noise", async () => {
  const random = rng(71);
  const rows = Array.from({ length: 320 }, (_, index) => {
    const x1 = normal(random);
    const x2 = normal(random);
    const x3 = normal(random);
    return {
      row_id: `reg-${index}`,
      x1,
      x2,
      x3,
      outcome: 4.2 * x1 - 2.6 * x2 + 0.8 * x3 + 0.25 * normal(random),
    };
  });
  const result = await runAudit(table("linear-regression.csv", rows), {
    target: "outcome",
    task: "regression",
    model: "regularized",
    repeats: 12,
    testSize: 0.25,
    seed: 991,
  });

  assertFiniteAudit(result);
  assert.ok(result.baseline.score > 0.95);
  const featureNoise = result.curves.find((curve) => curve.id === "feature-noise");
  assert.ok(featureNoise);
  assert.ok(featureNoise.points.at(-1)!.median < featureNoise.points[0].median - 0.1);

  const tinyUnits = await runAudit(
    table("linear-regression-tiny-units.csv", rows.map((row) => ({
      ...row,
      outcome: Number(row.outcome) * 1e-7,
    }))),
    {
      target: "outcome",
      task: "regression",
      model: "regularized",
      repeats: 12,
      testSize: 0.25,
      seed: 991,
    },
  );
  assertFiniteAudit(tinyUnits);
  assert.ok(Math.abs(tinyUnits.baseline.score - result.baseline.score) < 1e-10);
});

test("browser audit exposes model dependence on nonlinear XOR structure", async () => {
  const random = rng(101);
  const rows = Array.from({ length: 480 }, (_, index) => {
    const x1 = random() * 2 - 1;
    const x2 = random() * 2 - 1;
    const noisyLabel = (x1 > 0) !== (x2 > 0) !== (random() < 0.03);
    return { row_id: `xor-${index}`, x1, x2, target: noisyLabel ? "yes" : "no" };
  });
  const data = table("xor.csv", rows);
  const common = {
    target: "target",
    task: "classification" as const,
    repeats: 12,
    testSize: 0.25,
    seed: 808,
  };
  const linear = await runAudit(data, { ...common, model: "regularized" });
  const neighbor = await runAudit(data, { ...common, model: "nearest-neighbor" });

  assertFiniteAudit(linear);
  assertFiniteAudit(neighbor);
  assert.ok(linear.baseline.score > 0.35 && linear.baseline.score < 0.65);
  assert.ok(neighbor.baseline.score > 0.85);
  assert.ok(neighbor.baseline.score - linear.baseline.score > 0.3);
});

test("browser audit stays finite with imbalance and missing feature cells", async () => {
  const random = rng(151);
  const rows = Array.from({ length: 420 }, (_, index) => {
    const latent = normal(random);
    const x1 = random() < 0.09 ? null : latent;
    const x2 = random() < 0.07 ? null : normal(random);
    return {
      row_id: `sparse-${index}`,
      x1,
      x2,
      target: latent > 1.35 ? "rare" : "common",
    };
  });
  const result = await runAudit(table("imbalanced-missing.csv", rows), {
    target: "target",
    task: "classification",
    model: "regularized",
    repeats: 12,
    testSize: 0.3,
    seed: 1201,
  });

  assertFiniteAudit(result);
  assert.ok(result.warnings.some((warning) => warning.startsWith("Rare class:")));
  assert.ok(result.warnings.some((warning) => warning.includes("median-imputed")));
});

test("tie-adjusted null rank does not turn a degenerate tie into a pass", () => {
  const tied = correctedNullRank(0.5, Array(12).fill(0.5));
  assert.equal(tied, 50);
  assert.equal(correctedNullRank(0.8, Array(12).fill(0.5)), 100 * 12.5 / 13);
});

test("browser audit rejects unusable class counts and constant regression targets", async () => {
  const rareRows = Array.from({ length: 40 }, (_, index) => ({
    x: index,
    target: index === 0 ? "rare" : "common",
  }));
  await assert.rejects(
    runAudit(table("one-positive.csv", rareRows), {
      target: "target",
      task: "classification",
      model: "regularized",
      repeats: 4,
      testSize: 0.25,
      seed: 1,
    }),
    /at least 8 rows in each class/,
  );

  const constantRows = Array.from({ length: 80 }, (_, index) => ({ x: index, outcome: 3 }));
  await assert.rejects(
    runAudit(table("constant-target.csv", constantRows), {
      target: "outcome",
      task: "regression",
      model: "regularized",
      repeats: 4,
      testSize: 0.25,
      seed: 2,
    }),
    /non-zero variance/,
  );
});

test("browser audit surfaces duplicate-row and obvious target-proxy leakage risks", async () => {
  const rows = Array.from({ length: 160 }, (_, index) => {
    const target = index % 2;
    return { copied_feature: target, target: target ? "yes" : "no" };
  });
  const result = await runAudit(table("leakage-proxy.csv", rows), {
    target: "target",
    task: "classification",
    model: "nearest-neighbor",
    repeats: 8,
    testSize: 0.25,
    seed: 17,
  });

  assert.ok(result.warnings.some((warning) => warning.includes("predictor pattern")));
  assert.ok(result.warnings.some((warning) => warning.includes("nearly determines the target")));
  assert.equal(result.findings[0].title, "Check possible split leakage first");
});

test("browser audit distinguishes discrete repeats from repeated entities and unsupported columns", async () => {
  const discreteRows = Array.from({ length: 240 }, (_, index) => ({
    row_id: `row-${index}`,
    bucket: index % 4,
    cycle: index % 13,
    category: index % 2 ? "north" : "south",
    target: (index * 37) % 101 < 50 ? "yes" : "no",
  }));
  const discrete = await runAudit(table("ordinary-discrete.csv", discreteRows), {
    target: "target",
    task: "classification",
    model: "regularized",
    repeats: 8,
    testSize: 0.25,
    seed: 23,
  });
  assert.ok(discrete.warnings.some((warning) => warning.includes("predictor pattern")));
  assert.ok(discrete.warnings.some((warning) => warning.startsWith("Unsupported non-numeric or sparse columns excluded: category")));
  assert.notEqual(discrete.findings[0].title, "Check possible split leakage first");

  const repeatedEntityRows = Array.from({ length: 240 }, (_, index) => {
    const patient = Math.floor(index / 2);
    return {
      patient_id: `patient-${patient}`,
      visit_value: Math.sin(patient * 1.3) + (index % 2) * 0.01,
      target: patient % 2 ? "yes" : "no",
    };
  });
  const repeatedEntity = await runAudit(table("repeated-patients.csv", repeatedEntityRows), {
    target: "target",
    task: "classification",
    model: "regularized",
    repeats: 8,
    testSize: 0.25,
    seed: 29,
  });
  assert.ok(repeatedEntity.warnings.some((warning) => warning.includes("patient_id repeats across")));
  assert.equal(repeatedEntity.findings[0].title, "Check possible split leakage first");
});

test("browser source fingerprint includes rows after the first two thousand", async () => {
  const rows = Array.from({ length: 2_050 }, (_, index) => ({
    x: index,
    outcome: Math.sin(index / 7) + Math.cos(index / 19),
  }));
  const changed = rows.map((row) => ({ ...row }));
  changed[2_025].x += 0.125;
  const settings = {
    target: "outcome",
    task: "regression" as const,
    model: "regularized" as const,
    repeats: 4,
    testSize: 0.25,
    seed: 31,
  };
  const first = await runAudit(table("fingerprint-a.csv", rows), settings);
  const second = await runAudit(table("fingerprint-a.csv", changed), settings);
  assert.notEqual(first.protocol.sourceHash, second.protocol.sourceHash);
});
