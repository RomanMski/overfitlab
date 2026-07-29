export type CellValue = string | number | null;

export interface DataTable {
  name: string;
  headers: string[];
  rows: Record<string, CellValue>[];
}

export type TaskType = "classification" | "regression";
export type ModelKind = "regularized" | "nearest-neighbor";

export interface AuditSettings {
  target: string;
  task: TaskType;
  model: ModelKind;
  repeats: number;
  testSize: number;
  seed: number;
}

export interface CurvePoint {
  level: number;
  median: number;
  low: number;
  high: number;
}

export interface StressCurve {
  id: "feature-noise" | "label-noise" | "missingness" | "train-size";
  label: string;
  shortLabel: string;
  unit: string;
  points: CurvePoint[];
  stroke: string;
  dash: number[];
}

export interface StressSummary {
  id: StressCurve["id"];
  label: string;
  degradationArea: number;
  firstStepLoss: number;
  halfSkillAt: string;
  mode: "fixed model" | "refit";
}

export interface AuditFinding {
  kind: "generalization" | "robustness" | "falsification";
  eyebrow: string;
  title: string;
  detail: string;
  status: "stable" | "watch" | "warning";
}

export interface AuditResult {
  dataset: {
    name: string;
    rows: number;
    features: string[];
    ignored: string[];
    target: string;
    task: TaskType;
    classBalance?: number;
  };
  baseline: {
    scoreLabel: "AUROC" | "R²";
    score: number;
    lossLabel: "Brier loss" | "MSE";
    trainLoss: number;
    auditLoss: number;
    gap: number;
    splitSpread: number;
  };
  permutation: {
    observed: number;
    nullMedian: number;
    percentile: number;
    runs: number;
  };
  curves: StressCurve[];
  summaries: StressSummary[];
  findings: AuditFinding[];
  warnings: string[];
  protocol: {
    repeats: number;
    testSize: number;
    seed: number;
    model: ModelKind;
    sourceHash: string;
    generatedAt: string;
    browserEngine: string;
  };
}

interface PreparedData {
  X: number[][];
  y: number[];
  features: string[];
  ignored: string[];
  labels?: [string, string];
  classBalance?: number;
  missingRate: number;
}

interface FittedModel {
  predict: (X: number[][]) => number[];
}

interface ScaleStats {
  medians: number[];
  means: number[];
  scales: number[];
  robustScales: number[];
}

const FEATURE_LEVELS = [0, 0.1, 0.25, 0.5];
const LABEL_LEVELS = [0, 0.05, 0.1, 0.2];
const MISSING_LEVELS = [0, 0.05, 0.15, 0.3];
const SIZE_STRESS_LEVELS = [0, 0.25, 0.5, 0.75];

export function makeSampleDataset(): DataTable {
  const rng = mulberry32(1729);
  const rows: Record<string, CellValue>[] = [];
  for (let i = 0; i < 420; i += 1) {
    const age = 21 + Math.floor(rng() * 44);
    const tenure = Math.max(0, Math.round(12 * Math.pow(rng(), 1.7) * 10) / 10);
    const usage = Math.max(0, 7 + 4.2 * gaussian(rng) + 0.09 * (age - 40));
    const latency = Math.max(35, 180 + 58 * gaussian(rng) + 8 * Math.max(0, usage - 10));
    const support = Math.max(0, Math.floor(Math.exp(0.32 + 0.44 * gaussian(rng)) - 1));
    const price = Math.max(12, 48 + 14 * gaussian(rng) + 1.8 * tenure);
    const hiddenSegment = rng() < 0.28 ? 1 : 0;
    const logit =
      -1.4 +
      0.008 * (latency - 170) +
      0.31 * support -
      0.16 * (usage - 7) -
      0.055 * tenure +
      0.012 * (price - 50) +
      0.72 * hiddenSegment +
      0.35 * Math.sin(age / 6);
    const churn = rng() < sigmoid(logit) ? "churn" : "retain";
    rows.push({
      account_id: `A-${String(i + 1).padStart(4, "0")}`,
      age,
      tenure_months: tenure,
      weekly_sessions: rng() < 0.018 ? null : round(usage, 2),
      median_latency_ms: round(latency, 1),
      support_tickets_90d: support,
      monthly_price: round(price, 2),
      churn,
    });
  }
  return {
    name: "subscription_churn_sample.csv",
    headers: [
      "account_id",
      "age",
      "tenure_months",
      "weekly_sessions",
      "median_latency_ms",
      "support_tickets_90d",
      "monthly_price",
      "churn",
    ],
    rows,
  };
}

export function inferTarget(table: DataTable): string {
  const preferred = table.headers.find((header) =>
    /^(target|label|outcome|y|churn|default|response)$/i.test(header),
  );
  return preferred ?? table.headers.at(-1) ?? "";
}

export function inferTask(table: DataTable, target: string): TaskType {
  const values = table.rows
    .map((row) => row[target])
    .filter((value): value is string | number => value !== null && value !== "");
  const unique = new Set(values.map(String));
  if (unique.size === 2) return "classification";
  const numericShare = values.length
    ? values.filter((value) => Number.isFinite(Number(value))).length / values.length
    : 0;
  return numericShare > 0.95 ? "regression" : "classification";
}

export async function runAudit(
  table: DataTable,
  settings: AuditSettings,
  onProgress?: (fraction: number, label: string) => void,
): Promise<AuditResult> {
  const prepared = prepareData(table, settings);
  if (prepared.X.length < 40) {
    throw new Error("At least 40 usable rows are required after target filtering.");
  }
  if (prepared.features.length === 0) {
    throw new Error("No numeric feature columns were found. The browser lab currently audits numeric predictors.");
  }

  const repeats = clamp(Math.round(settings.repeats), 4, 40);
  const curveSamples = new Map<string, Map<number, number[]>>();
  for (const [id, levels] of [
    ["feature-noise", FEATURE_LEVELS],
    ["label-noise", LABEL_LEVELS],
    ["missingness", MISSING_LEVELS],
    ["train-size", SIZE_STRESS_LEVELS],
  ] as const) {
    curveSamples.set(id, new Map(levels.map((level) => [level, []])));
  }

  const trainLosses: number[] = [];
  const auditLosses: number[] = [];
  const baselineScores: number[] = [];
  const permutationScores: number[] = [];

  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const repeatSeed = mixSeed(settings.seed, repeat + 1);
    const rng = mulberry32(repeatSeed);
    const split = makeSplit(prepared.y, settings.task, settings.testSize, rng);
    const stats = fitScale(prepared.X, split.train);
    const trainX = split.train.map((index) => transformRow(prepared.X[index], stats));
    const testX = split.test.map((index) => transformRow(prepared.X[index], stats));
    const trainY = split.train.map((index) => prepared.y[index]);
    const testY = split.test.map((index) => prepared.y[index]);
    const model = fitModel(settings.model, settings.task, trainX, trainY, rng);
    const cleanTrainPrediction = model.predict(trainX);
    const cleanTestPrediction = model.predict(testX);
    const trainMetric = evaluate(settings.task, trainY, cleanTrainPrediction);
    const testMetric = evaluate(settings.task, testY, cleanTestPrediction);
    const referenceLoss = constantReferenceLoss(settings.task, trainY, testY);
    const advantage = referenceLoss - testMetric.loss;

    trainLosses.push(trainMetric.loss);
    auditLosses.push(testMetric.loss);
    baselineScores.push(testMetric.score);

    for (const id of curveSamples.keys()) {
      curveSamples.get(id)?.get(0)?.push(1);
    }

    for (const level of FEATURE_LEVELS.slice(1)) {
      const stressRng = mulberry32(mixSeed(repeatSeed, 1000 + Math.round(level * 1000)));
      const perturbed = addFeatureNoise(testX, stats.robustScales, level, stressRng);
      const stressedLoss = evaluate(settings.task, testY, model.predict(perturbed)).loss;
      curveSamples.get("feature-noise")?.get(level)?.push(retainedSkill(referenceLoss, testMetric.loss, stressedLoss, advantage));
    }

    for (const level of MISSING_LEVELS.slice(1)) {
      const stressRng = mulberry32(mixSeed(repeatSeed, 2000 + Math.round(level * 1000)));
      const perturbed = injectMissingness(testX, level, stressRng);
      const stressedLoss = evaluate(settings.task, testY, model.predict(perturbed)).loss;
      curveSamples.get("missingness")?.get(level)?.push(retainedSkill(referenceLoss, testMetric.loss, stressedLoss, advantage));
    }

    for (const level of LABEL_LEVELS.slice(1)) {
      const stressRng = mulberry32(mixSeed(repeatSeed, 3000 + Math.round(level * 1000)));
      const stressedY = corruptLabels(trainY, settings.task, level, stressRng);
      const stressedModel = fitModel(settings.model, settings.task, trainX, stressedY, stressRng);
      const stressedLoss = evaluate(settings.task, testY, stressedModel.predict(testX)).loss;
      curveSamples.get("label-noise")?.get(level)?.push(retainedSkill(referenceLoss, testMetric.loss, stressedLoss, advantage));
    }

    for (const severity of SIZE_STRESS_LEVELS.slice(1)) {
      const fraction = 1 - severity;
      const stressRng = mulberry32(mixSeed(repeatSeed, 4000 + Math.round(severity * 1000)));
      const subset = subsampleIndices(trainY, settings.task, fraction, stressRng);
      const subsetX = subset.map((index) => trainX[index]);
      const subsetY = subset.map((index) => trainY[index]);
      const stressedModel = fitModel(settings.model, settings.task, subsetX, subsetY, stressRng);
      const stressedLoss = evaluate(settings.task, testY, stressedModel.predict(testX)).loss;
      curveSamples.get("train-size")?.get(severity)?.push(retainedSkill(referenceLoss, testMetric.loss, stressedLoss, advantage));
    }

    const permutationRng = mulberry32(mixSeed(repeatSeed, 9001));
    const permutedY = shuffled(trainY, permutationRng);
    const nullModel = fitModel(settings.model, settings.task, trainX, permutedY, permutationRng);
    permutationScores.push(evaluate(settings.task, testY, nullModel.predict(testX)).score);

    onProgress?.((repeat + 1) / repeats, `Repeated split ${repeat + 1} of ${repeats}`);
    if (repeat % 2 === 1) await nextFrame();
  }

  const curves = buildCurves(curveSamples);
  const summaries = curves.map(summarizeCurve);
  const gapSamples = auditLosses.map((loss, index) => loss - trainLosses[index]);
  const scoreSpread = quantile(baselineScores, 0.95) - quantile(baselineScores, 0.05);
  const observed = median(baselineScores);
  const nullMedian = median(permutationScores);
  const percentile =
    (100 * (1 + permutationScores.filter((score) => score <= observed).length)) /
    (permutationScores.length + 1);
  const baseline = {
    scoreLabel: settings.task === "classification" ? ("AUROC" as const) : ("R²" as const),
    score: observed,
    lossLabel: settings.task === "classification" ? ("Brier loss" as const) : ("MSE" as const),
    trainLoss: median(trainLosses),
    auditLoss: median(auditLosses),
    gap: median(gapSamples),
    splitSpread: scoreSpread,
  };
  const warnings = buildWarnings(table, prepared, baseline, percentile);
  const findings = buildFindings(baseline, summaries, percentile, nullMedian, settings.task);

  return {
    dataset: {
      name: table.name,
      rows: prepared.X.length,
      features: prepared.features,
      ignored: prepared.ignored,
      target: settings.target,
      task: settings.task,
      classBalance: prepared.classBalance,
    },
    baseline,
    permutation: {
      observed,
      nullMedian,
      percentile,
      runs: permutationScores.length,
    },
    curves,
    summaries,
    findings,
    warnings,
    protocol: {
      repeats,
      testSize: settings.testSize,
      seed: settings.seed,
      model: settings.model,
      sourceHash: fingerprint(table),
      generatedAt: new Date().toISOString(),
      browserEngine: "StressFold browser protocol 0.1",
    },
  };
}

export function generateStressVariant(
  table: DataTable,
  target: string,
  family: "feature-noise" | "label-noise" | "missingness" | "bootstrap",
  level: number,
  seed: number,
): { table: DataTable; manifest: Record<string, unknown> } {
  const rng = mulberry32(seed);
  const rows = table.rows.map((row) => ({ ...row }));
  const numericFeatures = table.headers.filter((header) => {
    if (header === target) return false;
    const values = rows.map((row) => toNumber(row[header])).filter(Number.isFinite);
    return values.length >= Math.max(10, rows.length * 0.75);
  });

  if (family === "bootstrap") {
    const sampled = Array.from({ length: rows.length }, () => ({ ...rows[Math.floor(rng() * rows.length)] }));
    return {
      table: { ...table, name: `${stripExtension(table.name)}_bootstrap.csv`, rows: sampled },
      manifest: makeManifest(table, target, family, level, seed, numericFeatures),
    };
  }

  if (family === "feature-noise") {
    for (const feature of numericFeatures) {
      const values = rows.map((row) => toNumber(row[feature])).filter(Number.isFinite);
      const scale = robustScale(values);
      for (const row of rows) {
        const value = toNumber(row[feature]);
        if (Number.isFinite(value)) row[feature] = round(value + gaussian(rng) * scale * level, 6);
      }
    }
  } else if (family === "missingness") {
    for (const row of rows) {
      for (const feature of numericFeatures) if (rng() < level) row[feature] = null;
    }
  } else if (family === "label-noise") {
    const labels = [...new Set(rows.map((row) => String(row[target])).filter(Boolean))];
    if (labels.length === 2) {
      for (const row of rows) if (rng() < level) row[target] = String(row[target]) === labels[0] ? labels[1] : labels[0];
    } else {
      const values = rows.map((row) => toNumber(row[target])).filter(Number.isFinite);
      const scale = robustScale(values);
      for (const row of rows) {
        const value = toNumber(row[target]);
        if (Number.isFinite(value)) row[target] = round(value + gaussian(rng) * scale * level, 6);
      }
    }
  }

  return {
    table: { ...table, name: `${stripExtension(table.name)}_${family}_${formatLevel(level)}.csv`, rows },
    manifest: makeManifest(table, target, family, level, seed, numericFeatures),
  };
}

function prepareData(table: DataTable, settings: AuditSettings): PreparedData {
  const usableRows = table.rows.filter((row) => row[settings.target] !== null && row[settings.target] !== "");
  const features: string[] = [];
  const ignored: string[] = [];
  for (const header of table.headers) {
    if (header === settings.target) continue;
    const values = usableRows.map((row) => row[header]);
    const numeric = values.filter((value) => Number.isFinite(toNumber(value))).length;
    const looksLikeId = /(^id$|_id$|^index$|uuid|identifier|row_number)/i.test(header);
    if (!looksLikeId && numeric >= Math.max(10, usableRows.length * 0.75)) features.push(header);
    else ignored.push(header);
  }

  let missing = 0;
  const X = usableRows.map((row) =>
    features.map((feature) => {
      const value = toNumber(row[feature]);
      if (!Number.isFinite(value)) missing += 1;
      return value;
    }),
  );
  let y: number[];
  let labels: [string, string] | undefined;
  let classBalance: number | undefined;
  if (settings.task === "classification") {
    const unique = [...new Set(usableRows.map((row) => String(row[settings.target])))].sort();
    if (unique.length !== 2) throw new Error(`Binary classification requires exactly two target values; found ${unique.length}.`);
    labels = [unique[0], unique[1]];
    y = usableRows.map((row) => (String(row[settings.target]) === unique[1] ? 1 : 0));
    classBalance = mean(y);
  } else {
    y = usableRows.map((row) => toNumber(row[settings.target]));
    if (y.some((value) => !Number.isFinite(value))) throw new Error("Regression targets must be numeric.");
  }
  return {
    X,
    y,
    features,
    ignored,
    labels,
    classBalance,
    missingRate: X.length && features.length ? missing / (X.length * features.length) : 0,
  };
}

function makeSplit(y: number[], task: TaskType, testSize: number, rng: () => number) {
  const fraction = clamp(testSize, 0.15, 0.4);
  if (task === "classification") {
    const zero = shuffled(y.map((value, index) => ({ value, index })).filter((item) => item.value === 0).map((item) => item.index), rng);
    const one = shuffled(y.map((value, index) => ({ value, index })).filter((item) => item.value === 1).map((item) => item.index), rng);
    const zeroTest = clamp(Math.round(zero.length * fraction), 1, Math.max(1, zero.length - 2));
    const oneTest = clamp(Math.round(one.length * fraction), 1, Math.max(1, one.length - 2));
    return {
      test: shuffled([...zero.slice(0, zeroTest), ...one.slice(0, oneTest)], rng),
      train: shuffled([...zero.slice(zeroTest), ...one.slice(oneTest)], rng),
    };
  }
  const all = shuffled(Array.from({ length: y.length }, (_, index) => index), rng);
  const testCount = clamp(Math.round(y.length * fraction), 2, y.length - 4);
  return { test: all.slice(0, testCount), train: all.slice(testCount) };
}

function fitScale(X: number[][], indices: number[]): ScaleStats {
  const width = X[0].length;
  const medians: number[] = [];
  const means: number[] = [];
  const scales: number[] = [];
  const robustScales: number[] = [];
  for (let column = 0; column < width; column += 1) {
    const values = indices.map((index) => X[index][column]).filter(Number.isFinite);
    const med = median(values);
    const complete = indices.map((index) => (Number.isFinite(X[index][column]) ? X[index][column] : med));
    const avg = mean(complete);
    const scale = Math.sqrt(mean(complete.map((value) => (value - avg) ** 2))) || 1;
    medians.push(med);
    means.push(avg);
    scales.push(scale);
    robustScales.push(Math.max(0.15, robustScale(complete) / scale));
  }
  return { medians, means, scales, robustScales };
}

function transformRow(row: number[], stats: ScaleStats): number[] {
  return row.map((value, column) => {
    const complete = Number.isFinite(value) ? value : stats.medians[column];
    return (complete - stats.means[column]) / stats.scales[column];
  });
}

function fitModel(kind: ModelKind, task: TaskType, X: number[][], y: number[], rng: () => number): FittedModel {
  if (kind === "nearest-neighbor") return fitNearestNeighbor(task, X, y);
  return fitRegularizedModel(task, X, y, rng);
}

function fitRegularizedModel(task: TaskType, X: number[][], y: number[], rng: () => number): FittedModel {
  const width = X[0]?.length ?? 0;
  const weights = Array.from({ length: width }, () => (rng() - 0.5) * 0.01);
  let bias = 0;
  const l2 = task === "classification" ? 0.08 : 0.12;
  const iterations = task === "classification" ? 170 : 220;
  const learningRate = task === "classification" ? 0.12 : 0.055;
  const targetMean = task === "regression" ? mean(y) : 0;
  const targetScale = task === "regression" ? Math.sqrt(mean(y.map((value) => (value - targetMean) ** 2))) || 1 : 1;
  const trainTarget = task === "regression" ? y.map((value) => (value - targetMean) / targetScale) : y;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const gradient = Array(width).fill(0) as number[];
    let biasGradient = 0;
    for (let row = 0; row < X.length; row += 1) {
      let linear = bias;
      for (let column = 0; column < width; column += 1) linear += weights[column] * X[row][column];
      const prediction = task === "classification" ? sigmoid(linear) : linear;
      const error = prediction - trainTarget[row];
      biasGradient += error;
      for (let column = 0; column < width; column += 1) gradient[column] += error * X[row][column];
    }
    const rate = learningRate / (1 + iteration * 0.008);
    bias -= rate * biasGradient / X.length;
    for (let column = 0; column < width; column += 1) {
      weights[column] -= rate * (gradient[column] / X.length + l2 * weights[column]);
    }
  }

  return {
    predict: (rows) =>
      rows.map((row) => {
        let linear = bias;
        for (let column = 0; column < width; column += 1) linear += weights[column] * row[column];
        return task === "classification" ? clamp(sigmoid(linear), 0.001, 0.999) : linear * targetScale + targetMean;
      }),
  };
}

function fitNearestNeighbor(task: TaskType, X: number[][], y: number[]): FittedModel {
  const trainX = X.map((row) => [...row]);
  const trainY = [...y];
  const neighbors = task === "classification" ? 3 : 1;
  return {
    predict: (rows) =>
      rows.map((row) => {
        const distances = trainX
          .map((candidate, index) => ({
            index,
            distance: candidate.reduce((sum, value, column) => sum + (value - row[column]) ** 2, 0),
          }))
          .sort((a, b) => a.distance - b.distance)
          .slice(0, neighbors);
        return mean(distances.map((item) => trainY[item.index]));
      }),
  };
}

function evaluate(task: TaskType, truth: number[], prediction: number[]) {
  const loss = mean(truth.map((value, index) => (value - prediction[index]) ** 2));
  if (task === "classification") return { loss, score: auc(truth, prediction) };
  const variance = mean(truth.map((value) => (value - mean(truth)) ** 2));
  return { loss, score: variance > 1e-12 ? 1 - loss / variance : 0 };
}

function constantReferenceLoss(task: TaskType, trainY: number[], testY: number[]) {
  const constant = mean(trainY);
  return mean(testY.map((value) => (value - constant) ** 2));
}

function retainedSkill(reference: number, clean: number, stressed: number, advantage: number) {
  if (advantage > Math.max(1e-8, reference * 0.01)) return (reference - stressed) / advantage;
  return 1 - (stressed - clean) / Math.max(reference, clean, 1e-6);
}

function addFeatureNoise(X: number[][], scales: number[], level: number, rng: () => number) {
  return X.map((row) => row.map((value, column) => value + gaussian(rng) * scales[column] * level));
}

function injectMissingness(X: number[][], level: number, rng: () => number) {
  return X.map((row) => row.map((value) => (rng() < level ? 0 : value)));
}

function corruptLabels(y: number[], task: TaskType, level: number, rng: () => number) {
  if (task === "classification") return y.map((value) => (rng() < level ? 1 - value : value));
  const scale = robustScale(y) || 1;
  return y.map((value) => value + gaussian(rng) * scale * level);
}

function subsampleIndices(y: number[], task: TaskType, fraction: number, rng: () => number) {
  if (task === "classification") {
    const zero = shuffled(y.map((value, index) => ({ value, index })).filter((item) => item.value === 0).map((item) => item.index), rng);
    const one = shuffled(y.map((value, index) => ({ value, index })).filter((item) => item.value === 1).map((item) => item.index), rng);
    const zeroCount = clamp(Math.round(zero.length * fraction), 2, zero.length);
    const oneCount = clamp(Math.round(one.length * fraction), 2, one.length);
    return shuffled([...zero.slice(0, zeroCount), ...one.slice(0, oneCount)], rng);
  }
  const count = clamp(Math.round(y.length * fraction), 8, y.length);
  return shuffled(Array.from({ length: y.length }, (_, index) => index), rng).slice(0, count);
}

function buildCurves(samples: Map<string, Map<number, number[]>>): StressCurve[] {
  const styles: Record<string, Pick<StressCurve, "label" | "shortLabel" | "unit" | "stroke" | "dash">> = {
    "feature-noise": {
      label: "Numeric measurement noise",
      shortLabel: "Feature noise",
      unit: "robust σ",
      stroke: "#1f5c63",
      dash: [],
    },
    "label-noise": {
      label: "Training-label corruption",
      shortLabel: "Label noise",
      unit: "fraction",
      stroke: "#b56a28",
      dash: [8, 5],
    },
    missingness: {
      label: "Evaluation missingness",
      shortLabel: "Missingness",
      unit: "fraction",
      stroke: "#7c5a8d",
      dash: [3, 4],
    },
    "train-size": {
      label: "Training-set reduction",
      shortLabel: "Less training data",
      unit: "fraction removed",
      stroke: "#b19226",
      dash: [11, 4, 2, 4],
    },
  };
  return [...samples.entries()].map(([id, levelMap]) => ({
    id: id as StressCurve["id"],
    ...styles[id],
    points: [...levelMap.entries()].map(([level, values]) => ({
      level,
      median: median(values),
      low: quantile(values, 0.05),
      high: quantile(values, 0.95),
    })),
  }));
}

function summarizeCurve(curve: StressCurve): StressSummary {
  const maxLevel = Math.max(...curve.points.map((point) => point.level), 1e-9);
  let area = 0;
  for (let index = 1; index < curve.points.length; index += 1) {
    const left = curve.points[index - 1];
    const right = curve.points[index];
    area += ((2 - left.median - right.median) / 2) * ((right.level - left.level) / maxLevel);
  }
  const first = curve.points[1];
  const half = curve.points.find((point) => point.level > 0 && point.median <= 0.5);
  return {
    id: curve.id,
    label: curve.shortLabel,
    degradationArea: Math.max(0, area),
    firstStepLoss: first ? 1 - first.median : 0,
    halfSkillAt: half ? `${formatLevel(half.level)} ${curve.unit}` : "not reached",
    mode: curve.id === "feature-noise" || curve.id === "missingness" ? "fixed model" : "refit",
  };
}

function buildWarnings(
  table: DataTable,
  data: PreparedData,
  baseline: AuditResult["baseline"],
  percentile: number,
) {
  const warnings: string[] = [];
  if (data.X.length < 200) warnings.push("Small sample: Monte Carlo bands may be wide; prefer the Python audit preset for decisions.");
  if (data.classBalance !== undefined && Math.min(data.classBalance, 1 - data.classBalance) < 0.12) {
    warnings.push("Rare class: repeated splits can be unstable; consider grouped or nested evaluation.");
  }
  if (data.missingRate > 0.05) warnings.push(`${formatPercent(data.missingRate)} of numeric feature cells were median-imputed inside each training split.`);
  if (data.ignored.length) warnings.push(`${data.ignored.length} non-numeric or identifier-like column${data.ignored.length === 1 ? " was" : "s were"} excluded in the browser lab.`);
  if (baseline.gap > Math.max(0.03, baseline.auditLoss * 0.3)) warnings.push("The train-audit loss gap is large relative to audit loss.");
  if (percentile < 90) warnings.push("The observed score is not clearly separated from the label-permutation null in this quick run.");
  if (table.rows.length > data.X.length) warnings.push(`${table.rows.length - data.X.length} rows with missing targets were excluded.`);
  return warnings;
}

function buildFindings(
  baseline: AuditResult["baseline"],
  summaries: StressSummary[],
  percentile: number,
  nullMedian: number,
  task: TaskType,
): AuditFinding[] {
  const weakest = [...summaries].sort((a, b) => b.degradationArea - a.degradationArea)[0];
  const gapRatio = baseline.auditLoss ? baseline.gap / baseline.auditLoss : 0;
  const gapStatus = gapRatio > 0.35 ? "warning" : gapRatio > 0.18 ? "watch" : "stable";
  const robustnessStatus = weakest.degradationArea > 0.55 ? "warning" : weakest.degradationArea > 0.25 ? "watch" : "stable";
  const nullStatus = percentile >= 95 ? "stable" : percentile >= 85 ? "watch" : "warning";
  return [
    {
      kind: "generalization",
      eyebrow: "Generalization",
      title: gapStatus === "stable" ? "Clean split gap is contained" : "Clean split gap needs attention",
      detail: `Median audit loss exceeds training loss by ${formatNumber(baseline.gap, 3)} across paired splits (${formatPercent(Math.max(0, gapRatio))} of audit loss). This is descriptive evidence, not a proof of overfitting.`,
      status: gapStatus,
    },
    {
      kind: "robustness",
      eyebrow: "Robustness",
      title: `${weakest.label} is the sharpest failure path`,
      detail: `Its degradation area is ${formatNumber(weakest.degradationArea, 2)}. Compare the complete response curve before choosing a deployment tolerance; unrelated stressors are deliberately not collapsed into one score.`,
      status: robustnessStatus,
    },
    {
      kind: "falsification",
      eyebrow: "Falsification",
      title: nullStatus === "stable" ? "Observed signal clears the quick null" : "Null separation is inconclusive",
      detail: `The observed ${baseline.scoreLabel} sits at the ${formatNumber(percentile, 0)}th percentile of ${task === "classification" ? "label-permuted" : "target-permuted"} refits; null median ${formatNumber(nullMedian, 3)}. Increase permutations before publication or decision use.`,
      status: nullStatus,
    },
  ];
}

function makeManifest(
  table: DataTable,
  target: string,
  family: string,
  level: number,
  seed: number,
  features: string[],
) {
  return {
    schema_version: "stressfold.variant.v1",
    source_name: table.name,
    source_hash: fingerprint(table),
    generated_at: new Date().toISOString(),
    target,
    stressor: family,
    level,
    seed,
    features,
    rows: table.rows.length,
    note: "Stress-test variant; not a claim about the future data distribution.",
  };
}

function fingerprint(table: DataTable) {
  const text = JSON.stringify({ headers: table.headers, rows: table.rows.slice(0, 2000) });
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function robustScale(values: number[]) {
  if (!values.length) return 1;
  const iqrScale = (quantile(values, 0.75) - quantile(values, 0.25)) / 1.349;
  const med = median(values);
  const madScale = median(values.map((value) => Math.abs(value - med))) * 1.4826;
  return iqrScale > 1e-10 ? iqrScale : madScale > 1e-10 ? madScale : 1;
}

function auc(truth: number[], score: number[]) {
  const ranked = truth.map((value, index) => ({ value, score: score[index] })).sort((a, b) => a.score - b.score);
  let rankSum = 0;
  let cursor = 0;
  while (cursor < ranked.length) {
    let end = cursor + 1;
    while (end < ranked.length && ranked[end].score === ranked[cursor].score) end += 1;
    const averageRank = (cursor + 1 + end) / 2;
    for (let index = cursor; index < end; index += 1) if (ranked[index].value === 1) rankSum += averageRank;
    cursor = end;
  }
  const positives = truth.filter((value) => value === 1).length;
  const negatives = truth.length - positives;
  if (!positives || !negatives) return 0.5;
  return (rankSum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function quantile(values: number[], probability: number) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * probability;
  const base = Math.floor(position);
  const fraction = position - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + fraction * (sorted[base + 1] - sorted[base]);
}

function median(values: number[]) {
  return quantile(values, 0.5);
}

function mean(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function shuffled<T>(values: T[], rng: () => number) {
  const output = [...values];
  for (let index = output.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(rng() * (index + 1));
    [output[index], output[swap]] = [output[swap], output[index]];
  }
  return output;
}

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function mixSeed(seed: number, value: number) {
  let mixed = (seed ^ Math.imul(value, 0x9e3779b1)) >>> 0;
  mixed ^= mixed >>> 16;
  mixed = Math.imul(mixed, 0x85ebca6b);
  mixed ^= mixed >>> 13;
  return mixed >>> 0;
}

function gaussian(rng: () => number) {
  const left = Math.max(rng(), 1e-12);
  const right = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(left)) * Math.cos(2 * Math.PI * right);
}

function sigmoid(value: number) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function toNumber(value: CellValue) {
  if (typeof value === "number") return value;
  if (value === null || value === "") return Number.NaN;
  const normalized = String(value).trim().replace(/,/g, "");
  return Number(normalized);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function round(value: number, digits: number) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function stripExtension(name: string) {
  return name.replace(/\.[^.]+$/, "");
}

function formatLevel(value: number) {
  return value < 0.01 ? value.toFixed(3) : value < 0.1 ? value.toFixed(2) : value.toFixed(1);
}

function formatNumber(value: number, digits: number) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function nextFrame() {
  return new Promise<void>((resolve) => {
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });
}
