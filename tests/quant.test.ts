import assert from "node:assert/strict";
import test from "node:test";

import {
  autocorrelatedMarket,
  applyBlockOrder,
  blockPermutation,
  expectedMaximumSharpe,
  generateNoiseTrials,
  makeRng,
  normalDraws,
  normInv,
  polynomialFit,
  rootMeanSquaredError,
  sampleOverfitData,
  searchSnapshot,
  sharpe,
  trueFunction,
} from "../app/lib/quant.ts";

// Reference values produced by scipy.stats.norm.ppf, so the browser and the
// Python package cannot drift apart.
const NORM_INV_REFERENCE: [number, number][] = [
  [0.01, -2.326347874041],
  [0.25, -0.674489750196],
  [0.5, 0.0],
  [0.9, 1.281551565545],
  [0.99, 2.326347874041],
  [0.999, 3.090232306168],
];

test("normInv agrees with scipy to within the approximation's tolerance", () => {
  for (const [p, expected] of NORM_INV_REFERENCE) {
    assert.ok(
      Math.abs(normInv(p) - expected) < 1e-6,
      `normInv(${p}) gave ${normInv(p)}, expected ${expected}`,
    );
  }
});

// Reference values from overfitlab.expected_maximum_sharpe.
const EXPECTED_MAX_REFERENCE: [number, number][] = [
  [2, 0.051975534428],
  [10, 0.157459830135],
  [100, 0.25306028932],
  [1000, 0.325512151365],
  [10000, 0.386066485551],
];

test("expectedMaximumSharpe matches the Python implementation", () => {
  for (const [trials, expected] of EXPECTED_MAX_REFERENCE) {
    const got = expectedMaximumSharpe(trials, 0.01);
    assert.ok(
      Math.abs(got - expected) < 1e-6,
      `n=${trials} gave ${got}, expected ${expected}`,
    );
  }
});

test("a single trial has no selection bias to remove", () => {
  assert.equal(expectedMaximumSharpe(1, 0.25), 0);
  assert.equal(expectedMaximumSharpe(500, 0), 0);
});

test("the bar rises monotonically with the number of trials", () => {
  const bars = [2, 5, 20, 200, 2000].map((n) => expectedMaximumSharpe(n, 0.01));
  for (let i = 1; i < bars.length; i += 1) assert.ok(bars[i] > bars[i - 1]);
});

test("the random generator is deterministic for a fixed seed", () => {
  const a = makeRng(42);
  const b = makeRng(42);
  for (let i = 0; i < 20; i += 1) assert.equal(a(), b());
});

test("sharpe of a flat series is zero rather than infinite", () => {
  assert.equal(sharpe(new Array(50).fill(0.01)), 0);
});

test("a higher polynomial degree always fits the training dots better", () => {
  const { trainX, trainY } = sampleOverfitData(7);
  let previous = Infinity;
  for (const degree of [1, 2, 4, 8, 14]) {
    const fit = polynomialFit(trainX, trainY, degree);
    const error = rootMeanSquaredError(trainX, trainY, fit);
    assert.ok(error <= previous + 1e-9, `degree ${degree} fitted worse`);
    previous = error;
  }
});

test("the overfitting explainer actually overfits at high degree", () => {
  const { trainX, trainY, testX, testY } = sampleOverfitData(3);

  const sensible = polynomialFit(trainX, trainY, 4);
  const wild = polynomialFit(trainX, trainY, 15);

  const sensibleTrain = rootMeanSquaredError(trainX, trainY, sensible);
  const wildTrain = rootMeanSquaredError(trainX, trainY, wild);
  const sensibleTest = rootMeanSquaredError(testX, testY, sensible);
  const wildTest = rootMeanSquaredError(testX, testY, wild);

  // The wild fit tracks the dots it was given more closely.
  assert.ok(wildTrain < sensibleTrain);
  // And is worse on fresh points from the same process, which is the lesson.
  assert.ok(wildTest > sensibleTest);
});

test("a degree-4 fit stays close to the process it was sampled from", () => {
  const { trainX, trainY } = sampleOverfitData(11);
  const fit = polynomialFit(trainX, trainY, 4);
  for (const x of [-2.5, -1, 0, 1, 2.5]) {
    assert.ok(Math.abs(fit(x) - trueFunction(x)) < 0.45);
  }
});

test("generated trials contain no edge", () => {
  const trials = generateNoiseTrials(5, 200, 750);
  const sharpes = trials.map((trial) => trial.sharpe);
  const average = sharpes.reduce((a, b) => a + b, 0) / sharpes.length;
  // Mean Sharpe across trials sits at zero, because nothing has any signal.
  assert.ok(Math.abs(average) < 0.02, `mean sharpe was ${average}`);
});

test("searching more no-signal trials manufactures a better looking winner", () => {
  const trials = generateNoiseTrials(9, 400, 750);
  const few = searchSnapshot(trials, 5, 252);
  const many = searchSnapshot(trials, 400, 252);

  assert.ok(many.bestSharpe > few.bestSharpe);
  assert.ok(many.bestAnnualised > 1.0, "the winner should look investable");
  // The bar rises with it, so the excess never opens up into real evidence.
  assert.ok(many.expectedBest > few.expectedBest);
  assert.ok(Math.abs(many.excess) < 0.05);
});

test("the snapshot is stable when the slider returns to a position", () => {
  const trials = generateNoiseTrials(13, 300, 600);
  assert.deepEqual(searchSnapshot(trials, 120, 252), searchSnapshot(trials, 120, 252));
});

test("blockPermutation keeps every observation exactly once", () => {
  const rng = makeRng(3);
  const market = normalDraws(makeRng(9), 600).map((v) => v * 0.01);
  const sorted = [...market].sort((a, b) => a - b);

  for (const block of [1, 5, 60]) {
    const path = blockPermutation(market, block, rng);
    assert.equal(path.length, market.length);
    assert.deepEqual([...path].sort((a, b) => a - b), sorted);
  }
});

test("blockPermutation at block one destroys ordering", () => {
  const autocorrelation = (v: number[]) => {
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    let num = 0;
    let den = 0;
    for (let i = 0; i < v.length; i += 1) {
      den += (v[i] - m) ** 2;
      if (i > 0) num += (v[i] - m) * (v[i - 1] - m);
    }
    return num / den;
  };

  const market = autocorrelatedMarket(11, 3000, 0.4, 0);
  assert.ok(autocorrelation(market) > 0.25);

  const rng = makeRng(5);
  const shuffled = Array.from({ length: 20 }, () => blockPermutation(market, 1, rng));
  const mean = shuffled.map(autocorrelation).reduce((a, b) => a + b, 0) / shuffled.length;
  assert.ok(Math.abs(mean) < 0.05, `autocorrelation survived at ${mean}`);
});

test("block cutting matches the shared fixture", async () => {
  const { readFile } = await import("node:fs/promises");
  const url = new URL("./fixtures/block-order.json", import.meta.url);
  const { cases } = JSON.parse(await readFile(url, "utf8"));

  assert.ok(cases.length > 0, "fixture is empty");
  for (const testCase of cases) {
    const got = applyBlockOrder(testCase.source, testCase.block_size, testCase.order);
    assert.deepEqual(got, testCase.expected, testCase.name);
  }
});

test("applyBlockOrder rejects an ordering that is not a permutation", () => {
  assert.throws(() => applyBlockOrder([1, 2, 3, 4, 5, 6], 2, [0, 0, 1]), /permutation/);
});

test("blockPermutation keeps the multiset at awkward block sizes", () => {
  for (const [n, block] of [[100, 7], [101, 10], [13, 5], [9, 4], [12, 5]]) {
    const market = normalDraws(makeRng(n * 31 + block), n);
    const sorted = [...market].sort((a, b) => a - b);
    const rng = makeRng(3);
    for (let path = 0; path < 10; path += 1) {
      const got = blockPermutation(market, block, rng);
      assert.equal(got.length, n);
      assert.deepEqual([...got].sort((a, b) => a - b), sorted, `n=${n} block=${block}`);
    }
  }
});
