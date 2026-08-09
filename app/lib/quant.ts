/**
 * Browser-side maths for the interactive explainers.
 *
 * Everything here is deterministic given a seed, so a reader who drags a
 * slider back to where it was sees exactly what they saw before. The
 * expected-maximum-Sharpe expression mirrors the Python package, and there is
 * a test asserting the two agree.
 */

/** Deterministic 32-bit PRNG. Small, fast, and good enough for illustration. */
export function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal draws by the Box-Muller transform. */
export function normalDraws(rng: () => number, count: number): number[] {
  const out: number[] = [];
  while (out.length < count) {
    const u1 = Math.max(rng(), Number.EPSILON);
    const u2 = rng();
    const radius = Math.sqrt(-2 * Math.log(u1));
    out.push(radius * Math.cos(2 * Math.PI * u2));
    if (out.length < count) out.push(radius * Math.sin(2 * Math.PI * u2));
  }
  return out;
}

const ACKLAM_A = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
  1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
];
const ACKLAM_B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
  6.680131188771972e1, -1.328068155288572e1,
];
const ACKLAM_C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
  -2.549732539343734, 4.374664141464968, 2.938163982698783,
];
const ACKLAM_D = [
  7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
  3.754408661907416,
];

/** Inverse standard normal CDF, by Acklam's rational approximation. */
export function normInv(p: number): number {
  if (!(p > 0 && p < 1)) return Number.NaN;
  const low = 0.02425;
  if (p < low) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((ACKLAM_C[0] * q + ACKLAM_C[1]) * q + ACKLAM_C[2]) * q + ACKLAM_C[3]) * q +
        ACKLAM_C[4]) *
        q +
        ACKLAM_C[5]) /
      ((((ACKLAM_D[0] * q + ACKLAM_D[1]) * q + ACKLAM_D[2]) * q + ACKLAM_D[3]) * q + 1)
    );
  }
  if (p > 1 - low) return -normInv(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((ACKLAM_A[0] * r + ACKLAM_A[1]) * r + ACKLAM_A[2]) * r + ACKLAM_A[3]) * r +
      ACKLAM_A[4]) *
      r +
      ACKLAM_A[5]) *
      q) /
    (((((ACKLAM_B[0] * r + ACKLAM_B[1]) * r + ACKLAM_B[2]) * r + ACKLAM_B[3]) * r +
      ACKLAM_B[4]) *
      r +
      1)
  );
}

export function mean(values: number[]): number {
  if (!values.length) return 0;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

export function standardDeviation(values: number[], ddof = 1): number {
  const n = values.length;
  if (n - ddof <= 0) return 0;
  const average = mean(values);
  let sum = 0;
  for (const value of values) sum += (value - average) ** 2;
  return Math.sqrt(sum / (n - ddof));
}

/** Per-period Sharpe ratio. Returns 0 for a series that never moves. */
export function sharpe(values: number[]): number {
  const deviation = standardDeviation(values);
  const scale = Math.max(...values.map(Math.abs), 1);
  if (deviation <= 1e-12 * scale) return 0;
  return mean(values) / deviation;
}

const EULER_MASCHERONI = 0.5772156649015329;

/**
 * The Sharpe the best of `nTrials` reaches when none of them has any edge.
 *
 * This is the bar the deflated Sharpe ratio deflates against. It rises with
 * the number of trials, which is the whole point of the second explainer.
 */
export function expectedMaximumSharpe(nTrials: number, sharpeVariance: number): number {
  if (nTrials <= 1 || sharpeVariance <= 0) return 0;
  const deviation = Math.sqrt(sharpeVariance);
  const first = normInv(1 - 1 / nTrials);
  const second = normInv(1 - 1 / (nTrials * Math.E));
  return deviation * ((1 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second);
}

/** Cumulative sum, used to turn returns into an equity curve. */
export function cumulative(values: number[]): number[] {
  const out: number[] = [];
  let total = 0;
  for (const value of values) {
    total += value;
    out.push(total);
  }
  return out;
}

/**
 * Least-squares polynomial fit.
 *
 * The x values are mapped onto [-1, 1] before the Vandermonde matrix is built,
 * because raw x with a high degree produces a matrix too ill-conditioned to
 * solve in double precision. A very small ridge term keeps the highest degrees
 * numerically stable without visibly changing the fit.
 */
export function polynomialFit(
  xs: number[],
  ys: number[],
  degree: number,
): (x: number) => number {
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  const span = hi - lo || 1;
  const scale = (x: number) => (2 * (x - lo)) / span - 1;

  const terms = degree + 1;
  const normal: number[][] = Array.from({ length: terms }, () =>
    new Array(terms + 1).fill(0),
  );

  for (let i = 0; i < xs.length; i += 1) {
    const powers: number[] = [1];
    for (let p = 1; p < terms; p += 1) powers.push(powers[p - 1] * scale(xs[i]));
    for (let r = 0; r < terms; r += 1) {
      for (let c = 0; c < terms; c += 1) normal[r][c] += powers[r] * powers[c];
      normal[r][terms] += powers[r] * ys[i];
    }
  }
  for (let r = 0; r < terms; r += 1) normal[r][r] += 1e-9;

  // Gaussian elimination with partial pivoting.
  for (let col = 0; col < terms; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < terms; row += 1) {
      if (Math.abs(normal[row][col]) > Math.abs(normal[pivot][col])) pivot = row;
    }
    [normal[col], normal[pivot]] = [normal[pivot], normal[col]];
    const lead = normal[col][col];
    if (Math.abs(lead) < 1e-14) continue;
    for (let row = 0; row < terms; row += 1) {
      if (row === col) continue;
      const factor = normal[row][col] / lead;
      for (let c = col; c <= terms; c += 1) normal[row][c] -= factor * normal[col][c];
    }
  }

  const coefficients = normal.map((row, index) =>
    Math.abs(row[index]) < 1e-14 ? 0 : row[terms] / row[index],
  );

  return (x: number) => {
    const t = scale(x);
    let total = 0;
    let power = 1;
    for (let i = 0; i < terms; i += 1) {
      total += coefficients[i] * power;
      power *= t;
    }
    return total;
  };
}

export function rootMeanSquaredError(
  xs: number[],
  ys: number[],
  fit: (x: number) => number,
): number {
  if (!xs.length) return 0;
  let total = 0;
  for (let i = 0; i < xs.length; i += 1) total += (ys[i] - fit(xs[i])) ** 2;
  return Math.sqrt(total / xs.length);
}

/**
 * The smooth process the overfitting explainer samples from.
 *
 * Deliberately gentle, a little under one full period across the plotted
 * range. A wigglier truth would need a high degree to fit honestly, which
 * muddles the lesson: the point is that a low degree is enough and anything
 * beyond it is chasing noise.
 */
export function trueFunction(x: number): number {
  return Math.sin(x) * 0.65 + x * 0.16;
}

export interface OverfitSample {
  trainX: number[];
  trainY: number[];
  testX: number[];
  testY: number[];
}

export function sampleOverfitData(seed: number, nTrain = 22, nTest = 220): OverfitSample {
  const rng = makeRng(seed);
  const noise = normalDraws(rng, nTrain + nTest);
  const trainX: number[] = [];
  const trainY: number[] = [];
  for (let i = 0; i < nTrain; i += 1) {
    const x = -3 + (6 * i) / (nTrain - 1);
    trainX.push(x);
    trainY.push(trueFunction(x) + noise[i] * 0.28);
  }
  const testX: number[] = [];
  const testY: number[] = [];
  for (let i = 0; i < nTest; i += 1) {
    const x = -3 + (6 * i) / (nTest - 1);
    testX.push(x);
    testY.push(trueFunction(x) + noise[nTrain + i] * 0.28);
  }
  return { trainX, trainY, testX, testY };
}

export interface Trial {
  returns: number[];
  equity: number[];
  sharpe: number;
}

/** Generate `count` return series that contain no signal whatsoever. */
export function generateNoiseTrials(
  seed: number,
  count: number,
  periods: number,
  volatility = 0.01,
): Trial[] {
  const rng = makeRng(seed);
  const trials: Trial[] = [];
  for (let index = 0; index < count; index += 1) {
    const returns = normalDraws(rng, periods).map((value) => value * volatility);
    trials.push({ returns, equity: cumulative(returns), sharpe: sharpe(returns) });
  }
  return trials;
}

export interface SearchSnapshot {
  nTrials: number;
  bestIndex: number;
  bestSharpe: number;
  bestAnnualised: number;
  expectedBest: number;
  excess: number;
}

/**
 * What a search over the first `nTrials` of a pool would report, and what it
 * should have expected to report given that none of them has any edge.
 */
export function searchSnapshot(
  trials: Trial[],
  nTrials: number,
  periodsPerYear: number,
): SearchSnapshot {
  const used = trials.slice(0, Math.max(1, nTrials));
  let bestIndex = 0;
  for (let i = 1; i < used.length; i += 1) {
    if (used[i].sharpe > used[bestIndex].sharpe) bestIndex = i;
  }
  const sharpes = used.map((trial) => trial.sharpe);
  const variance = used.length > 1 ? standardDeviation(sharpes) ** 2 : 0;
  const expectedBest = expectedMaximumSharpe(used.length, variance);
  const bestSharpe = used[bestIndex].sharpe;
  return {
    nTrials: used.length,
    bestIndex,
    bestSharpe,
    bestAnnualised: bestSharpe * Math.sqrt(periodsPerYear),
    expectedBest,
    excess: bestSharpe - expectedBest,
  };
}

/* ---------------------------------------------------------------------------
 * Alternative histories. Mirrors overfitlab.synthetic.
 * Resampling with longer blocks keeps more ordering intact, so sweeping the
 * block length destroys market structure by degrees.
 * ------------------------------------------------------------------------- */

/** Resample with geometric block lengths (Politis and Romano). */
export function stationaryBootstrap(
  returns: number[],
  expectedBlock: number,
  rng: () => number,
): number[] {
  const n = returns.length;
  const restart = 1 / Math.max(expectedBlock, 1);
  const out: number[] = new Array(n);
  let index = Math.floor(rng() * n) % n;
  for (let step = 0; step < n; step += 1) {
    if (step > 0) {
      index = rng() < restart ? Math.floor(rng() * n) % n : (index + 1) % n;
    }
    out[step] = returns[index];
  }
  return out;
}

/** Hold yesterday's direction. Profitable only if returns persist. */
export function momentumStrategy(market: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < market.length; i += 1) {
    out.push(Math.sign(market[i - 1]) * market[i]);
  }
  return out;
}

export function buyAndHold(market: number[]): number[] {
  return market;
}

/** A market with tunable momentum: r_t = phi * r_{t-1} + shock + drift. */
export function autocorrelatedMarket(
  seed: number,
  periods: number,
  phi: number,
  drift = 0.0003,
  volatility = 0.011,
): number[] {
  const shocks = normalDraws(makeRng(seed), periods).map((v) => v * volatility);
  const out: number[] = [shocks[0]];
  for (let i = 1; i < periods; i += 1) {
    out.push(phi * out[i - 1] + shocks[i] + drift);
  }
  return out;
}

export interface PathStressLevel {
  blockSize: number;
  medianAnnualised: number;
  p95Annualised: number;
  percentile: number;
  pValue: number;
}

/**
 * Locate the real result inside the synthetic distribution. Mirrors
 * `_rank_statistics` in the Python package.
 *
 * A strategy that ignores ordering scores the same on every generated path,
 * and those scores differ from the real one only in the last bits, because
 * the returns are summed in a different order. Counting strictly below turns
 * that rounding into an arbitrary percentile, so ties are detected within a
 * tolerance and split evenly.
 */
function rankStatistics(scores: number[], observed: number): { percentile: number; pValue: number } {
  const tolerance = 1e-9 * Math.max(Math.abs(observed), 1);
  let below = 0;
  let tied = 0;
  let atLeast = 0;
  for (const score of scores) {
    if (Math.abs(score - observed) <= tolerance) {
      tied += 1;
      atLeast += 1;
    } else if (score < observed) {
      below += 1;
    } else {
      atLeast += 1;
    }
  }
  return {
    percentile: (100 * (below + 0.5 * tied)) / scores.length,
    pValue: (1 + atLeast) / (scores.length + 1),
  };
}

/** Rerun a strategy on synthetic markets at several block lengths. */
export function pathStress(
  strategy: (market: number[]) => number[],
  market: number[],
  blockSizes: number[],
  nPaths: number,
  seed: number,
  periodsPerYear = 252,
): { observedAnnualised: number; levels: PathStressLevel[] } {
  const annualiser = Math.sqrt(periodsPerYear);
  const observed = sharpe(strategy(market));
  const levels: PathStressLevel[] = [];

  for (let position = 0; position < blockSizes.length; position += 1) {
    const block = blockSizes[position];
    const rng = makeRng(seed * 7919 + position);
    const scores: number[] = [];
    for (let path = 0; path < nPaths; path += 1) {
      scores.push(sharpe(strategy(blockPermutation(market, block, rng))));
    }
    const ranks = rankStatistics(scores, observed);
    scores.sort((a, b) => a - b);
    const quantile = (q: number) =>
      scores[Math.min(scores.length - 1, Math.floor(q * scores.length))];
    levels.push({
      blockSize: block,
      medianAnnualised: quantile(0.5) * annualiser,
      p95Annualised: quantile(0.95) * annualiser,
      percentile: ranks.percentile,
      pValue: ranks.pValue,
    });
  }
  return { observedAnnualised: observed * annualiser, levels };
}

/* ---------------------------------------------------------------------------
 * The two selection statistics, in the browser. Mirrors the Python package.
 * ------------------------------------------------------------------------- */

function combinations(items: number[], choose: number): number[][] {
  const out: number[][] = [];
  const current: number[] = [];
  const walk = (start: number) => {
    if (current.length === choose) {
      out.push(current.slice());
      return;
    }
    for (let i = start; i < items.length; i += 1) {
      current.push(items[i]);
      walk(i + 1);
      current.pop();
    }
  };
  walk(0);
  return out;
}

export interface DeflatedResult {
  observedSharpe: number;
  observedAnnualised: number;
  expectedMax: number;
  expectedMaxAnnualised: number;
  deflated: number;
  nTrials: number;
}

/** Deflated Sharpe ratio for the best column of a trial matrix. */
export function deflatedSharpe(
  trials: number[][],
  periodsPerYear = 252,
): DeflatedResult {
  const nTrials = trials.length;
  const sharpes = trials.map(sharpe);
  let bestIndex = 0;
  for (let i = 1; i < nTrials; i += 1) {
    if (sharpes[i] > sharpes[bestIndex]) bestIndex = i;
  }
  const observed = sharpes[bestIndex];
  const variance = nTrials > 1 ? standardDeviation(sharpes) ** 2 : 0;
  const expectedMax = expectedMaximumSharpe(nTrials, variance);
  const best = trials[bestIndex];
  const n = best.length;

  // Probabilistic Sharpe ratio of the observed against the deflated benchmark,
  // using the standard error that accounts for skew and kurtosis.
  const m = mean(best);
  const sd = standardDeviation(best);
  let skew = 0;
  let kurt = 0;
  if (sd > 0) {
    for (const value of best) {
      skew += ((value - m) / sd) ** 3;
      kurt += ((value - m) / sd) ** 4;
    }
    skew /= n;
    kurt /= n;
  }
  const denominator = Math.sqrt(
    Math.max(1 - skew * observed + ((kurt - 1) / 4) * observed * observed, 1e-12),
  );
  const statistic = ((observed - expectedMax) * Math.sqrt(n - 1)) / denominator;
  const deflated = 0.5 * (1 + erf(statistic / Math.SQRT2));

  const annualiser = Math.sqrt(periodsPerYear);
  return {
    observedSharpe: observed,
    observedAnnualised: observed * annualiser,
    expectedMax,
    expectedMaxAnnualised: expectedMax * annualiser,
    deflated,
    nTrials,
  };
}

/** Abramowitz and Stegun 7.1.26 error function. */
export function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * z);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-z * z);
  return sign * y;
}

/** Probability of backtest overfitting by combinatorially symmetric CV. */
export function probabilityOfBacktestOverfitting(
  trials: number[][],
  nSplits = 10,
): { pbo: number; nCombinations: number } {
  const nConfigs = trials.length;
  const nPeriods = trials[0].length;
  const splits = Math.max(2, nSplits - (nSplits % 2));
  const size = Math.floor(nPeriods / splits);
  const blocks: number[][] = [];
  for (let i = 0; i < splits; i += 1) {
    blocks.push(
      Array.from({ length: size }, (_, k) => i * size + k),
    );
  }

  const half = splits / 2;
  const groups = combinations(
    Array.from({ length: splits }, (_, i) => i),
    half,
  );

  let below = 0;
  for (const inSample of groups) {
    const inSet = new Set(inSample);
    const isIndex: number[] = [];
    const oosIndex: number[] = [];
    for (let b = 0; b < splits; b += 1) {
      (inSet.has(b) ? isIndex : oosIndex).push(...blocks[b]);
    }
    const isScores = trials.map((col) => sharpe(isIndex.map((i) => col[i])));
    const oosScores = trials.map((col) => sharpe(oosIndex.map((i) => col[i])));

    let winner = 0;
    for (let c = 1; c < nConfigs; c += 1) {
      if (isScores[c] > isScores[winner]) winner = c;
    }
    // Where the in-sample winner ranks out of sample. Below the median means
    // the selection did worse than picking at random would have.
    const beaten = oosScores.filter((score) => score < oosScores[winner]).length;
    if ((beaten + 1) / (nConfigs + 1) < 0.5) below += 1;
  }
  return { pbo: below / groups.length, nCombinations: groups.length };
}

/**
 * Cut the series into consecutive blocks and permute their order.
 *
 * Samples without replacement, so every observation appears exactly once and
 * the mean, variance, skew and extremes are identical to the source. Only the
 * arrangement changes. The bootstrap above draws with replacement and does not
 * have that property, which is why this is what the generator uses.
 */
export function blockPermutation(
  returns: number[],
  blockSize: number,
  rng: () => number,
): number[] {
  const blockCount = Math.ceil(returns.length / blockSize);
  const order = Array.from({ length: blockCount }, (_, index) => index);
  // Fisher-Yates over the block order.
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return applyBlockOrder(returns, blockSize, order);
}

/**
 * Cut into consecutive blocks and concatenate them in the given order.
 *
 * Split out from the random draw so the block arithmetic can be checked
 * against a fixed expected output, including the ragged final block. The
 * Python package has the same function and both are tested against one shared
 * fixture, because a cross language test catches the two drifting apart but
 * not both agreeing on the same wrong thing.
 */
export function applyBlockOrder(
  returns: number[],
  blockSize: number,
  order: number[],
): number[] {
  if (blockSize < 1) throw new Error("blockSize must be at least 1");
  const blocks: number[][] = [];
  for (let start = 0; start < returns.length; start += blockSize) {
    blocks.push(returns.slice(start, start + blockSize));
  }
  const sorted = [...order].sort((a, b) => a - b);
  const expected = blocks.map((_, index) => index);
  if (sorted.length !== expected.length || sorted.some((v, i) => v !== expected[i])) {
    throw new Error(`order must be a permutation of 0..${blocks.length - 1}`);
  }
  return order.flatMap((index) => blocks[index]);
}

/* ---------------------------------------------------------------------------
 * Reading the two selection statistics together.
 * ------------------------------------------------------------------------- */

export type SelectionVerdict =
  | "clears both"
  | "fails both"
  | "fails the luck bar"
  | "unstable selection";

/**
 * Combine the deflated Sharpe ratio and the overfitting probability.
 *
 * Pulled out of the component so it can be tested. It was previously inline
 * and compared the overfitting result object against a number rather than its
 * `pbo` field, which is always false, so a set of trials that passed both
 * checks was told its selection was unreliable.
 */
export function selectionVerdict(deflated: number, pbo: number): SelectionVerdict {
  const survives = deflated >= 0.95;
  const stable = pbo <= 0.5;
  if (survives && stable) return "clears both";
  if (!survives && !stable) return "fails both";
  if (!survives) return "fails the luck bar";
  return "unstable selection";
}
