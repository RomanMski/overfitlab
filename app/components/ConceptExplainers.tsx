"use client";

import { useMemo, useState, type ReactNode } from "react";
import { correctedNullRank } from "../lib/analysis";

type SyntheticScenario = "marginals" | "structure" | "copies";

interface ConceptFrameProps {
  number: string;
  eyebrow: string;
  title: string;
  summary: string;
  formula: ReactNode;
  asks: string;
  reads: string;
  cannot: string;
  children: ReactNode;
}

interface Point {
  x: number;
  y: number;
}

const clamp = (value: number, minimum = 0, maximum = 1) =>
  Math.min(maximum, Math.max(minimum, value));

const asPercent = (value: number, digits = 0) =>
  `${(value * 100).toFixed(digits)}%`;

function deterministicNoise(index: number, salt: number) {
  const raw = Math.sin((index + 1) * 12.9898 + salt * 78.233) * 43758.5453;
  return (raw - Math.floor(raw)) * 2 - 1;
}

function quantile(values: number[], probability: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function ConceptFrame({
  number,
  eyebrow,
  title,
  summary,
  formula,
  asks,
  reads,
  cannot,
  children,
}: ConceptFrameProps) {
  return (
    <article className="sf-learn-card" aria-labelledby={`sf-learn-${number}-title`}>
      <header className="sf-learn-card__header">
        <span className="sf-learn-card__number" aria-hidden="true">
          {number}
        </span>
        <div>
          <p className="sf-learn-card__eyebrow">{eyebrow}</p>
          <h3 id={`sf-learn-${number}-title`}>{title}</h3>
          <p className="sf-learn-card__summary">{summary}</p>
        </div>
      </header>

      <div className="sf-learn-card__visual">{children}</div>

      <div className="sf-learn-formula">
        <span className="sf-learn-formula__label">Quick formula</span>
        <code>{formula}</code>
      </div>

      <dl className="sf-learn-interpretation">
        <div className="sf-learn-interpretation__item">
          <dt>Question</dt>
          <dd>{asks}</dd>
        </div>
        <div className="sf-learn-interpretation__item">
          <dt>Read the result</dt>
          <dd>{reads}</dd>
        </div>
        <div className="sf-learn-interpretation__item sf-learn-interpretation__item--limit">
          <dt>Limit</dt>
          <dd>{cannot}</dd>
        </div>
      </dl>
    </article>
  );
}

function OverfittingExplainer() {
  const [complexity, setComplexity] = useState(8);
  const points = useMemo(
    () =>
      Array.from({ length: 10 }, (_, index) => {
        const level = index + 1;
        const trainLoss = clamp(0.38 - level * 0.027, 0.08, 0.5);
        const auditLoss = clamp(0.36 - level * 0.05 + level * level * 0.006, 0.08, 0.6);
        return { level, trainLoss, auditLoss };
      }),
    [],
  );
  const active = points[complexity - 1];
  const gap = active.auditLoss - active.trainLoss;

  return (
    <ConceptFrame
      number="01"
      eyebrow="Train versus audit"
      title="Training loss versus unseen loss"
      summary="A flexible model can learn real structure, but it can also learn accidents that only occurred in its training rows."
      formula={
        <>
          G = L<sub>audit</sub> − L<sub>train</sub> = {gap.toFixed(3)}
        </>
      }
      asks="Does the model keep its performance on rows that had no part in fitting or tuning it?"
      reads="Move model flexibility to the right. Training loss usually falls. Concern begins when unseen loss rises and the paired loss gap opens."
      cannot="One small train/test gap does not certify a model. A lucky split, duplicated rows, leakage, or a shifted future population can still mislead it."
    >
      <div className="sf-learn-control">
        <label htmlFor="sf-learn-complexity">
          Model flexibility
          <output htmlFor="sf-learn-complexity">
            {complexity <= 3 ? "simple" : complexity <= 6 ? "moderate" : "high"}, {complexity}/10
          </output>
        </label>
        <input
          id="sf-learn-complexity"
          type="range"
          min="1"
          max="10"
          step="1"
          value={complexity}
          onChange={(event) => setComplexity(Number(event.target.value))}
        />
      </div>

      <div
        className="sf-learn-overfit-plot"
        role="img"
        aria-label={`At flexibility ${complexity}, training loss is ${active.trainLoss.toFixed(3)} and unseen loss is ${active.auditLoss.toFixed(3)}. The generalization gap is ${gap.toFixed(3)}.`}
      >
        <div className="sf-learn-axis-label sf-learn-axis-label--top">lower loss is better</div>
        <div className="sf-learn-overfit-plot__columns" aria-hidden="true">
          {points.map((point) => (
            <div
              className={`sf-learn-overfit-plot__column${
                point.level === complexity ? " sf-learn-overfit-plot__column--active" : ""
              }`}
              key={point.level}
            >
              <span
                className="sf-learn-overfit-plot__point sf-learn-overfit-plot__point--train"
                style={{ bottom: `${(1 - point.trainLoss) * 100}%` }}
              />
              <span
                className="sf-learn-overfit-plot__point sf-learn-overfit-plot__point--unseen"
                style={{ bottom: `${(1 - point.auditLoss) * 100}%` }}
              />
              <span className="sf-learn-overfit-plot__tick">{point.level}</span>
            </div>
          ))}
        </div>
        <div className="sf-learn-axis-label sf-learn-axis-label--bottom">
          model flexibility →
        </div>
      </div>

      <div className="sf-learn-legend" aria-label="Chart legend">
        <span>
          <i className="sf-learn-key sf-learn-key--train" aria-hidden="true" />
          Training loss <strong>{active.trainLoss.toFixed(3)}</strong>
        </span>
        <span>
          <i className="sf-learn-key sf-learn-key--unseen" aria-hidden="true" />
          Unseen loss <strong>{active.auditLoss.toFixed(3)}</strong>
        </span>
      </div>
    </ConceptFrame>
  );
}

function NoiseInjectionExplainer() {
  const [severity, setSeverity] = useState(35);
  const basePoints = useMemo<Point[]>(
    () =>
      Array.from({ length: 22 }, (_, index) => {
        const x = 7 + (index / 21) * 86;
        return {
          x,
          y: clamp(0.86 - x / 125 + deterministicNoise(index, 2) * 0.045, 0.08, 0.92) * 100,
        };
      }),
    [],
  );
  const noisyPoints = useMemo(
    () =>
      basePoints.map((point, index) => ({
        x: clamp(
          (point.x + deterministicNoise(index, 11) * severity * 0.28) / 100,
          0.03,
          0.97,
        ) * 100,
        y: point.y,
      })),
    [basePoints, severity],
  );
  const referenceLoss = 0.3;
  const cleanLoss = 0.18;
  const stressedLoss = cleanLoss + (severity / 100) * 0.09 + (severity / 100) ** 2 * 0.05;
  const normalization = referenceLoss - cleanLoss;
  const retainedSkill = 1 - (stressedLoss - cleanLoss) / normalization;

  return (
    <ConceptFrame
      number="02"
      eyebrow="Feature perturbation"
      title="Adding noise to numeric inputs"
      summary="We add measured errors to features, such as a noisy sensor or rounded value, without changing the hidden answer."
      formula={
        <>
          R = 1 − (L<sub>stress</sub> − L<sub>clean</sub>) ÷ D = {retainedSkill.toFixed(2)}
        </>
      }
      asks="Is the fitted model relying on a signal that collapses under small, plausible measurement errors?"
      reads="Increase severity gradually. A gentle downward curve is ordinary; a cliff at low severity is a robustness warning worth investigating."
      cannot="Sensitivity to noise is not, by itself, proof of overfitting. The perturbation may be unrealistic, or the real task may genuinely require precise measurements."
    >
      <div className="sf-learn-control">
        <label htmlFor="sf-learn-noise">
          Noise severity
          <output htmlFor="sf-learn-noise">{severity}% of each feature&apos;s typical training spread</output>
        </label>
        <input
          id="sf-learn-noise"
          type="range"
          min="0"
          max="100"
          step="5"
          value={severity}
          onChange={(event) => setSeverity(Number(event.target.value))}
        />
      </div>

      <div
        className="sf-learn-scatter"
        role="img"
        aria-label={`${basePoints.length} example observations before and after ${severity}% noise. The model retains ${asPercent(retainedSkill, 1)} of its clean skill.`}
      >
        <span className="sf-learn-scatter__ylabel" aria-hidden="true">
          outcome
        </span>
        <div className="sf-learn-scatter__field" aria-hidden="true">
          {basePoints.map((point, index) => (
            <span
              className="sf-learn-scatter__point sf-learn-scatter__point--clean"
              key={`clean-${index}`}
              style={{ left: `${point.x}%`, bottom: `${point.y}%` }}
            />
          ))}
          {noisyPoints.map((point, index) => (
            <span
              className="sf-learn-scatter__point sf-learn-scatter__point--stressed"
              key={`noise-${index}`}
              style={{ left: `${point.x}%`, bottom: `${point.y}%` }}
            />
          ))}
        </div>
        <span className="sf-learn-scatter__xlabel" aria-hidden="true">
          measured feature →
        </span>
      </div>

      <div className="sf-learn-readout">
        <span>Clean loss <strong>{cleanLoss.toFixed(3)}</strong></span>
        <span>Stressed loss <strong>{stressedLoss.toFixed(3)}</strong></span>
        <span>Performance retained <strong>{asPercent(retainedSkill, 1)}</strong></span>
      </div>
    </ConceptFrame>
  );
}

function MonteCarloExplainer() {
  const [runs, setRuns] = useState(16);
  const allScores = useMemo(
    () =>
      Array.from({ length: 40 }, (_, index) =>
        clamp(
          0.79 +
            deterministicNoise(index, 5) * 0.075 +
            deterministicNoise(index * 3, 17) * 0.025,
          0.58,
          0.94,
        ),
      ),
    [],
  );
  const scores = allScores.slice(0, runs);
  const center = quantile(scores, 0.5);
  const low = quantile(scores, 0.05);
  const high = quantile(scores, 0.95);

  return (
    <ConceptFrame
      number="03"
      eyebrow="Repeated holdout"
      title="Why the audit repeats the split"
      summary="A single train and audit split depends on which rows happened to land on each side. Repeated splits show that sensitivity."
      formula={
        <>
          typical score = median(score₁, …, score<sub>{runs}</sub>) = {center.toFixed(3)}
        </>
      }
      asks="How much would the reported model score change if different eligible rows happened to land in the unseen set?"
      reads="Each dot is one complete repeat. StressFold reports the median and the 5th to 95th percentile range. A wide cluster means the result depends heavily on the split."
      cannot="These holdouts overlap, so they are not independent new datasets. More repeats stabilize this resampling summary but do not repair bias, leakage, or time drift."
    >
      <div className="sf-learn-control">
        <label htmlFor="sf-learn-runs">
          Protocol repeats
          <output htmlFor="sf-learn-runs">{runs} runs</output>
        </label>
        <input
          id="sf-learn-runs"
          type="range"
          min="1"
          max="40"
          step="1"
          value={runs}
          onChange={(event) => setRuns(Number(event.target.value))}
        />
      </div>

      <div
        className="sf-learn-monte-carlo"
        role="img"
        aria-label={`${runs} repeated holdout scores. The first split scores ${asPercent(scores[0], 1)}. Their median is ${asPercent(center, 1)}, with a fifth to ninety-fifth percentile range from ${asPercent(low, 1)} to ${asPercent(high, 1)}.`}
      >
        <div className="sf-learn-monte-carlo__scale" aria-hidden="true">
          <span>60%</span>
          <span>70%</span>
          <span>80%</span>
          <span>90%</span>
        </div>
        <div className="sf-learn-monte-carlo__strip" aria-hidden="true">
          {scores.map((score, index) => (
            <span
              className={`sf-learn-monte-carlo__dot${
                index === 0 ? " sf-learn-monte-carlo__dot--first" : ""
              }`}
              key={index}
              style={{
                left: `${clamp((score - 0.58) / 0.38) * 100}%`,
                top: `${20 + ((index * 37) % 5) * 13}%`,
              }}
            >
              <span className="sf-learn-sr-only">
                Run {index + 1}: {asPercent(score, 1)}
              </span>
            </span>
          ))}
          <span
            className="sf-learn-monte-carlo__median"
            style={{ left: `${clamp((center - 0.58) / 0.38) * 100}%` }}
          />
        </div>
      </div>

      <div className="sf-learn-readout">
        <span>One split <strong>{asPercent(scores[0], 1)}</strong></span>
        <span>Median of {runs} <strong>{asPercent(center, 1)}</strong></span>
        <span>Middle 90% <strong>{asPercent(low, 1)} to {asPercent(high, 1)}</strong></span>
      </div>
    </ConceptFrame>
  );
}

const ORIGINAL_LABELS = ["Yes", "No", "No", "Yes", "No", "Yes", "Yes", "No"];
const PERMUTATION_ORDERS = [
  [1, 6, 3, 0, 7, 2, 5, 4],
  [4, 0, 6, 2, 5, 7, 1, 3],
  [2, 5, 7, 1, 3, 0, 4, 6],
  [7, 3, 0, 5, 1, 4, 6, 2],
];
const PERMUTED_SCORES = [0.51, 0.47, 0.53, 0.49];

function LabelPermutationExplainer() {
  const [round, setRound] = useState(0);
  const permutedLabels = PERMUTATION_ORDERS[round].map((index) => ORIGINAL_LABELS[index]);
  const originalScore = 0.84;
  const permutedScore = PERMUTED_SCORES[round];
  const correctedPercentile = correctedNullRank(originalScore, PERMUTED_SCORES);

  return (
    <ConceptFrame
      number="04"
      eyebrow="Null comparison"
      title="Why the audit shuffles answers"
      summary="Within each repeated split, we shuffle the training answers and refit the model on the same prepared feature rows."
      formula={
        <>
          P = 100 × (½ + # lower + ½ × # tied) ÷ (B + 1) = {correctedPercentile.toFixed(0)}
        </>
      }
      asks="Can this pipeline still appear predictive after we deliberately remove the real signal it is supposed to learn?"
      reads="For balanced yes or no data, shuffled AUROC should settle near 50%. The corrected percentile ranks the real score against all shuffled runs; exact ties receive half weight."
      cannot="A passed permutation check does not prove the original relationship is causal, useful, or stable. It only shows this particular null check did not expose the pipeline."
    >
      <div className="sf-learn-permutation">
        <div className="sf-learn-permutation__table-wrap">
          <table>
            <caption>Eight example rows before and after shuffling the answer</caption>
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Activity</th>
                <th scope="col">Real answer</th>
                <th scope="col">Shuffled answer</th>
              </tr>
            </thead>
            <tbody>
              {ORIGINAL_LABELS.map((label, index) => (
                <tr key={index}>
                  <th scope="row">{index + 1}</th>
                  <td>{[8, 2, 4, 9, 3, 7, 6, 1][index]} visits</td>
                  <td>{label}</td>
                  <td className={label === permutedLabels[index] ? "" : "sf-learn-permutation__changed"}>
                    {permutedLabels[index]}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="sf-learn-permutation__scores">
          <div>
            <span>Real answers</span>
            <div className="sf-learn-meter" aria-hidden="true">
              <i style={{ width: `${originalScore * 100}%` }} />
            </div>
            <strong>{asPercent(originalScore, 0)}</strong>
          </div>
          <div>
            <span>Shuffled answers</span>
            <div className="sf-learn-meter sf-learn-meter--null" aria-hidden="true">
              <i style={{ width: `${permutedScore * 100}%` }} />
              <b style={{ left: "50%" }} />
            </div>
            <strong>{asPercent(permutedScore, 0)}</strong>
          </div>
        </div>

        <button
          className="sf-learn-button"
          type="button"
          onClick={() => setRound((current) => (current + 1) % PERMUTATION_ORDERS.length)}
          aria-label={`Run another deterministic label shuffle. Current shuffle is ${round + 1} of ${PERMUTATION_ORDERS.length}.`}
        >
          Run another shuffle
          <span aria-hidden="true">↻</span>
        </button>
      </div>
    </ConceptFrame>
  );
}

const ORIGINAL_SYNTHETIC_POINTS: Point[] = Array.from({ length: 16 }, (_, index) => ({
  x: 8 + ((index * 17) % 84),
  y: clamp(
    0.16 + (((index * 17) % 84) / 100) * 0.72 + deterministicNoise(index, 31) * 0.08,
    0.06,
    0.94,
  ) * 100,
}));

const SYNTHETIC_SCENARIOS: Record<
  SyntheticScenario,
  {
    label: string;
    short: string;
    shape: number;
    relation: number;
    copying: "low" | "moderate" | "high";
    points: Point[];
    verdict: string;
  }
> = {
  marginals: {
    label: "Looks similar",
    short: "Same ranges, wrong relationships",
    shape: 0.93,
    relation: 0.24,
    copying: "low",
    points: ORIGINAL_SYNTHETIC_POINTS.map((point, index, all) => ({
      x: point.x,
      y: all[(index * 5 + 3) % all.length].y,
    })),
    verdict: "Useful for testing simple range changes. It is not a faithful replica of the process that produced the data.",
  },
  structure: {
    label: "Preserves structure",
    short: "Similar ranges and relationships",
    shape: 0.91,
    relation: 0.86,
    copying: "low",
    points: ORIGINAL_SYNTHETIC_POINTS.map((point, index) => ({
      x: clamp((point.x + deterministicNoise(index, 43) * 6) / 100, 0.04, 0.96) * 100,
      y: clamp((point.y + deterministicNoise(index, 47) * 7) / 100, 0.04, 0.96) * 100,
    })),
    verdict: "A plausible stress data candidate, provided downstream utility and subgroup behavior are checked too.",
  },
  copies: {
    label: "Copies records",
    short: "Excellent match, unsafe originality",
    shape: 0.99,
    relation: 0.98,
    copying: "high",
    points: ORIGINAL_SYNTHETIC_POINTS.map((point, index) => ({
      x: point.x + (index % 3 === 0 ? 0.6 : 0),
      y: point.y + (index % 3 === 0 ? -0.5 : 0),
    })),
    verdict: "Reject as independent evidence: near-duplicates can leak private rows and make evaluation look better than it is.",
  },
};

function SyntheticDataExplainer() {
  const [scenario, setScenario] = useState<SyntheticScenario>("marginals");
  const selected = SYNTHETIC_SCENARIOS[scenario];

  return (
    <ConceptFrame
      number="05"
      eyebrow="Optional generator check"
      title="Generated data is a separate validation problem"
      summary="A generator can create useful stress cases. Before trusting them, check distributions, relationships, downstream behavior, and whether it merely copied real rows."
      formula={
        <>
          generator audit = fidelity + coverage + utility − copying risk
        </>
      }
      asks="Is this generator suitable for creating controlled challenges without pretending its rows are fresh observations from the real world?"
      reads="Compare more than each column’s shape. Good stress data should preserve the relationships relevant to the task, cover important groups, and remain distinct from source records."
      cannot="A beautiful synthetic match cannot prove real-world generalization or increase the amount of independent evidence. The generator learned from the same finite history."
    >
      <div className="sf-learn-synthetic__tabs" aria-label="Synthetic generator examples">
        {(Object.keys(SYNTHETIC_SCENARIOS) as SyntheticScenario[]).map((key) => (
          <button
            type="button"
            key={key}
            aria-pressed={scenario === key}
            className={scenario === key ? "sf-learn-synthetic__tab sf-learn-synthetic__tab--active" : "sf-learn-synthetic__tab"}
            onClick={() => setScenario(key)}
          >
            <strong>{SYNTHETIC_SCENARIOS[key].label}</strong>
            <span>{SYNTHETIC_SCENARIOS[key].short}</span>
          </button>
        ))}
      </div>

      <div className="sf-learn-synthetic__comparison">
        <div>
          <p>Observed data</p>
          <div
            className="sf-learn-synthetic__plot"
            role="img"
            aria-label="Observed example data with a strong positive relationship between two features."
          >
            {ORIGINAL_SYNTHETIC_POINTS.map((point, index) => (
              <i
                key={index}
                className="sf-learn-synthetic__point sf-learn-synthetic__point--observed"
                style={{ left: `${point.x}%`, bottom: `${point.y}%` }}
              />
            ))}
          </div>
        </div>
        <div>
          <p>Generated data, {selected.label}</p>
          <div
            className="sf-learn-synthetic__plot"
            role="img"
            aria-label={`Generated example data. Column-shape fidelity is ${asPercent(selected.shape)}, relationship fidelity is ${asPercent(selected.relation)}, and copying risk is ${selected.copying}.`}
          >
            {selected.points.map((point, index) => (
              <i
                key={index}
                className="sf-learn-synthetic__point sf-learn-synthetic__point--generated"
                style={{ left: `${point.x}%`, bottom: `${point.y}%` }}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="sf-learn-synthetic__checks">
        <span>
          Column shapes <strong>{asPercent(selected.shape)}</strong>
        </span>
        <span>
          Relationships <strong>{asPercent(selected.relation)}</strong>
        </span>
        <span>
          Copying risk <strong className={`sf-learn-risk sf-learn-risk--${selected.copying}`}>{selected.copying}</strong>
        </span>
      </div>
      <p className="sf-learn-synthetic__verdict" aria-live="polite">
        <strong>Boundary:</strong> {selected.verdict}
      </p>
    </ConceptFrame>
  );
}

export function ConceptExplainers() {
  return (
    <section className="sf-learn" id="learn" aria-labelledby="sf-learn-title">
      <header className="sf-learn__intro">
        <p className="sf-learn__kicker">Visual explanations</p>
        <h2 id="sf-learn-title">Explore how each test behaves.</h2>
        <p>
          Move the controls to see the ideas behind the audit. The exact equations and symbol
          definitions are in the mathematics section above.
        </p>
      </header>

      <div className="sf-learn__sequence" aria-label="Generalization test sequence">
        <span>Fit</span>
        <i aria-hidden="true">→</i>
        <span>Hold out</span>
        <i aria-hidden="true">→</i>
        <span>Stress</span>
        <i aria-hidden="true">→</i>
        <span>Repeat</span>
        <i aria-hidden="true">→</i>
        <span>Shuffle labels</span>
      </div>

      <div className="sf-learn__cards">
        <OverfittingExplainer />
        <NoiseInjectionExplainer />
        <MonteCarloExplainer />
        <LabelPermutationExplainer />
        <SyntheticDataExplainer />
      </div>
    </section>
  );
}

export default ConceptExplainers;
