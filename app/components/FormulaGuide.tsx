"use client";

import { useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";

type FormulaId = "gap" | "noise" | "retained" | "monte-carlo" | "permutation";

interface FormulaTerm {
  id: string;
  symbol: string;
  name: string;
  definition: string;
}

interface EquationSegment {
  text: string;
  term?: string;
  spoken?: string;
}

interface FormulaDefinition {
  id: FormulaId;
  tab: string;
  title: string;
  question: string;
  convention?: string;
  equation: EquationSegment[];
  terms: FormulaTerm[];
  example: {
    values: string;
    steps: string[];
    result: string;
  };
  interpretation: string;
  unusual: string;
  boundary: string;
}

const FORMULAS: FormulaDefinition[] = [
  {
    id: "gap",
    tab: "Generalization gap",
    title: "Generalization gap within one audit block",
    question: "How much worse is the fitted model on held-out rows than on its training rows?",
    convention: "Loss is an error measure here, so smaller values are better.",
    equation: [
      { text: "G_b", term: "gap", spoken: "G sub b" },
      { text: " = " },
      { text: "L_audit,b", term: "audit-loss", spoken: "L audit sub b" },
      { text: " − " },
      { text: "L_train,b", term: "train-loss", spoken: "L train sub b" },
    ],
    terms: [
      {
        id: "gap",
        symbol: "G_b",
        name: "generalization gap",
        definition: "Audit loss minus training loss in block b, measured in the units of the selected loss.",
      },
      {
        id: "audit-loss",
        symbol: "L_audit,b",
        name: "audit loss",
        definition: "Loss on rows kept out of fitting and model selection in block b.",
      },
      {
        id: "train-loss",
        symbol: "L_train,b",
        name: "training loss",
        definition: "Loss on rows used to fit the corresponding model in block b, using the same loss function.",
      },
      {
        id: "block",
        symbol: "b",
        name: "audit block",
        definition: "The repeat, fold, time block, or group split defining one train-and-audit comparison.",
      },
      {
        id: "loss",
        symbol: "L",
        name: "loss function",
        definition: "A declared error measure such as log loss, Brier loss, or mean squared error.",
      },
    ],
    example: {
      values: "Training loss = 0.180; audit loss = 0.270.",
      steps: ["G_b = L_audit,b − L_train,b", "G_b = 0.270 − 0.180"],
      result: "G_b = 0.090 loss units",
    },
    interpretation: "A positive value means the model made more error on held-out rows. Larger positive gaps deserve attention, especially when they recur across blocks.",
    unusual: "A negative gap means the audit loss was lower. An easier audit sample, a small sample, or stricter training-time regularization can cause this. It is not evidence of reverse overfitting.",
    boundary: "The gap describes this split and loss. Alone, it cannot distinguish memorization from leakage, sampling luck, population shift, or a mismatched loss.",
  },
  {
    id: "noise",
    tab: "Feature noise",
    title: "Standardized feature-noise injection",
    question: "What does a controlled measurement error do to one standardized audit value?",
    equation: [
      { text: "z′_ij", term: "perturbed", spoken: "z prime i j" },
      { text: " = " },
      { text: "z_ij", term: "original", spoken: "z i j" },
      { text: " + " },
      { text: "λ", term: "severity", spoken: "lambda" },
      { text: " max(" },
      { text: "0.15", term: "floor", spoken: "zero point one five" },
      { text: ", " },
      { text: "r_j", term: "robust-scale", spoken: "r sub j" },
      { text: "/" },
      { text: "σ_j", term: "standard-scale", spoken: "sigma sub j" },
      { text: ") " },
      { text: "ε_ij", term: "draw", spoken: "epsilon i j" },
    ],
    terms: [
      {
        id: "perturbed",
        symbol: "z′_ij",
        name: "perturbed standardized value",
        definition: "The value supplied to the fixed fitted model for audit row i and feature j after noise is added.",
      },
      {
        id: "original",
        symbol: "z_ij",
        name: "clean standardized value",
        definition: "Feature j in audit row i after training-fold centering, imputation, and scaling, before noise.",
      },
      {
        id: "severity",
        symbol: "λ",
        name: "severity",
        definition: "A dimensionless multiplier fixed by the protocol. Zero means no disturbance.",
      },
      {
        id: "robust-scale",
        symbol: "r_j",
        name: "robust feature scale",
        definition: "Training-fold robust scale for feature j, using IQR/1.349 with a MAD fallback.",
      },
      {
        id: "standard-scale",
        symbol: "σ_j",
        name: "standard-deviation scale",
        definition: "Training-fold standard deviation for feature j used to standardize the feature.",
      },
      {
        id: "floor",
        symbol: "0.15",
        name: "minimum standardized noise scale",
        definition: "A fixed floor preventing a very small robust-to-standard scale ratio from making the stress vanish.",
      },
      {
        id: "draw",
        symbol: "ε_ij",
        name: "noise draw",
        definition: "A seeded standard-Gaussian draw for audit row i and feature j.",
      },
      {
        id: "indices",
        symbol: "i, j",
        name: "row and feature indices",
        definition: "i identifies an audit row and j identifies one numeric feature.",
      },
    ],
    example: {
      values: "z_ij = 0.80; λ = 0.25; r_j = 0.90; σ_j = 1.20; ε_ij = −0.60.",
      steps: [
        "max(0.15, r_j/σ_j) = max(0.15, 0.90/1.20) = 0.75",
        "z′_ij = 0.80 + (0.25)(0.75)(−0.60)",
        "z′_ij = 0.80 − 0.1125",
      ],
      result: "z′_ij = 0.6875",
    },
    interpretation: "The robust-to-standard scale ratio keeps the disturbance tied to each feature’s training-fold spread. The fixed model is evaluated again on these changed audit values.",
    unusual: "The 0.15 floor prevents near-zero stress. A large λ can still create implausible standardized values. Strong clipping or repeated boundary violations should be reported.",
    boundary: "This measures robustness to the stated noise model. It does not prove overfitting and may not represent a different deployment error mechanism.",
  },
  {
    id: "retained",
    tab: "Retained skill",
    title: "Skill retained after a stress test",
    question: "How much of the model’s improvement over a reference survives the disturbance?",
    convention: "Loss is smaller when the model is better. The denominator floor keeps weak-baseline cases finite.",
    equation: [
      { text: "D_b", term: "denominator", spoken: "D sub b" },
      { text: " = max(" },
      { text: "L_ref,b", term: "reference", spoken: "L reference sub b" },
      { text: " − " },
      { text: "L_clean,b", term: "clean", spoken: "L clean sub b" },
      { text: ", 0.01 L_ref,b, 10^-8);  " },
      { text: "R_b(λ)", term: "retained", spoken: "R sub b of lambda" },
      { text: " = 1 − (" },
      { text: "L_stress,b(λ)", term: "stress", spoken: "L stress sub b of lambda" },
      { text: " − " },
      { text: "L_clean,b", term: "clean", spoken: "L clean sub b" },
      { text: ") / " },
      { text: "D_b", term: "denominator", spoken: "D sub b" },
    ],
    terms: [
      {
        id: "denominator",
        symbol: "D_b",
        name: "protected clean-skill denominator",
        definition: "The largest of clean improvement over reference, 1% of reference loss, and 10^-8 in audit block b.",
      },
      {
        id: "retained",
        symbol: "R_b(λ)",
        name: "retained skill",
        definition: "The fraction of clean skill retained at severity λ in audit block b.",
      },
      {
        id: "reference",
        symbol: "L_ref,b",
        name: "reference loss",
        definition: "Loss of the training-fold constant reference on audit rows in block b.",
      },
      {
        id: "stress",
        symbol: "L_stress,b(λ)",
        name: "stressed-model loss",
        definition: "Loss under the selected stress at severity λ in audit block b.",
      },
      {
        id: "clean",
        symbol: "L_clean,b",
        name: "clean-model loss",
        definition: "Loss from the corresponding clean model evaluation in audit block b.",
      },
      {
        id: "severity",
        symbol: "λ",
        name: "stress severity",
        definition: "The declared strength of the selected stress operator.",
      },
      {
        id: "block",
        symbol: "b",
        name: "audit block",
        definition: "One complete split, fit, clean evaluation, and stressed evaluation.",
      },
      {
        id: "floors",
        symbol: "0.01, 10^-8",
        name: "denominator floors",
        definition: "Relative and absolute safeguards that prevent division by zero or an unstable tiny clean-skill denominator.",
      },
    ],
    example: {
      values: "In block b: reference loss = 0.693; clean loss = 0.410; stressed loss at λ = 0.495.",
      steps: [
        "D_b = max(0.693 − 0.410, 0.01(0.693), 10^-8)",
        "D_b = max(0.283, 0.00693, 0.00000001) = 0.283",
        "R_b(λ) = 1 − (0.495 − 0.410) / 0.283",
        "R_b(λ) = 1 − 0.085 / 0.283",
      ],
      result: "R_b(λ) = 0.700, so 70.0% of clean skill remains",
    },
    interpretation: "R = 1 means stressed and clean loss match. R = 0 means the stress consumed the protected clean-skill denominator. Values between zero and one show partial retention.",
    unusual: "R < 0 means the loss increase exceeded D_b. R > 1 means the stressed run beat the clean run. When clean improvement is tiny, a denominator floor is active and even small loss changes can produce extreme values.",
    boundary: "Retained skill is a paired, normalized response within block b. It is not a universal pass mark and depends on the reference, loss, denominator rule, and stress mechanism. The browser withholds curve rankings when the clean baseline does not reliably beat the constant reference.",
  },
  {
    id: "monte-carlo",
    tab: "Monte Carlo interval",
    title: "Median and central Monte Carlo range",
    question: "What result is typical across repeated valid splits, and how widely do the results vary?",
    equation: [
      { text: "M", term: "median", spoken: "M" },
      { text: " = median(" },
      { text: "S_1,…,S_B", term: "score", spoken: "scores S one through S B" },
      { text: "); [" },
      { text: "Q_0.05", term: "quantile", spoken: "Q point zero five" },
      { text: ", " },
      { text: "Q_0.95", term: "quantile", spoken: "Q point nine five" },
      { text: "]" },
    ],
    terms: [
      {
        id: "median",
        symbol: "M",
        name: "Monte Carlo median",
        definition: "The 50th percentile after ordering the B repeated results. With an even count, StressFold averages the two central values.",
      },
      {
        id: "score",
        symbol: "S_b",
        name: "score from repeat b",
        definition: "The declared metric from the complete pipeline in repeat b.",
      },
      {
        id: "repeat-count",
        symbol: "B",
        name: "number of repeats",
        definition: "The number of valid Monte Carlo repetitions included in the summary.",
      },
      {
        id: "quantile",
        symbol: "Q_p",
        name: "empirical quantile",
        definition: "The value below which proportion p of repeated scores falls, using the declared interpolation rule.",
      },
      {
        id: "probability",
        symbol: "p",
        name: "quantile probability",
        definition: "The requested ordered position. StressFold uses p = 0.05 and p = 0.95 for its central range.",
      },
      {
        id: "block",
        symbol: "b",
        name: "repeat index",
        definition: "One complete valid split, fit, stress, and evaluation run, from 1 through B.",
      },
    ],
    example: {
      values: "Ordered scores: 0.72, 0.76, 0.79, 0.81, 0.83, 0.86, 0.90; B = 7.",
      steps: [
        "M is the fourth value: M = 0.810",
        "Q_0.05: position 1.30; 0.72 + 0.30(0.76 − 0.72) = 0.732",
        "Q_0.95: position 6.70; 0.86 + 0.70(0.90 − 0.86) = 0.888",
      ],
      result: "M = 0.810; [Q_0.05, Q_0.95] = [0.732, 0.888]",
    },
    interpretation: "The median is the typical repeat. The 5th and 95th quantiles show the central spread produced by the declared split-and-seed procedure.",
    unusual: "A wide range signals split sensitivity. Identical endpoints may reflect stability, rounding, or too few distinct repeats. With small B, tail quantiles are coarse.",
    boundary: "This is Monte Carlo variation under the protocol. The repeats reuse one finite dataset, so they are not independent new samples. The range is not automatically a population confidence interval, and it omits untested shifts.",
  },
  {
    id: "permutation",
    tab: "Permutation null",
    title: "Observed score within a permutation null",
    question: "Where does the observed score rank among label-shuffled pipeline scores, with exact ties handled neutrally?",
    convention: "S is higher-is-better here: AUROC for classification or R² for regression. The browser uses one shuffled-target refit per repeated split.",
    equation: [
      { text: "S_obs = median(S_clean,1,…,S_clean,B);  " },
      { text: "P_null", term: "percentile", spoken: "P null" },
      { text: " = " },
      { text: "100", term: "scale", spoken: "one hundred" },
      { text: "(" },
      { text: "½", term: "correction", spoken: "one-half observed-score contribution" },
      { text: " + Σ[" },
      { text: "I", term: "indicator", spoken: "indicator I" },
      { text: "(" },
      { text: "S_π", term: "null-loss", spoken: "S pi" },
      { text: " < " },
      { text: "S_obs", term: "observed-loss", spoken: "S observed" },
      { text: ") + ½" },
      { text: "I", term: "indicator", spoken: "indicator I" },
      { text: "(" },
      { text: "S_π", term: "null-loss", spoken: "S pi" },
      { text: " = " },
      { text: "S_obs", term: "observed-loss", spoken: "S observed" },
      { text: ")]) / (" },
      { text: "K", term: "count", spoken: "K" },
      { text: " + " },
      { text: "1", spoken: "one observed score" },
      { text: ")" },
    ],
    terms: [
      {
        id: "percentile",
        symbol: "P_null",
        name: "null percentile",
        definition: "Corrected midrank of the observed score among the shuffled-target scores. Lower null scores count fully and exact ties count one half.",
      },
      {
        id: "count",
        symbol: "K",
        name: "number of permutations",
        definition: "Number of shuffled-target refits. In the browser, K equals the repeat count and each repeated split contributes one null score.",
      },
      {
        id: "index",
        symbol: "π",
        name: "permutation index",
        definition: "One label reassignment valid under the declared exchangeability scheme.",
      },
      {
        id: "null-loss",
        symbol: "S_π",
        name: "permuted score",
        definition: "AUROC or R² after the model is refit on shuffled training targets in one repeated split. Target-independent preprocessing from that split is reused.",
      },
      {
        id: "observed-loss",
        symbol: "S_obs",
        name: "observed score",
        definition: "Median clean AUROC or R² across the B repeated splits using observed, unshuffled targets.",
      },
      {
        id: "indicator",
        symbol: "I",
        name: "indicator function",
        definition: "Returns 1 when its comparison is true and 0 otherwise. A strictly lower null score contributes 1, and an exact tie contributes ½.",
      },
      {
        id: "correction",
        symbol: "+½",
        name: "finite-sample midpoint correction",
        definition: "Includes the observed score as the additional item and gives it half weight, so an all-tied null lands exactly at 50.",
      },
      {
        id: "scale",
        symbol: "100",
        name: "percentage scale",
        definition: "Converts the corrected fraction to percentage points.",
      },
    ],
    example: {
      values: "K = 24; 21 shuffled refits score lower than the median observed score and 2 tie it.",
      steps: [
        "P_null = 100(½ + 21 + ½ × 2) / (24 + 1)",
        "P_null = 2250 / 25",
      ],
      result: "P_null = 90.0%",
    },
    interpretation: "A percentile near 100% means the observed score ranks above most shuffled-label scores. Near 50% looks ordinary under this null. A low percentile means the observed pipeline scored below much of the null distribution.",
    unusual: "With K = 24, strict comparisons move the rank in 4 percentage-point steps and tie half-weights move it in 2-point steps. If every score ties, the rank stays near 50 rather than falsely clearing the null.",
    boundary: "This browser statistic ranks one null score from each repeated split against the median observed score. It is not a paired permutation p-value. It does not establish causality, eliminate leakage, or prove deployment generalization.",
  },
];

function Equation(props: {
  formula: FormulaDefinition;
  selected: string;
  onSelect: (term: string) => void;
}) {
  return (
    <div
      className="sf-formula-equation"
      aria-label={props.formula.title + ". Select a term for its definition."}
    >
      {props.formula.equation.map((segment, index) =>
        segment.term ? (
          <button
            type="button"
            key={segment.term + "-" + index}
            className={
              "sf-formula-equation__term" +
              (props.selected === segment.term ? " sf-formula-equation__term--selected" : "")
            }
            aria-label={"Explain " + segment.spoken}
            aria-pressed={props.selected === segment.term}
            aria-controls={
              "sf-formula-definition-" + props.formula.id + "-" + segment.term
            }
            onClick={() => props.onSelect(segment.term as string)}
          >
            {segment.text}
          </button>
        ) : (
          <span key={"operator-" + index}>{segment.text}</span>
        ),
      )}
    </div>
  );
}

function RemainingOperators() {
  return (
    <section className="sf-formula-operators" aria-labelledby="sf-formula-operators-title">
      <header>
        <div>
          <p className="sf-formula-section-label">Operator and summary reference</p>
          <h3 id="sf-formula-operators-title">Stress operators and curve summaries</h3>
        </div>
        <p>
          These operators modify data inside one audit run. Monte Carlo repeats sit outside them.
          The final block defines the summaries shown in the lab&apos;s advanced results table.
        </p>
      </header>

      <details className="sf-formula-operators__details">
        <summary>Open the full operator, metric, and curve summary reference</summary>
        <div className="sf-formula-operators__grid">
        <article>
          <p>Feature missingness</p>
          <div className="sf-formula-operators__equation">
            m_ij ∼ Bernoulli(λ); x′_ij = (1−m_ij)x_ij + m_ij median_train,j
          </div>
          <dl>
            <div><dt>m_ij</dt><dd>Binary mask draw for audit row i and feature j.</dd></div>
            <div><dt>λ</dt><dd>Probability that the cell is replaced by its training-fold median.</dd></div>
            <div><dt>x_ij, x′_ij</dt><dd>Original and filled audit feature values.</dd></div>
            <div><dt>median_train,j</dt><dd>Feature-j median fitted only on the training fold.</dd></div>
            <div><dt>Bernoulli</dt><dd>A binary draw equal to 1 with probability λ.</dd></div>
            <div><dt>i, j</dt><dd>Audit-row and feature indices.</dd></div>
          </dl>
        </article>

        <article>
          <p>Target corruption</p>
          <div className="sf-formula-operators__equation sf-formula-operators__equation--stacked">
            <span>
              Binary: z_i ∼ Bernoulli(λ); y′_i = (1−z_i)y_i + z_i(1−y_i)
            </span>
            <span>Regression: y′_i = y_i + λs_yε_i</span>
          </div>
          <dl>
            <div><dt>z_i</dt><dd>Binary draw deciding whether target i flips.</dd></div>
            <div><dt>y_i, y′_i</dt><dd>Original and corrupted targets, where binary y is 0 or 1.</dd></div>
            <div><dt>λ</dt><dd>Flip probability for classification or scaled-noise severity for regression.</dd></div>
            <div><dt>s_y</dt><dd>Training-only scale of a regression target.</dd></div>
            <div><dt>ε_i</dt><dd>Seeded zero-mean, unit-scale noise draw.</dd></div>
            <div><dt>i</dt><dd>Training-row index. The model is refit after target corruption.</dd></div>
          </dl>
        </article>

        <article>
          <p>Training-set reduction</p>
          <div className="sf-formula-operators__equation sf-formula-operators__equation--stacked">
            <span>Classification: n′_c = clamp(round((1 − λ)n_c), 2, n_c); n′ = Σ_c n′_c</span>
            <span>Regression: n′ = clamp(round((1 − λ)n), 8, n)</span>
            <span>Refit preprocessing and the model on the retained rows</span>
          </div>
          <dl>
            <div><dt>n</dt><dd>Available training rows before reduction.</dd></div>
            <div><dt>n_c</dt><dd>Available training rows in binary class c before reduction.</dd></div>
            <div><dt>λ</dt><dd>Fraction of training rows removed, from 0 to less than 1.</dd></div>
            <div><dt>n′</dt><dd>Total rows retained. Classification samples each class separately, and regression samples all rows together.</dd></div>
            <div><dt>round</dt><dd>The declared nearest-integer rounding rule.</dd></div>
            <div><dt>clamp</dt><dd>Enforces at least two rows per class or eight regression rows, without exceeding those available.</dd></div>
          </dl>
        </article>

        <article>
          <p>Curve summaries</p>
          <div className="sf-formula-operators__equation sf-formula-operators__equation--stacked">
            <span>A = max(0, Σ_k [(2 − R_(k−1) − R_k) / 2] [(λ_k − λ_(k−1)) / λ_m])</span>
            <span>first-step loss = 1 − R_1</span>
            <span>half-skill point = min&#123;λ_k &gt; 0 : R_k ≤ 0.5&#125;</span>
          </div>
          <dl>
            <div><dt>A</dt><dd>Normalized trapezoid area between the retained-performance curve and 1.</dd></div>
            <div><dt>k</dt><dd>Index of a tested severity, where k = 0 is the clean point and m is the last tested point.</dd></div>
            <div><dt>R_k</dt><dd>Median performance retained at tested severity λ_k.</dd></div>
            <div><dt>λ_m</dt><dd>Largest tested severity, used to normalize the horizontal scale.</dd></div>
            <div><dt>max</dt><dd>Truncates the summary at zero when the complete curve improves rather than degrades.</dd></div>
            <div><dt>min</dt><dd>First tested point at or below 50%. StressFold does not interpolate between points.</dd></div>
          </dl>
          <p className="sf-formula-operators__boundary">
            <strong>Comparison limit.</strong> The area summarizes one chosen grid. Areas from unlike stress mechanisms help organize the plots, but they do not rank real-world risk.
          </p>
        </article>

        <article>
          <p>Browser losses</p>
          <div className="sf-formula-operators__equation sf-formula-operators__equation--stacked">
            <span>Classification: L_Brier = (1/n) Σ_i (y_i − p_i)²</span>
            <span>Regression: L_MSE = (1/n) Σ_i (y_i − ŷ_i)²</span>
            <span>c = mean(y_train); L_ref = (1/n_audit) Σ_i (y_i − c)²</span>
          </div>
          <dl>
            <div><dt>y_i</dt><dd>Observed target on audit row i. It is 0 or 1 for classification and numeric for regression.</dd></div>
            <div><dt>p_i</dt><dd>Predicted probability of class 1 for audit row i.</dd></div>
            <div><dt>ŷ_i</dt><dd>Predicted numeric target for audit row i.</dd></div>
            <div><dt>n</dt><dd>Number of rows in the evaluated set.</dd></div>
            <div><dt>c</dt><dd>Constant reference prediction: the mean target from training rows only.</dd></div>
          </dl>
        </article>

        <article>
          <p>Browser scores</p>
          <div className="sf-formula-operators__equation sf-formula-operators__equation--stacked">
            <span>AUROC = P(p_positive &gt; p_negative) + 0.5 P(p_positive = p_negative)</span>
            <span>R² = 1 − MSE / [(1/n) Σ_i (y_i − mean(y_audit))²]</span>
          </div>
          <dl>
            <div><dt>AUROC</dt><dd>Probability that a random positive row ranks above a random negative row, with half credit for a tie.</dd></div>
            <div><dt>R²</dt><dd>Regression improvement over predicting the audit-target mean. One is perfect, and zero matches that mean benchmark.</dd></div>
            <div><dt>P</dt><dd>Probability over positive and negative row pairs in the evaluated audit set.</dd></div>
            <div><dt>role</dt><dd>Scores drive the headline and permutation rank. Stress curves use Brier loss or MSE.</dd></div>
          </dl>
        </article>
        </div>

        <p className="sf-formula-operators__boundary">
          <strong>Separation of roles.</strong> An operator defines what changes within a run.
          Monte Carlo defines how often the complete train, stress, and evaluation procedure repeats.
        </p>
      </details>
    </section>
  );
}

export function FormulaGuide() {
  const [activeId, setActiveId] = useState<FormulaId>("gap");
  const [selected, setSelected] = useState<Record<FormulaId, string>>({
    gap: "gap",
    noise: "perturbed",
    retained: "retained",
    "monte-carlo": "median",
    permutation: "percentile",
  });
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);

  const activeIndex = FORMULAS.findIndex((formula) => formula.id === activeId);
  const formula = FORMULAS[activeIndex];
  const selectedId = selected[activeId];
  const selectedTerm =
    formula.terms.find((candidate) => candidate.id === selectedId) ?? formula.terms[0];

  function chooseFormula(index: number, focus = false) {
    const next = (index + FORMULAS.length) % FORMULAS.length;
    setActiveId(FORMULAS[next].id);
    if (focus) tabRefs.current[next]?.focus();
  }

  function onTabKey(event: ReactKeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number | null = null;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = index + 1;
    if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = index - 1;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = FORMULAS.length - 1;
    if (next === null) return;
    event.preventDefault();
    chooseFormula(next, true);
  }

  function chooseTerm(term: string) {
    setSelected((current) => ({ ...current, [activeId]: term }));
  }

  return (
    <section className="sf-formula-guide" id="math" aria-labelledby="sf-formula-guide-title">
      <header className="sf-formula-guide__intro">
        <div>
          <p className="sf-formula-guide__kicker">Interactive notation guide</p>
          <h2 id="sf-formula-guide-title">Read the audit equations term by term</h2>
        </div>
        <p>
          Select any symbol to see what it means. Each equation has a numerical calculation,
          guidance for ordinary and unusual values, and the limit of the claim it supports.
        </p>
      </header>

      <div className="sf-formula-tabs" role="tablist" aria-label="StressFold equations">
        {FORMULAS.map((candidate, index) => {
          const active = candidate.id === activeId;
          return (
            <button
              key={candidate.id}
              ref={(node) => { tabRefs.current[index] = node; }}
              id={"sf-formula-tab-" + candidate.id}
              type="button"
              role="tab"
              aria-selected={active}
              aria-controls={"sf-formula-panel-" + candidate.id}
              tabIndex={active ? 0 : -1}
              className={"sf-formula-tabs__tab" + (active ? " sf-formula-tabs__tab--active" : "")}
              onClick={() => chooseFormula(index)}
              onKeyDown={(event) => onTabKey(event, index)}
            >
              <span>{String(index + 1).padStart(2, "0")}</span>
              {candidate.tab}
            </button>
          );
        })}
      </div>

      <article
        id={"sf-formula-panel-" + formula.id}
        className="sf-formula-panel"
        role="tabpanel"
        aria-labelledby={"sf-formula-tab-" + formula.id}
      >
        <header className="sf-formula-panel__header">
          <div>
            <p className="sf-formula-panel__index">
              Equation {String(activeIndex + 1).padStart(2, "0")} / 05
            </p>
            <h3>{formula.title}</h3>
          </div>
          <p>{formula.question}</p>
        </header>

        <div className="sf-formula-equation-block">
          <p className="sf-formula-equation-block__instruction">Select an equation term</p>
          <Equation formula={formula} selected={selectedTerm.id} onSelect={chooseTerm} />
          {formula.convention ? (
            <p className="sf-formula-equation-block__convention">{formula.convention}</p>
          ) : null}
        </div>

        <div className="sf-formula-selected" aria-live="polite" aria-atomic="true">
          <span className="sf-formula-selected__symbol" aria-hidden="true">
            {selectedTerm.symbol}
          </span>
          <div>
            <p>{selectedTerm.name}</p>
            <span>{selectedTerm.definition}</span>
          </div>
        </div>

        <div className="sf-formula-panel__body">
          <section
            className="sf-formula-worked"
            aria-labelledby={"sf-formula-worked-" + formula.id}
          >
            <p className="sf-formula-section-label">Worked calculation</p>
            <h4 id={"sf-formula-worked-" + formula.id}>Substitute, then simplify</h4>
            <p className="sf-formula-worked__values">{formula.example.values}</p>
            <ol>
              {formula.example.steps.map((step) => (
                <li key={step}><code>{step}</code></li>
              ))}
            </ol>
            <p className="sf-formula-worked__result">{formula.example.result}</p>
          </section>

          <section
            className="sf-formula-reading"
            aria-labelledby={"sf-formula-reading-" + formula.id}
          >
            <p className="sf-formula-section-label">Reading the output</p>
            <h4 id={"sf-formula-reading-" + formula.id}>What the value means</h4>
            <dl>
              <div>
                <dt>Ordinary reading</dt>
                <dd>{formula.interpretation}</dd>
              </div>
              <div className="sf-formula-reading__unusual">
                <dt>Unusual values</dt>
                <dd>{formula.unusual}</dd>
              </div>
            </dl>
          </section>
        </div>

        <section
          className="sf-formula-glossary"
          aria-labelledby={"sf-formula-glossary-" + formula.id}
        >
          <div className="sf-formula-glossary__header">
            <p className="sf-formula-section-label">Complete symbol key</p>
            <h4 id={"sf-formula-glossary-" + formula.id}>Every symbol in this equation</h4>
          </div>
          <dl>
            {formula.terms.map((term) => {
              const isSelected = term.id === selectedTerm.id;
              return (
                <div
                  id={"sf-formula-definition-" + formula.id + "-" + term.id}
                  className={"sf-formula-glossary__term" + (isSelected ? " sf-formula-glossary__term--selected" : "")}
                  key={term.id}
                >
                  <dt>
                    <button
                      type="button"
                      aria-pressed={isSelected}
                      aria-label={"Explain " + term.symbol}
                      onClick={() => chooseTerm(term.id)}
                    >
                      {term.symbol}
                    </button>
                  </dt>
                  <dd>
                    <strong>{term.name}</strong>
                    <span>{term.definition}</span>
                  </dd>
                </div>
              );
            })}
          </dl>
        </section>

        <aside className="sf-formula-boundary" aria-label="Claim boundary">
          <span>Claim boundary</span>
          <p>{formula.boundary}</p>
        </aside>
      </article>

      <RemainingOperators />
    </section>
  );
}

export default FormulaGuide;
