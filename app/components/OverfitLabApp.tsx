"use client";

import { useState } from "react";

import { CurveLab } from "./CurveLab";
import { GeneratorLab } from "./GeneratorLab";
import { PathLab } from "./PathLab";
import { SearchLab } from "./SearchLab";
import { SharpeLab } from "./SharpeLab";
import { TrialLab } from "./TrialLab";

type View = "tool" | "concepts";

export function OverfitLabApp() {
  const [view, setView] = useState<View>("tool");

  return (
    <div className="page">
      <style>{`
        .view-tabs{display:flex;gap:8px}
        .view-tabs button{border:1px solid var(--rule,#d9ddd8);background:var(--panel,#fffdfa);border-radius:999px;padding:7px 16px;cursor:pointer;font:inherit}
        .view-tabs button[aria-current=page]{background:#20272e;color:#fff;border-color:#20272e}
      `}</style>

      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>OverfitLab</span>
        </div>
        <nav className="view-tabs" aria-label="Sections">
          <button type="button" aria-current={view === "tool" ? "page" : undefined}
                  onClick={() => setView("tool")}>
            The tool
          </button>
          <button type="button" aria-current={view === "concepts" ? "page" : undefined}
                  onClick={() => setView("concepts")}>
            Concepts
          </button>
        </nav>
      </header>

      {view === "tool" ? (
        <main id="top">
          <section className="hero section-boundary">
            <div className="hero-copy">
              <div className="kicker">
                <span>Backtest overfitting</span><span>Selection bias</span>
              </div>
              <h1>How much of your backtest is the search?</h1>
              <p className="hero-lede">
                Give it a price or return series. It builds hundreds of
                alternative versions of that history, each keeping a different
                amount of the original ordering, and hands them back as CSV
                files. Run your own model on them and compare. It never needs to
                see your model.
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href="#generate">
                  Generate datasets <span aria-hidden="true">→</span>
                </a>
                <a className="button button-quiet" href="#test">Score my backtest</a>
                <a className="button button-quiet" href="./paper/overfitlab.pdf"
                   target="_blank" rel="noreferrer">Read the paper</a>
              </div>
            </div>
          </section>

          <section className="lab-section" id="generate">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Generate datasets</div>
                <h2>Make versions of your history to test against</h2>
              </div>
              <p>
                Drop in a series and choose how many versions you want. Nothing
                is uploaded and no model is involved.
              </p>
            </div>
            <GeneratorLab />
          </section>

          <section className="lab-section" id="paths">
            <div className="section-heading">
              <div>
                <div className="eyebrow">What the output shows</div>
                <h2>Reading the result</h2>
              </div>
              <p>
                The same generated series, with a strategy run across them so you
                can see what the comparison looks like once you have done it on
                your own model.
              </p>
            </div>
            <PathLab />
          </section>

          <section className="lab-section" id="test">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Score it directly</div>
                <h2>Or hand over your trial results</h2>
              </div>
              <p>
                If you would rather not run the comparison yourself, upload the
                period returns of every configuration you tested, one column
                each, and get the two selection statistics back.
              </p>
            </div>
            <TrialLab />
          </section>

          <section className="lab-section" id="package">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Package</div>
                <h2>Or from Python</h2>
              </div>
              <p>Same maths, scriptable.</p>
            </div>
            <pre className="code-block"><code>{`from overfitlab import (
    deflated_sharpe_ratio,
    path_stress,
    probability_of_backtest_overfitting,
    write_datasets,
)

# one CSV per block length, plus a manifest
write_datasets(returns, "generated/", block_sizes=(1, 5, 20, 60), n_paths=100)

# trials: (periods, configurations), every configuration you tried
deflated_sharpe_ratio(trials, periods_per_year=252)
probability_of_backtest_overfitting(trials, n_splits=16)

# or let it rerun a strategy for you
path_stress(strategy, market_returns, block_sizes=(1, 5, 20, 60))`}</code></pre>
            <p>
              Pass every configuration you tried, including the ones that did
              badly. Dropping those is the exact bias these numbers exist to
              catch.
            </p>
          </section>
        </main>
      ) : (
        <main id="top">
          <section className="hero section-boundary">
            <div className="hero-copy">
              <div className="kicker">
                <span>Reference</span><span>Interactive</span>
              </div>
              <h1>Concepts</h1>
              <p className="hero-lede">
                The ideas the tool rests on, each with something you can move.
                Written for someone who wants to understand what the numbers
                mean rather than take them on trust.
              </p>
            </div>
          </section>

          <section className="lab-section" id="overfitting">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Concept</div>
                <h2>Overfitting</h2>
              </div>
              <p>
                A curve through 22 noisy dots, and you control how bendy it is
                allowed to be.
              </p>
            </div>
            <CurveLab />
          </section>

          <section className="lab-section" id="selection">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Concept</div>
                <h2>Selection bias</h2>
              </div>
              <p>
                500 strategies, none with any edge. You control how many you get
                to look at before picking the best one.
              </p>
            </div>
            <SearchLab />
          </section>

          <section className="lab-section" id="sharpe">
            <div className="section-heading">
              <div>
                <div className="eyebrow">Concept</div>
                <h2>The Sharpe ratio</h2>
              </div>
              <p>
                What it measures, what annualising assumes, and the two things it
                cannot see.
              </p>
            </div>
            <SharpeLab />
          </section>
        </main>
      )}

      <footer className="site-footer">
        <div className="brand footer-brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>OverfitLab</span>
        </div>
        <p>Selection bias in backtests, measured rather than assumed.</p>
      </footer>
    </div>
  );
}
