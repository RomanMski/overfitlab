"use client";

import { CurveLab } from "./CurveLab";
import { PathLab } from "./PathLab";
import { SearchLab } from "./SearchLab";
import { TrialLab } from "./TrialLab";

export function OverfitLabApp() {
  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>OverfitLab</span>
        </div>
        <div className="header-status"><span /> v1.0.0</div>
      </header>

      <main id="top">
        <section className="hero section-boundary">
          <div className="hero-copy">
            <div className="kicker">
              <span>Backtest overfitting</span><span>Selection bias</span>
            </div>
            <h1>How much of your backtest is the search?</h1>
            <p className="hero-lede">
              You tried 200 parameter combinations and kept the best one. So did
              the best of 200 random ones. This tells you which of those two you
              have. Drag the sliders below, then drop in your own backtest.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#search">
                See it happen <span aria-hidden="true">→</span>
              </a>
              <a className="button button-quiet" href="#test">Test your own backtest</a>
            </div>
          </div>
        </section>

        <section className="lab-section" id="overfit">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lab one</div>
              <h2>What overfitting actually is</h2>
            </div>
            <p>
              No markets yet. A curve through 22 noisy dots, and you control how
              bendy it is allowed to be.
            </p>
          </div>
          <CurveLab />
        </section>

        <section className="lab-section" id="search">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lab two</div>
              <h2>How searching manufactures a strategy</h2>
            </div>
            <p>
              500 strategies. None of them has any edge. You control how many you
              get to look at before picking the best one.
            </p>
          </div>
          <SearchLab />
        </section>

        <section className="lab-section" id="paths">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lab three</div>
              <h2>Markets that never happened</h2>
            </div>
            <p>
              Your backtest ran on one price path. This rebuilds it hundreds of
              times and reruns the strategy on each, keeping less of the ordering
              every step, so you can see what the strategy actually needs.
            </p>
          </div>
          <PathLab />
        </section>

        <section className="lab-section" id="test">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Test your own</div>
              <h2>Run it on your backtest</h2>
            </div>
            <p>
              One column per configuration you tried, one row per period. Drop it
              below. It is read in this page and never leaves your machine.
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
            <p>
              Same numbers, scriptable.
            </p>
          </div>
          <pre className="code-block"><code>{`from overfitlab import (
    deflated_sharpe_ratio,
    path_stress,
    probability_of_backtest_overfitting,
)

# trials: (n_periods, n_configurations) of returns
print(deflated_sharpe_ratio(trials, periods_per_year=252))
print(probability_of_backtest_overfitting(trials, n_splits=16))

# rerun your strategy on markets that never happened
print(path_stress(strategy, market_returns, block_sizes=(1, 5, 20, 60)))`}</code></pre>
          <p>
            The deflated Sharpe works out what the best of that many tries reaches
            when nothing has an edge, then checks whether yours beats it. The
            overfitting probability cuts your history in half every possible way. It
            finds the winner in one half and looks at where it ranks in the other.
          </p>
          <p>
            Pass every configuration you tried. Dropping the bad ones is the exact
            bias these numbers exist to catch.
          </p>
        </section>
      </main>

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
