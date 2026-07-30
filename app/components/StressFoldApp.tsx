"use client";

import { OverfitLab } from "./OverfitLab";
import { SearchLab } from "./SearchLab";

export function StressFoldApp() {
  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>StressFold</span>
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
              If you tried two hundred parameter combinations and reported the best
              one, the number you reported is not the number you have. StressFold
              measures the difference. The two labs below show why it matters, and
              the Python package measures it on your own trials.
            </p>
            <div className="hero-actions">
              <a className="button button-primary" href="#search">
                See it happen <span aria-hidden="true">→</span>
              </a>
              <a className="button button-quiet" href="#package">Use the package</a>
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
              Before anything about markets, the idea in its simplest form. A curve
              is fitted to a handful of noisy points. You control how bendy it is
              allowed to be.
            </p>
          </div>
          <OverfitLab />
        </section>

        <section className="lab-section" id="search">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Lab two</div>
              <h2>How searching manufactures a strategy</h2>
            </div>
            <p>
              The same idea, pointed at backtests. Five hundred strategies, none of
              which has any edge at all. You control how many of them you are
              allowed to look at before picking the best.
            </p>
          </div>
          <SearchLab />
        </section>

        <section className="lab-section" id="package">
          <div className="section-heading">
            <div>
              <div className="eyebrow">Python package</div>
              <h2>Measure it on your own trials</h2>
            </div>
            <p>
              Hand over a table of period returns with one column per configuration
              you backtested. The package never runs your strategy and never sees
              your data feed.
            </p>
          </div>
          <pre className="code-block"><code>{`from stressfold import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

# trials: (n_periods, n_configurations) of returns
print(deflated_sharpe_ratio(trials, periods_per_year=252))
print(probability_of_backtest_overfitting(trials, n_splits=16))`}</code></pre>
          <p>
            The deflated Sharpe ratio works out what the best of that many trials
            reaches when nothing has an edge, then asks whether yours clears it. The
            probability of backtest overfitting takes every way of splitting your
            history in half and measures how often the in-sample winner lands below
            the median of its peers out of sample.
          </p>
          <p>
            Pass every configuration you evaluated. Dropping the ones that did badly
            is exactly the bias both statistics exist to measure.
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <div className="brand footer-brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>StressFold</span>
        </div>
        <p>Selection bias in backtests, measured rather than assumed.</p>
      </footer>
    </div>
  );
}
