# StressFold

**How much of your backtest is the search, and how much is the strategy?**

If you tried two hundred parameter combinations and reported the best one, the number you reported is not the number you have. This measures the difference.

## The demonstration

`examples/noise_that_looks_like_alpha.py` searches 200 configurations over data with **no signal in it at all**. Every series is a random walk, so no configuration is better than any other.

The best one looks like this:

```
Sharpe 0.0840 per period, 1.33 annualised
probability the Sharpe beats zero, ignoring the search: 0.996
```

An annualised Sharpe of 1.33, and a standard statistic is 99.6% confident it is real. The same trials with the search accounted for:

```
Deflated Sharpe ratio: 0.472
  200 trials raise the bar to 0.0862 before any edge is credited
  observed 0.0840 over 1000 observations

Probability of backtest overfitting: 0.384
  selected config averaged 0.120 in sample and -0.000 out of sample
```

The bar the winner had to clear was 0.0862. It reached 0.0840. Searching 200 things was worth more than the edge it found.

## The interactive version

`npm run dev` opens two labs that make the same point without any code.

**What overfitting is.** A curve fitted to noisy dots, with a slider for how bendy it may be. Drag right and the error on the dots it was given keeps falling while the error on fresh data bottoms out and climbs. The gap between those two numbers is the overfitting.

**How searching manufactures a strategy.** Five hundred equity curves, none with any edge. A slider for how many you are allowed to look at before picking the best. At one trial the winner shows an annualised Sharpe of 0.60. At five hundred it shows 1.54, and the bar for what luck alone reaches has risen to 1.70.

## Using it on your own trials

```python
from stressfold import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)

# trials: (n_periods, n_configurations) of returns
print(deflated_sharpe_ratio(trials, periods_per_year=252))
print(probability_of_backtest_overfitting(trials, n_splits=16))
```

A table of period-by-period returns, one column per configuration you backtested. The package never runs your strategy, never sees your data feed and never needs to know how your backtester works.

Pass **every** configuration you evaluated. Dropping the ones that did badly is precisely the bias both statistics exist to measure.

## The two measurements

**Deflated Sharpe ratio.** A Sharpe from a finite sample is an estimate, and a biased one once you have taken the best of several trials. The deflation works out the Sharpe you would expect the winner to show when nothing has any edge, given how many things you tried and how much they varied, then asks whether the observed one clears that bar. It also corrects for skew, fat tails and sample length. Bailey and Lopez de Prado, 2014.

**Probability of backtest overfitting.** Cut the history into blocks, take every way of choosing half of them, find the configuration that wins on that half, and see where it ranks on the other. If the in-sample winner lands below the median of its peers about half the time, your selection procedure carries no information. Bailey, Borwein, Lopez de Prado and Zhu, 2016.

They answer different questions and are deliberately not combined into one score.

## What these cannot tell you

**PBO is blind to chronology.** CSCV enumerates every way of choosing half the blocks, and that set does not change when the blocks are relabelled, so the statistic is exactly invariant to the order of your history. A strategy that worked until some structural break and stopped will not be flagged, because a symmetric split averages the good and bad regimes on both sides. Detecting that needs walk-forward. A test in the suite asserts this invariance so it cannot regress quietly.

**One PBO number is noisy.** Across independent datasets from the same null the estimate carries a standard deviation near 0.17 for a few dozen trials over several hundred periods. Treat 0.45 and 0.55 as the same answer.

**Neither is a profitability test.** No transaction costs, no slippage, no capacity, no market impact, no borrow.

**Both assume your trial set is honest.** They measure selection bias given the number of trials you declare. Run 5,000 and pass 50, and the arithmetic will cheerfully under-correct.

## Install

```bash
python -m pip install -e ".[dev]"
python -m pytest        # 28 tests
npm ci && npm test      # 13 tests, browser maths and rendering
```

Python 3.10 or newer, and Node 22.13 or newer for the labs.

## References

Bailey, D. and Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*.

Bailey, D. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.

Bailey, D., Borwein, J., Lopez de Prado, M. and Zhu, Q. (2016). The Probability of Backtest Overfitting. *Journal of Computational Finance*.

MIT licensed.
