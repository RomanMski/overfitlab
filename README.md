# OverfitLab

**How much of your backtest is the search?**

You tried 200 parameter combinations and kept the best one. It shows a Sharpe of 1.5.

So does the best of 200 random ones. That is the problem.

This tells you which of those two you have.

![Destroy the market's ordering and see what survives](docs/images/structure-sweep.png)

Same market, two strategies, both rerun on hundreds of resampled histories with less and less of the original ordering left. The trend follower needs that ordering. Buy and hold does not notice, because shuffling returns cannot change their mean.

## Three things it measures

**Did you search too hard?** The deflated Sharpe works out what the best of N tries reaches when nothing has an edge, then checks whether yours beats it. Run it on 60 configurations with no signal at all and the winner looks like 1.29 annualised. The bar is 1.37. So that result is worse than nothing.

**Does your winner keep winning?** Cut the history in half every possible way, find the best configuration in one half, look at where it ranks in the other. Above 0.5 means your selection did worse than picking at random.

**Is there anything in the market at all?** Rebuild the price history hundreds of times, rerun the strategy on each, and keep less of the ordering each round:

| block | what survives | trend follower | buy and hold |
| --- | --- | --- | --- |
| 1 | nothing, returns shuffled | 0.04 | 0.38 |
| 5 | runs of 5 | 0.75 | 0.33 |
| 20 | runs of 20 | 0.92 | 0.39 |
| 60 | runs of 60 | 1.00 | 0.38 |
| | **the real market** | **1.00** | **0.36** |

The trend follower dies at block 1 and comes back as the runs return. That is what a timing edge looks like. Buy and hold sits flat the whole way, so whatever it earns, it is not from timing.

## Use it

```python
from overfitlab import (
    deflated_sharpe_ratio,
    path_stress,
    probability_of_backtest_overfitting,
)

# trials: (periods, configurations), every configuration you tried
deflated_sharpe_ratio(trials, periods_per_year=252)
probability_of_backtest_overfitting(trials, n_splits=16)

# your strategy on markets that never happened
path_stress(strategy, market_returns, block_sizes=(1, 5, 20, 60))
```

Pass every configuration you tried. Dropping the bad ones is the exact bias these numbers exist to catch.

```bash
python -m pip install -e .
```

Python 3.10 or newer. NumPy, pandas, SciPy.

## Or in the browser

Four labs, nothing installed, nothing uploaded.

1. **What overfitting is.** A curve through noisy dots with a flexibility slider. The error on the dots falls to zero. The error on new dots climbs.
2. **How searching makes a strategy.** 500 strategies with no edge. Drag how many you get to look at and watch the winner get better.
3. **Markets that never happened.** The block sweep above, with a toggle between the two strategies.
4. **Your own backtest.** Drop a CSV of returns, one column per configuration, get both numbers.

```bash
npm ci && npm run dev
```

## What it cannot do

Both selection statistics assume your trials are roughly interchangeable. If you stopped early on the bad ones, the number of trials you ran is not the number that counts, and both will flatter you.

The overfitting probability ignores time order. It shuffles blocks around and does not care which came first, so a strategy that worked until the regime changed looks fine to it. That is how the published method works. It is why it is not the only thing here.

Resampling keeps volatility clustering inside a block and loses it across blocks. Real markets cluster for longer, so the synthetic paths are calmer in the tails than reality.

No transaction costs anywhere. A daily trend follower can lose most of a 1.00 Sharpe to spread and slippage and none of these numbers will say a word about it.

Passing everything is not evidence of an edge. It only means selection alone does not explain your result. That is a much smaller claim and it is the only one the maths supports.

## References

Bailey and López de Prado (2014), *The Deflated Sharpe Ratio*. Bailey, Borwein, López de Prado and Zhu (2016), *The Probability of Backtest Overfitting*. Politis and Romano (1994), *The Stationary Bootstrap*.

MIT licensed.
