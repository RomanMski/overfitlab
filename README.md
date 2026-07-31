# StressFold

**How much of your backtest is the search?**

If you tried two hundred parameter combinations and reported the best one, the number you reported is not the number you have. Some of it is skill and some of it is the search. StressFold measures the difference.

![Destroy the market's ordering and see what survives](docs/images/structure-sweep.png)

Two strategies on the same market. Both were rerun on hundreds of resampled histories, with less and less of the original ordering left intact. The trend follower needs that ordering and collapses without it. Buy and hold does not notice, because reordering returns cannot change their mean. `scripts/make_figures.py` regenerates this.

```python
from stressfold import deflated_sharpe_ratio, path_stress

# trials: (n_periods, n_configurations) of returns, every configuration you tried
print(deflated_sharpe_ratio(trials, periods_per_year=252))

# and rerun the strategy on markets that never happened
print(path_stress(strategy, market_returns, block_sizes=(1, 5, 20, 60)))
```

## The problem

You sweep a moving-average crossover over 200 parameter pairs. The best one shows an annualised Sharpe of 1.5 and a clean equity curve.

The catch is that the best of 200 random strategies also shows about 1.5. You did not find an edge, you ran a competition, and something always wins a competition.

Two searches can report the same number and mean completely different things. Nothing in a standard backtest report separates them.

## What it measures

**Did you search too hard?** The deflated Sharpe ratio works out what the best of N trials reaches when none of them has any edge, then asks whether yours clears that bar. The probability of backtest overfitting takes every way of cutting your history in half, finds the winner in one half, and measures how often it lands below the median of its peers in the other. Above 0.5 means your selection process is worse than picking at random.

**Is there anything there at all?** `path_stress` rebuilds your price history hundreds of times over and reruns your strategy on each one, destroying market structure by degrees:

| block size | what survives |
| --- | --- |
| 1 | nothing, returns drawn independently |
| 5 | runs of 5 periods kept intact |
| 20 | runs of 20 |
| 60 | runs of 60 |

Reading that gradient tells you what the strategy depends on, not just whether it works. A real timing edge should die when ordering is destroyed and recover as the blocks lengthen. Here is a trend follower on a market with genuine momentum:

```
  block  keeps                       median  the real result beats
      1  nothing, pure noise           0.04   100.0% of them
      5  runs of 5 periods             0.75    78.0%
     20  runs of 20 periods            0.92    60.7%
     60  runs of 60 periods            1.00    50.0%

Structure dependence 0.96
```

Sharpe 1.00 on the real path, 0.04 once the ordering goes, back to 1.00 when the runs return. Buy and hold on the same market scores 0.36 and scores 0.38 shuffled, giving a structure dependence of **-0.06**, because reordering cannot change a mean. The sweep says plainly that none of that came from timing.

## Interactive explainers

The site runs three labs in the browser, no install and no data leaving the page.

1. **What overfitting actually is.** A curve fitted to noisy points, with a flexibility slider. Watch the error on the fitted points fall to zero while the error on fresh points climbs.
2. **How searching manufactures a strategy.** Five hundred strategies with no edge at all. Drag the number you are allowed to look at and watch the winner's Sharpe climb past anything you would call investable.
3. **Markets that never happened.** The real market against synthetic ones, with the block-size sweep and a toggle between a trend follower and buy and hold.

```bash
npm ci && npm run dev
```

## Install

```bash
python -m pip install -e .
```

Python 3.10 or newer, with NumPy, pandas and SciPy.

## Honest limits

The two selection statistics assume your trials are roughly exchangeable. If you searched adaptively, stopping early on the bad ones, the effective number of trials is not the number you ran and both statistics will be optimistic.

The probability of backtest overfitting is blind to time order. It draws combinations of blocks and does not care which came first, so it cannot see a strategy that worked until a regime changed and then stopped. That is a property of the published method, not a bug here, and it is why it is not the only thing reported.

Resampling preserves volatility clustering only within blocks. Real markets have persistence that outlives that, so the synthetic paths are tamer than reality in the tails.

Nothing here models transaction costs. A strategy that trades daily can lose most of its measured edge to spread and slippage, and none of these statistics will notice.

Passing every check is not evidence of an edge. It means selection alone does not explain your result, which is a much weaker claim and the only one the maths supports.

## References

Bailey and López de Prado (2014), *The Deflated Sharpe Ratio*. Bailey, Borwein, López de Prado and Zhu (2016), *The Probability of Backtest Overfitting*. Politis and Romano (1994), *The Stationary Bootstrap*.

MIT licensed.
