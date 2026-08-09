# The sweep on real markets

Everything else here runs on simulated series, where the right answer is fixed
before the test starts. That is how you show a tool measures what it claims and
it is not evidence about any real market. This is the run on real data.

Five liquid ETFs, daily adjusted closes, roughly 24 to 26 years each, four
strategies chosen before the data was downloaded. Nothing was fitted, nothing
was dropped after the fact, and the two directional strategies have a known
sign in the published literature, so disagreeing with them would have been a
sign the tool was broken rather than a discovery.

Reproduce with `python scripts/real_data.py`. The prices are not committed here
because they are not mine to redistribute, so the script downloads them.

```
SPY  2000-01-03 to 2026-07-31  6683 days
  strategy               real   b1      b5      b20     b60       D      p
  buy and hold           0.51     0.51    0.51    0.51    0.51    0.00  1.000
  1 day trend           -0.45     0.05   -0.31   -0.33   -0.41    1.12  0.995
  60 day trend           0.20     0.08    0.02   -0.05   -0.03    0.60  0.262
  volatility targeted    0.61     0.49    0.51    0.53    0.60    0.21  0.030

QQQ  2000-01-03 to 2026-07-31  6683 days
  strategy               real   b1      b5      b20     b60       D      p
  buy and hold           0.44     0.44    0.44    0.44    0.44    0.00  1.000
  1 day trend           -0.27     0.05   -0.28   -0.22   -0.26    1.18  0.930
  60 day trend           0.13     0.06   -0.05   -0.09    0.03    0.52  0.377
  volatility targeted    0.72     0.41    0.46    0.54    0.63    0.42  0.002

TLT  2002-07-30 to 2026-07-31  6039 days
  strategy               real   b1      b5      b20     b60       D      p
  buy and hold           0.31     0.31    0.31    0.31    0.31    0.00  1.000
  1 day trend           -0.39     0.01   -0.23   -0.26   -0.37    1.03  0.968
  60 day trend           0.17     0.04   -0.03   -0.13    0.08    0.74  0.277
  volatility targeted    0.37     0.30    0.34    0.37    0.38    0.19  0.070

GLD  2004-11-18 to 2026-07-31  5457 days
  strategy               real   b1      b5      b20     b60       D      p
  buy and hold           0.63     0.63    0.63    0.63    0.63    0.00  1.000
  1 day trend           -0.29     0.04   -0.07   -0.24   -0.25    1.13  0.955
  60 day trend           0.32     0.13    0.11    0.10    0.13    0.58  0.180
  volatility targeted    0.69     0.61    0.62    0.65    0.71    0.11  0.095

EEM  2003-04-14 to 2026-07-31  5861 days
  strategy               real   b1      b5      b20     b60       D      p
  buy and hold           0.48     0.48    0.48    0.48    0.48    0.00  1.000
  1 day trend           -0.68     0.02   -0.35   -0.57   -0.67    1.04  1.000
  60 day trend           0.26     0.08   -0.09   -0.21   -0.09    0.67  0.202
  volatility targeted    0.50     0.47    0.47    0.56    0.50    0.07  0.284
```

`real` is the annualised Sharpe on the history that happened. The `b` columns
are the median across generated markets at that block length. `D` is structure
dependence and `p` is how often the fully shuffled markets reached the real
result.

## What it says

Buy and hold returns 0.00 dependence and a p-value of 1.000 on all five, which
is the invariance claim holding on real fat tailed data rather than on a
Gaussian simulation. This is the check that had to pass and it is also the
check that found a defect, described at the bottom.

The one day trend follower loses on every series, and loses by more than almost
every shuffled version of the same series. Its dependence is above one on all
five, which is what happens when destroying the ordering improves a strategy.
That is short horizon reversal in daily equity returns, it is well documented,
and the sweep recovers it with the right sign without being told to look for it.

The 60 day trend follower is the interesting one. Its dependence runs from 0.52
to 0.74, which by the ratio alone reads as a strategy that lives on ordering.
Its p-value never goes below 0.18. The shuffled markets reach its result between
a fifth and two fifths of the time on every series tested. The ratio is large
because the median shuffle earns little, and the p-value is large because the
spread of shuffles is wide, and the spread is what decides whether the gap means
anything.

Volatility targeting is the same disagreement pointing the other way. It has no
view on direction at all. It is long the whole time and only changes how much.
Its dependence is 0.42 on QQQ and 0.07 on EEM, all small, because a strategy
that stays long keeps most of its Sharpe when the returns are shuffled. Yet on
QQQ not one of the 400 generated markets reached its result, which is the
strongest number in the table, and on SPY 11 of 400 did. Volatility clusters,
sizing down into that clustering is worth something real, and destroying the
ordering destroys it.

The p-value floor is set by how many paths you generate. Zero hits out of 400
gives 0.002 and zero out of 2000 gives 0.0005, and neither is a measurement of
how much smaller than that the truth might be. The figure below uses 2000.

So the two cases sit at 0.52 and 0.42 dependence, close enough to be called the
same, with p-values of 0.29 and 0.0005. That is the figure below and it is the
argument for reporting both numbers.

![Structure dependence says one thing and the shuffles say another](images/real-data.png)

Neither of these is a timing edge. The trend follower does not clear chance, and
the volatility targeting clears it comfortably while having no directional view
to be right about. A strategy passing the sweep has shown that its result needs
the ordering of the market, which is a smaller claim than an edge and the only
one the arithmetic supports.

## What this run changed in the code

Running on real data found a defect that simulated data had hidden. An order
invariant strategy scores identically on every generated path, so its synthetic
scores differ from the real one only in the last bits, because the same returns
are summed in a different order. The percentile counted how many scored strictly
below, which turned that rounding into a percentile of 23 on SPY and 11 on TLT
when the only truthful answer is 50. Ties are now detected within a tolerance
and split, and buy and hold reports 50 on all five series.

It also showed that structure dependence should never have been the only number
reported. Both disagreements above were invisible in it. The p-value is now
reported next to it at every block length, and the written verdict takes both.
