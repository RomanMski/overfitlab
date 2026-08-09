Draft post. Not published anywhere by the repository, kept here so the claims
in it stay next to the code that produces them. Image is docs/images/real-data.png.

---

I built a tool that tests whether a backtest depends on the order history
arrived in, then ran it on real market data and it told me one of my own
numbers was misleading.

The method is simple. Cut a return series into blocks, reorder the blocks
without replacement so every observation still appears exactly once, and rerun
the strategy. The mean, the variance and every fat tail are identical to the
real data. Only the arrangement changes. A strategy that scores the same on the
reordered version was not using time structure, whatever it claims.

I was reporting one number from that, the fraction of the result that
disappears once the ordering is destroyed. Five liquid ETFs on daily closes
since 2000 showed that number failing in both directions.

A 60 day trend follower on SPY loses half its Sharpe to shuffling, which reads
as a strategy that lives on the arrangement of history. The shuffled markets
reach its result 29% of the time, so it does not clear chance.

Volatility targeting on QQQ loses less than that, and not one of two thousand
shuffled markets reached it. That strategy has no view on direction at all. It
is long the whole time and only varies how much. What it picks up is volatility
clustering, not any ability to time anything.

Almost the same effect size, p-values of 0.29 and 0.0005. So the tool reports
both numbers now, and the write up says why.

The same run found a bug that simulated data had hidden. Buy and hold scores
identically on every reordering, so its scores differ from the real one only in
the last bits of a sum taken in a different order. I was counting how many
scored strictly below, which turned that rounding into a percentile of 23 on
one series and 11 on another when the only honest answer is 50.

Code, an eight page write up and a browser demo that runs on your own CSV
without uploading it.

https://github.com/RomanMski/overfitlab
