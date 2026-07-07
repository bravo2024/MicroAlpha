# MicroAlpha

A signal research and backtesting engine for hunting small alphas.

MicroAlpha generates trading signals from price history, evaluates their
predictive content via Information Coefficient (IC) and Information Ratio (IR),
and backtests a long-short strategy with proportional transaction costs. A
Streamlit dashboard renders the full dark-theme tearsheet.

## Methodology

### Signal generation

Two complementary signals are computed from the price panel:

1. **Time-series momentum** (Moskowitz, Ooi & Pedersen, 2012) — the
   cumulative return over a `lookback`-day window. Positive signal →
   asset is trending upward.

2. **Mean-reversion** — the negative z-score of price relative to its
   moving average. Positive signal → price is below its average and
   expected to revert upward.

Signals are combined (60% momentum, 40% reversion) and converted to
cross-sectional ranks for market-neutral position sizing.

### Signal evaluation

* **Information Coefficient (IC)** — Spearman rank correlation between
  the signal and the forward `horizon`-day return, computed at each
  rebalance date.
* **Information Ratio (IR)** — mean(IC) / std(IC), annualised.
* **IC hit rate** — fraction of dates where IC > 0.
* **IC decay** — how predictive power decays as the horizon lengthens.

### Backtesting

* Long-short, dollar-neutral weights from cross-sectional ranks.
* Proportional transaction costs (`cost_bps` basis points per dollar traded).
* Rebalance frequency configurable (default: monthly = 21 days).
* Full performance tearsheet: Sharpe, Sortino, Calmar, max drawdown,
  hit rate, profit factor, VaR, expected shortfall.

## Synthetic data

The synthetic generator creates a 6-asset price panel where:
* 3 assets exhibit **momentum** (AR(1) coefficient > 0)
* 3 assets exhibit **mean-reversion** (AR(1) coefficient < 0)

This ensures both signals have genuine predictive content, producing
meaningful IC values rather than noise.

## Quickstart

```bash
pip install -r requirements.txt
python train.py          # run the full signal-research pipeline
pytest -q                # verify the pipeline
streamlit run app.py     # launch the dashboard
```

## References

* Moskowitz, T., Ooi, Y., & Pedersen, L. (2012). "Time Series Momentum."
  *Journal of Financial Economics*.
* Asness, C., Moskowitz, T., & Pedersen, L. (2013). "Value and Momentum
  Everywhere." *Journal of Finance*.
* Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*, 2nd ed.

## License

MIT