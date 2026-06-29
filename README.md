# MicroAlpha

> Quantitative alpha research platform with portfolio optimisation and risk analytics.

Generates synthetic OHLCV data for a multi-asset universe (AAPL, GOOGL, MSFT, AMZN, META) using correlated Geometric Brownian Motion. Computes portfolio performance metrics (Sharpe, Sortino, Calmar), risk measures (VaR, CVaR, drawdown), and optimises weights via mean-variance and risk-parity approaches.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

6-asset equal-weight portfolio metrics (252-day trading window):

| Metric | Value |
|---|---|
| Annual Return | 3.07% |
| Annual Volatility | 6.75% |
| Sharpe Ratio | 0.454 |
| VaR (95%) | 0.68% |
| Expected Shortfall (95%) | 0.89% |

## Features

| Component | What it does |
|---|---|
| **Portfolio Overview** | Equity curves, correlation matrix, return distribution, drawdown chart |
| **Risk Dashboard** | VaR/CVaR (historical + parametric), rolling volatility, stress test scenarios |
| **Optimisation** | Mean-variance efficient frontier, max-Sharpe and min-volatility portfolios, risk parity weights |
| **Performance** | Rolling Sharpe, factor decomposition, turnover analysis, rebalancing calendar |
| **Monte Carlo** | GBM path simulation for forward-looking risk assessment |

## Repo Structure

```
MicroAlpha/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic OHLCV data generated from ticker-specific GBM parameters calibrated to approximate real AAPL, GOOGL, MSFT, AMZN, META behaviour. 756 trading days with realistic drift and volatility.

## License

MIT
