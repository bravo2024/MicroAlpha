"""data.py — Synthetic multi-asset price panel with embedded alpha signals.

Generates a (T x N) price matrix where:
  * Half the assets exhibit **momentum** (positive serial correlation).
  * Half exhibit **mean-reversion** (negative serial correlation).

This structure lets both momentum and mean-reversion signals have
genuine predictive content, so the signal-research pipeline in
``model.py`` produces meaningful IC and IR values — not just noise.

Swap ``make_synthetic`` for ``load_real`` (yfinance) in production.
"""
from __future__ import annotations
import numpy as np

ASSETS = ["MOM_A", "MOM_B", "MOM_C", "MR_X", "MR_Y", "MR_Z"]
TRADING_DAYS = 252


def make_synthetic(n_days=756, seed=42):
    """Generate a synthetic price panel with momentum and mean-reversion.

    Momentum assets follow a trending random walk where positive returns
    increase the drift for the next period (AR(1) coefficient > 0).
    Mean-reversion assets oscillate around a slow trend (AR(1) < 0).

    Returns dict with 'prices' (T,N), 'returns' (T,N), 'assets', and
    metadata about which assets are momentum vs mean-reversion.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(ASSETS)
    n_mom = 3  # first 3 are momentum, last 3 are mean-reversion

    dt = 1.0 / TRADING_DAYS
    base_drift = np.array([0.10, 0.08, 0.12, 0.06, 0.07, 0.05]) / TRADING_DAYS
    base_vol = np.array([0.22, 0.25, 0.30, 0.18, 0.20, 0.16]) / np.sqrt(TRADING_DAYS)

    returns = np.zeros((n_days, n_assets))
    # AR(1) coefficient: positive for momentum, negative for reversion
    ar_coef = np.array([0.12, 0.10, 0.15, -0.15, -0.12, -0.18])

    for i in range(n_assets):
        shocks = rng.standard_normal(n_days) * base_vol[i]
        r = np.zeros(n_days)
        r[0] = base_drift[i] + shocks[0]
        for t in range(1, n_days):
            r[t] = base_drift[i] + ar_coef[i] * (r[t - 1] - base_drift[i]) + shocks[t]
        returns[:, i] = r

    prices = np.cumprod(1.0 + returns, axis=0) * 100.0

    return {
        "prices": prices,
        "returns": returns,
        "assets": ASSETS,
        "n_assets": n_assets,
        "n_days": n_days,
        "momentum_assets": ASSETS[:n_mom],
        "reversion_assets": ASSETS[n_mom:],
        "ar_coefficients": ar_coef.tolist(),
    }


def load_real(tickers, period="3y"):
    """Load real price data via yfinance (optional, not required for tests)."""
    import yfinance as yf
    px = yf.download(tickers, period=period)["Adj Close"].dropna()
    returns = px.pct_change().dropna().to_numpy()
    prices = px.to_numpy()
    return {
        "prices": prices,
        "returns": returns,
        "assets": list(px.columns),
        "n_assets": len(px.columns),
        "n_days": len(px),
    }