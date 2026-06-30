"""core.py — Quantitative performance & signal-evaluation metrics (pure NumPy).

Building blocks for the signal-research and backtesting pipeline in model.py.
Nothing depends on sklearn so the smoke test stays fast and dependency-light.

Convention
----------
* ``returns`` — 1-D array of simple period returns.
* ``cumulative`` — 1-D array of cumulative wealth, e.g. ``(1+r).cumprod()``.
* ``freq`` — periods per year (252 daily, 12 monthly).
"""
from __future__ import annotations
import numpy as np
from scipy import stats as _stats


# ── Return-level performance metrics ─────────────────────────────────────────

def sharpe_ratio(returns, rf=0.0, freq=252):
    """Annualised Sharpe: (mean_excess / std) * sqrt(freq). rf is *annualised*."""
    r = np.asarray(returns, float)
    if r.size < 2:
        return 0.0
    excess = r - rf / freq
    sd = float(r.std(ddof=1))
    return float(excess.mean() / sd * np.sqrt(freq)) if sd > 1e-12 else 0.0


def sortino_ratio(returns, rf=0.0, freq=252):
    """Annualised Sortino — downside deviation only."""
    r = np.asarray(returns, float)
    if r.size < 2:
        return 0.0
    downside = np.minimum(r - rf / freq, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float((r.mean() - rf / freq) * freq / (dd * np.sqrt(freq))) if dd > 1e-12 else 0.0


def cagr(cumulative, freq=252):
    """Compound annual growth rate from a cumulative-wealth curve."""
    c = np.asarray(cumulative, float)
    if c.size < 2 or c[0] <= 0:
        return 0.0
    years = (c.size - 1) / freq
    return float((c[-1] / c[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0


def max_drawdown(cumulative):
    """Maximum peak-to-trough drawdown (negative fraction)."""
    c = np.asarray(cumulative, float)
    if c.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(c)
    return float(((c - running_max) / running_max).min())


def calmar_ratio(returns, freq=252):
    """Calmar = CAGR / |max drawdown|."""
    r = np.asarray(returns, float)
    if r.size < 2:
        return 0.0
    cum = np.cumprod(1.0 + r)
    mdd = abs(max_drawdown(cum))
    return float(cagr(cum, freq) / mdd) if mdd > 1e-12 else 0.0


def hit_rate(returns):
    """Fraction of periods with positive returns."""
    r = np.asarray(returns, float)
    return float(np.mean(r > 0)) if r.size else 0.0


def profit_factor(returns):
    """Gross profit / gross loss."""
    r = np.asarray(returns, float)
    gains = float(r[r > 0].sum())
    losses = float(abs(r[r < 0].sum()))
    return float(gains / losses) if losses > 1e-12 else float("inf")


def annualised_volatility(returns, freq=252):
    r = np.asarray(returns, float)
    return float(r.std(ddof=1) * np.sqrt(freq)) if r.size > 1 else 0.0

# ── Signal-evaluation metrics (IC / IR) ──────────────────────────────────────

def information_coefficient(signal, forward_returns):
    """Rank IC — Spearman correlation between signal and forward return
    for a single cross-section (both length-N, one per asset)."""
    s = np.asarray(signal, float)
    f = np.asarray(forward_returns, float)
    if s.size < 3 or s.size != f.size:
        return 0.0
    rho, _ = _stats.spearmanr(s, f)
    return float(rho) if np.isfinite(rho) else 0.0


def information_ratio(ic_series, freq=12):
    """IR = mean(IC) / std(IC) * sqrt(freq)."""
    ic = np.asarray(ic_series, float)
    if ic.size < 2:
        return 0.0
    sd = float(ic.std(ddof=1))
    return float(ic.mean() / sd * np.sqrt(freq)) if sd > 1e-12 else 0.0


def ic_decay(signal, returns_matrix, horizons=(1, 5, 10, 21)):
    """How rank-IC decays as the forward horizon lengthens."""
    s = np.asarray(signal, float)
    R = np.asarray(returns_matrix, float)
    out = {}
    for h in horizons:
        if h >= R.shape[0]:
            out[h] = 0.0
            continue
        fwd = R[:h].sum(axis=0)
        out[h] = information_coefficient(s, fwd)
    return out


# ── Risk metrics ─────────────────────────────────────────────────────────────

def value_at_risk(returns, alpha=0.05):
    """Historical VaR — the -alpha quantile of returns."""
    return float(-np.quantile(np.asarray(returns, float), alpha))


def expected_shortfall(returns, alpha=0.05):
    """Expected shortfall (CVaR) — mean of the tail beyond VaR."""
    r = np.asarray(returns, float)
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if tail.size else float(-q)


def turnover(weights_prev, weights_curr):
    """One-sided portfolio turnover: 0.5 * sum(|w_curr - w_prev|)."""
    return float(0.5 * np.abs(np.asarray(weights_curr, float) - np.asarray(weights_prev, float)).sum())


# ── Convenience: full metric bundle ──────────────────────────────────────────

def performance_summary(returns, rf=0.0, freq=252):
    """Compute the full tearsheet of metrics for a return series."""
    r = np.asarray(returns, float)
    cum = np.cumprod(1.0 + r)
    return {
        "cagr": cagr(cum, freq),
        "annual_volatility": annualised_volatility(r, freq),
        "sharpe": sharpe_ratio(r, rf, freq),
        "sortino": sortino_ratio(r, rf, freq),
        "calmar": calmar_ratio(r, freq),
        "max_drawdown": max_drawdown(cum),
        "hit_rate": hit_rate(r),
        "profit_factor": profit_factor(r),
        "var_95": value_at_risk(r, 0.05),
        "es_95": expected_shortfall(r, 0.05),
        "n_periods": int(r.size),
    }