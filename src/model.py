"""model.py — Signal research and backtesting engine for alpha discovery.

Unlike a portfolio optimiser (QuantRisk) or a Bayesian allocator
(AssetAllocator), MicroAlpha focuses on **signal research**: generating
trading signals, evaluating their predictive content via Information
Coefficient (IC), and backtesting a long-short strategy with realistic
transaction costs.

References
----------
* Moskowitz, Ooi & Pedersen (2012). "Time Series Momentum." JFE.
* Grinold & Kahn (2000). *Active Portfolio Management*, 2nd ed. (IC/IR).
"""
from __future__ import annotations
import numpy as np
from src.core import (
    information_coefficient, information_ratio, ic_decay,
    turnover, performance_summary,
)


# ── Signal generators ────────────────────────────────────────────────────────

def momentum_signal(prices, lookback=21):
    """Time-series momentum: past ``lookback``-day return per asset.
    Returns (T, N) array of signals (NaN during warm-up)."""
    P = np.asarray(prices, float)
    T, N = P.shape
    sig = np.full_like(P, np.nan)
    for t in range(lookback, T):
        sig[t] = P[t] / P[t - lookback] - 1.0
    return sig


def mean_reversion_signal(prices, lookback=21):
    """Mean-reversion signal: negative z-score of price vs moving average.
    Positive → price below average → expected to revert up.
    Returns (T, N) array (NaN during warm-up)."""
    P = np.asarray(prices, float)
    T, N = P.shape
    sig = np.full_like(P, np.nan)
    for t in range(lookback, T):
        window = P[t - lookback + 1: t + 1]
        mu = window.mean(axis=0)
        sd = window.std(axis=0, ddof=1)
        z = np.where(sd > 1e-12, (P[t] - mu) / sd, 0.0)
        sig[t] = -z
    return sig


# ── Signal evaluation ───────────────────────────────────────────────────────

def evaluate_signal(signal, returns_matrix, horizon=5):
    """Evaluate a signal's predictive power via rank-IC over time.

    For each date t, compute Spearman rank correlation between signal[t]
    and the ``horizon``-day forward return. Returns IC stats + decay."""
    S = np.asarray(signal, float)
    R = np.asarray(returns_matrix, float)
    T, N = S.shape
    ic_list = []
    for t in range(T - horizon):
        s_t = S[t]
        if np.all(np.isnan(s_t)) or np.sum(~np.isnan(s_t)) < 3:
            continue
        valid = ~np.isnan(s_t)
        fwd = R[t + 1: t + 1 + horizon].sum(axis=0)
        ic_list.append(information_coefficient(s_t[valid], fwd[valid]))
    ic_arr = np.array(ic_list) if ic_list else np.array([0.0])
    return {
        "ic_series": ic_arr.tolist(),
        "mean_ic": float(np.mean(ic_arr)),
        "ic_std": float(np.std(ic_arr, ddof=1)) if ic_arr.size > 1 else 0.0,
        "ic_ir": information_ratio(ic_arr),
        "ic_hit_rate": float(np.mean(ic_arr > 0)),
        "n_observations": int(ic_arr.size),
    }


# ── Backtesting engine ──────────────────────────────────────────────────────

def backtest(signal, returns_matrix, cost_bps=5.0, rebal_freq=21, max_leverage=1.0):
    """Backtest a long-short strategy from a signal panel.

    At each rebalance date the cross-sectional rank is converted to
    dollar-neutral weights (sum=0). Transaction costs at ``cost_bps``
    basis points per dollar traded (one-way). Returns dict with
    strategy_returns, cumulative_wealth, turnover, and performance.
    """
    S = np.asarray(signal, float)
    R = np.asarray(returns_matrix, float)
    T, N = S.shape
    cost_rate = cost_bps / 10000.0

    weights = np.zeros(N)
    strategy_returns = np.zeros(T)
    weights_history = []
    turnover_total = 0.0
    n_rebalances = 0

    for t in range(T):
        s_t = S[t]
        if t % rebal_freq == 0 and not np.all(np.isnan(s_t)):
            valid = ~np.isnan(s_t)
            n_valid = int(valid.sum())
            if n_valid < 2:
                weights = np.zeros(N)
            else:
                order = np.argsort(s_t[valid])
                ranks = np.empty(n_valid)
                ranks[order] = np.arange(1, n_valid + 1)
                norm = 2.0 * (ranks - 1) / max(n_valid - 1, 1) - 1.0
                new_w = np.zeros(N)
                new_w[valid] = norm / (n_valid / 2.0)
                new_w = np.clip(new_w, -max_leverage / N, max_leverage / N)
                turnover_total += turnover(weights, new_w)
                weights = new_w
                n_rebalances += 1
            weights_history.append(weights.copy())

        daily_ret = float(np.nansum(weights * R[t]))
        strategy_returns[t] = daily_ret

    cum_wealth = np.cumprod(1.0 + strategy_returns)
    return {
        "strategy_returns": strategy_returns,
        "cumulative_wealth": cum_wealth,
        "turnover_total": turnover_total,
        "avg_turnover": turnover_total / max(n_rebalances, 1),
        "n_rebalances": n_rebalances,
        "weights_history": np.array(weights_history) if weights_history else np.zeros((0, N)),
        "performance": performance_summary(strategy_returns),
    }


# ── Convenience: full pipeline ───────────────────────────────────────────────

def fit_and_evaluate(data, lookback=21, horizon=5, cost_bps=5.0, rebal_freq=21):
    """Run the full signal-research pipeline on a data dict.

    Generates momentum + mean-reversion signals, evaluates IC, and
    backtests a combined long-short strategy. Returns (model, metrics).
    """
    prices = np.asarray(data["prices"], float)
    returns = np.asarray(data["returns"], float)

    mom = momentum_signal(prices, lookback=lookback)
    rev = mean_reversion_signal(prices, lookback=lookback)
    combined = 0.6 * mom + 0.4 * rev

    mom_eval = evaluate_signal(mom, returns, horizon=horizon)
    rev_eval = evaluate_signal(rev, returns, horizon=horizon)
    combined_eval = evaluate_signal(combined, returns, horizon=horizon)
    bt = backtest(combined, returns, cost_bps=cost_bps, rebal_freq=rebal_freq)

    model = {
        "momentum_signal": mom,
        "mean_reversion_signal": rev,
        "combined_signal": combined,
        "weights_history": bt["weights_history"],
        "assets": data.get("assets", [f"asset_{i}" for i in range(prices.shape[1])]),
    }
    metrics = {
        "assets": len(model["assets"]),
        "lookback": lookback,
        "horizon": horizon,
        "cost_bps": cost_bps,
        "rebal_freq": rebal_freq,
        "momentum_ic": mom_eval["mean_ic"],
        "momentum_ir": mom_eval["ic_ir"],
        "reversion_ic": rev_eval["mean_ic"],
        "reversion_ir": rev_eval["ic_ir"],
        "combined_ic": combined_eval["mean_ic"],
        "combined_ir": combined_eval["ic_ir"],
        "n_rebalances": bt["n_rebalances"],
        "avg_turnover": bt["avg_turnover"],
        **bt["performance"],
    }
    return model, metrics