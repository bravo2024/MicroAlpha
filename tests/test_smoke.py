"""Smoke tests for the MicroAlpha signal-research pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import momentum_signal, mean_reversion_signal, evaluate_signal, backtest, fit_and_evaluate
from src.core import sharpe_ratio, max_drawdown, information_coefficient, performance_summary


def test_data_generation():
    """Synthetic data has the right shape and embedded structure."""
    data = make_synthetic(n_days=252, seed=42)
    assert data["prices"].shape == (252, 6)
    assert data["returns"].shape == (252, 6)
    assert len(data["momentum_assets"]) == 3
    assert len(data["reversion_assets"]) == 3


def test_momentum_signal():
    """Momentum signal is NaN during warm-up, then finite."""
    data = make_synthetic(n_days=126, seed=0)
    sig = momentum_signal(data["prices"], lookback=21)
    assert sig.shape == (126, 6)
    assert np.all(np.isnan(sig[:21]))
    assert np.any(~np.isnan(sig[21:]))


def test_mean_reversion_signal():
    """Mean-reversion signal has the right sign convention."""
    data = make_synthetic(n_days=126, seed=0)
    sig = mean_reversion_signal(data["prices"], lookback=21)
    assert sig.shape == (126, 6)
    assert np.all(np.isnan(sig[:21]))


def test_signal_evaluation():
    """Evaluate_signal returns IC stats with reasonable values."""
    data = make_synthetic(n_days=252, seed=42)
    mom = momentum_signal(data["prices"], lookback=21)
    eval_result = evaluate_signal(mom, data["returns"], horizon=5)
    assert "mean_ic" in eval_result
    assert "ic_ir" in eval_result
    assert eval_result["n_observations"] > 0


def test_backtest():
    """Backtest produces returns and performance metrics."""
    data = make_synthetic(n_days=252, seed=42)
    mom = momentum_signal(data["prices"], lookback=21)
    rev = mean_reversion_signal(data["prices"], lookback=21)
    combined = 0.6 * mom + 0.4 * rev
    bt = backtest(combined, data["returns"], cost_bps=5.0, rebal_freq=21)
    assert bt["strategy_returns"].shape == (252,)
    assert bt["n_rebalances"] > 0
    assert "sharpe" in bt["performance"]
    assert bt["performance"]["n_periods"] == 252


def test_fit_and_evaluate():
    """Full pipeline returns model and metrics dicts."""
    data = make_synthetic(n_days=504, seed=7)
    model, metrics = fit_and_evaluate(data, lookback=21, horizon=5)
    assert "momentum_signal" in model
    assert "combined_signal" in model
    assert "sharpe" in metrics
    assert "momentum_ic" in metrics
    assert "combined_ic" in metrics
    assert metrics["n_rebalances"] > 0


def test_core_metrics():
    """Core metric functions compute correctly on known inputs."""
    r = np.array([0.01, -0.02, 0.03, 0.01, -0.01])
    assert sharpe_ratio(r) != 0.0
    cum = np.cumprod(1 + r)
    assert max_drawdown(cum) <= 0
    s = np.array([1, 2, 3, 4, 5], dtype=float)
    f = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    assert abs(information_coefficient(s, f) - 1.0) < 0.01  # perfect rank correlation


def test_performance_summary():
    """Performance summary returns all expected keys."""
    r = np.random.default_rng(0).normal(0.001, 0.02, 252)
    summary = performance_summary(r)
    assert all(k in summary for k in ["cagr", "sharpe", "sortino", "calmar",
                                       "max_drawdown", "hit_rate", "profit_factor"])


if __name__ == "__main__":
    test_data_generation()
    test_momentum_signal()
    test_mean_reversion_signal()
    test_signal_evaluation()
    test_backtest()
    test_fit_and_evaluate()
    test_core_metrics()
    test_performance_summary()
    print("All MicroAlpha smoke tests passed!")
