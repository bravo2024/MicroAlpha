import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

st.set_page_config(page_title="MicroAlpha | Quant Alpha Research", layout="wide", page_icon="📊")

plt.style.use("dark_background")
DARK_BG   = "#0e1117"
PANEL_BG  = "#1a1d27"
ACCENT    = "#00d4ff"
GREEN     = "#00ff88"
RED       = "#ff4444"
YELLOW    = "#ffd700"

TICKERS   = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
SEED      = 42
TRADING_DAYS = 252

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    selected_tickers = st.multiselect("Tickers", TICKERS, default=TICKERS)
    if not selected_tickers:
        selected_tickers = TICKERS
    lookback = st.slider("Lookback (days)", 63, 756, 504)
    rebal_freq = st.selectbox("Rebalance Frequency", ["Monthly (21d)", "Weekly (5d)", "Quarterly (63d)"], index=0)
    rf_rate = st.number_input("Risk-Free Rate (annual)", 0.0, 0.10, 0.04, 0.005, format="%.3f")
    var_conf = st.selectbox("VaR Confidence Level", [0.95, 0.99], index=0)
    st.markdown("---")
    st.markdown("**MicroAlpha** v2.0  \nQuantitative Alpha Research")

rebal_map = {"Monthly (21d)": 21, "Weekly (5d)": 5, "Quarterly (63d)": 63}
REBAL = rebal_map[rebal_freq]

# ── Data Generation ─────────────────────────────────────────────────────────────
@st.cache_data
def generate_ohlcv(tickers, n_days=756, seed=SEED):
    rng = np.random.default_rng(seed)
    params = {
        "AAPL":  {"S0": 150.0, "mu": 0.22, "sigma": 0.28},
        "GOOGL": {"S0": 120.0, "mu": 0.18, "sigma": 0.30},
        "MSFT":  {"S0": 280.0, "mu": 0.20, "sigma": 0.25},
        "AMZN":  {"S0": 130.0, "mu": 0.25, "sigma": 0.35},
        "META":  {"S0":  90.0, "mu": 0.30, "sigma": 0.40},
    }
    dates = pd.bdate_range(end="2024-12-31", periods=n_days)
    result = {}
    for t in tickers:
        p = params[t]
        S0, mu, sigma = p["S0"], p["mu"], p["sigma"]
        dt = 1 / TRADING_DAYS
        W  = rng.standard_normal(n_days)
        log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
        close = S0 * np.exp(np.cumsum(log_ret))
        noise = rng.uniform(0.005, 0.020, n_days)
        high   = close * (1 + noise)
        low    = close * (1 - noise)
        open_  = np.roll(close, 1); open_[0] = S0
        vol    = rng.integers(5_000_000, 50_000_000, n_days).astype(float)
        result[t] = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                                   "Close": close, "Volume": vol}, index=dates)
    return result

@st.cache_data
def compute_returns(tickers, n_days=756):
    data = generate_ohlcv(tickers, n_days)
    closes = pd.DataFrame({t: data[t]["Close"] for t in tickers})
    return closes.pct_change().dropna()

# ── Helper: styled figure ───────────────────────────────────────────────────────
def _fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#333")
    return fig, ax

def _fig2(w=12, h=8):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(w, h), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
    return fig, ax1, ax2

def sharpe(r, rf=0.0):
    er = r.mean() * TRADING_DAYS - rf
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return er / vol if vol > 1e-9 else 0.0

def max_drawdown(cum):
    roll_max = np.maximum.accumulate(cum)
    dd = (cum - roll_max) / roll_max
    return dd.min()

def cagr(cum, n_days):
    years = n_days / TRADING_DAYS
    return (cum[-1] / cum[0]) ** (1 / years) - 1 if years > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# HEADER KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1 style='color:#00d4ff;text-align:center;'>📊 MicroAlpha | Quantitative Alpha Research</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888;text-align:center;'>Walk-forward backtesting · Factor engineering · Portfolio optimization · Risk attribution</p>", unsafe_allow_html=True)

rets_all = compute_returns(selected_tickers, 756)
rets_hdr = rets_all.tail(lookback)

eq_curve_hdr = (1 + rets_hdr.mean(axis=1)).cumprod().values
ann_ret_hdr  = cagr(eq_curve_hdr, len(eq_curve_hdr))
sh_hdr       = sharpe(rets_hdr.mean(axis=1), rf_rate / TRADING_DAYS)
mdd_hdr      = max_drawdown(eq_curve_hdr)
wins_hdr     = (rets_hdr.mean(axis=1) > 0).mean()
calmar_hdr   = ann_ret_hdr / abs(mdd_hdr) if abs(mdd_hdr) > 1e-9 else 0.0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Ann. Return",      f"{ann_ret_hdr:.1%}")
c2.metric("Sharpe Ratio",     f"{sh_hdr:.2f}")
c3.metric("Max Drawdown",     f"{mdd_hdr:.1%}")
c4.metric("Win Rate",         f"{wins_hdr:.1%}")
c5.metric("Calmar Ratio",     f"{calmar_hdr:.2f}")
c6.metric("Tickers",          str(len(selected_tickers)))

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Data Explorer",
    "⚙️ Alpha Factor Engineering",
    "🧪 Walk-Forward Backtesting",
    "📈 Portfolio Construction",
    "💰 Risk & Performance Attribution",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Market Data Explorer — Synthetic OHLCV (GBM)")
    st.latex(r"S(t) = S(0)\,\exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)t + \sigma W(t)\right]")

    ticker_sel = st.selectbox("Select ticker for OHLC chart", selected_tickers, key="t1_ticker")
    data_all   = generate_ohlcv(selected_tickers, 756)
    df_tick    = data_all[ticker_sel].tail(lookback)

    # OHLC bar chart
    fig, ax1_t, ax2_t = _fig2(14, 8)
    idx = np.arange(len(df_tick))
    up   = df_tick["Close"] >= df_tick["Open"]
    down = ~up
    ax1_t.bar(idx[up],   df_tick["Close"].values[up]  - df_tick["Open"].values[up],
              bottom=df_tick["Open"].values[up],   color=GREEN, width=0.8, alpha=0.85)
    ax1_t.bar(idx[down], df_tick["Close"].values[down] - df_tick["Open"].values[down],
              bottom=df_tick["Open"].values[down], color=RED,   width=0.8, alpha=0.85)
    ax1_t.vlines(idx[up],   df_tick["Low"].values[up],   df_tick["High"].values[up],   color=GREEN, lw=0.6)
    ax1_t.vlines(idx[down], df_tick["Low"].values[down], df_tick["High"].values[down], color=RED,   lw=0.6)
    xt = np.linspace(0, len(df_tick)-1, 6, dtype=int)
    ax1_t.set_xticks(xt)
    ax1_t.set_xticklabels([str(df_tick.index[i].date()) for i in xt], rotation=30, color="white", fontsize=8)
    ax1_t.set_title(f"{ticker_sel} — OHLC Bar Chart", color="white", fontsize=13)
    ax1_t.set_ylabel("Price (USD)", color="white")

    ax2_t.bar(idx, df_tick["Volume"].values / 1e6, color=ACCENT, alpha=0.7)
    ax2_t.set_xticks(xt)
    ax2_t.set_xticklabels([str(df_tick.index[i].date()) for i in xt], rotation=30, color="white", fontsize=8)
    ax2_t.set_title("Volume (M shares)", color="white", fontsize=11)
    ax2_t.set_ylabel("Volume (M)", color="white")
    fig.tight_layout(pad=2)
    st.pyplot(fig); plt.close(fig)

    # Rolling correlation heatmap
    st.markdown("#### 30-Day Rolling Correlation Heatmap")
    rets_t1 = rets_all[selected_tickers].tail(lookback)
    corr30  = rets_t1.rolling(30).corr().dropna().groupby(level=0).last()

    fig2, ax2 = _fig(10, 4)
    n = len(selected_tickers)
    corr_mat = corr30.values.reshape(n, n) if corr30.shape == (n, n) else rets_t1.corr().values
    im = ax2.imshow(corr_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels(selected_tickers, color="white"); ax2.set_yticklabels(selected_tickers, color="white")
    for i in range(n):
        for j in range(n):
            ax2.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.046)
    ax2.set_title("30-Day Rolling Correlation Matrix", color="white")
    fig2.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    # Summary statistics
    st.markdown("#### Summary Statistics")
    rows = []
    for t in selected_tickers:
        r = rets_all[t].tail(lookback)
        cum = (1 + r).cumprod().values
        rows.append({
            "Ticker": t,
            "Ann. Return": f"{cagr(cum, len(cum)):.1%}",
            "Ann. Volatility": f"{r.std()*np.sqrt(TRADING_DAYS):.1%}",
            "Sharpe": f"{sharpe(r, rf_rate/TRADING_DAYS):.2f}",
            "Max Drawdown": f"{max_drawdown(cum):.1%}",
            "Skewness": f"{stats.skew(r):.2f}",
            "Kurtosis": f"{stats.kurtosis(r):.2f}",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Alpha Factor Engineering
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Alpha Factor Engineering — 6 Factors (Pure NumPy)")
    st.latex(r"\text{IC} = \text{rank\_corr}(f_t,\; r_{t+1})")

    @st.cache_data
    def compute_factors(tickers, n_days=756):
        data  = generate_ohlcv(tickers, n_days)
        close = pd.DataFrame({t: data[t]["Close"] for t in tickers})
        vol   = pd.DataFrame({t: data[t]["Volume"] for t in tickers})
        rets  = close.pct_change()
        out   = {}

        # 1. Momentum 12-1
        mom = close.pct_change(252).shift(21)
        out["Momentum_12_1"] = mom

        # 2. RSI(14)
        def rsi14(s):
            delta = s.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / (loss + 1e-9)
            return 100 - 100 / (1 + rs)
        out["RSI_14"] = close.apply(rsi14)

        # 3. Volatility reversal (low vol → long)
        out["Vol_Reversal"] = -rets.rolling(20).std()

        # 4. Price-Volume divergence (negated corr)
        def pv_corr(df_price, df_volume, w=20):
            result = pd.DataFrame(index=df_price.index, columns=df_price.columns, dtype=float)
            for t in df_price.columns:
                result[t] = -df_price[t].rolling(w).corr(df_volume[t])
            return result
        out["PV_Divergence"] = pv_corr(close, vol)

        # 5. Mean reversion (negated z-score)
        def mean_rev(s, w=20):
            ma  = s.rolling(w).mean()
            std = s.rolling(w).std()
            return -(s - ma) / (std + 1e-9)
        out["Mean_Reversion"] = close.apply(mean_rev)

        # 6. Earnings momentum proxy: 60d - 20d return
        out["Earnings_Mom"] = close.pct_change(60) - close.pct_change(20)

        return out, rets

    factors, rets_f = compute_factors(tuple(selected_tickers), 756)
    factor_names    = list(factors.keys())

    # IC computation
    def compute_ic(factor_df, rets_df, horizon=1):
        fwd = rets_df.shift(-horizon)
        ic_series = []
        idx = factor_df.dropna(how="all").index
        for date in idx:
            if date not in fwd.index: continue
            f = factor_df.loc[date].dropna()
            r = fwd.loc[date].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 3: continue
            ic, _ = stats.spearmanr(f[common], r[common])
            ic_series.append((date, ic))
        if not ic_series: return pd.Series(dtype=float)
        dates, vals = zip(*ic_series)
        return pd.Series(vals, index=pd.DatetimeIndex(dates))

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_factor = st.selectbox("Select factor to inspect", factor_names)
    with col_f2:
        ic_horizon = st.selectbox("IC horizon (days)", [1, 5, 10, 21], index=0)

    ic_s = compute_ic(factors[sel_factor][selected_tickers], rets_f[selected_tickers], ic_horizon)

    # IC time series
    if len(ic_s) > 0:
        fig, ax = _fig(13, 4)
        ic_roll = ic_s.rolling(21).mean()
        ax.bar(range(len(ic_s)), ic_s.values, color=np.where(ic_s.values > 0, GREEN, RED), alpha=0.5, width=1)
        ax.plot(range(len(ic_roll)), ic_roll.values, color=YELLOW, lw=1.5, label="21d Rolling IC")
        ax.axhline(0, color="white", lw=0.8, ls="--")
        ax.set_title(f"{sel_factor} — IC at lag {ic_horizon}d  |  Mean IC = {ic_s.mean():.3f}", color="white")
        ax.set_ylabel("IC (Spearman)", color="white"); ax.legend(facecolor=PANEL_BG)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    # IC heatmap across factors
    st.markdown("#### IC Heatmap: Factors × Lag")
    lags = [1, 5, 10, 21]
    ic_matrix = np.zeros((len(factor_names), len(lags)))
    for i, fn in enumerate(factor_names):
        for j, lag in enumerate(lags):
            ic_tmp = compute_ic(factors[fn][selected_tickers], rets_f[selected_tickers], lag)
            ic_matrix[i, j] = ic_tmp.mean() if len(ic_tmp) > 0 else 0.0

    fig3, ax3 = _fig(10, 5)
    im3 = ax3.imshow(ic_matrix, cmap="RdYlGn", vmin=-0.1, vmax=0.1, aspect="auto")
    ax3.set_xticks(range(len(lags))); ax3.set_xticklabels([f"Lag {l}d" for l in lags], color="white")
    ax3.set_yticks(range(len(factor_names))); ax3.set_yticklabels(factor_names, color="white")
    for i in range(len(factor_names)):
        for j in range(len(lags)):
            ax3.text(j, i, f"{ic_matrix[i,j]:.3f}", ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.set_title("Mean IC: Factors × Lag (lower = factor decays faster)", color="white")
    fig3.tight_layout(); st.pyplot(fig3); plt.close(fig3)

    # Factor decay chart
    st.markdown("#### Factor Decay (IC by Lag)")
    fig4, ax4 = _fig(10, 4)
    colors_f = [ACCENT, GREEN, YELLOW, RED, "#ff88ff", "#88ffcc"]
    for i, fn in enumerate(factor_names):
        ax4.plot(lags, ic_matrix[i], marker="o", color=colors_f[i % len(colors_f)], label=fn, lw=1.5)
    ax4.axhline(0, color="white", lw=0.8, ls="--")
    ax4.set_xlabel("Lag (days)", color="white"); ax4.set_ylabel("Mean IC", color="white")
    ax4.set_title("Factor Decay: IC vs Prediction Horizon", color="white")
    ax4.legend(facecolor=PANEL_BG, fontsize=8); fig4.tight_layout()
    st.pyplot(fig4); plt.close(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Walk-Forward Backtesting
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Walk-Forward Backtesting — Expanding Window")
    st.latex(r"S = \frac{\bar{R} - r_f}{\sigma}\,\sqrt{252}")
    st.latex(r"\text{MDD} = \max_t \frac{\max_{s \leq t} V_s - V_t}{\max_{s \leq t} V_s}")

    @st.cache_data
    def run_backtest(tickers, rebal, n_days=756):
        data  = generate_ohlcv(tickers, n_days)
        close = pd.DataFrame({t: data[t]["Close"] for t in tickers})
        rets  = close.pct_change().dropna()
        n     = len(rets)
        train_win = 252
        test_win  = rebal

        # composite factor: equal weight of all 6
        factors_bt, _ = compute_factors(tickers, n_days)
        composite = sum(
            factors_bt[fn][list(tickers)].rank(axis=1, pct=True).fillna(0.5)
            for fn in factors_bt
        ) / len(factors_bt)
        composite = composite.reindex(rets.index).fillna(0.5)

        pnl_list  = []
        dates_out = []
        turnover_list = []
        prev_w = None

        for start in range(train_win, n - test_win, test_win):
            test_idx  = rets.index[start: start + test_win]
            score_row = composite.iloc[start - 1]
            q20 = score_row.quantile(0.2); q80 = score_row.quantile(0.8)
            longs  = (score_row >= q80).astype(float)
            shorts = (score_row <= q20).astype(float)
            n_l, n_s = longs.sum(), shorts.sum()
            if n_l == 0 or n_s == 0: continue
            w = longs / n_l - shorts / n_s
            w = w / (w.abs().sum() + 1e-9)
            if prev_w is not None:
                turnover_list.append((w - prev_w).abs().sum())
            prev_w = w.copy()
            period_rets = rets.loc[test_idx]
            pnl = period_rets @ w
            pnl_list.append(pnl)
            dates_out.append(test_idx)

        if not pnl_list:
            return pd.Series(dtype=float), {}

        all_pnl = pd.concat(pnl_list).sort_index()
        cum     = (1 + all_pnl).cumprod()
        roll_sh = all_pnl.rolling(63).apply(lambda x: sharpe(x, 0.0))
        roll_max = cum.expanding().max()
        dd_series = (cum - roll_max) / roll_max

        metrics = {
            "CAGR":           cagr(cum.values, len(cum)),
            "Sharpe":         sharpe(all_pnl),
            "Max Drawdown":   max_drawdown(cum.values),
            "Calmar":         cagr(cum.values, len(cum)) / abs(max_drawdown(cum.values) + 1e-9),
            "Avg Turnover":   float(np.mean(turnover_list)) if turnover_list else 0.0,
            "Win Rate":       float((all_pnl > 0).mean()),
        }
        return all_pnl, cum, dd_series, roll_sh, metrics

    bt_result = run_backtest(tuple(selected_tickers), REBAL, 756)
    if isinstance(bt_result[0], pd.Series) and len(bt_result[0]) == 0:
        st.warning("Insufficient data for backtest. Try selecting more tickers or reducing lookback.")
    else:
        bt_rets, bt_cum, bt_dd, bt_roll_sh, bt_metrics = bt_result

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("CAGR",         f"{bt_metrics['CAGR']:.1%}")
        m2.metric("Sharpe",       f"{bt_metrics['Sharpe']:.2f}")
        m3.metric("Max DD",       f"{bt_metrics['Max Drawdown']:.1%}")
        m4.metric("Calmar",       f"{bt_metrics['Calmar']:.2f}")
        m5.metric("Win Rate",     f"{bt_metrics['Win Rate']:.1%}")
        m6.metric("Avg Turnover", f"{bt_metrics['Avg Turnover']:.2f}")

        # Equity curve
        fig_bt, axes_bt = plt.subplots(3, 1, figsize=(13, 10), facecolor=DARK_BG)
        for ax in axes_bt:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white"); ax.title.set_color("white")
            for spine in ax.spines.values(): spine.set_edgecolor("#333")

        axes_bt[0].plot(bt_cum.index, bt_cum.values, color=ACCENT, lw=1.5)
        axes_bt[0].fill_between(bt_cum.index, 1, bt_cum.values, alpha=0.15, color=ACCENT)
        axes_bt[0].set_title("Strategy Equity Curve (Long/Short Quintile)", color="white")
        axes_bt[0].set_ylabel("Cumulative NAV", color="white")

        axes_bt[1].fill_between(bt_dd.index, bt_dd.values, 0, color=RED, alpha=0.7)
        axes_bt[1].set_title("Drawdown", color="white"); axes_bt[1].set_ylabel("Drawdown", color="white")

        if bt_roll_sh is not None:
            valid = bt_roll_sh.dropna()
            axes_bt[2].plot(valid.index, valid.values, color=GREEN, lw=1.2)
            axes_bt[2].axhline(0, color="white", lw=0.8, ls="--")
            axes_bt[2].set_title("Rolling 63-Day Sharpe Ratio", color="white")
            axes_bt[2].set_ylabel("Sharpe", color="white")

        fig_bt.tight_layout(pad=2)
        st.pyplot(fig_bt); plt.close(fig_bt)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Portfolio Construction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Portfolio Construction — Markowitz Mean-Variance Optimization")
    st.latex(r"\min_w\; w^\top \Sigma w \quad \text{s.t.} \quad w^\top \mu = \mu^*, \quad w^\top \mathbf{1} = 1, \quad w \geq 0")

    rets_mv = rets_all[selected_tickers].tail(lookback)
    mu_vec  = rets_mv.mean().values * TRADING_DAYS
    cov_mat = rets_mv.cov().values * TRADING_DAYS
    n_assets = len(selected_tickers)

    def port_stats(w):
        r = float(w @ mu_vec)
        v = float(np.sqrt(w @ cov_mat @ w))
        s = (r - rf_rate) / v if v > 1e-9 else 0.0
        return r, v, s

    # Random portfolios for efficient frontier
    rng_mv = np.random.default_rng(SEED)
    n_ports = 3000
    rnd_w   = rng_mv.dirichlet(np.ones(n_assets), n_ports)
    rnd_r   = np.array([port_stats(w)[0] for w in rnd_w])
    rnd_v   = np.array([port_stats(w)[1] for w in rnd_w])
    rnd_s   = np.array([port_stats(w)[2] for w in rnd_w])

    # Min-variance via gradient descent (pure NumPy, constrained)
    def min_var_port(cov, n_iter=2000, lr=0.01):
        w = np.ones(n_assets) / n_assets
        for _ in range(n_iter):
            grad = 2 * cov @ w
            w = w - lr * grad
            w = np.clip(w, 0, None)
            w /= w.sum() + 1e-12
        return w

    def max_sharpe_port(mu, cov, rf, n_iter=2000, lr=0.005):
        w = np.ones(n_assets) / n_assets
        for _ in range(n_iter):
            p_r = w @ mu; p_v = np.sqrt(w @ cov @ w + 1e-12)
            grad_r = mu; grad_v = cov @ w / p_v
            grad_sh = (grad_r * p_v - (p_r - rf) * grad_v) / (p_v**2)
            w = w + lr * grad_sh
            w = np.clip(w, 0, None)
            w /= w.sum() + 1e-12
        return w

    w_minv = min_var_port(cov_mat)
    w_maxs = max_sharpe_port(mu_vec, cov_mat, rf_rate)
    r_minv, v_minv, s_minv = port_stats(w_minv)
    r_maxs, v_maxs, s_maxs = port_stats(w_maxs)

    col4a, col4b = st.columns([3, 2])
    with col4a:
        # Efficient frontier
        fig_ef, ax_ef = _fig(10, 6)
        sc = ax_ef.scatter(rnd_v, rnd_r, c=rnd_s, cmap="plasma", s=6, alpha=0.6)
        plt.colorbar(sc, ax=ax_ef, label="Sharpe Ratio")
        ax_ef.scatter(v_minv, r_minv, color=GREEN,  s=200, marker="*", zorder=5, label=f"Min Variance (Sh={s_minv:.2f})")
        ax_ef.scatter(v_maxs, r_maxs, color=YELLOW, s=200, marker="*", zorder=5, label=f"Max Sharpe  (Sh={s_maxs:.2f})")
        ax_ef.set_xlabel("Ann. Volatility", color="white"); ax_ef.set_ylabel("Ann. Return", color="white")
        ax_ef.set_title("Efficient Frontier (3,000 Random Portfolios)", color="white")
        ax_ef.legend(facecolor=PANEL_BG, fontsize=9)
        fig_ef.tight_layout(); st.pyplot(fig_ef); plt.close(fig_ef)

    with col4b:
        st.markdown("#### Optimal Weights")
        wt_df = pd.DataFrame({
            "Ticker": selected_tickers,
            "Min Variance": [f"{v:.1%}" for v in w_minv],
            "Max Sharpe":   [f"{v:.1%}" for v in w_maxs],
        }).set_index("Ticker")
        st.dataframe(wt_df, use_container_width=True)
        st.markdown("#### Min-Variance Portfolio")
        st.metric("Ann. Return",   f"{r_minv:.1%}")
        st.metric("Ann. Volatility", f"{v_minv:.1%}")
        st.metric("Sharpe",        f"{s_minv:.2f}")
        st.markdown("#### Max-Sharpe Portfolio")
        st.metric("Ann. Return",   f"{r_maxs:.1%}")
        st.metric("Ann. Volatility", f"{v_maxs:.1%}")
        st.metric("Sharpe",        f"{s_maxs:.2f}")

    # Covariance heatmap
    st.markdown("#### Covariance Matrix Heatmap")
    fig_cov, ax_cov = _fig(9, 4)
    im_cov = ax_cov.imshow(cov_mat, cmap="coolwarm", aspect="auto")
    ax_cov.set_xticks(range(n_assets)); ax_cov.set_yticks(range(n_assets))
    ax_cov.set_xticklabels(selected_tickers, color="white"); ax_cov.set_yticklabels(selected_tickers, color="white")
    for i in range(n_assets):
        for j in range(n_assets):
            ax_cov.text(j, i, f"{cov_mat[i,j]:.4f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im_cov, ax=ax_cov, fraction=0.046)
    ax_cov.set_title("Annualised Covariance Matrix", color="white")
    fig_cov.tight_layout(); st.pyplot(fig_cov); plt.close(fig_cov)

    # Risk decomposition
    st.markdown("#### Risk Decomposition — Asset Contribution to Portfolio Variance")
    w_use = w_maxs
    port_var = float(w_use @ cov_mat @ w_use)
    mrc  = cov_mat @ w_use
    ctv  = w_use * mrc
    pctv = ctv / (port_var + 1e-12)

    fig_rd, ax_rd = _fig(9, 3.5)
    bars = ax_rd.bar(selected_tickers, pctv * 100, color=[ACCENT, GREEN, YELLOW, RED, "#ff88ff"][:n_assets], alpha=0.85)
    ax_rd.set_ylabel("% Contribution to Variance", color="white")
    ax_rd.set_title("Risk Decomposition (Max-Sharpe Portfolio)", color="white")
    for bar, val in zip(bars, pctv):
        ax_rd.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:.1%}", ha="center", va="bottom", color="white", fontsize=9)
    fig_rd.tight_layout(); st.pyplot(fig_rd); plt.close(fig_rd)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Risk & Performance Attribution
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Risk & Performance Attribution — Fama-French Style Factor Model")
    st.latex(r"r_p = \alpha + \beta_{\text{MKT}} r_{\text{MKT}} + \beta_{\text{SMB}} r_{\text{SMB}} + \beta_{\text{HML}} r_{\text{HML}} + \varepsilon")
    st.latex(r"\text{VaR}_\alpha = Q_\alpha(\text{losses}), \quad \text{CVaR}_\alpha = \mathbb{E}[\text{losses} \mid \text{losses} > \text{VaR}_\alpha]")
    st.latex(r"S_{\text{Sortino}} = \frac{\bar{R} - r_f}{\sigma_{\text{down}}}\,\sqrt{252}, \quad \sigma_{\text{down}} = \text{std}(r_t\,[r_t < 0])")

    @st.cache_data
    def generate_ff_factors(n_days=756, seed=SEED):
        rng = np.random.default_rng(seed + 99)
        dates = pd.bdate_range(end="2024-12-31", periods=n_days)
        mkt  = rng.normal(0.0004, 0.012, n_days)
        smb  = rng.normal(0.0001, 0.006, n_days)
        hml  = rng.normal(0.0001, 0.006, n_days)
        rf_d = np.full(n_days, 0.04 / TRADING_DAYS)
        return pd.DataFrame({"MKT_RF": mkt - rf_d, "SMB": smb, "HML": hml, "RF": rf_d}, index=dates)

    ff = generate_ff_factors(756)
    rets_t5 = rets_all[selected_tickers].tail(lookback)

    # Equal-weight portfolio
    port_rets = rets_t5.mean(axis=1)
    common_idx = port_rets.index.intersection(ff.index)
    port_rets_c = port_rets.loc[common_idx]
    ff_c = ff.loc[common_idx]

    # OLS factor regression
    Y = port_rets_c.values - ff_c["RF"].values
    X = np.column_stack([np.ones(len(Y)), ff_c["MKT_RF"].values, ff_c["SMB"].values, ff_c["HML"].values])
    beta_hat, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    alpha_ann = beta_hat[0] * TRADING_DAYS
    betas = beta_hat[1:]
    residuals = Y - X @ beta_hat
    r2 = 1 - np.var(residuals) / (np.var(Y) + 1e-12)

    col5a, col5b, col5c, col5d = st.columns(4)
    col5a.metric("Alpha (ann.)",   f"{alpha_ann:.2%}")
    col5b.metric("Beta MKT",       f"{betas[0]:.2f}")
    col5c.metric("Beta SMB",       f"{betas[1]:.2f}")
    col5d.metric("Beta HML",       f"{betas[2]:.2f}")

    # Rolling Beta
    fig_rb, ax_rb = _fig(13, 4)
    roll_beta = []
    window_rb = 63
    pr_arr = port_rets_c.values
    mkt_arr = ff_c["MKT_RF"].values
    for i in range(window_rb, len(pr_arr)):
        y_w = pr_arr[i-window_rb:i]; x_w = mkt_arr[i-window_rb:i]
        xb  = np.column_stack([np.ones(window_rb), x_w])
        bw, _, _, _ = np.linalg.lstsq(xb, y_w, rcond=None)
        roll_beta.append(bw[1])
    rb_idx = port_rets_c.index[window_rb:]
    ax_rb.plot(rb_idx, roll_beta, color=ACCENT, lw=1.3)
    ax_rb.axhline(1.0, color="white", lw=0.8, ls="--", label="Beta=1")
    ax_rb.axhline(0.0, color=RED,     lw=0.8, ls=":")
    ax_rb.set_title("Rolling 63-Day Market Beta", color="white")
    ax_rb.set_ylabel("Beta (MKT)", color="white"); ax_rb.legend(facecolor=PANEL_BG)
    fig_rb.tight_layout(); st.pyplot(fig_rb); plt.close(fig_rb)

    # VaR and CVaR
    st.markdown("#### Value-at-Risk & Conditional VaR")
    losses = -port_rets_c.values
    var95  = np.quantile(losses, var_conf)
    cvar95 = losses[losses >= var95].mean() if (losses >= var95).any() else var95
    var99  = np.quantile(losses, 0.99)
    cvar99 = losses[losses >= var99].mean() if (losses >= var99).any() else var99

    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    col_v1.metric(f"VaR  ({var_conf:.0%})",  f"{var95:.2%}")
    col_v2.metric(f"CVaR ({var_conf:.0%})",  f"{cvar95:.2%}")
    col_v3.metric("VaR  (99%)",  f"{var99:.2%}")
    col_v4.metric("CVaR (99%)",  f"{cvar99:.2%}")

    # Loss distribution
    fig_var, ax_var = _fig(12, 4)
    ax_var.hist(losses * 100, bins=60, color=ACCENT, alpha=0.6, edgecolor="none", label="Daily Losses (%)")
    ax_var.axvline(var95 * 100,  color=YELLOW, lw=2, ls="--", label=f"VaR  {var_conf:.0%} = {var95:.2%}")
    ax_var.axvline(cvar95 * 100, color=RED,    lw=2, ls="-",  label=f"CVaR {var_conf:.0%} = {cvar95:.2%}")
    ax_var.axvline(var99 * 100,  color="#ff8800", lw=1.5, ls=":", label=f"VaR  99% = {var99:.2%}")
    ax_var.set_xlabel("Daily Loss (%)", color="white"); ax_var.set_ylabel("Frequency", color="white")
    ax_var.set_title("Loss Distribution with VaR / CVaR", color="white")
    ax_var.legend(facecolor=PANEL_BG, fontsize=9)
    fig_var.tight_layout(); st.pyplot(fig_var); plt.close(fig_var)

    # Full performance tearsheet
    st.markdown("#### Performance Tearsheet")
    cum_t5 = (1 + port_rets_c).cumprod().values
    neg_rets = port_rets_c.values[port_rets_c.values < 0]
    sigma_down = neg_rets.std() * np.sqrt(TRADING_DAYS) if len(neg_rets) > 1 else 1e-9
    sortino = (port_rets_c.mean() * TRADING_DAYS - rf_rate) / sigma_down if sigma_down > 1e-9 else 0.0

    daily_pnl = np.diff(cum_t5, prepend=cum_t5[0])
    wins_t5   = (port_rets_c.values > 0)
    profits   = port_rets_c.values[wins_t5].sum()
    gross_loss = abs(port_rets_c.values[~wins_t5].sum())
    profit_factor = profits / (gross_loss + 1e-9)

    tearsheet = {
        "CAGR":               f"{cagr(cum_t5, len(cum_t5)):.2%}",
        "Sharpe Ratio":       f"{sharpe(port_rets_c, rf_rate/TRADING_DAYS):.2f}",
        "Sortino Ratio":      f"{sortino:.2f}",
        "Calmar Ratio":       f"{cagr(cum_t5, len(cum_t5)) / (abs(max_drawdown(cum_t5)) + 1e-9):.2f}",
        "Max Drawdown":       f"{max_drawdown(cum_t5):.2%}",
        "Win Rate":           f"{wins_t5.mean():.2%}",
        "Profit Factor":      f"{profit_factor:.2f}",
        "Ann. Volatility":    f"{port_rets_c.std()*np.sqrt(TRADING_DAYS):.2%}",
        "Skewness":           f"{stats.skew(port_rets_c):.2f}",
        "Kurtosis":           f"{stats.kurtosis(port_rets_c):.2f}",
        "Alpha (ann.)":       f"{alpha_ann:.2%}",
        "R² (factor model)":  f"{r2:.3f}",
    }
    ts_df = pd.DataFrame(list(tearsheet.items()), columns=["Metric", "Value"])
    col_ts1, col_ts2 = st.columns(2)
    half = len(ts_df) // 2
    col_ts1.dataframe(ts_df.iloc[:half].set_index("Metric"), use_container_width=True)
    col_ts2.dataframe(ts_df.iloc[half:].set_index("Metric"), use_container_width=True)

    # Factor attribution bar chart
    st.markdown("#### Factor Attribution — Return Decomposition")
    factor_labels   = ["Alpha", "MKT-RF", "SMB", "HML"]
    factor_mean_ret = np.array([
        alpha_ann,
        betas[0] * ff_c["MKT_RF"].mean() * TRADING_DAYS,
        betas[1] * ff_c["SMB"].mean()     * TRADING_DAYS,
        betas[2] * ff_c["HML"].mean()     * TRADING_DAYS,
    ])
    fig_fa, ax_fa = _fig(9, 3.5)
    colors_fa = [GREEN if v >= 0 else RED for v in factor_mean_ret]
    ax_fa.bar(factor_labels, factor_mean_ret * 100, color=colors_fa, alpha=0.85, edgecolor="none")
    ax_fa.axhline(0, color="white", lw=0.8); ax_fa.set_ylabel("Ann. Contribution (%)", color="white")
    ax_fa.set_title("Factor Attribution — Annualised Return Contribution", color="white")
    for i, v in enumerate(factor_mean_ret):
        ax_fa.text(i, v * 100 + (0.1 if v >= 0 else -0.3), f"{v:.2%}", ha="center", color="white", fontsize=10)
    fig_fa.tight_layout(); st.pyplot(fig_fa); plt.close(fig_fa)

st.markdown("---")
st.markdown("<p style='color:#555;text-align:center;font-size:11px;'>MicroAlpha v2.0 · All data is synthetic (GBM) · No external data feeds · For research purposes only</p>", unsafe_allow_html=True)
