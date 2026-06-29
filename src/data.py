
"""data.py - synthetic multi-asset return panel; swap make_synthetic for yfinance in prod."""
import numpy as np
ASSETS=["EQ_US","EQ_EU","EQ_EM","BOND","GOLD","REIT"]
def make_synthetic(n_days=756,seed=42):
    rng=np.random.default_rng(seed); k=len(ASSETS)
    mu=np.array([0.08,0.07,0.10,0.03,0.05,0.06])/252; vol=np.array([0.18,0.20,0.28,0.06,0.16,0.22])/np.sqrt(252)
    L=np.linalg.cholesky(0.3*np.ones((k,k))+0.7*np.eye(k))
    R=mu+(rng.normal(size=(n_days,k))@L.T)*vol
    return {"returns":R,"assets":ASSETS}
def load_real(tickers,period="3y"):
    import yfinance as yf; px=yf.download(tickers,period=period)["Adj Close"].dropna()
    return {"returns":px.pct_change().dropna().to_numpy(),"assets":list(px.columns)}
