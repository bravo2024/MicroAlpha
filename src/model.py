
"""model.py - mean-variance optimizer + VaR / Expected-Shortfall risk engine."""
import numpy as np
PREDICT_KIND="quant"
def mean_variance_weights(mu,cov,risk_aversion=8.0,long_only=True):
    w=np.linalg.pinv(risk_aversion*cov)@mu
    if long_only: w=np.clip(w,0,None)
    s=w.sum(); return w/s if s else np.ones_like(w)/len(w)
def value_at_risk(p,a=0.05): return float(-np.quantile(p,a))
def expected_shortfall(p,a=0.05):
    q=np.quantile(p,a); tail=p[p<=q]; return float(-tail.mean()) if len(tail) else float(-q)
def fit_and_evaluate(data):
    R=np.asarray(data["returns"],float); mu,cov=R.mean(0),np.cov(R.T); w=mean_variance_weights(mu,cov); port=R@w
    ar=float(port.mean()*252); av=float(port.std()*np.sqrt(252))
    metrics={"assets":len(data["assets"]),"annual_return":ar,"annual_vol":av,"sharpe":float(ar/av) if av else 0.0,
             "VaR_95":value_at_risk(port),"ES_95":expected_shortfall(port)}
    return {"weights":w.tolist(),"assets":data["assets"]},metrics
