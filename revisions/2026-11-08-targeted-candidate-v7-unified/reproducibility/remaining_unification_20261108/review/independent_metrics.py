"""Independent metric verification (no sklearn); validation of supplied predictions only."""
import numpy as np

def metrics(y,p,weights=None):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(p,dtype=np.float64)
    assert y.ndim==p.ndim==1 and len(y)==len(p)
    assert set(np.unique(y))<={0,1} and np.isfinite(p).all()
    w=np.ones(len(y),dtype=np.float64) if weights is None else np.asarray(weights,dtype=np.float64)
    assert len(w)==len(y) and np.isfinite(w).all() and np.all(w>=0)
    positive=w*y;negative=w*(1-y);np_=positive.sum();nn=negative.sum()
    if not np_ or not nn:return np.array([np.nan,np.nan])
    # AUC via weighted concordant pairs, ties given half credit.
    order=np.argsort(p,kind='stable');v=p[order]
    start=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1]
    gp=np.add.reduceat(positive[order],start);gn=np.add.reduceat(negative[order],start)
    auc=float(np.dot(gp,np.cumsum(gn)-gn/2)/(np_*nn))
    # AP via threshold-group precision and incremental positive mass.
    gp=gp[::-1];gn=gn[::-1];cum_p=np.cumsum(gp);cum_all=cum_p+np.cumsum(gn)
    precision=np.divide(cum_p,cum_all,out=np.zeros_like(cum_p),where=cum_all>0)
    ap=float(np.dot(gp,precision)/np_)
    return np.array([auc,ap])

if __name__=='__main__':
    # Analytic tie/no-tie/nonuniform multiplicity cases; only self-checks.
    assert np.allclose(metrics([0,1],[.1,.9]),[1,1])
    assert np.allclose(metrics([0,1],[.9,.1]),[0,.5])
    assert np.allclose(metrics([0,1],[.5,.5]),[.5,.5])
    assert np.allclose(metrics([0,1],[.5,.5],[3,1]),[.5,.25])
    print('independent metric analytic checks passed')
