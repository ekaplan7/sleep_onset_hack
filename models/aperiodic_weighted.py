"""Aperiodic/spectral features + bin-weighted gradient boosting (numpy fast path).

Identical features and trained weights as the torch version, but feature
extraction is pure numpy with all per-window constants (Hann window, frequency
grid, band masks) precomputed ONCE at import. The streaming evaluator calls
``predict`` one window at a time (B=1); torch's per-call dispatch + per-call
allocation on tiny (C, T) arrays was the entire runtime cost, so this cuts
per-window time ~1-2 orders of magnitude and clears the compute budget.

torch's hann_window is PERIODIC (0.5 - 0.5*cos(2*pi*n/T)); replicated exactly
so features match the trained model to float precision.

Self-contained: numpy + sklearn + joblib + benchmark_utils only. No torch.
"""

import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {
    "delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0), "beta": (16.0, 30.0),
}
FIT_LO, FIT_HI = 2.0, 40.0
EPS = 1e-8
CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])

# ---- per-(T, sfreq) constants, cached so streaming eval allocates nothing ----
_CACHE = {}


def _const(T, sf):
    key = (int(T), float(sf))
    c = _CACHE.get(key)
    if c is not None:
        return c
    n = np.arange(T)
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / T)).astype(np.float32)  # periodic
    freqs = np.fft.rfftfreq(T, d=1.0 / float(sf))
    tot = (freqs >= 1.0) & (freqs <= 40.0)
    bm = {name: ((freqs >= lo) & (freqs < hi)) for name, (lo, hi) in BANDS.items()}
    fm = (freqs >= FIT_LO) & (freqs <= FIT_HI)
    lf = np.log10(np.clip(freqs[fm], EPS, None))
    lfc = lf - lf.mean()
    lfcss = float((lfc ** 2).sum())
    fb = freqs[tot]
    c = dict(hann=hann, tot=tot, bm=bm, fm=fm, lf=lf, lfc=lfc,
             lfcss=max(lfcss, EPS), lfmean=float(lf.mean()), fb=fb)
    _CACHE[key] = c
    return c


def spectral_features(X, sfreq, device=None):
    """(B, C, T) or (C, T) -> numpy (B, C*k) of per-channel spectral features."""
    if hasattr(X, "detach"):          # torch tensor -> numpy (duck-typed, no import)
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    B, C, T = X.shape
    c = _const(T, sfreq)
    xw = (X - X.mean(axis=-1, keepdims=True)) * c["hann"]
    spec = np.fft.rfft(xw, axis=-1)
    psd = spec.real ** 2 + spec.imag ** 2

    feats = []
    total = np.clip(psd[..., c["tot"]].sum(-1), EPS, None)
    bp = {}
    for name in BANDS:
        p = np.clip(psd[..., c["bm"][name]].sum(-1), EPS, None)
        bp[name] = p
        feats.append(np.log(p))
    for name in BANDS:
        feats.append(bp[name] / total)
    feats.append(bp["delta"] / bp["beta"])
    feats.append((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))

    ly = np.log10(np.clip(psd[..., c["fm"]], EPS, None))
    lyc = ly - ly.mean(axis=-1, keepdims=True)
    slope = (lyc * c["lfc"]).sum(-1) / c["lfcss"]
    feats.append(-slope)
    feats.append(ly.mean(-1) - slope * c["lfmean"])

    p_b = psd[..., c["tot"]]
    t2 = np.clip(p_b.sum(-1, keepdims=True), EPS, None)
    pn = p_b / t2
    feats.append(-(pn * np.log(pn + EPS)).sum(-1))
    cum = np.cumsum(p_b, axis=-1) / t2
    idx = np.clip((cum < 0.95).sum(-1), 0, c["fb"].size - 1)
    feats.append(c["fb"][idx])

    out = np.stack(feats, axis=-1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1)


class FeatureGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")
        self.pipe = pipe

    def fit(self, train_loader):
        feats, tgts = [], []
        for X, y, _info in train_loader:
            feats.append(spectral_features(X, self.sfreq))
            tgts.append(to_numpy(y).ravel())
        feats = np.concatenate(feats)
        y = np.concatenate(tgts)
        idx = np.clip(np.digitize(y, BIN_EDGES[1:-1], right=False), 0, 3)
        sw = BIN_WEIGHTS[idx]
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HistGradientBoostingRegressor(loss="absolute_error")),
        ])
        self.pipe.fit(feats, y, reg__sample_weight=sw)
        return self

    def predict(self, X):
        f = spectral_features(X, self.sfreq)
        return np.clip(self.pipe.predict(f), 0.0, CAP)


class Solver(CompetSolver):

    name = "Aperiodic-Weighted"

    requirements = []

    def load_model(self, meta):
        model = FeatureGBM(meta)
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            model.pipe = joblib.load(weights)
        elif self.train_loader is None:
            raise FileNotFoundError(
                "Aperiodic-Weighted needs weights.joblib; train with "
                "training=True and upload the exported folder."
            )
        return model

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
