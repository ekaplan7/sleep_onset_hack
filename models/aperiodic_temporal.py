"""Spectral features + TEMPORAL-TREND features -> bin-weighted GBM.

Adds, to the per-window spectral features, how a few key quantities are
*changing* over recent windows within a recording -- the trajectory toward
sleep that single-window snapshots miss. Tracked quantities (channel-mean):
aperiodic exponent, spindle (sigma) relative power, delta/beta slowing, alpha
relative power, log total power. Trend features = per-quantity delta from the
previous window + slope over the last ~3 windows. Only CHANGES are used
(no elapsed-time / window-count), so this stays content-driven.

Streaming-aware: keeps a per-recording history in predict, cleared by
reset_state() at each new recording (see benchmark_utils.streaming). At train
time the sequence is reconstructed from info[record_id, onset]. Submittable.
"""
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FIT_LO, FIT_HI = 2.0, 40.0
EPS = 1e-8
CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
M = 5  # tracked quantities


def compute(X, sfreq, device):
    """Return (feat64 (B,64), tracked (B,5)) as numpy."""
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 2:
        X = X[None]
    B, C, T = X.shape
    xw = (X - X.mean(dim=-1, keepdim=True)) * torch.hann_window(T, device=X.device, dtype=X.dtype)
    spec = torch.fft.rfft(xw, dim=-1)
    psd = spec.real ** 2 + spec.imag ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0 / float(sfreq)).to(X.device)
    feats = []
    tmask = (freqs >= 1.0) & (freqs <= 40.0)
    total = psd[..., tmask].sum(-1).clamp_min(EPS)
    bp = {}
    for name, (lo, hi) in BANDS.items():
        p = psd[..., (freqs >= lo) & (freqs < hi)].sum(-1).clamp_min(EPS)
        bp[name] = p; feats.append(torch.log(p))
    for name in BANDS:
        feats.append(bp[name] / total)
    feats.append(bp["delta"] / bp["beta"])
    feats.append((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))
    m = (freqs >= FIT_LO) & (freqs <= FIT_HI)
    lf = torch.log10(freqs[m].clamp_min(EPS)); ly = torch.log10(psd[..., m].clamp_min(EPS))
    lf_c = lf - lf.mean(); ly_c = ly - ly.mean(dim=-1, keepdim=True)
    slope = (ly_c * lf_c).sum(-1) / (lf_c ** 2).sum().clamp_min(EPS)
    exponent = -slope
    feats.append(exponent); feats.append(ly.mean(dim=-1) - slope * lf.mean())
    p_b = psd[..., tmask]; f_b = freqs[tmask]
    tot = p_b.sum(-1, keepdim=True).clamp_min(EPS); pn = p_b / tot
    feats.append(-(pn * torch.log(pn + EPS)).sum(-1))
    cum = torch.cumsum(p_b, dim=-1) / tot
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)])
    out = torch.stack(feats, dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    feat64 = out.reshape(B, -1).detach().cpu().numpy()

    # tracked quantities (channel-mean)
    tracked = torch.stack([
        exponent.mean(1),
        (bp["sigma"] / total).mean(1),
        torch.log(bp["delta"] / bp["beta"]).mean(1),
        (bp["alpha"] / total).mean(1),
        torch.log(total).mean(1),
    ], dim=1)
    tracked = torch.nan_to_num(tracked, nan=0.0, posinf=0.0, neginf=0.0)
    return feat64, tracked.detach().cpu().numpy()


def trend(hist, cur):
    """delta from previous + slope over last <=3 windows. hist: list of (M,)."""
    prev = hist[-1] if hist else cur
    delta = cur - prev
    seq = np.array((hist[-2:] + [cur]) if hist else [cur])  # (p, M)
    if len(seq) >= 2:
        x = np.arange(len(seq), dtype=np.float64); xc = x - x.mean()
        sl = (xc[:, None] * (seq - seq.mean(0))).sum(0) / (xc ** 2).sum()
    else:
        sl = np.zeros(cur.shape[0])
    return np.concatenate([delta, sl])  # (2M,)


class TemporalGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu")
        self.pipe = pipe
        self._hist = []

    def reset_state(self):
        self._hist = []

    def fit(self, loader):
        f64, trk, rid, ons, Y = [], [], [], [], []
        for X, y, info in loader:
            a, b = compute(X, self.sfreq, self.device)
            f64.append(a); trk.append(b)
            try:
                rid.append(to_numpy(info["record_id"]).reshape(-1))
                ons.append(to_numpy(info["onset"]).reshape(-1))
            except Exception:
                n = len(a); rid.append(np.zeros(n)); ons.append(np.arange(n))
            Y.append(to_numpy(y).ravel())
        f64 = np.concatenate(f64); trk = np.concatenate(trk)
        rid = np.concatenate(rid); ons = np.concatenate(ons); y = np.concatenate(Y)

        # reconstruct per-recording sequences, compute trend in onset order
        Tr = np.zeros((len(y), 2 * M))
        for r in np.unique(rid):
            idx = np.where(rid == r)[0]
            idx = idx[np.argsort(ons[idx])]
            hist = []
            for i in idx:
                Tr[i] = trend(hist, trk[i])
                hist.append(trk[i])
        Xfull = np.concatenate([f64, Tr], axis=1)

        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        self.pipe = Pipeline([("scaler", StandardScaler()),
                              ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Xfull, y, reg__sample_weight=sw)
        return self

    def predict(self, X):
        f64, trk = compute(X, self.sfreq, self.device)
        rows = []
        for i in range(len(f64)):
            tr = trend(self._hist, trk[i]); self._hist.append(trk[i])
            rows.append(np.concatenate([f64[i], tr]))
        return np.clip(self.pipe.predict(np.array(rows)), 0.0, CAP)


class Solver(CompetSolver):
    name = "Aperiodic-Temporal"
    requirements = []

    def load_model(self, meta):
        m = TemporalGBM(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists():
            m.pipe = joblib.load(w)
        elif self.train_loader is None:
            raise FileNotFoundError("Aperiodic-Temporal needs weights.joblib; train with training=True.")
        return m

    def fit(self, model, loader):
        model.fit(loader)

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
