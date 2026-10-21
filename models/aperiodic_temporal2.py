"""Fixed temporal-trend model: adjacency-enforced trends + weighted GBM.

Fix for the bin-sampler bug (duplicates + gaps in the train loader):
  1. dedup to the unique (recording, onset) windows -> the true sequence
  2. compute trends ONLY within contiguous 5-s runs (reset history at a gap),
     so every trend is a real neighbour change, never a corrupted jump
  3. map each training row (incl. oversampled duplicates) to its correct trend
     -> keeps the bin-balancing AND gives clean trends
At inference (streaming) windows arrive dense + in order, so trends match.
Base = the 64 weighted spectral/aperiodic features. Self-contained.
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
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
M = 5; STEP = 5.0


def compute(X, sfreq, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 2: X = X[None]
    B, C, T = X.shape; sf = float(sfreq)
    xw = (X - X.mean(-1, keepdim=True)) * torch.hann_window(T, device=X.device, dtype=X.dtype)
    spec = torch.fft.rfft(xw, dim=-1); psd = spec.real ** 2 + spec.imag ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0 / sf).to(X.device)
    feats = []; tmask = (freqs >= 1.0) & (freqs <= 40.0); total = psd[..., tmask].sum(-1).clamp_min(EPS); bp = {}
    for name, (lo, hi) in BANDS.items():
        p = psd[..., (freqs >= lo) & (freqs < hi)].sum(-1).clamp_min(EPS); bp[name] = p; feats.append(torch.log(p))
    for name in BANDS: feats.append(bp[name] / total)
    feats.append(bp["delta"] / bp["beta"]); feats.append((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))
    m = (freqs >= FIT_LO) & (freqs <= FIT_HI); lf = torch.log10(freqs[m].clamp_min(EPS)); ly = torch.log10(psd[..., m].clamp_min(EPS))
    lf_c = lf - lf.mean(); ly_c = ly - ly.mean(-1, keepdim=True)
    slope = (ly_c * lf_c).sum(-1) / (lf_c ** 2).sum().clamp_min(EPS); exponent = -slope; offset = ly.mean(-1) - slope * lf.mean()
    feats.append(exponent); feats.append(offset)
    p_b = psd[..., tmask]; f_b = freqs[tmask]; tot = p_b.sum(-1, keepdim=True).clamp_min(EPS); pn = p_b / tot
    feats.append(-(pn * torch.log(pn + EPS)).sum(-1)); cum = torch.cumsum(p_b, -1) / tot
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)])
    feat = torch.nan_to_num(torch.stack(feats, -1), nan=0.0, posinf=0.0, neginf=0.0).reshape(B, -1).detach().cpu().numpy()
    tracked = torch.stack([exponent.mean(1), (bp["sigma"] / total).mean(1),
                           torch.log(bp["delta"] / bp["beta"]).mean(1), (bp["alpha"] / total).mean(1),
                           torch.log(total).mean(1)], dim=1)
    tracked = torch.nan_to_num(tracked, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
    return feat, tracked


def trend(hist, cur):
    prev = hist[-1] if hist else cur
    delta = cur - prev
    seq = np.array((hist[-2:] + [cur]) if hist else [cur])
    if len(seq) >= 2:
        x = np.arange(len(seq), dtype=np.float64); xc = x - x.mean()
        sl = (xc[:, None] * (seq - seq.mean(0))).sum(0) / (xc ** 2).sum()
    else:
        sl = np.zeros(cur.shape[0])
    return np.concatenate([delta, sl])


class Temporal2:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu")
        self.pipe = pipe; self._hist = []

    def reset_state(self):
        self._hist = []

    def fit(self, loader):
        f64, trk, rid, ons, Y = [], [], [], [], []
        for X, y, info in loader:
            a, b = compute(X, self.sfreq, self.device); f64.append(a); trk.append(b)
            rid.append(to_numpy(info["record_id"]).reshape(-1)); ons.append(to_numpy(info["onset"]).reshape(-1))
            Y.append(to_numpy(y).ravel())
        f64 = np.concatenate(f64); trk = np.concatenate(trk)
        rid = np.concatenate(rid).astype(np.int64); ons = np.concatenate(ons).astype(np.float64); y = np.concatenate(Y)
        # first occurrence of each (rid,onset) -> tracked vector
        first = {}
        for i in range(len(y)):
            k = (int(rid[i]), round(float(ons[i]), 3))
            if k not in first: first[k] = i
        # per recording, build trends on the unique ordered sequence (gap-reset)
        tmap = {}
        for r in np.unique(rid):
            os_r = sorted({round(float(o), 3) for o in ons[rid == r]})
            hist = []; prev = None
            for o in os_r:
                t = trk[first[(int(r), o)]]
                if prev is not None and abs(o - (prev + STEP)) > 1e-6:
                    hist = []
                tmap[(int(r), o)] = trend(hist, t); hist.append(t); prev = o
        Tr = np.stack([tmap[(int(rid[i]), round(float(ons[i]), 3))] for i in range(len(y))])
        Xfull = np.concatenate([f64, Tr], axis=1)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        self.pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Xfull, y, reg__sample_weight=sw)
        return self

    def predict(self, X):
        f64, trk = compute(X, self.sfreq, self.device)
        rows = []
        for i in range(len(f64)):
            tr = trend(self._hist, trk[i]); self._hist.append(trk[i]); rows.append(np.concatenate([f64[i], tr]))
        return np.clip(self.pipe.predict(np.array(rows)), 0.0, CAP)


class Solver(CompetSolver):
    name = "Aperiodic-Temporal2"; requirements = []
    def load_model(self, meta):
        m = Temporal2(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists(): m.pipe = joblib.load(w)
        elif self.train_loader is None: raise FileNotFoundError("Aperiodic-Temporal2 needs weights.joblib; train with training=True.")
        return m
    def fit(self, model, loader): model.fit(loader)
    def save_model(self, model, path): joblib.dump(model.pipe, path / "weights.joblib")
