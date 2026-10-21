"""FFT-light submission champion: spectral/aperiodic + Hjorth + 1 broadband
shape feature + smoothing + countdown. ~3 FFTs/window (vs 16 in smooth_fast) so
it clears the B=1 streaming time budget, while keeping the free wins (smoothing,
countdown). Self-contained, submittable.
"""
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
STEP = 5.0
A_FAST = 1.0 - 0.5 ** (1.0 / 6.0)
A_SLOW = 1.0 - 0.5 ** (1.0 / 24.0)
ALPHA = 0.4


def _skew(x):
    mu = x.mean(-1, keepdim=True); d = x - mu
    return (d ** 3).mean(-1) / ((d ** 2).mean(-1) ** 1.5 + EPS)


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
    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mob = torch.sqrt(v1 / v0); feats.append(mob); feats.append(torch.sqrt(v2 / v1) / mob.clamp_min(EPS))
    Xd = X - X.mean(-1, keepdim=True); Sp = torch.fft.rfft(Xd, dim=-1)
    s = torch.fft.irfft(Sp * ((freqs >= 1.0) & (freqs <= 40.0)).to(Sp.dtype), n=T, dim=-1)
    dd = torch.diff(s, dim=-1); feats.append(_skew(s)); feats.append(dd.abs().sum(-1) / (s.std(-1).clamp_min(EPS) * T))
    out = torch.stack(feats, -1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(B, -1).detach().cpu().numpy()
    tracked = torch.stack([exponent.mean(1), (bp["sigma"] / total).mean(1),
                           torch.log(bp["delta"] / bp["beta"]).mean(1), (bp["alpha"] / total).mean(1),
                           torch.log(total).mean(1)], dim=1)
    tracked = torch.nan_to_num(tracked, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
    return out, tracked


def _best_const(y):
    gr = np.arange(0.0, 600.01, 2.0); idx = np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)
    ne = [b for b in range(4) if np.any(idx == b)]
    s = [(BIN_WEIGHTS[ne] * np.array([np.abs(c - y)[idx == b].mean() for b in ne])).sum() / BIN_WEIGHTS[ne].sum() for c in gr]
    return float(gr[int(np.argmin(s))])


class LeanCD:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.pipe = pipe
        self._ef = None; self._es = None; self._cd = None
    def reset_state(self): self._ef = None; self._es = None; self._cd = None
    def _assemble(self, loader):
        f, trk, rid, ons, Y = [], [], [], [], []
        for X, y, info in loader:
            o, t = compute(X, self.sfreq, self.device)
            f.append(o); trk.append(t)
            rid.append(to_numpy(info["record_id"]).reshape(-1)); ons.append(to_numpy(info["onset"]).reshape(-1)); Y.append(to_numpy(y).ravel())
        f = np.concatenate(f); trk = np.concatenate(trk)
        rid = np.concatenate(rid).astype(np.int64); ons = np.concatenate(ons).astype(np.float64); y = np.concatenate(Y)
        first = {}
        for i in range(len(y)):
            k = (int(rid[i]), round(float(ons[i]), 3))
            if k not in first: first[k] = i
        smap = {}
        for r in np.unique(rid):
            os_r = sorted({round(float(o), 3) for o in ons[rid == r]}); ef = es = None; prev = None
            for o in os_r:
                t = trk[first[(int(r), o)]]
                if prev is None or abs(o - (prev + STEP)) > 1e-6:
                    ef = t.copy(); es = t.copy()
                else:
                    ef = A_FAST * t + (1 - A_FAST) * ef; es = A_SLOW * t + (1 - A_SLOW) * es
                smap[(int(r), o)] = np.concatenate([ef, es]); prev = o
        S = np.stack([smap[(int(rid[i]), round(float(ons[i]), 3))] for i in range(len(y))])
        return np.concatenate([f, S], axis=1), y
    def fit(self, loader):
        Xf, y = self._assemble(loader)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        rng = np.random.default_rng(33); idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, ho = idx[:cut], idx[cut:]
        dg = Pipeline([("s", StandardScaler()), ("r", HistGradientBoostingRegressor(loss="absolute_error"))])
        dg.fit(Xf[tr], y[tr], r__sample_weight=sw[tr])
        c = _best_const(y[tr]); yh = y[ho]; pg = np.clip(dg.predict(Xf[ho]), 0, 600); bi = np.clip(np.digitize(yh, BIN_EDGES[1:-1]), 0, 3)
        print(f"[lean per-bin] c={c:.1f}  n_features={Xf.shape[1]}")
        for b, lab in enumerate(["[0,40)", "[40,90)", "[90,300)", "[300,600]"]):
            mk = bi == b
            if mk.any(): print(f"{lab:12s} GBM {np.abs(pg[mk]-yh[mk]).mean():6.1f}  const {np.abs(c-yh[mk]).mean():6.1f}")
        self.pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Xf, y, reg__sample_weight=sw); return self
    def predict(self, X):
        o, trk = compute(X, self.sfreq, self.device); rows = []
        for i in range(len(o)):
            t = trk[i]
            if self._ef is None:
                self._ef = t.copy(); self._es = t.copy()
            else:
                self._ef = A_FAST * t + (1 - A_FAST) * self._ef; self._es = A_SLOW * t + (1 - A_SLOW) * self._es
            rows.append(np.concatenate([o[i], self._ef, self._es]))
        raw = np.clip(self.pipe.predict(np.array(rows)), 0.0, CAP)
        out = np.empty_like(raw)
        for i in range(len(raw)):
            est = raw[i] if self._cd is None else ALPHA * raw[i] + (1 - ALPHA) * max(self._cd - STEP, 0.0)
            est = min(max(est, 0.0), CAP); self._cd = est; out[i] = est
        return out


class Solver(CompetSolver):
    name = "Smooth-Lean"; requirements = []
    def load_model(self, meta):
        mdl = LeanCD(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists(): mdl.pipe = joblib.load(w)
        elif self.train_loader is None: raise FileNotFoundError("Smooth-Lean needs weights.joblib; train with training=True.")
        return mdl
    def fit(self, model, loader): model.fit(loader)
    def save_model(self, model, path): joblib.dump(model.pipe, path / "weights.joblib")
