"""Full-vector causal smoothing of the multiband feature set.

Complexity-Smooth (48.90, best) showed that past-only EMA denoising of the
key discriminators is the live lever -- the gains landed in the heavily
weighted near bins ([0,40) +7.6, [40,90) woke up +4.2). This pushes it:
smooth the ENTIRE multiband feature vector (every band power / shape feature),
with a proven 120s EMA AND a longer 240s EMA (does more memory help the middle
bin?), plus deviation features (base - smoothed) -- a denoised version of the
trend idea, built on clean levels. Raw complexity features dropped (hurt in
isolation). Per-recording + gap reset so train matches streaming. Self-
contained, submittable. fit() prints a per-bin GBM-vs-constant check.
"""
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
SHAPE_BANDS = [(1.0, 8.0), (12.0, 16.0), (16.0, 30.0), (1.0, 40.0)]
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
STEP = 5.0
A1 = 1.0 - 0.5 ** (1.0 / 24.0)   # ~120 s half-life (proven)
A2 = 1.0 - 0.5 ** (1.0 / 48.0)   # ~240 s half-life (longer memory)


def _skew(x):
    mu = x.mean(-1, keepdim=True); d = x - mu
    return (d ** 3).mean(-1) / ((d ** 2).mean(-1) ** 1.5 + EPS)
def _shape(Sp, freqs, lo, hi, T):
    s = torch.fft.irfft(Sp * ((freqs >= lo) & (freqs <= hi)).to(Sp.dtype), n=T, dim=-1)
    dd = torch.diff(s, dim=-1); ll = dd.abs().sum(-1) / (s.std(-1).clamp_min(EPS) * T)
    return torch.stack([_skew(s), ll], dim=-1)


def _feat(X, sfreq, device):
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
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)]); base = torch.stack(feats, -1)
    Xd = X - X.mean(-1, keepdim=True); Sp = torch.fft.rfft(Xd, dim=-1)
    sig = torch.fft.irfft(Sp * ((freqs >= 11.0) & (freqs <= 16.0)).to(Sp.dtype), n=T, dim=-1)
    w = max(int(0.25 * sf), 1); ker = torch.ones(1, 1, w, device=X.device, dtype=X.dtype) / w
    env = torch.sqrt(F.conv1d((sig ** 2).reshape(B * C, 1, T), ker, padding=w // 2).clamp_min(0) + EPS).reshape(B, C, -1)
    sp_peak = env.amax(-1)
    log_f = torch.log10(freqs.clamp_min(EPS)); ap = 10.0 ** (offset.unsqueeze(-1) - exponent.unsqueeze(-1) * log_f.view(1, 1, -1))
    osc = torch.log((psd - ap).clamp_min(0)[..., (freqs >= 12.0) & (freqs < 16.0)].sum(-1) + EPS)
    kc = torch.fft.irfft(Sp * ((freqs >= 0.5) & (freqs <= 4.0)).to(Sp.dtype), n=T, dim=-1)
    kc = (kc - kc.mean(-1, keepdim=True)) / kc.std(-1, keepdim=True).clamp_min(EPS); L = max(int(1.2 * sf), 1); kcf = kc.reshape(B * C, 1, T)
    p2p = (F.max_pool1d(kcf, L, stride=1) + F.max_pool1d(-kcf, L, stride=1)).reshape(B, C, -1)
    kc_max = p2p.amax(-1); kc_mean = p2p.mean(-1)
    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mob = torch.sqrt(v1 / v0); cplx = torch.sqrt(v2 / v1) / mob.clamp_min(EPS)
    kept = torch.stack([sp_peak, osc, kc_max, kc_mean, mob, cplx], -1)
    shape = torch.cat([_shape(Sp, freqs, lo, hi, T) for (lo, hi) in SHAPE_BANDS], dim=-1)
    out = torch.cat([base, kept, shape], -1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(B, -1).detach().cpu().numpy()


def _rows(base, s1, s2):
    return np.concatenate([base, s1, s2, base - s1])


def _best_const(y):
    gr = np.arange(0.0, 600.01, 2.0); idx = np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)
    ne = [b for b in range(4) if np.any(idx == b)]
    s = [(BIN_WEIGHTS[ne] * np.array([np.abs(c - y)[idx == b].mean() for b in ne])).sum() / BIN_WEIGHTS[ne].sum() for c in gr]
    return float(gr[int(np.argmin(s))])


class SmoothFull:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.pipe = pipe
        self._s1 = None; self._s2 = None
    def reset_state(self): self._s1 = None; self._s2 = None
    def _assemble(self, loader):
        f, rid, ons, Y = [], [], [], []
        for X, y, info in loader:
            f.append(_feat(X, self.sfreq, self.device))
            rid.append(to_numpy(info["record_id"]).reshape(-1)); ons.append(to_numpy(info["onset"]).reshape(-1)); Y.append(to_numpy(y).ravel())
        f = np.concatenate(f); rid = np.concatenate(rid).astype(np.int64); ons = np.concatenate(ons).astype(np.float64); y = np.concatenate(Y)
        first = {}
        for i in range(len(y)):
            k = (int(rid[i]), round(float(ons[i]), 3))
            if k not in first: first[k] = i
        smap = {}
        for r in np.unique(rid):
            os_r = sorted({round(float(o), 3) for o in ons[rid == r]}); s1 = s2 = None; prev = None
            for o in os_r:
                v = f[first[(int(r), o)]]
                if prev is None or abs(o - (prev + STEP)) > 1e-6:
                    s1 = v.copy(); s2 = v.copy()
                else:
                    s1 = A1 * v + (1 - A1) * s1; s2 = A2 * v + (1 - A2) * s2
                smap[(int(r), o)] = _rows(v, s1, s2); prev = o
        Xf = np.stack([smap[(int(rid[i]), round(float(ons[i]), 3))] for i in range(len(y))])
        return Xf, y
    def fit(self, loader):
        Xf, y = self._assemble(loader)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        rng = np.random.default_rng(33); idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, ho = idx[:cut], idx[cut:]
        dg = Pipeline([("s", StandardScaler()), ("r", HistGradientBoostingRegressor(loss="absolute_error"))])
        dg.fit(Xf[tr], y[tr], r__sample_weight=sw[tr])
        c = _best_const(y[tr]); yh = y[ho]; pg = np.clip(dg.predict(Xf[ho]), 0, 600); bi = np.clip(np.digitize(yh, BIN_EDGES[1:-1]), 0, 3)
        print(f"[smooth-full per-bin] optimal constant c={c:.1f}s  n_features={Xf.shape[1]}")
        print(f"{'bin':12s}{'GBM_MAE':>10s}{'const_MAE':>11s}{'gain':>8s}")
        for b, lab in enumerate(["[0,40)", "[40,90)", "[90,300)", "[300,600]"]):
            mk = bi == b
            if not mk.any(): continue
            g = np.abs(pg[mk] - yh[mk]).mean(); cc = np.abs(c - yh[mk]).mean()
            print(f"{lab:12s}{g:>10.1f}{cc:>11.1f}{cc - g:>8.1f}")
        self.pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Xf, y, reg__sample_weight=sw); return self
    def predict(self, X):
        f = _feat(X, self.sfreq, self.device); rows = []
        for i in range(len(f)):
            v = f[i]
            if self._s1 is None:
                self._s1 = v.copy(); self._s2 = v.copy()
            else:
                self._s1 = A1 * v + (1 - A1) * self._s1; self._s2 = A2 * v + (1 - A2) * self._s2
            rows.append(_rows(v, self._s1, self._s2))
        return np.clip(self.pipe.predict(np.array(rows)), 0.0, CAP)


class Solver(CompetSolver):
    name = "Aperiodic-Smooth-Full"; requirements = []
    def load_model(self, meta):
        mdl = SmoothFull(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists(): mdl.pipe = joblib.load(w)
        elif self.train_loader is None: raise FileNotFoundError("Aperiodic-Smooth-Full needs weights.joblib; train with training=True.")
        return mdl
    def fit(self, model, loader): model.fit(loader)
    def save_model(self, model, path): joblib.dump(model.pipe, path / "weights.joblib")
