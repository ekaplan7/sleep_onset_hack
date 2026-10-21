"""Aperiodic-Weighted + spindle-detection & sleep features -> weighted GBM.

Extends the winning model's feature set (per window, torch-only, worker-safe):
original 64 spectral/aperiodic features PLUS, per channel:
  - spindle envelope: max, mean, density (burst count), fraction-in-spindle
  - oscillatory spindle power (sigma power ABOVE the 1/f aperiodic fit)
  - slow (11-13) and fast (13-16) spindle power
  - Hjorth mobility & complexity
Same bin-weighted absolute-error GBM head. Self-contained (submittable).
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
FIT_LO, FIT_HI = 2.0, 40.0
EPS = 1e-8
CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])


def features(X, sfreq, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 2:
        X = X[None]
    B, C, T = X.shape
    sf = float(sfreq)

    # ---------- windowed PSD for spectral features ----------
    xw = (X - X.mean(dim=-1, keepdim=True)) * torch.hann_window(T, device=X.device, dtype=X.dtype)
    spec = torch.fft.rfft(xw, dim=-1)
    psd = spec.real ** 2 + spec.imag ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0 / sf).to(X.device)

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
    offset = ly.mean(dim=-1) - slope * lf.mean()
    feats.append(exponent); feats.append(offset)
    p_b = psd[..., tmask]; f_b = freqs[tmask]
    tot = p_b.sum(-1, keepdim=True).clamp_min(EPS); pn = p_b / tot
    feats.append(-(pn * torch.log(pn + EPS)).sum(-1))
    cum = torch.cumsum(p_b, dim=-1) / tot
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)])
    base = torch.stack(feats, dim=-1)  # (B, C, 16)

    # ---------- spindle detection (sigma-band envelope) ----------
    Xd = X - X.mean(dim=-1, keepdim=True)
    Sp = torch.fft.rfft(Xd, dim=-1)
    smask = ((freqs >= 12.0) & (freqs <= 16.0)).to(Sp.dtype)
    sig = torch.fft.irfft(Sp * smask, n=T, dim=-1)              # sigma-filtered
    w = max(int(0.25 * sf), 1)
    kernel = torch.ones(1, 1, w, device=X.device, dtype=X.dtype) / w
    env = torch.sqrt(F.conv1d((sig ** 2).reshape(B * C, 1, T), kernel,
                              padding=w // 2).clamp_min(0) + EPS).reshape(B, C, -1)
    e_mean = env.mean(-1); e_std = env.std(-1).clamp_min(EPS); e_max = env.amax(-1)
    thr = (e_mean + 2.0 * e_std).unsqueeze(-1)
    above = env > thr
    frac_above = above.float().mean(-1)
    density = (above[..., 1:] & ~above[..., :-1]).float().sum(-1)   # burst count

    # ---------- oscillatory spindle power (above 1/f fit) ----------
    log_f = torch.log10(freqs.clamp_min(EPS))
    ap_psd = 10.0 ** (offset.unsqueeze(-1) - exponent.unsqueeze(-1) * log_f.view(1, 1, -1))
    resid = (psd - ap_psd).clamp_min(0.0)
    sig_mask = (freqs >= 12.0) & (freqs < 16.0)
    osc_sigma = torch.log(resid[..., sig_mask].sum(-1) + EPS)
    slow_sp = torch.log(psd[..., (freqs >= 11.0) & (freqs < 13.0)].sum(-1) + EPS)
    fast_sp = torch.log(psd[..., (freqs >= 13.0) & (freqs < 16.0)].sum(-1) + EPS)

    # ---------- Hjorth ----------
    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mobility = torch.sqrt(v1 / v0)
    complexity = torch.sqrt(v2 / v1) / mobility.clamp_min(EPS)

    extra = torch.stack([e_max, e_mean, density, frac_above, osc_sigma,
                         slow_sp, fast_sp, mobility, complexity], dim=-1)  # (B,C,9)

    out = torch.cat([base, extra], dim=-1)                     # (B, C, 25)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1).detach().cpu().numpy()           # (B, C*25)


class SpindleGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.pipe = pipe

    def fit(self, loader):
        Fx, Y = [], []
        for X, y, _ in loader:
            Fx.append(features(X, self.sfreq, self.device)); Y.append(to_numpy(y).ravel())
        Fx = np.concatenate(Fx); y = np.concatenate(Y)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        self.pipe = Pipeline([("scaler", StandardScaler()),
                              ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Fx, y, reg__sample_weight=sw); return self

    def predict(self, X):
        return np.clip(self.pipe.predict(features(X, self.sfreq, self.device)), 0.0, CAP)


class Solver(CompetSolver):
    name = "Aperiodic-Spindle"
    requirements = []

    def load_model(self, meta):
        m = SpindleGBM(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists():
            m.pipe = joblib.load(w)
        elif self.train_loader is None:
            raise FileNotFoundError("Aperiodic-Spindle needs weights.joblib; train with training=True.")
        return m

    def fit(self, model, loader):
        model.fit(loader)

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
