"""Aperiodic-Events + waveform-shape (ByCycle-spirit) features -> weighted GBM.

Adds a genuinely ORTHOGONAL axis to the power-based features: nonsinusoidal
waveform shape (Cole & Voytek, ByCycle), reimplemented package-free in torch
so it runs in the scoring container. Per channel, on the 1-8 Hz band:
  - rise-decay asymmetry  (fraction of signal movement that is rising)
  - derivative skewness   (rise vs decay steepness -- the rdsym idea)
  - value skewness        (peak vs trough amplitude asymmetry)
  - normalized line length (waveform complexity)
On top of the full event-detection feature set. Self-contained, submittable.
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


def _upcross(mask):
    m = mask.float()
    return (m[..., 1:] * (1 - m[..., :-1])).sum(-1) + m[..., 0]


def _skew(x):  # (B,C,T) -> (B,C)
    mu = x.mean(-1, keepdim=True); d = x - mu
    m2 = (d ** 2).mean(-1); m3 = (d ** 3).mean(-1)
    return m3 / (m2 ** 1.5 + EPS)


def features(X, sfreq, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 2:
        X = X[None]
    B, C, T = X.shape
    sf = float(sfreq)
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
    exponent = -slope; offset = ly.mean(dim=-1) - slope * lf.mean()
    feats.append(exponent); feats.append(offset)
    p_b = psd[..., tmask]; f_b = freqs[tmask]
    tot = p_b.sum(-1, keepdim=True).clamp_min(EPS); pn = p_b / tot
    feats.append(-(pn * torch.log(pn + EPS)).sum(-1))
    cum = torch.cumsum(p_b, dim=-1) / tot
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)])
    base = torch.stack(feats, dim=-1)

    Xd = X - X.mean(dim=-1, keepdim=True)
    Sp = torch.fft.rfft(Xd, dim=-1)

    # spindle events
    sig = torch.fft.irfft(Sp * ((freqs >= 11.0) & (freqs <= 16.0)).to(Sp.dtype), n=T, dim=-1)
    w = max(int(0.25 * sf), 1)
    ker = torch.ones(1, 1, w, device=X.device, dtype=X.dtype) / w
    env = torch.sqrt(F.conv1d((sig ** 2).reshape(B * C, 1, T), ker, padding=w // 2).clamp_min(0) + EPS).reshape(B, C, -1)
    thr = (env.mean(-1) + 1.5 * env.std(-1).clamp_min(EPS)).unsqueeze(-1)
    above = (env > thr).float().reshape(B * C, 1, -1)
    wmin = max(int(0.4 * sf), 1)
    core = (-F.max_pool1d(-above, kernel_size=wmin, stride=1)).reshape(B, C, -1) > 0.5
    sp_count = _upcross(core); sp_frac = core.float().mean(-1); sp_peak = env.amax(-1)

    log_f = torch.log10(freqs.clamp_min(EPS))
    ap_psd = 10.0 ** (offset.unsqueeze(-1) - exponent.unsqueeze(-1) * log_f.view(1, 1, -1))
    resid = (psd - ap_psd).clamp_min(0.0)
    osc_sigma = torch.log(resid[..., (freqs >= 12.0) & (freqs < 16.0)].sum(-1) + EPS)

    # K-complex events
    kc = torch.fft.irfft(Sp * ((freqs >= 0.5) & (freqs <= 4.0)).to(Sp.dtype), n=T, dim=-1)
    kc = (kc - kc.mean(-1, keepdim=True)) / kc.std(-1, keepdim=True).clamp_min(EPS)
    L = max(int(1.2 * sf), 1); kcf = kc.reshape(B * C, 1, T)
    p2p = (F.max_pool1d(kcf, L, stride=1) + F.max_pool1d(-kcf, L, stride=1)).reshape(B, C, -1)
    kc_max = p2p.amax(-1); kc_mean = p2p.mean(-1)
    kthr = (p2p.mean(-1) + 2.0 * p2p.std(-1).clamp_min(EPS)).unsqueeze(-1)
    kc_count = _upcross(p2p > kthr)

    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mob = torch.sqrt(v1 / v0); cplx = torch.sqrt(v2 / v1) / mob.clamp_min(EPS)

    extra = torch.stack([sp_count, sp_frac, sp_peak, osc_sigma,
                         kc_count, kc_max, kc_mean, mob, cplx], dim=-1)

    # ---- waveform shape (ByCycle-spirit), 1-8 Hz, package-free ----
    lfsig = torch.fft.irfft(Sp * ((freqs >= 1.0) & (freqs <= 8.0)).to(Sp.dtype), n=T, dim=-1)
    dd = torch.diff(lfsig, dim=-1)
    rise = dd.clamp_min(0).sum(-1); fall = (-dd).clamp_min(0).sum(-1)
    rdsym = rise / (rise + fall + EPS)
    deriv_skew = _skew(dd)
    val_skew = _skew(lfsig)
    line_len = dd.abs().sum(-1) / (lfsig.std(-1).clamp_min(EPS) * T)
    wave = torch.stack([rdsym, deriv_skew, val_skew, line_len], dim=-1)

    out = torch.cat([base, extra, wave], dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1).detach().cpu().numpy()


class WaveGBM:
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
    name = "Aperiodic-Waveform"
    requirements = []

    def load_model(self, meta):
        mdl = WaveGBM(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists():
            mdl.pipe = joblib.load(w)
        elif self.train_loader is None:
            raise FileNotFoundError("Aperiodic-Waveform needs weights.joblib; train with training=True.")
        return mdl

    def fit(self, model, loader):
        model.fit(loader)

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
