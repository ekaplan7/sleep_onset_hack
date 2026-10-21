"""Aperiodic-Events + FAITHFUL cycle-by-cycle (ByCycle) features -> weighted GBM.

Real cycle segmentation (Cole & Voytek), reimplemented package-free in numpy:
on the 1-8 Hz band, find peaks/troughs, segment peak-to-peak cycles, and per
cycle compute rise-decay symmetry, amplitude and period; aggregate per window.
Per channel: rdsym mean/std, amplitude mean, period CV, cycle count, amplitude
consistency. On top of the spectral+event features. Self-contained, submittable.
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
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
NCYC = 6


def _up(mask):
    m = mask.float(); return (m[..., 1:] * (1 - m[..., :-1])).sum(-1) + m[..., 0]


def _cycle_feats(band):  # band: (B, C, T) numpy -> (B, C, NCYC)
    B, C, T = band.shape
    out = np.zeros((B, C, NCYC), dtype=np.float64)
    for b in range(B):
        for c in range(C):
            x = band[b, c]
            pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
            tr = np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1
            if len(pk) < 2 or len(tr) == 0:
                continue
            rd, amp, per = [], [], []
            for j in range(len(pk) - 1):
                p1, p2 = pk[j], pk[j + 1]
                tb = tr[(tr > p1) & (tr < p2)]
                if len(tb) == 0:
                    continue
                t0 = tb[np.argmin(x[tb])]
                decay = t0 - p1; rise = p2 - t0
                if rise + decay <= 0:
                    continue
                rd.append(rise / (rise + decay))
                amp.append(((x[p1] - x[t0]) + (x[p2] - x[t0])) / 2.0)
                per.append(p2 - p1)
            if not per:
                continue
            rd = np.array(rd); amp = np.array(amp); per = np.array(per, float)
            out[b, c, 0] = rd.mean(); out[b, c, 1] = rd.std()
            out[b, c, 2] = amp.mean(); out[b, c, 3] = per.std() / (per.mean() + EPS)
            out[b, c, 4] = len(per); out[b, c, 5] = 1.0 - amp.std() / (np.abs(amp.mean()) + EPS)
    return out


def features(X, sfreq, device):
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
    thr = (env.mean(-1) + 1.5 * env.std(-1).clamp_min(EPS)).unsqueeze(-1); above = (env > thr).float().reshape(B * C, 1, -1)
    core = (-F.max_pool1d(-above, kernel_size=max(int(0.4 * sf), 1), stride=1)).reshape(B, C, -1) > 0.5
    sp_count = _up(core); sp_frac = core.float().mean(-1); sp_peak = env.amax(-1)
    log_f = torch.log10(freqs.clamp_min(EPS)); ap = 10.0 ** (offset.unsqueeze(-1) - exponent.unsqueeze(-1) * log_f.view(1, 1, -1))
    osc = torch.log((psd - ap).clamp_min(0)[..., (freqs >= 12.0) & (freqs < 16.0)].sum(-1) + EPS)
    kc = torch.fft.irfft(Sp * ((freqs >= 0.5) & (freqs <= 4.0)).to(Sp.dtype), n=T, dim=-1)
    kc = (kc - kc.mean(-1, keepdim=True)) / kc.std(-1, keepdim=True).clamp_min(EPS); L = max(int(1.2 * sf), 1); kcf = kc.reshape(B * C, 1, T)
    p2p = (F.max_pool1d(kcf, L, stride=1) + F.max_pool1d(-kcf, L, stride=1)).reshape(B, C, -1)
    kc_max = p2p.amax(-1); kc_mean = p2p.mean(-1); kc_count = _up(p2p > (p2p.mean(-1) + 2.0 * p2p.std(-1).clamp_min(EPS)).unsqueeze(-1))
    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mob = torch.sqrt(v1 / v0); cplx = torch.sqrt(v2 / v1) / mob.clamp_min(EPS)
    extra = torch.stack([sp_count, sp_frac, sp_peak, osc, kc_count, kc_max, kc_mean, mob, cplx], -1)
    spec_np = torch.cat([base, extra], -1).detach().cpu().numpy()  # (B,C,25)
    # faithful cycle features on 1-8 Hz
    lfsig = torch.fft.irfft(Sp * ((freqs >= 1.0) & (freqs <= 8.0)).to(Sp.dtype), n=T, dim=-1).detach().cpu().numpy()
    cyc = _cycle_feats(lfsig)  # (B,C,6)
    out = np.concatenate([spec_np, cyc], axis=-1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1)


class ByCycleGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.pipe = pipe
    def fit(self, loader):
        Fx, Y = [], []
        for X, y, _ in loader:
            Fx.append(features(X, self.sfreq, self.device)); Y.append(to_numpy(y).ravel())
        Fx = np.concatenate(Fx); y = np.concatenate(Y)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        self.pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Fx, y, reg__sample_weight=sw); return self
    def predict(self, X):
        return np.clip(self.pipe.predict(features(X, self.sfreq, self.device)), 0.0, CAP)


class Solver(CompetSolver):
    name = "Aperiodic-ByCycle"; requirements = []
    def load_model(self, meta):
        mdl = ByCycleGBM(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists(): mdl.pipe = joblib.load(w)
        elif self.train_loader is None: raise FileNotFoundError("Aperiodic-ByCycle needs weights.joblib; train with training=True.")
        return mdl
    def fit(self, model, loader): model.fit(loader)
    def save_model(self, model, path): joblib.dump(model.pipe, path / "weights.joblib")
