"""Diagnostics for Aperiodic-Waveform: permutation importance + per-bin gains.
Analysis only (writes CSVs + prints)."""
import csv
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
PERCH = ["logbp_delta","logbp_theta","logbp_alpha","logbp_sigma","logbp_beta",
         "rel_delta","rel_theta","rel_alpha","rel_sigma","rel_beta","ratio_db","ratio_slowfast",
         "exponent","offset","entropy","edge","sp_count","sp_frac","sp_peak","osc_sigma",
         "kc_count","kc_max","kc_mean","hjorth_mob","hjorth_cplx","rdsym","deriv_skew","val_skew","line_len"]


def _up(mask):
    m = mask.float(); return (m[..., 1:] * (1 - m[..., :-1])).sum(-1) + m[..., 0]
def _skew(x):
    mu = x.mean(-1, keepdim=True); d = x - mu
    return (d ** 3).mean(-1) / ((d ** 2).mean(-1) ** 1.5 + EPS)
def _wb(y, p):
    y = np.asarray(y, float).ravel(); e = np.abs(np.asarray(p, float).ravel() - y)
    idx = np.digitize(y, BIN_EDGES[1:-1]); ne = [b for b in range(4) if np.any(idx == b)]
    return float((BIN_WEIGHTS[ne] * np.array([e[idx == b].mean() for b in ne])).sum() / BIN_WEIGHTS[ne].sum())


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
    lfsig = torch.fft.irfft(Sp * ((freqs >= 1.0) & (freqs <= 8.0)).to(Sp.dtype), n=T, dim=-1); dd = torch.diff(lfsig, dim=-1)
    rise = dd.clamp_min(0).sum(-1); fall = (-dd).clamp_min(0).sum(-1); rdsym = rise / (rise + fall + EPS)
    wave = torch.stack([rdsym, _skew(dd), _skew(lfsig), dd.abs().sum(-1) / (lfsig.std(-1).clamp_min(EPS) * T)], -1)
    out = torch.cat([base, extra, wave], -1); out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1).detach().cpu().numpy()


class _M:
    def __init__(self, meta): self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.value = 300.0
    def predict(self, X): return np.full(len(to_numpy(X)), self.value)


class Solver(CompetSolver):
    name = "Aperiodic-Waveform-Diag"; requirements = []
    def load_model(self, meta): self._out = meta["submission_dir"]; return _M(meta)
    def fit(self, model, loader):
        Fx, Y = [], []
        for X, y, _ in loader:
            Fx.append(features(X, model.sfreq, model.device)); Y.append(to_numpy(y).ravel())
        Fx = np.concatenate(Fx); y = np.concatenate(Y)
        rng = np.random.default_rng(33); idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, ho = idx[:cut], idx[cut:]
        sw = BIN_WEIGHTS[np.clip(np.digitize(y[tr], BIN_EDGES[1:-1]), 0, 3)]
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        pipe.fit(Fx[tr], y[tr], reg__sample_weight=sw)
        C = Fx.shape[1] // len(PERCH); names = [f"ch{c}_{n}" for c in range(C) for n in PERCH]
        r = permutation_importance(pipe, Fx[ho], y[ho], scoring=lambda e, Xv, yv: -_wb(yv, e.predict(Xv)), n_repeats=8, random_state=33)
        agg = {}
        for i, nm in enumerate(names):
            b = nm.split("_", 1)[1]; agg[b] = agg.get(b, 0.0) + r.importances_mean[i]
        agg = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        with open(self._out / "wave_importance_by_type.csv", "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["feature_type", "importance"]); wr.writerows([[b, round(v, 4)] for b, v in agg])
        print("[importance] top feature types (W-bMAE drop when shuffled):")
        for b, v in agg[:12]: print(f"    {b:16s} {v:.3f}")
        # per-bin gains vs constant
        gr = np.arange(0.0, 600.01, 1.0); ic = np.clip(np.digitize(y[tr], BIN_EDGES[1:-1]), 0, 3); nec = [b for b in range(4) if np.any(ic == b)]
        wbc = lambda c: (BIN_WEIGHTS[nec] * np.array([np.abs(c - y[tr])[ic == b].mean() for b in nec])).sum() / BIN_WEIGHTS[nec].sum()
        cst = float(gr[int(np.argmin([wbc(c) for c in gr]))])
        yh = y[ho]; pg = np.clip(pipe.predict(Fx[ho]), 0, 600); bi = np.clip(np.digitize(yh, BIN_EDGES[1:-1]), 0, 3)
        print(f"[per-bin] optimal constant c={cst:.1f}s")
        print(f"{'bin':12s}{'GBM_MAE':>10s}{'const_MAE':>11s}{'gain':>8s}")
        for b, lab in enumerate(["[0,40)", "[40,90)", "[90,300)", "[300,600]"]):
            mk = bi == b
            if not mk.any(): continue
            g = np.abs(pg[mk] - yh[mk]).mean(); cc = np.abs(cst - yh[mk]).mean()
            print(f"{lab:12s}{g:>10.1f}{cc:>11.1f}{cc-g:>8.1f}")
    def save_model(self, model, path): joblib.dump({"value": model.value}, path / "weights.joblib")
