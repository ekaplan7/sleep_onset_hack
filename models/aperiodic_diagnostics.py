"""Per-bin diagnostics: where does the weighted feature model beat the constant?
Analysis only."""
import csv
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}
FIT_LO, FIT_HI = 2.0, 40.0
EPS = 1e-8
CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])


def spectral_features(X, sfreq, device):
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
    feats.append(-slope); feats.append(ly.mean(dim=-1) - slope * lf.mean())
    p_b = psd[..., tmask]; f_b = freqs[tmask]
    tot = p_b.sum(-1, keepdim=True).clamp_min(EPS); pn = p_b / tot
    feats.append(-(pn * torch.log(pn + EPS)).sum(-1))
    cum = torch.cumsum(p_b, dim=-1) / tot
    feats.append(f_b[(cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)])
    out = torch.stack(feats, dim=-1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.reshape(B, -1).detach().cpu().numpy()


def _best_constant(y):
    grid = np.arange(0.0, 600.01, 1.0)
    idx = np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)
    ne = [b for b in range(4) if np.any(idx == b)]
    def wb(c):
        err = np.abs(c - y)
        maes = np.array([err[idx == b].mean() for b in ne])
        return (BIN_WEIGHTS[ne] * maes).sum() / BIN_WEIGHTS[ne].sum()
    return float(grid[int(np.argmin([wb(c) for c in grid]))])


class _M:
    def __init__(self, meta):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.value = 300.0
    def predict(self, X): return np.full(len(to_numpy(X)), self.value)


class Solver(CompetSolver):
    name = "Aperiodic-Diagnostics"; requirements = []

    def load_model(self, meta):
        self._out = meta["submission_dir"]; return _M(meta)

    def fit(self, model, loader):
        Fx, Y = [], []
        for X, y, _ in loader:
            Fx.append(spectral_features(X, model.sfreq, model.device)); Y.append(to_numpy(y).ravel())
        Fx = np.concatenate(Fx); y = np.concatenate(Y)
        rng = np.random.default_rng(33); idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, ho = idx[:cut], idx[cut:]
        sw = BIN_WEIGHTS[np.clip(np.digitize(y[tr], BIN_EDGES[1:-1]), 0, 3)]
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        pipe.fit(Fx[tr], y[tr], reg__sample_weight=sw)
        c = _best_constant(y[tr]); model.value = c
        yh = y[ho]; pg = np.clip(pipe.predict(Fx[ho]), 0, 600); pc = np.full(len(yh), c)
        b_idx = np.clip(np.digitize(yh, BIN_EDGES[1:-1]), 0, 3)
        labels = ["[0,40)", "[40,90)", "[90,300)", "[300,600]"]; rows = []
        print(f"[diagnostics] optimal constant c={c:.1f}s  (holdout n={len(yh)})")
        print(f"{'bin':12s}{'n':>7s}{'GBM_MAE':>10s}{'const_MAE':>11s}{'gain':>8s}")
        for b in range(4):
            m = b_idx == b; n = int(m.sum())
            if n == 0: continue
            g = float(np.abs(pg[m] - yh[m]).mean()); cc = float(np.abs(pc[m] - yh[m]).mean())
            print(f"{labels[b]:12s}{n:>7d}{g:>10.1f}{cc:>11.1f}{cc-g:>8.1f}")
            rows.append([labels[b], n, round(g, 2), round(cc, 2), round(cc - g, 2)])
        with open(self._out / "per_bin_diagnostics.csv", "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["bin", "n", "gbm_mae", "const_mae", "gain"]); wr.writerows(rows)

    def save_model(self, model, path): joblib.dump({"value": model.value}, path / "weights.joblib")
