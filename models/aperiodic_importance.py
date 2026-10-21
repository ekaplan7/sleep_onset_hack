"""Aperiodic-Weighted + permutation feature importance (analysis solver).

Trains the exact winning model (features + bin-weighted abs-error GBM), then
on a held-out split computes permutation importance scored with W-bMAE, and
writes a ranked CSV + per-feature-type aggregate to the output folder. Use it
to see which features drive the score; submit `aperiodic_weighted` for the
leaderboard.
"""

import csv
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
BASE = ["logbp_delta", "logbp_theta", "logbp_alpha", "logbp_sigma",
        "logbp_beta", "rel_delta", "rel_theta", "rel_alpha", "rel_sigma",
        "rel_beta", "ratio_delta_beta", "ratio_slowfast", "exponent",
        "offset", "entropy", "edge"]


def _wbmae(y, p):
    y = np.asarray(y, float).ravel(); err = np.abs(np.asarray(p, float).ravel() - y)
    idx = np.digitize(y, BIN_EDGES[1:-1], right=False)
    inr = (y >= BIN_EDGES[0]) & (y <= BIN_EDGES[-1])
    ne = [b for b in range(4) if np.any(inr & (idx == b))]
    if not ne:
        return np.inf
    maes = np.array([err[inr & (idx == b)].mean() for b in ne])
    return float((BIN_WEIGHTS[ne] * maes).sum() / BIN_WEIGHTS[ne].sum())


def spectral_features(X, sfreq, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.ndim == 2:
        X = X[None]
    B, C, T = X.shape
    xw = (X - X.mean(dim=-1, keepdim=True)) * torch.hann_window(
        T, device=X.device, dtype=X.dtype)
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


class FeatureGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu")
        self.pipe = pipe

    def predict(self, X):
        f = spectral_features(X, self.sfreq, self.device)
        return np.clip(self.pipe.predict(f), 0.0, CAP)


class Solver(CompetSolver):

    name = "Aperiodic-Importance"
    requirements = []

    def load_model(self, meta):
        self._out = meta["submission_dir"]
        model = FeatureGBM(meta)
        w = meta["submission_dir"] / "weights.joblib"
        if w.exists():
            model.pipe = joblib.load(w)
        elif self.train_loader is None:
            raise FileNotFoundError("needs weights.joblib; train with training=True.")
        return model

    def fit(self, model, train_loader):
        feats, tgts = [], []
        for X, y, _info in train_loader:
            feats.append(spectral_features(X, model.sfreq, model.device))
            tgts.append(to_numpy(y).ravel())
        F = np.concatenate(feats); y = np.concatenate(tgts)
        rng = np.random.default_rng(33)
        idx = rng.permutation(len(y)); cut = int(0.8 * len(y))
        tr, ho = idx[:cut], idx[cut:]
        sw = BIN_WEIGHTS[np.clip(np.digitize(y[tr], BIN_EDGES[1:-1]), 0, 3)]
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        pipe.fit(F[tr], y[tr], reg__sample_weight=sw)
        model.pipe = pipe

        def scorer(est, Xv, yv):
            return -_wbmae(yv, est.predict(Xv))

        r = permutation_importance(pipe, F[ho], y[ho], scoring=scorer,
                                   n_repeats=10, random_state=33)
        n_ch = F.shape[1] // len(BASE)
        names = [f"ch{c}_{b}" for c in range(n_ch) for b in BASE]

        # ranked per-feature CSV
        order = np.argsort(r.importances_mean)[::-1]
        with open(self._out / "feature_importance.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["feature", "importance_mean", "importance_std"])
            for i in order:
                wr.writerow([names[i], f"{r.importances_mean[i]:.4f}",
                             f"{r.importances_std[i]:.4f}"])

        # aggregate by feature type (summed over channels)
        agg = {}
        for i, nm in enumerate(names):
            base = nm.split("_", 1)[1]
            agg[base] = agg.get(base, 0.0) + r.importances_mean[i]
        agg_sorted = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        with open(self._out / "feature_importance_by_type.csv", "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["feature_type", "importance_sum_over_channels"])
            for b, v in agg_sorted:
                wr.writerow([b, f"{v:.4f}"])

        print("[importance] top feature types (W-bMAE drop when shuffled):")
        for b, v in agg_sorted[:8]:
            print(f"    {b:18s} {v:.3f}")
        print(f"[importance] full tables in {self._out}")

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
