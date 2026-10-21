"""REVE embedding + aperiodic features -> bin-weighted abs-error GBM.

The "add features to the foundation model, done right" model: mean-pooled
frozen REVE embedding concatenated with the spectral/aperiodic features, fed
to a gradient-boosting head trained with absolute-error loss and W-bMAE bin
sample-weights. Tests whether REVE adds anything on top of features once the
objective is correct. Self-contained (submittable).
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


class _REVE:
    REVE_SFREQ = 200
    PATCH_SIZE = 200

    def __init__(self, meta):
        from braindecode.models import REVE
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu")
        self._n_times = max(round(meta["n_times"] * self.REVE_SFREQ / self.sfreq), 1)
        if self._n_times < self.PATCH_SIZE:
            raise ValueError("REVE needs >= 1 s after resampling to 200 Hz")
        self.model = REVE.from_pretrained(
            "brain-bzh/reve-base", n_outputs=1, n_chans=meta["n_chans"],
            n_times=self._n_times, sfreq=self.REVE_SFREQ, chs_info=None).to(self.device)
        names = [str(n).removeprefix("EEG ").split("-", 1)[0]
                 for n in (meta.get("ch_names") or [])]
        known = self.model._position_bank.mapping
        if len(names) != self.model.n_chans or any(n not in known for n in names):
            raise ValueError("REVE needs one standard EEG channel name per input channel")
        self.model.default_pos = self.model.get_positions(names)
        self.model.eval(); self.model.requires_grad_(False)

    def embed(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.shape[-1] != self._n_times:
            X = F.interpolate(X, size=self._n_times, mode="linear", align_corners=False)
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = ((X - mean) / std).clamp(-15.0, 15.0)
        with torch.inference_mode():
            out = self.model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 4:
            feats = feats.mean(dim=1)
        if feats.ndim == 3:
            feats = feats.mean(dim=1)                 # mean-pool over T'
        return feats.detach().cpu().numpy()           # (B, D)


class REVEFeatGBM:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu")
        self.enc = _REVE(meta); self.pipe = pipe

    def _feat(self, X):
        emb = self.enc.embed(X)
        sp = spectral_features(X, self.sfreq, self.device)
        return np.concatenate([emb, sp], axis=1)

    def fit(self, loader):
        Fx, Y = [], []
        for X, y, _ in loader:
            Fx.append(self._feat(X)); Y.append(to_numpy(y).ravel())
        Fx = np.concatenate(Fx); y = np.concatenate(Y)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        self.pipe = Pipeline([("scaler", StandardScaler()),
                              ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Fx, y, reg__sample_weight=sw)
        return self

    def predict(self, X):
        return np.clip(self.pipe.predict(self._feat(X)), 0.0, CAP)


class Solver(CompetSolver):
    name = "REVE-Weighted"
    requirements = ["pip::braindecode", "pip::safetensors"]

    def load_model(self, meta):
        m = REVEFeatGBM(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists():
            m.pipe = joblib.load(w)
        elif self.train_loader is None:
            raise FileNotFoundError("REVE-Weighted needs weights.joblib; train with training=True.")
        return m

    def fit(self, model, loader):
        model.fit(loader)

    def save_model(self, model, path):
        joblib.dump(model.pipe, path / "weights.joblib")
