"""Aperiodic/spectral features only -> ridge head (no REVE).

Isolates how far interpretable, domain-motivated features go on their own.
Same features as ``reve_plus_aperiodic`` but fed straight to the ridge head
with no foundation model -- the cleanest test of "features rival deep nets",
and the lightest/fastest model for inference.

Self-contained: torch + sklearn + benchmark_utils only (all in the worker
image). No braindecode, no specparam/FOOOF -- the aperiodic exponent/offset
come from a log-log least-squares fit of the power spectrum.
"""

import joblib
import torch
from sklearn.linear_model import Ridge

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.linear_probe import Encoder, LinearProbe


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),   # spindle band
    "beta":  (16.0, 30.0),
}
FIT_LO, FIT_HI = 2.0, 40.0
EPS = 1e-8


class FeatureEncoder(Encoder):
    """Compute per-channel spectral features; return (B, 1, k)."""

    def __init__(self, meta):
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")

    def _spectral_features(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        if X.ndim == 2:
            X = X[None]
        B, C, T = X.shape
        sf = float(self.sfreq)

        xw = (X - X.mean(dim=-1, keepdim=True)) * torch.hann_window(
            T, device=X.device, dtype=X.dtype
        )
        spec = torch.fft.rfft(xw, dim=-1)
        psd = spec.real ** 2 + spec.imag ** 2            # (B, C, F)
        freqs = torch.fft.rfftfreq(T, d=1.0 / sf).to(X.device)

        feats = []
        total_mask = (freqs >= 1.0) & (freqs <= 40.0)
        total = psd[..., total_mask].sum(-1).clamp_min(EPS)
        bp = {}
        for name, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            p = psd[..., m].sum(-1).clamp_min(EPS)
            bp[name] = p
            feats.append(torch.log(p))
        for name in BANDS:
            feats.append(bp[name] / total)
        feats.append(bp["delta"] / bp["beta"])
        feats.append((bp["delta"] + bp["theta"])
                     / (bp["alpha"] + bp["beta"]))

        m = (freqs >= FIT_LO) & (freqs <= FIT_HI)
        lf = torch.log10(freqs[m].clamp_min(EPS))
        ly = torch.log10(psd[..., m].clamp_min(EPS))
        lf_c = lf - lf.mean()
        ly_c = ly - ly.mean(dim=-1, keepdim=True)
        slope = (ly_c * lf_c).sum(-1) / (lf_c ** 2).sum().clamp_min(EPS)
        feats.append(-slope)                                  # exponent
        feats.append(ly.mean(dim=-1) - slope * lf.mean())     # offset

        p_b = psd[..., total_mask]
        f_b = freqs[total_mask]
        tot = p_b.sum(-1, keepdim=True).clamp_min(EPS)
        pn = p_b / tot
        feats.append(-(pn * torch.log(pn + EPS)).sum(-1))     # entropy
        cum = torch.cumsum(p_b, dim=-1) / tot
        idx = (cum < 0.95).sum(-1).clamp(max=f_b.numel() - 1)
        feats.append(f_b[idx])                                # spectral edge

        out = torch.stack(feats, dim=-1)                      # (B, C, k)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.reshape(B, -1)                             # (B, C*k)

    def encode(self, X):
        feats = self._spectral_features(X)                    # (B, k)
        return feats[:, None, :]                              # (B, 1, k)


class Solver(CompetSolver):

    name = "Aperiodic-Only"

    requirements = []   # torch + sklearn provided by the benchmark env

    def load_model(self, meta):
        probe = LinearProbe(FeatureEncoder(meta), Ridge())
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            probe.head = joblib.load(weights)
        elif self.train_loader is None:
            raise FileNotFoundError(
                "Aperiodic-Only inference needs weights.joblib. Train this "
                "solver with training=True and upload the exported folder."
            )
        return probe

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.head, path / "weights.joblib")
