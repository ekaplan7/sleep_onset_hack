"""REVE encoder + aperiodic/spectral features -> ridge head.

Extends the REVE probe: the frozen pretrained REVE embedding is concatenated
with interpretable, domain-motivated features from the raw window, and the
ridge head regresses onset latency on the combination.

Fusion point: ``encode(X)`` returns ``(B, T', D + k)`` -- REVE's ``D`` dims
plus ``k`` spectral features per channel, tiled across time so the probe's
mean-pool leaves them unchanged. Only the ridge head is trained.

Self-contained: only imports ``benchmark_utils`` (in the worker image),
braindecode, torch and sklearn. No specparam/FOOOF -- the aperiodic
exponent/offset come from a log-log least-squares fit of the power spectrum,
safe inside the scoring container. (Validate on lab data that this cheap slope
tracks the specparam exponent.)
"""

import joblib
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.linear_probe import Encoder, LinearProbe


# ---- feature config -------------------------------------------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),   # spindle band
    "beta":  (16.0, 30.0),
}
FIT_LO, FIT_HI = 2.0, 40.0   # aperiodic (1/f) fit range
EPS = 1e-8


# ---- frozen REVE encoder (inlined so the submission is self-contained) ----
class REVEEncoder(Encoder):
    """Frozen pretrained braindecode REVE (200 Hz) -> (B, T', D)."""

    REVE_SFREQ = 200
    PATCH_SIZE = 200

    def __init__(self, meta):
        from braindecode.models import REVE
        self.sfreq = float(meta["sfreq"])
        self.device = meta.get("device", "cpu")
        self._n_times = max(
            round(meta["n_times"] * self.REVE_SFREQ / self.sfreq), 1
        )
        if self._n_times < self.PATCH_SIZE:
            raise ValueError(
                "REVE needs at least one second of EEG after resampling to "
                f"200 Hz; this window has {self._n_times} samples"
            )
        self.model = REVE.from_pretrained(
            "brain-bzh/reve-base",
            n_outputs=1,
            n_chans=meta["n_chans"],
            n_times=self._n_times,
            sfreq=self.REVE_SFREQ,
            chs_info=None,
        ).to(self.device)
        self._set_channel_positions(meta.get("ch_names"))
        self.model.eval()
        self.model.requires_grad_(False)

    def _set_channel_positions(self, ch_names):
        if not ch_names or len(ch_names) != self.model.n_chans:
            raise ValueError(
                "REVE needs one standard EEG channel name per input channel"
            )
        names = [str(name).removeprefix("EEG ").split("-", 1)[0]
                 for name in ch_names]
        known = self.model._position_bank.mapping
        unknown = [name for name in names if name not in known]
        if unknown:
            raise ValueError(
                "REVE cannot locate these EEG channels: " + ", ".join(unknown)
            )
        self.model.default_pos = self.model.get_positions(names)

    def encode(self, X):
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        if X.shape[-1] != self._n_times:
            X = F.interpolate(
                X, size=self._n_times, mode="linear", align_corners=False
            )
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = ((X - mean) / std).clamp(-15.0, 15.0)
        with torch.inference_mode():
            out = self.model(X, return_features=True)
        feats = out["features"] if isinstance(out, dict) else out
        if feats.ndim == 4:
            feats = feats.mean(dim=1)
        elif feats.ndim == 2:
            feats = feats[:, None, :]
        return feats


# ---- REVE + aperiodic/spectral features -----------------------------------
class REVEAperiodicEncoder(REVEEncoder):
    """REVE embedding augmented with per-channel spectral features."""

    def _spectral_features(self, X):
        # Features computed at the ORIGINAL sampling rate (self.sfreq),
        # before REVE's internal resample to 200 Hz.
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
        freqs = torch.fft.rfftfreq(T, d=1.0 / sf).to(X.device)  # (F,)

        feats = []

        total_mask = (freqs >= 1.0) & (freqs <= 40.0)
        total = psd[..., total_mask].sum(-1).clamp_min(EPS)   # (B, C)
        bp = {}
        for name, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs < hi)
            p = psd[..., m].sum(-1).clamp_min(EPS)
            bp[name] = p
            feats.append(torch.log(p))                        # log band power
        for name in BANDS:
            feats.append(bp[name] / total)                    # relative power
        feats.append(bp["delta"] / bp["beta"])                # slow/fast
        feats.append((bp["delta"] + bp["theta"])
                     / (bp["alpha"] + bp["beta"]))            # slow/fast 2

        # aperiodic exponent + offset via log-log least squares
        m = (freqs >= FIT_LO) & (freqs <= FIT_HI)
        lf = torch.log10(freqs[m].clamp_min(EPS))             # (F',)
        ly = torch.log10(psd[..., m].clamp_min(EPS))          # (B, C, F')
        lf_c = lf - lf.mean()
        ly_c = ly - ly.mean(dim=-1, keepdim=True)
        slope = (ly_c * lf_c).sum(-1) / (lf_c ** 2).sum().clamp_min(EPS)
        feats.append(-slope)                                  # exponent
        feats.append(ly.mean(dim=-1) - slope * lf.mean())     # offset

        # spectral entropy + 95% spectral edge
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
        emb = super().encode(X)                               # (B, T', D)
        feats = self._spectral_features(X)                    # (B, k)
        feats = feats[:, None, :].expand(-1, emb.shape[1], -1)
        return torch.cat([emb, feats.to(emb.dtype)], dim=-1)  # (B, T', D+k)


class Solver(CompetSolver):

    name = "REVE-Aperiodic"

    requirements = ["pip::braindecode", "pip::safetensors"]

    def load_model(self, meta):
        probe = LinearProbe(REVEAperiodicEncoder(meta), Ridge())
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            print(f"[loading] {weights} into {self.name}")
            probe.head = joblib.load(weights)
        elif self.train_loader is None:
            raise FileNotFoundError(
                "REVE-Aperiodic inference needs weights.joblib. Train this "
                "solver with training=True and upload the exported folder."
            )
        return probe

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.head, path / "weights.joblib")
