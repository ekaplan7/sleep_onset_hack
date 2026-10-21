"""Context ridge, time-based: deduplicated training windows, 60 s context by time, metric weights."""

import joblib
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

CAP_S = 600.0
SFREQ = 128
WIN_S = 5.0
BANDS = [(1, 4), (4, 8), (8, 12), (12, 16), (16, 30)]  # delta, theta, alpha, sigma, beta
TREND_S = 60.0      # trend = windows that started in the last 60 s
BASELINE_S = 60.0   # baseline = windows in the recording's first 60 s
N_PASSES = 5        # passes over the (randomly sampled) training loader
BIN_EDGES = [40, 90, 300]
BIN_WEIGHTS = [10, 5, 3, 1]


def band_powers(X):
    """Log power in each band, per channel: (B, 4, 640) -> (B, 20)."""
    x = to_numpy(X)
    freqs, psd = welch(x, fs=SFREQ, nperseg=256, axis=-1)
    out = []
    for lo, hi in BANDS:
        in_band = (freqs >= lo) & (freqs < hi)
        out.append(psd[..., in_band].mean(axis=-1))
    bp = np.stack(out, axis=-1)
    return np.log(bp + 1e-12).reshape(len(x), -1)


def context_features(bp, times):
    """One recording's windows, sorted by time, no duplicates.

    Context is defined in seconds, not window counts, so it means the same
    thing whether windows are back to back (test) or have gaps (training).
    Row i only uses windows that started at or before times[i].
    """
    n = len(bp)
    csum = np.vstack([np.zeros((1, bp.shape[1])), np.cumsum(bp, axis=0)])
    idx = np.arange(n)
    start = np.searchsorted(times, times - TREND_S, side="right")
    trend = (csum[idx + 1] - csum[start]) / (idx + 1 - start)[:, None]
    base_count = max(np.searchsorted(times, times[0] + BASELINE_S, side="left"), 1)
    base_end = np.minimum(idx + 1, base_count)
    baseline = csum[base_end] / base_end[:, None]
    return np.hstack([bp, trend, trend - baseline, times[:, None] / 60.0])


def metric_weights(y_r):
    """Per-window training weights that mimic W-bMAE for one recording."""
    bins = np.digitize(y_r, BIN_EDGES)
    w = np.zeros(len(y_r))
    present = np.unique(bins)
    total = sum(BIN_WEIGHTS[k] for k in present)
    for k in present:
        in_bin = bins == k
        w[in_bin] = BIN_WEIGHTS[k] / (in_bin.sum() * total)
    return w


class ContextRidgeT:

    def __init__(self, estimator=None):
        self.fitted = estimator is not None
        self.clf = estimator or make_pipeline(StandardScaler(), Ridge())
        self.reset_state()

    def reset_state(self):
        self._bp = []

    def fit(self, train_loader):
        bps, ys, recs, onsets = [], [], [], []
        seen = set()
        for p in range(N_PASSES):
            for X, y, info in train_loader:
                r = np.asarray(to_numpy(info["record_id"])).reshape(-1)
                o = np.asarray(to_numpy(info["onset"])).reshape(-1).astype(float)
                keep = []
                for i, key in enumerate(zip(r.tolist(), o.tolist())):
                    if key not in seen:          # skip windows already collected
                        seen.add(key)
                        keep.append(i)
                if not keep:
                    continue
                keep = np.array(keep)
                bps.append(band_powers(to_numpy(X)[keep]))
                ys.append(to_numpy(y).ravel()[keep])
                recs.append(r[keep])
                onsets.append(o[keep])
            print(f"pass {p + 1}: unique training windows so far: {len(seen)}")
        bp, y, rec, onset = map(np.concatenate, (bps, ys, recs, onsets))

        feats, targets, weights, gaps = [], [], [], []
        for r in np.unique(rec):
            sel = np.where(rec == r)[0]
            sel = sel[np.argsort(onset[sel])]
            y_r = y[sel]
            hits = np.where(y_r <= 0)[0]
            if len(hits):
                sel, y_r = sel[:hits[0] + 1], y_r[:hits[0] + 1]
            feats.append(context_features(bp[sel], onset[sel]))
            targets.append(y_r)
            weights.append(metric_weights(y_r))
            gaps.append(np.diff(onset[sel]))
        gaps = np.concatenate(gaps)
        print(f"after dedup: median gap {np.median(gaps):.1f}s  share exactly 5s: {np.mean(gaps == 5):.2f}")

        w = np.concatenate(weights)
        w *= len(w) / w.sum()
        self.clf.fit(np.vstack(feats), np.concatenate(targets), ridge__sample_weight=w)
        self.fitted = True
        return self

    def predict(self, X):
        preds = []
        for row in band_powers(X):
            self._bp.append(row)
            if not self.fitted:
                preds.append(CAP_S / 2)
                continue
            seq = np.array(self._bp)
            times = WIN_S * np.arange(1, len(seq) + 1)
            feats = context_features(seq, times)[-1:]
            preds.append(self.clf.predict(feats)[0])
        return np.clip(np.array(preds, dtype=np.float64), 0.0, CAP_S)


class Solver(CompetSolver):

    name = "sanaa-context-ridge-t"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            return ContextRidgeT(estimator=joblib.load(weights))
        return ContextRidgeT()

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")