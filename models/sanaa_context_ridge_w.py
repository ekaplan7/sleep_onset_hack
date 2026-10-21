"""Context ridge: band powers + last-minute trend + change from recording start + elapsed time."""

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
TREND_WINDOWS = 12     # 12 x 5 s = the last minute
BASELINE_WINDOWS = 12  # the recording's first minute


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
    """Features for one recording's windows, in time order.

    bp: (n, 20) band powers; times: (n,) seconds since recording start.
    Row i uses only windows 0..i, so it never looks into the future.
    """
    n = len(bp)
    csum = np.vstack([np.zeros((1, bp.shape[1])), np.cumsum(bp, axis=0)])
    idx = np.arange(n)
    start = np.maximum(0, idx - TREND_WINDOWS + 1)
    trend = (csum[idx + 1] - csum[start]) / (idx + 1 - start)[:, None]
    base_end = np.minimum(idx + 1, BASELINE_WINDOWS)
    baseline = csum[base_end] / base_end[:, None]
    return np.hstack([bp, trend, trend - baseline, times[:, None] / 60.0])

BIN_EDGES = [40, 90, 300]        # boundaries of the metric's ranges (seconds)
BIN_WEIGHTS = [10, 5, 3, 1]      # weights for [0,40), [40,90), [90,300), [300,600]


def metric_weights(y_r):
    """Per-window training weights that mimic W-bMAE for one recording."""
    bins = np.digitize(y_r, BIN_EDGES)  # 0..3: which range each window is in
    w = np.zeros(len(y_r))
    present = np.unique(bins)
    total = sum(BIN_WEIGHTS[k] for k in present)
    for k in present:
        in_bin = bins == k
        w[in_bin] = BIN_WEIGHTS[k] / (in_bin.sum() * total)
    return w 

class ContextRidge:

    def __init__(self, estimator=None):
        self.fitted = estimator is not None
        self.clf = estimator or make_pipeline(StandardScaler(), Ridge())
        self.reset_state()

    def reset_state(self):
        """Called by the evaluator at the start of every recording."""
        self._bp = []  # band powers of the windows seen so far in this recording

    def fit(self, train_loader):
        bps, ys, recs, onsets = [], [], [], []
        for X, y, info in train_loader:
            bps.append(band_powers(X))
            ys.append(to_numpy(y).ravel())
            recs.append(np.asarray(to_numpy(info["record_id"])).reshape(-1))
            onsets.append(np.asarray(to_numpy(info["onset"])).reshape(-1).astype(float))
        bp, y, rec, onset = map(np.concatenate, (bps, ys, recs, onsets))

        feats, targets, weights = [], [], [] 
        for r in np.unique(rec):
            sel = np.where(rec == r)[0]
            sel = sel[np.argsort(onset[sel])]   # put this recording back in time order
            y_r = y[sel]
            hits = np.where(y_r <= 0)[0]
            if len(hits):                        # drop windows after first N2
                sel, y_r = sel[:hits[0] + 1], y_r[:hits[0] + 1]
            feats.append(context_features(bp[sel], onset[sel]))
            targets.append(y_r)
            weights.append(metric_weights(y_r))

        F, Y = np.vstack(feats), np.concatenate(targets)
        w = np.concatenate(weights)
        w *= len(w) / w.sum()  # rescale so the average weight is 1
        self.clf.fit(F, Y, ridge__sample_weight=w)

        # --- diagnostic: compare weighted vs unweighted fits on the training data ---
        from sklearn.base import clone
        plain = clone(self.clf).fit(F, Y)
        bins = np.digitize(Y, BIN_EDGES)
        print(f"weights: min {w.min():.3f}  max {w.max():.1f}")
        for label, model in [("weighted", self.clf), ("plain", plain)]:
            P = np.clip(model.predict(F), 0, CAP_S)
            print(label)
            for k, rng in enumerate(["0-40", "40-90", "90-300", "300-600"]):
                m = bins == k
                print(f"  {rng:>7}s  n={m.sum():6d}  mean true {Y[m].mean():6.1f}  "
                      f"mean pred {P[m].mean():6.1f}  MAE {np.abs(P[m] - Y[m]).mean():6.1f}")
                gaps = []
                
        for r in np.unique(rec):
            o = np.sort(onset[rec == r])
            gaps.append(np.diff(o))
        gaps = np.concatenate(gaps)
        print(f"gap between consecutive training windows: median {np.median(gaps):.1f}s  "
              f"share under 5s: {np.mean(gaps < 5):.2f}  share over 5s: {np.mean(gaps > 5):.2f}")
        
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
            feats = context_features(seq, times)[-1:]  # only the newest window
            preds.append(self.clf.predict(feats)[0])
        return np.clip(np.array(preds, dtype=np.float64), 0.0, CAP_S)


class Solver(CompetSolver):

    name = "sanaa-context-ridge-w"

    def load_model(self, meta):
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            return ContextRidge(estimator=joblib.load(weights))
        return ContextRidge()

    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        joblib.dump(model.clf, path / "weights.joblib")