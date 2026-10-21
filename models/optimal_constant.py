"""Optimal-constant baseline: the single value that minimizes W-bMAE.

The plain Median minimizes *unweighted* error. This scans for the constant
that minimizes the competition's *weighted* binned MAE on the train targets.
Because near-onset errors count 10x, that constant is typically below the
median -- a trivial model that can beat the Median baseline on W-bMAE, and
proof that matching the metric matters more than the model here.
"""

import joblib
import numpy as np

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy

BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0])
BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])


def _wbmae_const(y, c):
    y = np.asarray(y, dtype=np.float64).ravel()
    err = np.abs(c - y)
    idx = np.digitize(y, BIN_EDGES[1:-1], right=False)
    in_range = (y >= BIN_EDGES[0]) & (y <= BIN_EDGES[-1])
    ne = [b for b in range(4) if np.any(in_range & (idx == b))]
    if not ne:
        return np.inf
    maes = np.array([err[in_range & (idx == b)].mean() for b in ne])
    w = BIN_WEIGHTS[ne]
    return float((w * maes).sum() / w.sum())


class ConstantRegressor:
    def __init__(self, value=300.0):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(to_numpy(X)), self.value, dtype=np.float64)


class Solver(CompetSolver):

    name = "Optimal-Constant"

    def load_model(self, meta):
        model = ConstantRegressor()
        weights = meta["submission_dir"] / "weights.joblib"
        if weights.exists():
            model.value = float(joblib.load(weights)["value"])
        elif self.train_loader is None:
            raise FileNotFoundError(
                "Optimal-Constant needs weights.joblib; train with "
                "training=True and upload the exported folder."
            )
        return model

    def fit(self, model, train_loader):
        tgts = [to_numpy(y).ravel() for _X, y, _info in train_loader]
        y = np.concatenate(tgts)
        grid = np.arange(0.0, 600.01, 1.0)
        scores = [_wbmae_const(y, c) for c in grid]
        model.value = float(grid[int(np.argmin(scores))])
        print(f"[Optimal-Constant] best c={model.value:.1f}s  "
              f"train W-bMAE={min(scores):.2f}  (median={np.median(y):.1f}s)")

    def save_model(self, model, path):
        joblib.dump({"value": model.value}, path / "weights.joblib")
