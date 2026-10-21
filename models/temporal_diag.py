"""Diagnose how the TRAIN loader hands us windows (for temporal features).

Prints: total windows, unique (record_id, onset) pairs (duplicate rate),
number of recordings, windows/recording, and whether each recording's onsets
are contiguous (constant step). Tells us whether we can reconstruct clean
sequences for trend features, or whether the bin sampler drops/dupes windows.
Analysis only.
"""
import numpy as np
from collections import Counter
from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import to_numpy


class _M:
    def predict(self, X):
        return np.zeros(len(to_numpy(X)))


class Solver(CompetSolver):
    name = "Temporal-Diag"; requirements = []

    def load_model(self, meta):
        return _M()

    def fit(self, model, loader):
        rid, ons = [], []
        has_info = True
        for X, y, info in loader:
            try:
                rid.append(to_numpy(info["record_id"]).reshape(-1))
                ons.append(to_numpy(info["onset"]).reshape(-1))
            except Exception as e:
                has_info = False
                print(f"[temporal-diag] no record_id/onset in info: {e}; keys={list(info)}")
                break
        if not has_info:
            return
        rid = np.concatenate(rid); ons = np.concatenate(ons)
        n = len(rid)
        pairs = set(zip(rid.tolist(), ons.tolist()))
        recs = np.unique(rid)
        wpr = Counter(rid.tolist())
        counts = np.array(list(wpr.values()))
        print(f"[temporal-diag] total windows ........ {n}")
        print(f"[temporal-diag] unique (rec,onset) ... {len(pairs)}  (dup rate {1 - len(pairs)/n:.3f})")
        print(f"[temporal-diag] recordings ........... {len(recs)}")
        print(f"[temporal-diag] windows/recording .... min {counts.min()}  median {int(np.median(counts))}  max {counts.max()}")
        # onset step / contiguity for a few recordings
        print("[temporal-diag] sample recordings (sorted onset diffs):")
        for r in recs[:5]:
            o = np.sort(ons[rid == r])
            d = np.diff(o)
            uniq = len(o) == len(set(o.tolist()))
            if len(d):
                print(f"    rec {int(r)}: n={len(o)} unique_onsets={uniq} "
                      f"onset[min,max]=[{o.min()},{o.max()}] diff[min,med,max]=[{d.min()},{int(np.median(d))},{d.max()}]")
            else:
                print(f"    rec {int(r)}: n={len(o)} (single window)")

    def save_model(self, model, path):
        import joblib
        joblib.dump({}, path / "weights.joblib")
