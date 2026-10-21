"""Champion (Complexity-Smooth, 48.90) + OUTPUT countdown filter.

The streaming target decreases by exactly STEP(=5s) each window within a
recording. So instead of emitting independent per-window guesses, carry a
running estimate and enforce the countdown: est = a*raw + (1-a)*(prev-5),
corrected by each window's fresh prediction. Denoises the output trajectory
where it matters most -- the heavily weighted near-onset bin. Predict-time
only; training is identical to the champion.

On top of Aperiodic-Complexity, add a CAUSAL (past-only) exponential moving
average of the key staging discriminators -- fast (~30s) and slow (~2min,
matching YASA's past-only window). This DENOISES the per-window estimate
(different from the slope/trend idea, which measured velocity and was
redundant). EMA is reset per recording (streaming reset_state) and gap-reset
in fit so train matches inference. Self-contained, submittable. fit() prints
a per-bin GBM-vs-constant check.
"""
import math
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
SHAPE_BANDS = [(1.0, 8.0), (12.0, 16.0), (16.0, 30.0), (1.0, 40.0)]
FIT_LO, FIT_HI = 2.0, 40.0; EPS = 1e-8; CAP = 600.0
BIN_EDGES = np.array([0.0, 40.0, 90.0, 300.0, 600.0]); BIN_WEIGHTS = np.array([10.0, 5.0, 3.0, 1.0])
PE_ORDER = 4; HFD_KMAX = 8; STEP = 5.0
A_FAST = 1.0 - 0.5 ** (1.0 / 6.0)    # ~30 s half-life
A_SLOW = 1.0 - 0.5 ** (1.0 / 24.0)   # ~120 s half-life
ALPHA = 0.4   # output filter: weight on fresh per-window pred vs the countdown prior


def _skew(x):
    mu = x.mean(-1, keepdim=True); d = x - mu
    return (d ** 3).mean(-1) / ((d ** 2).mean(-1) ** 1.5 + EPS)
def _shape(Sp, freqs, lo, hi, T):
    s = torch.fft.irfft(Sp * ((freqs >= lo) & (freqs <= hi)).to(Sp.dtype), n=T, dim=-1)
    dd = torch.diff(s, dim=-1); ll = dd.abs().sum(-1) / (s.std(-1).clamp_min(EPS) * T)
    return torch.stack([_skew(s), ll], dim=-1)


def compute(X, sfreq, device):
    """Spectral/shape multi-band features (out) + tracked discriminators."""
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
    sp_peak = env.amax(-1)
    log_f = torch.log10(freqs.clamp_min(EPS)); ap = 10.0 ** (offset.unsqueeze(-1) - exponent.unsqueeze(-1) * log_f.view(1, 1, -1))
    osc = torch.log((psd - ap).clamp_min(0)[..., (freqs >= 12.0) & (freqs < 16.0)].sum(-1) + EPS)
    kc = torch.fft.irfft(Sp * ((freqs >= 0.5) & (freqs <= 4.0)).to(Sp.dtype), n=T, dim=-1)
    kc = (kc - kc.mean(-1, keepdim=True)) / kc.std(-1, keepdim=True).clamp_min(EPS); L = max(int(1.2 * sf), 1); kcf = kc.reshape(B * C, 1, T)
    p2p = (F.max_pool1d(kcf, L, stride=1) + F.max_pool1d(-kcf, L, stride=1)).reshape(B, C, -1)
    kc_max = p2p.amax(-1); kc_mean = p2p.mean(-1)
    d1 = torch.diff(X, dim=-1); d2 = torch.diff(d1, dim=-1)
    v0 = X.var(-1).clamp_min(EPS); v1 = d1.var(-1).clamp_min(EPS); v2 = d2.var(-1).clamp_min(EPS)
    mob = torch.sqrt(v1 / v0); cplx = torch.sqrt(v2 / v1) / mob.clamp_min(EPS)
    kept = torch.stack([sp_peak, osc, kc_max, kc_mean, mob, cplx], -1)
    shape = torch.cat([_shape(Sp, freqs, lo, hi, T) for (lo, hi) in SHAPE_BANDS], dim=-1)
    out = torch.cat([base, kept, shape], -1)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(B, -1).detach().cpu().numpy()
    tracked = torch.stack([exponent.mean(1), (bp["sigma"] / total).mean(1),
                           torch.log(bp["delta"] / bp["beta"]).mean(1), (bp["alpha"] / total).mean(1),
                           torch.log(total).mean(1)], dim=1)
    tracked = torch.nan_to_num(tracked, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().numpy()
    return out, tracked


def _higuchi(x, kmax):
    B, C, N = x.shape; lnL = np.zeros((B, C, kmax)); lnk = np.zeros(kmax)
    for ki, k in enumerate(range(1, kmax + 1)):
        Lk = np.zeros((B, C))
        for mm in range(k):
            nmax = (N - mm - 1) // k
            if nmax < 1: continue
            idx = mm + np.arange(0, nmax + 1) * k
            seg = x[..., idx]; diff = np.abs(np.diff(seg, axis=-1)).sum(-1)
            Lk += diff * (N - 1.0) / (nmax * k)
        Lk = Lk / k
        lnL[..., ki] = np.log(Lk + 1e-12); lnk[ki] = np.log(1.0 / k)
    lk = lnk - lnk.mean(); ll = lnL - lnL.mean(-1, keepdims=True)
    return (ll * lk).sum(-1) / (lk ** 2).sum()


def _perm_entropy(x, m):
    B, C, N = x.shape; nseg = N - m + 1
    emb = np.stack([x[..., i:i + nseg] for i in range(m)], axis=-1)
    order = np.argsort(emb, axis=-1)
    weights = m ** np.arange(m); code = (order * weights).sum(-1).astype(np.int64)
    maxc = m ** m; denom = math.log(math.factorial(m)); pe = np.zeros((B, C))
    for b in range(B):
        for c in range(C):
            cnt = np.bincount(code[b, c], minlength=maxc).astype(np.float64); p = cnt[cnt > 0] / nseg
            pe[b, c] = -(p * np.log(p)).sum() / denom
    return pe


def _complexity(X, device):
    x = (X.detach().cpu().numpy() if torch.is_tensor(X) else np.asarray(X)).astype(np.float64)
    if x.ndim == 2: x = x[None]
    B, C, T = x.shape
    xm = x - x.mean(-1, keepdims=True)
    zc = (np.sign(xm[..., 1:]) != np.sign(xm[..., :-1])).sum(-1) / (T - 1.0)
    d = np.diff(x, axis=-1); Nd = (np.sign(d[..., 1:]) != np.sign(d[..., :-1])).sum(-1).astype(np.float64)
    Nf = float(T); pfd = np.log10(Nf) / (np.log10(Nf) + np.log10(Nf / (Nf + 0.4 * Nd)))
    hfd = _higuchi(x, HFD_KMAX); pe = _perm_entropy(x, PE_ORDER)
    out = np.stack([pe, hfd, pfd, zc], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(B, -1)


def _best_const(y):
    gr = np.arange(0.0, 600.01, 2.0); idx = np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)
    ne = [b for b in range(4) if np.any(idx == b)]
    s = [(BIN_WEIGHTS[ne] * np.array([np.abs(c - y)[idx == b].mean() for b in ne])).sum() / BIN_WEIGHTS[ne].sum() for c in gr]
    return float(gr[int(np.argmin(s))])


class CxSmoothCD:
    def __init__(self, meta, pipe=None):
        self.sfreq = float(meta["sfreq"]); self.device = meta.get("device", "cpu"); self.pipe = pipe
        self._ef = None; self._es = None; self._cd = None
    def reset_state(self): self._ef = None; self._es = None; self._cd = None
    def _assemble(self, loader):
        f, trk, rid, ons, Y = [], [], [], [], []
        for X, y, info in loader:
            o, t = compute(X, self.sfreq, self.device); cx = _complexity(X, self.device)
            f.append(np.concatenate([o, cx], axis=1)); trk.append(t)
            rid.append(to_numpy(info["record_id"]).reshape(-1)); ons.append(to_numpy(info["onset"]).reshape(-1)); Y.append(to_numpy(y).ravel())
        f = np.concatenate(f); trk = np.concatenate(trk)
        rid = np.concatenate(rid).astype(np.int64); ons = np.concatenate(ons).astype(np.float64); y = np.concatenate(Y)
        first = {}
        for i in range(len(y)):
            k = (int(rid[i]), round(float(ons[i]), 3))
            if k not in first: first[k] = i
        smap = {}
        for r in np.unique(rid):
            os_r = sorted({round(float(o), 3) for o in ons[rid == r]}); ef = es = None; prev = None
            for o in os_r:
                t = trk[first[(int(r), o)]]
                if prev is None or abs(o - (prev + STEP)) > 1e-6:
                    ef = t.copy(); es = t.copy()
                else:
                    ef = A_FAST * t + (1 - A_FAST) * ef; es = A_SLOW * t + (1 - A_SLOW) * es
                smap[(int(r), o)] = np.concatenate([ef, es]); prev = o
        S = np.stack([smap[(int(rid[i]), round(float(ons[i]), 3))] for i in range(len(y))])
        return np.concatenate([f, S], axis=1), y
    def fit(self, loader):
        Xf, y = self._assemble(loader)
        sw = BIN_WEIGHTS[np.clip(np.digitize(y, BIN_EDGES[1:-1]), 0, 3)]
        rng = np.random.default_rng(33); idx = rng.permutation(len(y)); cut = int(0.8 * len(y)); tr, ho = idx[:cut], idx[cut:]
        dg = Pipeline([("s", StandardScaler()), ("r", HistGradientBoostingRegressor(loss="absolute_error"))])
        dg.fit(Xf[tr], y[tr], r__sample_weight=sw[tr])
        c = _best_const(y[tr]); yh = y[ho]; pg = np.clip(dg.predict(Xf[ho]), 0, 600); bi = np.clip(np.digitize(yh, BIN_EDGES[1:-1]), 0, 3)
        print(f"[complexity-smooth per-bin] optimal constant c={c:.1f}s")
        print(f"{'bin':12s}{'GBM_MAE':>10s}{'const_MAE':>11s}{'gain':>8s}")
        for b, lab in enumerate(["[0,40)", "[40,90)", "[90,300)", "[300,600]"]):
            mk = bi == b
            if not mk.any(): continue
            g = np.abs(pg[mk] - yh[mk]).mean(); cc = np.abs(c - yh[mk]).mean()
            print(f"{lab:12s}{g:>10.1f}{cc:>11.1f}{cc - g:>8.1f}")
        self.pipe = Pipeline([("scaler", StandardScaler()), ("reg", HistGradientBoostingRegressor(loss="absolute_error"))])
        self.pipe.fit(Xf, y, reg__sample_weight=sw); return self
    def predict(self, X):
        o, trk = compute(X, self.sfreq, self.device); cx = _complexity(X, self.device); rows = []
        for i in range(len(o)):
            t = trk[i]
            if self._ef is None:
                self._ef = t.copy(); self._es = t.copy()
            else:
                self._ef = A_FAST * t + (1 - A_FAST) * self._ef; self._es = A_SLOW * t + (1 - A_SLOW) * self._es
            rows.append(np.concatenate([o[i], cx[i], self._ef, self._es]))
        raw = np.clip(self.pipe.predict(np.array(rows)), 0.0, CAP)
        out = np.empty_like(raw)
        for i in range(len(raw)):
            est = raw[i] if self._cd is None else ALPHA * raw[i] + (1 - ALPHA) * max(self._cd - STEP, 0.0)
            est = min(max(est, 0.0), CAP); self._cd = est; out[i] = est
        return out


class Solver(CompetSolver):
    name = "Smooth-Countdown"; requirements = []
    def load_model(self, meta):
        mdl = CxSmoothCD(meta); w = meta["submission_dir"] / "weights.joblib"
        if w.exists(): mdl.pipe = joblib.load(w)
        elif self.train_loader is None: raise FileNotFoundError("Smooth-Countdown needs weights.joblib; train with training=True.")
        return mdl
    def fit(self, model, loader): model.fit(loader)
    def save_model(self, model, path): joblib.dump(model.pipe, path / "weights.joblib")
