# -*- coding: utf-8 -*-
"""
train_deep_tcn_clean_v2_minimal_artifacts.py — Minimal outputs (models/*.pkl, dataset/holdout_probs.csv)
- Keep ALL model logic/behavior identical to original training/eval path
- Remove non‑essential validation exports (threshold grids, FP CSVs, etc.)
- Save ONE consolidated artifact file: models/model_artifacts.pkl
- Save holdout probabilities CSV to: dataset/holdout_probs.csv
"""

import os, argparse, math, random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F

# ----------------------------
# Global flip helper (kept)
# ----------------------------
FLIP_LOGITS = 1  # default

def maybe_flip_logits(logits, args=None):
    try:
        flip = int(getattr(args, "flip_logits", FLIP_LOGITS))
    except Exception:
        flip = FLIP_LOGITS
    return -logits if flip == 1 else logits

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ts_to_minutes(ts: pd.Series) -> np.ndarray:
    ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    return (ns // 10**9) // 60

def human_count(n):
    return f"{n:,}"

# ----------------------------
# Labeling
# ----------------------------
def make_pre_event_labels(df: pd.DataFrame, ts_col: str, label_col: str,
                          delta_min: int, pos_dilate_back_steps: int,
                          resample_minutes: int) -> np.ndarray:
    y = np.zeros(len(df), dtype=np.int8)
    ts_min = ts_to_minutes(df[ts_col])
    pos_idx = np.where(df[label_col].values.astype(int) == 1)[0]
    if len(pos_idx) == 0:
        return y
    pos_ts = ts_min[pos_idx]
    delta = int(delta_min)
    for pt in pos_ts:
        start_min = pt - delta
        mask = (ts_min >= start_min) & (ts_min <= pt)
        y[mask] = 1
    if pos_dilate_back_steps > 0:
        pos_where = np.where(y == 1)[0]
        for k in range(1, pos_dilate_back_steps + 1):
            prev_idx = pos_where - k
            y[prev_idx[prev_idx >= 0]] = 1
    return y

# ----------------------------
# Auto Features (past-only)
# ----------------------------
def build_auto_features(df: pd.DataFrame, feature_cols: list, ts_col: str, enable: bool):
    if not enable:
        return df, feature_cols

    base = df[feature_cols].copy()
    feats = [base]

    for k in (1, 5, 15):
        d = base.diff(k); d.columns = [f"{c}_d{k}" for c in d.columns]; feats.append(d)
    for k in (1, 5):
        p = base.pct_change(k); p.columns = [f"{c}_pct{k}" for c in p.columns]; feats.append(p)
    for w in (10, 30, 60):
        rstd = base.rolling(w, min_periods=2).std(); rstd.columns = [f"{c}_rstd{w}" for c in rstd.columns]; feats.append(rstd)
        rmax = base.rolling(w, min_periods=2).max(); rmin = base.rolling(w, min_periods=2).min()
        rrange = rmax - rmin; rrange.columns = [f"{c}_rrange{w}" for c in rrange.columns]; feats.append(rrange)
    diff1 = base.diff(1).abs(); ema_span = 10
    ema = diff1.ewm(span=ema_span, adjust=False).mean(); ema.columns = [f"{c}_ema_absdiff1_{ema_span}" for c in ema.columns]
    feats.append(ema)

    g = pd.concat(feats, axis=1); g.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_aug = pd.concat([df.drop(columns=feature_cols), g], axis=1)
    new_cols = list(g.columns)
    return df_aug, new_cols

# ----------------------------
# Sequence Builder
# ----------------------------
@dataclass
class SeqIndex:
    start: int; end: int; label: int; nearest_event_dist_min: int

def build_windows(df: pd.DataFrame, ts_col: str, feature_cols: list,
                  labels: np.ndarray, seq_len: int, stride: int,
                  pre_event_minutes: int):
    X = df[feature_cols].values.astype(np.float32)
    ts_min = ts_to_minutes(df[ts_col])
    pos_idx = np.where(labels == 1)[0]
    event_ts = ts_min[pos_idx] if len(pos_idx) else np.array([], dtype=np.int64)

    seqs = []; N = len(df); i = 0
    while i + seq_len <= N:
        s, e = i, i + seq_len
        y = 1 if labels[s:e].max() == 1 else 0
        if len(event_ts):
            end_min = ts_min[e - 1]
            future_diffs = event_ts - end_min
            future_diffs = future_diffs[future_diffs >= 0]
            nearest = int(future_diffs.min()) if len(future_diffs) else 10**9
        else:
            nearest = 10**9
        seqs.append(SeqIndex(s, e, int(y), nearest))
        i += stride
    return seqs, X, ts_min

# ----------------------------
# Time cutoff split (pos-aware embargo)
# ----------------------------
def embargo_posaware_split(df: pd.DataFrame, ts_col: str, labels: np.ndarray,
                           holdout_days: float, embargo_minutes: int,
                           min_pos_train: int, min_pos_holdout: int, *,
                           seqs_for_split, min_train_windows: int = 32,
                           min_holdout_windows: int = 32,
                           target_train_frac: float = 0.7,
                           train_frac_weight: float = 10.0):
    ts = df[ts_col]
    cutoff_base = ts.max() - pd.Timedelta(days=float(holdout_days))

    idx_to_ts = ts.values
    win_starts = np.array([idx_to_ts[s.start] for s in seqs_for_split], dtype="datetime64[ns]")
    win_ends   = np.array([idx_to_ts[s.end - 1] for s in seqs_for_split], dtype="datetime64[ns]")
    win_labels = np.array([s.label for s in seqs_for_split], dtype=int)

    candidates = []
    lo = ts.min() + pd.Timedelta(hours=2)
    hi = ts.max() - pd.Timedelta(hours=2)
    for shift_min in range(-72*60, 72*60 + 1, 15):
        c = cutoff_base + pd.Timedelta(minutes=shift_min)
        if c < lo or c > hi: continue
        candidates.append(c)

    def counts_at_cut(cut):
        tr_mask_w = (win_ends <= (cut.to_datetime64() - np.timedelta64(embargo_minutes, 'm')))
        ho_mask_w = (win_starts >= cut.to_datetime64())
        tr_n = int(tr_mask_w.sum()); ho_n = int(ho_mask_w.sum())
        tr_pos = int(win_labels[tr_mask_w].sum()); ho_pos = int(win_labels[ho_mask_w].sum())
        return tr_mask_w, ho_mask_w, tr_n, ho_n, tr_pos, ho_pos

    def eval_cut(cut):
        tr_m, ho_m, tr_n, ho_n, tr_pos, ho_pos = counts_at_cut(cut)
        feasible = (tr_n >= min_train_windows) and (ho_n >= min_holdout_windows) and \
                   (tr_pos >= min_pos_train) and (ho_pos >= min_pos_holdout)
        train_frac = tr_n / max(1, (tr_n + ho_n))
        score = (10.0 * ho_pos + 5.0 * tr_pos
                 - train_frac_weight * abs(train_frac - target_train_frac) * (tr_n + ho_n)
                 - (abs(tr_n - min_train_windows) + abs(ho_n - min_holdout_windows)))
        return feasible, score, (cut, tr_m, ho_m, tr_n, ho_n, tr_pos, ho_pos, train_frac)

    best_pack, best_score, best_feasible = None, -1e18, False
    for c in candidates:
        feasible, score, pack = eval_cut(c)
        if feasible and (score > best_score):
            best_score, best_pack, best_feasible = score, pack, True
        elif (not best_feasible) and (score > best_score):
            best_score, best_pack, best_feasible = score, pack, False

    if best_pack is None:
        raise RuntimeError("No candidate cut evaluated.")

    cut, tr_m, ho_m, tr_n, ho_n, tr_pos, ho_pos, train_frac = best_pack

    moved = False; step = pd.Timedelta(minutes=5); max_steps = 200
    if train_frac < target_train_frac:
        for _ in range(max_steps):
            c2 = cut + step
            tr_m2, ho_m2, tr_n2, ho_n2, tr_pos2, ho_pos2 = counts_at_cut(c2)
            feasible2 = (tr_n2 >= min_train_windows) and (ho_n2 >= min_holdout_windows) and \
                        (tr_pos2 >= min_pos_train) and (ho_pos2 >= min_pos_holdout)
            if not feasible2: break
            train_frac2 = tr_n2 / max(1, (tr_n2 + ho_n2))
            if train_frac2 <= train_frac: break
            cut, tr_m, ho_m, tr_n, ho_n, tr_pos, ho_pos, train_frac = c2, tr_m2, ho_m2, tr_n2, ho_n2, tr_pos2, ho_pos2, train_frac2
            moved = True
            if train_frac >= target_train_frac: break
    elif train_frac > (target_train_frac + 0.02):
        for _ in range(max_steps):
            c2 = cut - step
            tr_m2, ho_m2, tr_n2, ho_n2, tr_pos2, ho_pos2 = counts_at_cut(c2)
            feasible2 = (tr_n2 >= min_train_windows) and (ho_n2 >= min_holdout_windows) and \
                        (tr_pos2 >= min_pos_train) and (ho_pos2 >= min_pos_holdout)
            if not feasible2: break
            train_frac2 = tr_n2 / max(1, (tr_n2 + ho_n2))
            if train_frac2 >= train_frac: break
            cut, tr_m, ho_m, tr_n, ho_n, tr_pos, ho_pos, train_frac = c2, tr_m2, ho_m2, tr_n2, ho_n2, tr_pos2, ho_pos2, train_frac2
            moved = True
            if train_frac <= target_train_frac: break

    status = "OK" if best_feasible else "WARN"
    note = " (nudged)" if moved else ""
    print(f"[SPLIT] cutoff={cut} | embargo={embargo_minutes}min | train_win={tr_n} (pos={tr_pos}) | holdout_win={ho_n} (pos={ho_pos}) | {status}{note}")

    train_mask_pts = df[ts_col].values <= (cut.to_datetime64() - np.timedelta64(embargo_minutes, 'm'))
    hold_mask_pts  = df[ts_col].values >=  cut.to_datetime64()

    if tr_pos == 0: print("[WARN] Train has 0 positives after correction.")
    if ho_pos == 0: print("[WARN] Holdout has 0 positives after correction.")
    return train_mask_pts, hold_mask_pts

# ----------------------------
# Event-based split (kept)
# ----------------------------
def event_based_split(df: pd.DataFrame, ts_col: str, labels: np.ndarray, seqs_for_split: list,
                      pre_event_minutes: int, holdout_frac: float = 0.3, holdout_min_events: int = 4,
                      seed: int = 42, extra_min: int = 0, target_holdout_frac: float = 0.0,
                      target_holdout_max: float = 0.5, pos_dilate_back_steps: int = 0, tail_min: int = 30):
    y = labels.astype(int)
    nxt = np.concatenate((y[1:], [0]))
    ends_idx = np.where((y == 1) & (nxt == 0))[0]
    if len(ends_idx) == 0:
        print("[WARN] No events found for event-based split. Fallback to time split.")
        return None, None

    m = len(ends_idx)
    k = max(holdout_min_events, int(round(m * holdout_frac)))
    k = min(k, m)
    rng = np.random.default_rng(seed)
    ho_events = rng.choice(ends_idx, size=k, replace=False)

    ts = df[ts_col].values
    ts_min = (ts.astype("datetime64[s]").astype("int64")) // 60

    dilate_extra = max(0, int(pos_dilate_back_steps))
    delta = int(pre_event_minutes) + dilate_extra
    extra = max(0, int(extra_min))
    tail = max(1, int(tail_min))
    tail = min(tail, delta + extra)

    ho_point_mask = np.zeros(len(df), dtype=bool)
    for eidx in ho_events:
        t_e = ts_min[eidx]
        mask = (ts_min >= (t_e - tail - extra)) & (ts_min <= (t_e + extra))
        ho_point_mask |= mask

    tr_idx, ho_idx = [], []
    for i, s in enumerate(seqs_for_split):
        rows = np.arange(s.start, s.end)
        if ho_point_mask[rows].any():
            ho_idx.append(i)
        else:
            tr_idx.append(i)

    total_win = len(seqs_for_split)
    if target_holdout_frac and total_win > 0:
        cap = min(max(0.0, target_holdout_max), 0.95)
        tgt = min(cap, max(0.0, float(target_holdout_frac)))
        need = int(round(total_win * tgt)) - len(ho_idx)
        if need > 0:
            neg_train_idx = [i for i in tr_idx if seqs_for_split[i].label == 0]
            if len(neg_train_idx) > 0:
                add = rng.choice(neg_train_idx, size=min(need, len(neg_train_idx)), replace=False).tolist()
                for a in add:
                    tr_idx.remove(a)
                ho_idx.extend(add)

    y_tr = np.array([seqs_for_split[i].label for i in tr_idx], dtype=int)
    y_ho = np.array([seqs_for_split[i].label for i in ho_idx], dtype=int)
    print(f"[SPLIT-EVENT] events_total={m} | events_holdout={k} | train_win={len(tr_idx)} (pos={int(y_tr.sum())}) | holdout_win={len(ho_idx)} (pos={int(y_ho.sum())})")
    return np.array(tr_idx, dtype=int), np.array(ho_idx, dtype=int)

# ----------------------------
# Sampler/Dataset (kept)
# ----------------------------
class HardNegPosSampler(Sampler):
    def __init__(self, seqs, batch_size=128, pos_oversample=8, hard_neg_frac=0.3, neg_frac=0.3, shuffle=True):
        self.batch_size = batch_size; self.shuffle = shuffle
        self.pos_idx = [i for i, s in enumerate(seqs) if s.label == 1]
        self.neg_idx = [i for i, s in enumerate(seqs) if s.label == 0]
        hard_thresh = 90
        self.hard_neg_idx = [i for i, s in enumerate(seqs) if (s.label == 0 and s.nearest_event_dist_min <= hard_thresh)]
        self.other_neg_idx = list(set(self.neg_idx) - set(self.hard_neg_idx))

        self.hard_neg_quota = int(round(self.batch_size * hard_neg_frac))
        self.neg_quota = int(round(self.batch_size * neg_frac))
        if self.hard_neg_quota > self.neg_quota: self.hard_neg_quota = self.neg_quota
        self.other_neg_quota = max(0, self.neg_quota - self.hard_neg_quota)
        self.pos_quota = max(1, self.batch_size - (self.hard_neg_quota + self.other_neg_quota))
        self.pos_oversample = int(max(1, pos_oversample))

        self.hard_pool = self.hard_neg_idx.copy(); self.other_pool = self.other_neg_idx.copy()
        if self.shuffle:
            random.shuffle(self.hard_pool); random.shuffle(self.other_pool)

        denom = max(1, (self.hard_neg_quota + self.other_neg_quota))
        self.num_batches = max(1, math.ceil(len(self.neg_idx) / denom))

    def __len__(self): return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            if len(self.hard_pool) < self.hard_neg_quota:
                self.hard_pool = self.hard_neg_idx.copy()
                if self.shuffle: random.shuffle(self.hard_pool)
            batch += self.hard_pool[:self.hard_neg_quota]
            self.hard_pool = self.hard_pool[self.hard_neg_quota:]

            if len(self.other_pool) < self.other_neg_quota:
                self.other_pool = self.other_neg_idx.copy()
                if self.shuffle: random.shuffle(self.other_pool)
            batch += self.other_pool[:self.other_neg_quota]
            self.other_pool = self.other_pool[self.other_neg_quota:]

            if len(self.pos_idx) == 0:
                neg_pool = self.hard_neg_idx + self.other_neg_idx
                pos_take = [] if len(neg_pool) == 0 else np.random.choice(neg_pool, size=self.pos_quota, replace=True).tolist()
            else:
                pos_take = np.random.choice(self.pos_idx, size=self.pos_quota, replace=True).tolist()
            batch += pos_take

            if self.shuffle: random.shuffle(batch)
            yield batch

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, seqs: list):
        self.X = X; self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        s = self.seqs[idx]
        x = self.X[s.start:s.end]; y = s.label
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# ----------------------------
# Models (kept)
# ----------------------------
class GRUModel(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=layers,
                          batch_first=True, bidirectional=False)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h).squeeze(1)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, dropout=0.2):
        super().__init__()
        self.k, self.d = k, d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=d, padding=0, bias=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, dilation=d, padding=0, bias=True)
        self.relu = nn.ReLU(); self.drop = nn.Dropout(p=dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def _causal_pad(self, x):
        pad = (self.k - 1) * self.d
        return F.pad(x, (pad, 0))
    def forward(self, x):
        y = self.conv1(self._causal_pad(x)); y = self.relu(y); y = self.drop(y)
        y = self.conv2(self._causal_pad(y)); y = self.relu(y); y = self.drop(y)
        return y + self.down(x)

class TCNModel(nn.Module):
    def __init__(self, in_dim, channels=(64, 64, 64), dropout=0.2):
        super().__init__()
        layers = []; c_in = in_dim; d = 1
        for c in channels:
            layers.append(TCNBlock(c_in, c, k=3, d=d, dropout=dropout))
            c_in = c; d *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        last = y[:, :, -1]
        return self.head(last).squeeze(1)

# ----------------------------
# Losses (kept)
# ----------------------------
class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2.0, alpha_pos=0.5, label_smoothing=0.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma; self.alpha_pos = alpha_pos; self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32))
    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            targets = targets * (1 - eps) + 0.5 * eps
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(logits); y = targets
        alpha_t = self.alpha_pos * y + (1 - self.alpha_pos) * (1 - y)
        p_t = p * y + (1 - p) * (1 - y)
        focal_weight = alpha_t * torch.pow((1 - p_t).clamp(1e-6, 1.0), self.gamma)
        return (focal_weight * bce).mean()

def batch_pairwise_rank_loss(logits, targets, margin=0.2):
    with torch.no_grad():
        pos_mask = targets > 0.5
        neg_mask = targets < 0.5
    if (pos_mask.sum() == 0) or (neg_mask.sum() == 0):
        return logits.new_zeros(())
    pos_scores = logits[pos_mask]
    neg_scores = logits[neg_mask]
    max_pairs = 4096
    P = pos_scores.numel(); N = neg_scores.numel()
    if P * N == 0: return logits.new_zeros(())
    k_neg = min(128, N)
    neg_topk, _ = torch.topk(neg_scores, k_neg)
    k_pos = min(256, P)
    pos_sel = pos_scores if P <= k_pos else torch.topk(pos_scores, k_pos).values
    diff = pos_sel.view(-1,1) - neg_topk.view(1,-1)
    loss = torch.relu(margin - diff)
    if loss.numel() > max_pairs:
        idx = torch.randperm(loss.numel(), device=loss.device)[:max_pairs]
        loss = loss.view(-1)[idx]
    return loss.mean()

# ----------------------------
# Simple augs (kept)
# ----------------------------
def aug_mixup(x, y, alpha=0.0):
    if alpha <= 0 or len(x) < 2: return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x2 = x[perm]; y2 = y[perm]
    return lam * x + (1 - lam) * x2, lam * y + (1 - lam) * y2

def aug_jitter(x, std=0.0):
    if std <= 0: return x
    return x + torch.randn_like(x) * std

def aug_time_cutout(x, prob=0.0):
    if prob <= 0: return x
    B, T, F_ = x.shape
    for b in range(B):
        if random.random() < prob:
            k = max(1, int(T * random.uniform(0.05, 0.1)))
            start = random.randint(0, T - k)
            x[b, start:start + k, :] = 0.0
    return x

# ----------------------------
# Postprocess (kept)
# ----------------------------
def k_consecutive_filter(yhat_bin: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return yhat_bin
    y = yhat_bin.copy(); n = len(y); i = 0
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1: j += 1
            if (j - i) < k: y[i:j] = 0
            i = j
        else: i += 1
    return y

def ema_series(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1: return x.copy()
    alpha = 2.0 / (span + 1.0)
    y = np.zeros_like(x, dtype=float); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def hysteresis_cooldown(prob: np.ndarray, win_ends_ts: np.ndarray,
                        open_th: float, close_th: float, cooldown_min: int) -> np.ndarray:
    assert prob.shape[0] == len(win_ends_ts)
    y = np.zeros(len(prob), dtype=int)
    state = 0; last_alarm_end_ts = None
    for i, p in enumerate(prob):
        t = win_ends_ts[i]
        if state == 0:
            ok_cool = True
            if last_alarm_end_ts is not None:
                ok_cool = (t >= last_alarm_end_ts + np.timedelta64(int(cooldown_min), 'm'))
            if ok_cool and (p >= open_th):
                state = 1; y[i] = 1
        else:
            y[i] = 1
            if p <= close_th:
                state = 0
                last_alarm_end_ts = t
    return y

# ----------------------------
# Calibration/threshold selection (kept)
# ----------------------------

def _safe_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calibrate_probs(y_prob, y_true, method="none"):
    if method == "none":
        return y_prob
    y_true = y_true.astype(int)
    if method == "platt":
        X = _safe_logit(y_prob).reshape(-1, 1)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, y_true)
        return lr.predict_proba(X)[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob, y_true)
        return iso.transform(y_prob)
    return y_prob


def choose_threshold_with_k(y_true, y_prob, k=1, recall_floor=None, precision_floor=None, fp_budget=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)
    best_f1, chosen = -1.0, None
    fp_allowed = fp_budget if fp_budget is not None else 10**9
    for t in thresholds:
        yhat = (y_prob >= t).astype(int)
        yhat = k_consecutive_filter(yhat, k)
        tp = int(((yhat == 1) & (y_true == 1)).sum())
        fp = int(((yhat == 1) & (y_true == 0)).sum())
        fn = int(((yhat == 0) & (y_true == 1)).sum())
        p = tp / max(1, (tp + fp)); r = tp / max(1, (tp + fn))
        if (recall_floor is not None and r < recall_floor) or \
           (precision_floor is not None and p < precision_floor) or \
           (fp > fp_allowed): continue
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if f1 > best_f1: best_f1, chosen = f1, t
    if chosen is None:
        for t in thresholds:
            yhat = (y_prob >= t).astype(int)
            yhat = k_consecutive_filter(yhat, k)
            tp = int(((yhat == 1) & (y_true == 1)).sum())
            fp = int(((yhat == 1) & (y_true == 0)).sum())
            fn = int(((yhat == 0) & (y_true == 1)).sum())
            p = tp / max(1, (tp + fp)); r = tp / max(1, (tp + fn))
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            if f1 > best_f1: best_f1, chosen = f1, t
    return chosen

def _save_eval_plots_and_report(y_true, y_prob, args):
    """
    검증 결과 시각화 및 리포트 저장 (evals 폴더).
    - PR Curve (AUPRC)
    - ROC Curve (ROC-AUC)
    - Calibration (Reliability curve)
    - Threshold sweep (Precision/Recall/F1 vs threshold)
    - Probability histogram (pos/neg)
    - metrics.txt (정량 지표 요약)
    """
    import os
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc

    os.makedirs("evals", exist_ok=True)

    # 안전 마스킹 (NaN/inf 제거)
    y = np.asarray(y_true).reshape(-1).astype(int)
    p = np.asarray(y_prob).reshape(-1).astype(float)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]; p = p[m]

    report_lines = []

    # 1) PR Curve + AUPRC
    prec, rec, thr = precision_recall_curve(y, p)
    auprc_curve = auc(rec, prec) if (len(rec) > 1) else float("nan")

    plt.figure()
    plt.step(rec, prec, where="post")
    base = (y == 1).mean() if len(y) else 0.0
    plt.hlines(base, 0, 1, linestyles="dashed")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (baseline={base:.3f}, AUC={auprc_curve:.4f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evals/pr_curve.png", dpi=150)
    plt.close()

    # 2) ROC Curve + ROC-AUC
    roc_auc = float("nan")
    if (y.max() != y.min()):  # 두 클래스 모두 있을 때만
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = roc_auc_score(y, p)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("evals/roc_curve.png", dpi=150)
        plt.close()

    # 3) Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
        plt.figure()
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Mean predicted value")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration Curve (Reliability)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("evals/calibration_curve.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # 4) Threshold sweep (P/R/F1)
    thr_sweep = np.unique(np.clip(np.concatenate([thr, [0.0, 1.0]]), 0, 1))
    pr_list, re_list, f1_list = [], [], []
    for t in thr_sweep:
        yhat = (p >= t).astype(int)
        if hasattr(args, "post_k") and int(args.post_k) > 1:
            yhat = k_consecutive_filter(yhat, int(args.post_k))
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        prec_t = tp / max(1, tp + fp)
        rec_t  = tp / max(1, tp + fn)
        f1_t   = (2*prec_t*rec_t/max(1e-12, (prec_t + rec_t))) if (prec_t + rec_t) > 0 else 0.0
        pr_list.append(prec_t); re_list.append(rec_t); f1_list.append(f1_t)

    plt.figure()
    plt.plot(thr_sweep, pr_list, label="Precision")
    plt.plot(thr_sweep, re_list, label="Recall")
    plt.plot(thr_sweep, f1_list, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep (Precision / Recall / F1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evals/threshold_sweep.png", dpi=150)
    plt.close()

    # 5) Score histogram (pos/neg)
    plt.figure()
    if (y == 1).any():
        plt.hist(p[y == 1], bins=30, alpha=0.6, label="pos")
    if (y == 0).any():
        plt.hist(p[y == 0], bins=30, alpha=0.6, label="neg")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Histogram by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evals/prob_hist.png", dpi=150)
    plt.close()

    # 6) 임계값 선택 + 최종 점수
    try:
        t_star = choose_threshold_with_k(
            y_true=y, y_prob=p,
            k=int(getattr(args, "post_k", 1)),
            recall_floor=getattr(args, "recall_floor", None),
            precision_floor=getattr(args, "precision_floor", None),
            fp_budget=getattr(args, "fp_budget", None),
        )
    except Exception:
        t_star = None
    if (t_star is None) or (not hasattr(__import__('numpy'), 'isfinite') or not __import__('numpy').isfinite(t_star)):
        try:
            import numpy as _np
            if (t_star is None) or (not _np.isfinite(t_star)):
                t_star = 0.5
        except Exception:
            t_star = 0.5

    yhat = (p >= t_star).astype(int)
    if hasattr(args, "post_k") and int(args.post_k) > 1:
        yhat = k_consecutive_filter(yhat, int(args.post_k))

    prec_star = precision_score(y, yhat, zero_division=0)
    rec_star  = recall_score(y, yhat, zero_division=0)
    f1_star   = f1_score(y, yhat, zero_division=0)

    report_lines.append(f"AUPRC_curve: {auprc_curve:.6f}")
    report_lines.append(f"ROC_AUC: {roc_auc:.6f}" if (roc_auc == roc_auc) else "ROC_AUC: nan")
    report_lines.append(f"Chosen_threshold: {t_star:.6f}")
    report_lines.append(f"Precision@th: {prec_star:.6f}")
    report_lines.append(f"Recall@th: {rec_star:.6f}")
    report_lines.append(f"F1@th: {f1_star:.6f}")
    report_lines.append(f"post_k: {int(getattr(args, 'post_k', 1))}")

    with open("evals/metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("[EVALS] Saved: evals/pr_curve.png, roc_curve.png, calibration_curve.png, threshold_sweep.png, prob_hist.png, metrics.txt")


# ----------------------------
# Evaluation (kept)
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, args=None):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            logits = maybe_flip_logits(logits, args)
            prob = torch.sigmoid(logits).squeeze(-1)
            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())
    if not ys:
        return float('nan'), None, None
    y = np.asarray(np.concatenate(ys)).reshape(-1).astype(int)
    p = np.asarray(np.concatenate(ps)).reshape(-1).astype(float)
    # safety squash
    if np.nanmin(p) < 0 or np.nanmax(p) > 1:
        p = 1.0 / (1.0 + np.exp(-p))
    # guard polarity
    try:
        mpos = float(np.nanmean(p[y==1])) if (y==1).any() else np.nan
        mneg = float(np.nanmean(p[y==0])) if (y==0).any() else np.nan
        if np.isfinite(mpos) and np.isfinite(mneg) and (mpos < mneg):
            print("[WARN] mean(pos) < mean(neg) → flipping scores for evaluation")
            p = 1.0 - p
    except Exception:
        pass
    try:
        ap = average_precision_score(y, p)
    except Exception:
        ap = float('nan')
    return ap, y, p

# ----------------------------
# Training loop (kept)
# ----------------------------
def train_one_epoch(model, loader, optimizer, focal_loss_fn, device,
                    mixup_alpha=0.0, jitter_std=0.0, time_cutout_prob=0.0,
                    loss_mode="focal", rank_weight=0.5, rank_margin=0.2, args=None):
    model.train(); losses = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        x = aug_jitter(x, std=jitter_std)
        x = aug_time_cutout(x, prob=time_cutout_prob)
        x, y = aug_mixup(x, y, alpha=mixup_alpha)
        logits = model(x)
        logits = maybe_flip_logits(logits, args)
        loss = focal_loss_fn(logits, y)
        if "rank" in loss_mode:
            rank_loss = batch_pairwise_rank_loss(logits.detach().clone(), y.detach().clone(), margin=rank_margin)
            loss = (1 - rank_weight) * loss + rank_weight * rank_loss
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

# ----------------------------
# MAIN
# ----------------------------
def parse_channels(ch_str: str):
    try: return tuple(int(s.strip()) for s in ch_str.split(",") if s.strip())
    except Exception: return (64, 64, 64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--ts-col", type=str, default="datetime")
    parser.add_argument("--id-cols", nargs="*", default=["_grp"])
    parser.add_argument("--label-col", type=str, default="passorfail")
    parser.add_argument("--resample", type=str, default="1min")

    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--label-mode", type=str, default="pre_event", choices=["pre_event"])
    parser.add_argument("--label-delta-minutes", type=int, default=60)
    parser.add_argument("--pos-dilate-back-steps", type=int, default=2)
    parser.add_argument("--horizon-minutes", type=int, default=15)

    parser.add_argument("--val-holdout-days", type=float, default=3.0)
    parser.add_argument("--embargo-minutes", type=int, default=30)
    parser.add_argument("--min-pos-train", type=int, default=10)
    parser.add_argument("--min-pos-holdout", type=int, default=2)
    parser.add_argument("--target-train-frac", type=float, default=0.7)
    parser.add_argument("--train-frac-weight", type=float, default=10.0)

    parser.add_argument("--split-mode", type=str, default="time", choices=["time", "event"])
    parser.add_argument("--event-holdout-frac", type=float, default=0.35)
    parser.add_argument("--event-holdout-min", type=int, default=4)
    parser.add_argument("--event-holdout-extra-min", type=int, default=0)
    parser.add_argument("--target-holdout-frac", type=float, default=0.0)
    parser.add_argument("--target-holdout-max", type=float, default=0.5)
    parser.add_argument("--event-holdout-tail-min", type=int, default=30)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "tcn"])
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--tcn-channels", type=str, default="64,64,64")

    parser.add_argument("--pos-oversample", type=int, default=10)
    parser.add_argument("--neg-frac", type=float, default=0.3)
    parser.add_argument("--hard-neg-frac", type=float, default=0.3)

    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=-1.0, help="if <0, auto compute from class balance")
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--jitter-std", type=float, default=0.0)
    parser.add_argument("--time-cutout-prob", type=float, default=0.0)

    parser.add_argument("--recall-floor", type=float, default=0.95)
    parser.add_argument("--precision-floor", type=float, default=None)
    parser.add_argument("--fp-budget", type=int, default=10)
    parser.add_argument("--use-best-ap", action="store_true", default=True)

    parser.add_argument("--post-k", type=int, default=1)
    parser.add_argument("--event-eval", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flip-logits", type=int, default=1)
    parser.add_argument("--use-raw-events", action="store_true", default=False)

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--early-stop-patience", type=int, default=15)

    parser.add_argument("--loss-mode", type=str, default="focal", choices=["focal", "focal+rank"])
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-margin", type=float, default=0.2)

    parser.add_argument("--ema-span", type=int, default=0)
    parser.add_argument("--open-th", type=float, default=0.85)
    parser.add_argument("--close-th", type=float, default=0.65)
    parser.add_argument("--cooldown-min", type=int, default=0)

    parser.add_argument("--auto-features", type=int, default=0)
    parser.add_argument("--save-probs", type=int, default=1)
    parser.add_argument("--probs-path", type=str, default="dataset/holdout_probs.csv")

    parser.add_argument("--calibration", type=str, default="none", choices=["none","platt","isotonic"])
    parser.add_argument("--th-objective", type=str, default="window_f1",
                        choices=["window_f1","event_f1","lead_p50"])

    # NEW: single consolidated artifact
    parser.add_argument("--save-artifacts", type=int, default=1)
    parser.add_argument("--artifacts-path", type=str, default="models/model_artifacts.pkl")

    args = parser.parse_args()
    set_seed(args.seed)

    # ensure output dirs
    probs_dir = os.path.dirname(args.probs_path) or "."
    art_dir = os.path.dirname(args.artifacts_path) or "."
    os.makedirs(probs_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    # ----- Load -----
    df = pd.read_csv(args.input)
    if args.ts_col not in df.columns: raise ValueError(f"ts_col {args.ts_col} not in data.")
    if args.label_col not in df.columns: raise ValueError(f"label_col {args.label_col} not in data.")
    df[args.ts_col] = pd.to_datetime(df[args.ts_col])
    df = df.sort_values(args.ts_col).reset_index(drop=True)

    excl = set([args.ts_col, args.label_col] + (args.id_cols or []))
    base_feature_cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    if not base_feature_cols: raise ValueError("No numeric feature columns found.")

    if args.resample:
        tmp = df.set_index(args.ts_col)
        agg = {c: "mean" for c in base_feature_cols}; agg[args.label_col] = "max"
        df = tmp.resample(args.resample).agg(agg).reset_index()

    if args.label_mode == "pre_event":
        y = make_pre_event_labels(df, args.ts_col, args.label_col,
                                  delta_min=args.label_delta_minutes,
                                  pos_dilate_back_steps=args.pos_dilate_back_steps,
                                  resample_minutes=1)
    else:
        raise NotImplementedError("only pre_event supported")

    total_pos = int(y.sum()); total_all = len(y)
    print(f"[DEBUG] total samples={human_count(total_all)} | positives={human_count(total_pos)}")

    df, feature_cols = build_auto_features(df, base_feature_cols, args.ts_col, bool(args.auto_features))

    seqs_all, _X_dummy, _ = build_windows(df, args.ts_col, feature_cols, y,
                                          seq_len=args.seq_len, stride=args.stride,
                                          pre_event_minutes=args.label_delta_minutes)

    # fit/transform scaler stats using ONLY train rows
    def _fit_transform_by_train_rows(tr_win_indices, feature_cols_in):
        tr_point_mask = np.zeros(len(df), dtype=bool)
        for i_idx in tr_win_indices:
            s = seqs_all[i_idx]; tr_point_mask[s.start:s.end] = True
        train_std = df.loc[tr_point_mask, feature_cols_in].std(numeric_only=True)
        keep_cols = [c for c in feature_cols_in if (c in train_std.index and not pd.isna(train_std[c]) and train_std[c] > 0)]
        dropped = [c for c in feature_cols_in if c not in keep_cols]
        if dropped: print(f"[PREP] drop zero-variance cols on train: {len(dropped)} dropped")
        train_mean = df.loc[tr_point_mask, keep_cols].mean(numeric_only=True)
        df[keep_cols] = df[keep_cols].fillna(train_mean)
        sc_mean = df.loc[tr_point_mask, keep_cols].mean()
        sc_std  = df.loc[tr_point_mask, keep_cols].std().replace(0, 1.0)
        df[keep_cols] = (df[keep_cols] - sc_mean) / sc_std
        X_all_local = df[keep_cols].values.astype(np.float32)
        return keep_cols, X_all_local, sc_mean.to_dict(), sc_std.to_dict()

    # split
    if args.split_mode == "event":
        tr_idx_arr, ho_idx_arr = event_based_split(df, args.ts_col, y, seqs_all,
                                                   pre_event_minutes=args.label_delta_minutes,
                                                   holdout_frac=args.event_holdout_frac,
                                                   holdout_min_events=args.event_holdout_min,
                                                   seed=args.seed,
                                                   extra_min=args.event_holdout_extra_min,
                                                   target_holdout_frac=args.target_holdout_frac,
                                                   target_holdout_max=args.target_holdout_max,
                                                   pos_dilate_back_steps=args.pos_dilate_back_steps,
                                                   tail_min=args.event_holdout_tail_min)
        if tr_idx_arr is None:
            tr_mask_pts, ho_mask_pts = embargo_posaware_split(df, args.ts_col, y,
                holdout_days=args.val_holdout_days, embargo_minutes=args.embargo_minutes,
                min_pos_train=args.min_pos_train, min_pos_holdout=args.min_pos_holdout,
                seqs_for_split=seqs_all, min_train_windows=32, min_holdout_windows=32,
                target_train_frac=args.target_train_frac, train_frac_weight=args.train_frac_weight)
            tr_idx = [i for i, s in enumerate(seqs_all) if tr_mask_pts[s.start] and tr_mask_pts[s.end - 1]]
            ho_idx = [i for i, s in enumerate(seqs_all) if ho_mask_pts[s.start] and ho_mask_pts[s.end - 1]]
            feature_cols, X_all, sc_mean, sc_std = _fit_transform_by_train_rows(tr_idx, feature_cols)
            tr_seqs = [seqs_all[i] for i in tr_idx]; ho_seqs = [seqs_all[i] for i in ho_idx]
        else:
            feature_cols, X_all, sc_mean, sc_std = _fit_transform_by_train_rows(tr_idx_arr.tolist(), feature_cols)
            tr_seqs = [seqs_all[i] for i in tr_idx_arr.tolist()]
            ho_seqs = [seqs_all[i] for i in ho_idx_arr.tolist()]
    else:
        tr_mask_pts, ho_mask_pts = embargo_posaware_split(df, args.ts_col, y,
            holdout_days=args.val_holdout_days, embargo_minutes=args.embargo_minutes,
            min_pos_train=args.min_pos_train, min_pos_holdout=args.min_pos_holdout,
            seqs_for_split=seqs_all, min_train_windows=32, min_holdout_windows=32,
            target_train_frac=args.target_train_frac, train_frac_weight=args.train_frac_weight)
        tr_idx = [i for i, s in enumerate(seqs_all) if tr_mask_pts[s.start] and tr_mask_pts[s.end - 1]]
        ho_idx = [i for i, s in enumerate(seqs_all) if ho_mask_pts[s.start] and ho_mask_pts[s.end - 1]]
        feature_cols, X_all, sc_mean, sc_std = _fit_transform_by_train_rows(tr_idx, feature_cols)
        tr_seqs = [seqs_all[i] for i in tr_idx]; ho_seqs = [seqs_all[i] for i in ho_idx]

    y_tr = np.array([s.label for s in tr_seqs], dtype=int)
    y_ho = np.array([s.label for s in ho_seqs], dtype=int)
    print(f"[SPLIT-FINAL] train windows={human_count(len(tr_seqs))} (pos={human_count(int(y_tr.sum()))}) | holdout windows={human_count(len(ho_seqs))} (pos={human_count(int(y_ho.sum()))})")
    if y_tr.sum() == 0: print("[WARN] Train has 0 positives.")
    if y_ho.sum() == 0: print("[WARN] Holdout has 0 positives. AP/threshold selection may be invalid.")

    X_all = df[feature_cols].values.astype(np.float32)
    ds_tr = WindowDataset(X_all, tr_seqs); ds_ho = WindowDataset(X_all, ho_seqs)
    sampler = HardNegPosSampler(tr_seqs, batch_size=args.batch_size,
                                pos_oversample=args.pos_oversample,
                                hard_neg_frac=args.hard_neg_frac,
                                neg_frac=args.neg_frac, shuffle=True)
    dl_tr = DataLoader(ds_tr, batch_sampler=sampler, num_workers=0)
    dl_ho = DataLoader(ds_ho, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = len(feature_cols)
    if args.model == "gru":
        model = GRUModel(in_dim, hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    else:
        channels = parse_channels(args.tcn_channels)
        model = TCNModel(in_dim, channels=channels, dropout=args.dropout)
    model.to(device)

    npos = int(y_tr.sum()); nneg = int(len(y_tr) - npos)
    pos_weight = (nneg / max(1, npos)) if npos > 0 else 1.0
    alpha_pos = (nneg / max(1, (nneg + npos))) if (args.alpha < 0 and (nneg + npos) > 0) else float(max(0.0, args.alpha))
    print(f"[INFO] Focal params: gamma={args.gamma}, alpha_pos={alpha_pos:.4f}, pos_weight={pos_weight:.3f}")
    focal_loss = FocalLossWithLogits(gamma=args.gamma, alpha_pos=alpha_pos,
                                     label_smoothing=args.label_smoothing, pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(20, args.epochs // 2), eta_min=3e-6)

    warmup_epochs = min(5, max(1, args.epochs // 5))
    best_ap, best_y_hold, best_p_hold = -1.0, None, None
    patience = max(3, int(args.early_stop_patience))
    bad = 0

    for epoch in range(1, args.epochs + 1):
        mixup = args.mixup_alpha if epoch > warmup_epochs else 0.0
        jitter = args.jitter_std if epoch > warmup_epochs else 0.0
        cutout = args.time_cutout_prob if epoch > warmup_epochs else 0.0

        tr_loss = train_one_epoch(model, dl_tr, optim, focal_loss, device,
                                  mixup_alpha=mixup, jitter_std=jitter, time_cutout_prob=cutout,
                                  loss_mode=args.loss_mode, rank_weight=args.rank_weight, rank_margin=args.rank_margin, args=args)
        ap, y_hold_cur, p_hold_cur = evaluate(model, dl_ho, device, args=args)
        ap_str = "nan" if (ap is None or (isinstance(ap, float) and np.isnan(ap))) else f"{ap:.6f}"
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | AP={ap_str}")

        improved = False
        if args.use_best_ap and (ap is not None) and not (isinstance(ap, float) and np.isnan(ap)):
            if ap > best_ap + 1e-4:
                best_ap = ap; best_y_hold = y_hold_cur.copy(); best_p_hold = p_hold_cur.copy(); improved = True
        scheduler.step()
        if improved: bad = 0
        else:
            bad += 1
            if bad >= args.early_stop_patience:
                print(f"[EARLY-STOP] patience={args.early_stop_patience}, best_AP={best_ap:.6f}")
                break

    final_y, final_p = (best_y_hold, best_p_hold) if (args.use_best_ap and (best_y_hold is not None)) \
                        else evaluate(model, dl_ho, device, args=args)[1:3]

    # Optional calibration on holdout
    if final_y is not None and final_p is not None and final_y.sum() >= 0:
        if args.calibration and args.calibration != "none":
            final_p = calibrate_probs(final_p, final_y, method=args.calibration)

        # Save holdout probabilities CSV (dataset/)
        if args.save_probs:
            y_arr = np.asarray(final_y).reshape(-1).astype(int)
            p_arr = np.asarray(final_p).reshape(-1).astype(float)
            assert len(y_arr) == len(p_arr), f"len mismatch y={len(y_arr)} p={len(p_arr)}"
            out = pd.DataFrame({"y_true": y_arr, "p_raw": p_arr, "p_hat": p_arr})
            out.to_csv(args.probs_path, index=False)
            print(f"[SAVE] holdout probabilities → {args.probs_path}")

        # === EVALS: plots + metrics report ===
        _save_eval_plots_and_report(final_y, final_p, args)

    # === SINGLE ARTIFACT SAVE (models/*.pkl) ===
    if int(getattr(args, "save_artifacts", 1)) == 1:
        artifact = {
            "model_type": args.model,
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler_mean": sc_mean,   # dict[col] = mean on train
            "scaler_std": sc_std,     # dict[col] = std on train (zeros replaced with 1.0)
            "args": vars(args),
            "pytorch_version": torch.__version__,
            "seed": args.seed,
        }
        torch.save(artifact, args.artifacts_path)
        print(f"[SAVE] model artifacts (single file) → {args.artifacts_path}")

if __name__ == "__main__":
    import sys
    sys.argv = [
        "train_deep_tcn_clean_v2.py",
        "--input", "dataset/train_features_60m.csv",
        "--model", "tcn",
        "--tcn-channels", "64,64,96",
        "--resample", "",
        "--seq-len", "300",
        "--stride", "1",
        "--label-mode", "pre_event",
        "--label-delta-minutes", "180",
        "--pos-dilate-back-steps", "6",
        "--split-mode", "time",
        "--val-holdout-days", "1.5",
        "--embargo-minutes", "30",
        "--min-pos-holdout", "300",
        "--target-holdout-frac", "0.20",
        "--target-holdout-max", "0.35",
        "--batch-size", "128",
        "--epochs", "20",
        "--lr", "5e-5",
        "--weight-decay", "1e-4",
        "--dropout", "0.25",
        "--pos-oversample", "6",
        "--neg-frac", "0.3",
        "--hard-neg-frac", "0.7",
        "--label-smoothing", "0.01",
        "--loss-mode", "focal+rank",
        "--rank-weight", "0.20",
        "--rank-margin", "0.15",
        "--post-k", "4",
        "--ema-span", "6",
        "--open-th", "0.57",
        "--close-th", "0.33",
        "--cooldown-min", "120",
        "--early-stop-patience", "2",
        "--auto-features", "0",
        "--calibration", "isotonic",
        "--th-objective", "event_f1",
        "--precision-floor", "0.90",
        "--fp-budget", "40",
        "--save-probs", "1",
        "--probs-path", "dataset/holdout_probs.csv",
        "--flip-logits", "1",
        "--seed", "42",
        "--use-best-ap"
    ]
    main()

