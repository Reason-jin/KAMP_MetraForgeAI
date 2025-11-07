
# -*- coding: utf-8 -*-
"""
Hybrid Ensemble Evaluation (Soft-vote & Cascade)
- Uses existing artifacts only (no modification to original training scripts).
- Recreates TCN time-aware split & windows by importing user's module.
- Evaluates:
    * Soft-vote:  P_ens = w_tab * P_tab + w_dl * P_dl
    * Cascade:    P_ens = 1{P_dl >= gate_th} * P_tab

Outputs (per mode) in evals_ensemble/<mode>/ :
    pr_*.png, roc_*.png, hist_*.png, metrics_<mode>.txt, holdout_probs_<mode>.csv

Run examples:
    python hybrid_ensemble_eval_v2.py --mode cascade --gate-th 0.55
    python hybrid_ensemble_eval_v2.py --mode soft --w-tab 0.7 --w-dl 0.3

Expected files:
- dataset/train_features_60m.csv    (has columns: datetime, passorfail, [features...])
- dataset/holdout_probs.csv         (TCN holdout probabilities; prefer column 'p_hat')
- models/model_RF_calibrated.pkl
- models/model_LightGBM.pkl
- train_deep_tcn_clean_v2_evals.py  (imported for labeling/splitting utilities)
"""

import os, sys, warnings, argparse
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, auc,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)

import importlib.util
import joblib

# ------------------------
# Default Config
# ------------------------
CFG = dict(
    data_path="dataset/train_features_60m.csv",
    probs_path_candidates=["dataset/holdout_probs.csv", "holdout_probs.csv"],
    ts_col="datetime",
    label_col="passorfail",
    id_cols=["_grp"],
    # window/label settings (mirror the user's run)
    seq_len=300, stride=1,
    label_mode="pre_event",
    label_delta_minutes=180,
    pos_dilate_back_steps=6,
    resample="",
    # split
    val_holdout_days=1.5,
    embargo_minutes=30,
    min_pos_train=10,
    min_pos_holdout=300,
    target_train_frac=0.7,
    train_frac_weight=10.0,
    # policy / thresholds
    post_k=4,
    open_th=0.57,
    close_th=0.33,
    cooldown_min=120,
    # soft-vote weights
    w_tab=0.6,
    w_dl=0.4,
    # cascade gate
    gate_th=0.5,
    # selection constraint
    precision_floor=0.90,
    # output
    out_root="evals_ensemble",
)

# ------------------------
# Utilities
# ------------------------
def _import_user_module(py_path: str):
    spec = importlib.util.spec_from_file_location("user_tcn_module", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def load_base_dataframe(path: str, ts_col: str, label_col: str, id_cols: list) -> Tuple[pd.DataFrame, list]:
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"ts_col '{ts_col}' not found in {path}")
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in {path}")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    excl = set([ts_col, label_col] + (id_cols or []))
    feat_cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("No numeric feature columns found for tabular models.")
    return df, feat_cols

def make_labels_and_windows(tcn, df: pd.DataFrame, feat_cols: list, cfg: dict):
    if cfg["resample"]:
        tmp = df.set_index(cfg["ts_col"])
        agg = {c: "mean" for c in feat_cols}; agg[cfg["label_col"]] = "max"
        df = tmp.resample(cfg["resample"]).agg(agg).reset_index()
        excl = set([cfg["ts_col"], cfg["label_col"]] + (cfg["id_cols"] or []))
        feat_cols = [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]

    if cfg["label_mode"] != "pre_event":
        raise NotImplementedError("only pre_event supported")
    y = tcn.make_pre_event_labels(
        df, cfg["ts_col"], cfg["label_col"],
        delta_min=cfg["label_delta_minutes"],
        pos_dilate_back_steps=cfg["pos_dilate_back_steps"],
        resample_minutes=1
    )

    seqs_all, X_dummy, ts_min = tcn.build_windows(
        df, cfg["ts_col"], feat_cols, y,
        seq_len=cfg["seq_len"], stride=cfg["stride"],
        pre_event_minutes=cfg["label_delta_minutes"]
    )

    tr_mask_pts, ho_mask_pts = tcn.embargo_posaware_split(
        df, cfg["ts_col"], y,
        holdout_days=cfg["val_holdout_days"], embargo_minutes=cfg["embargo_minutes"],
        min_pos_train=cfg["min_pos_train"], min_pos_holdout=cfg["min_pos_holdout"],
        seqs_for_split=seqs_all, min_train_windows=32, min_holdout_windows=32,
        target_train_frac=cfg["target_train_frac"], train_frac_weight=cfg["train_frac_weight"]
    )
    tr_idx = [i for i, s in enumerate(seqs_all) if tr_mask_pts[s.start] and tr_mask_pts[s.end - 1]]
    ho_idx = [i for i, s in enumerate(seqs_all) if ho_mask_pts[s.start] and ho_mask_pts[s.end - 1]]
    win_end_row = np.array([s.end - 1 for s in seqs_all], dtype=int)
    ho_end_rows = win_end_row[ho_idx]

    return df, feat_cols, y, seqs_all, tr_idx, ho_idx, ho_end_rows

def _select_feature_columns(df: pd.DataFrame, model, cfg):
    if hasattr(model, "feature_names_in_"):
        cols = [c for c in model.feature_names_in_ if c in df.columns]
        if cols: return cols
    if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        try:
            names = list(model.booster_.feature_name())
            cols = [c for c in names if c in df.columns]
            if cols: return cols
        except Exception:
            pass
    excl = set([cfg["ts_col"], cfg["label_col"]] + (cfg["id_cols"] or []))
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]

def predict_proba_tabular(df_hold: pd.DataFrame, cfg: dict):
    # load models
    rf_path  = "models/model_RF_calibrated.pkl" if Path("models/model_RF_calibrated.pkl").exists() else "model_RF_calibrated.pkl"
    lgb_path = "models/model_LightGBM.pkl"      if Path("models/model_LightGBM.pkl").exists()      else "model_LightGBM.pkl"
    models = []
    if Path(rf_path).exists():
        try: models.append(joblib.load(rf_path))
        except Exception as e: warnings.warn(f"RF load failed: {e}")
    if Path(lgb_path).exists():
        try: models.append(joblib.load(lgb_path))
        except Exception as e: warnings.warn(f"LGBM load failed: {e}")
    if not models:
        raise FileNotFoundError("No tabular models could be loaded.")

    feat_sets = [set(_select_feature_columns(df_hold, m, cfg)) for m in models]
    common_feats = set.intersection(*feat_sets) if feat_sets else set()
    if not common_feats: common_feats = feat_sets[0]
    cols = sorted(list(common_feats))

    X = df_hold[cols].copy()
    imp = SimpleImputer(strategy="median")
    X_imp_arr = imp.fit_transform(X)
    # restore column names to remove LGBM warnings
    X_imp = pd.DataFrame(X_imp_arr, columns=X.columns, index=X.index)

    probs = []
    for m in models:
        p = m.predict_proba(X_imp)
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        probs.append(p.astype(float))
    P_tab = np.mean(np.vstack(probs), axis=0) if len(probs) > 1 else probs[0]
    return P_tab, cols

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

def save_eval_plots(y_true, p_tab, p_dl, p_ens, outdir: Path, tag: str):
    outdir.mkdir(parents=True, exist_ok=True)
    # PR
    for name, p in [("tab", p_tab), ("dl", p_dl), (tag, p_ens)]:
        prec, rec, _ = precision_recall_curve(y_true, p)
        au = auc(rec, prec) if len(rec) > 1 else float("nan")
        plt.figure()
        plt.step(rec, prec, where="post")
        base = (np.asarray(y_true)==1).mean()
        plt.hlines(base, 0, 1, linestyles="dashed")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve [{name}] AUC={au:.4f} (baseline={base:.3f})")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(outdir / f"pr_{name}.png", dpi=150); plt.close()
    # ROC
    for name, p in [("tab", p_tab), ("dl", p_dl), (tag, p_ens)]:
        if (np.max(y_true) != np.min(y_true)):
            fpr, tpr, _ = roc_curve(y_true, p)
            aucv = roc_auc_score(y_true, p)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={aucv:.4f}")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title(f"ROC Curve [{name}]")
            plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(outdir / f"roc_{name}.png", dpi=150); plt.close()
    # Hist
    for name, p in [("tab", p_tab), ("dl", p_dl), (tag, p_ens)]:
        plt.figure()
        y = np.asarray(y_true).astype(int)
        if (y==1).any(): plt.hist(p[y==1], bins=30, alpha=0.6, label="pos")
        if (y==0).any(): plt.hist(p[y==0], bins=30, alpha=0.6, label="neg")
        plt.xlabel("Predicted probability"); plt.ylabel("Count")
        plt.title(f"Probability Histogram [{name}]"); plt.legend(); plt.tight_layout()
        plt.savefig(outdir / f"hist_{name}.png", dpi=150); plt.close()

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["soft", "cascade"], default="soft")
    parser.add_argument("--w-tab", type=float, default=CFG["w_tab"])
    parser.add_argument("--w-dl", type=float, default=CFG["w_dl"])
    parser.add_argument("--gate-th", type=float, default=CFG["gate_th"])
    parser.add_argument("--precision-floor", type=float, default=CFG["precision_floor"])
    parser.add_argument("--out-root", type=str, default=CFG["out_root"])
    args = parser.parse_args()

    # merge cfg with args
    CFG["w_tab"] = args.w_tab
    CFG["w_dl"] = args.w_dl
    CFG["gate_th"] = args.gate_th
    CFG["precision_floor"] = args.precision_floor
    out_root = Path(args.out_root)
    outdir = out_root / args.mode
    outdir.mkdir(parents=True, exist_ok=True)

    # Import user's TCN utils
    user_mod_path = "train_deep_tcn_clean_v2_evals.py"
    if not Path(user_mod_path).exists():
        user_mod_path = str(Path(__file__).parent / "train_deep_tcn_clean_v2_evals.py")
    tcn = _import_user_module(user_mod_path)

    # Load df and reproduce windows/split
    df, feat_cols = load_base_dataframe(CFG["data_path"], CFG["ts_col"], CFG["label_col"], CFG["id_cols"])
    df2, feat_cols2, y, seqs_all, tr_idx, ho_idx, ho_end_rows = make_labels_and_windows(tcn, df.copy(), feat_cols, dict(
        ts_col=CFG["ts_col"], label_col=CFG["label_col"], id_cols=CFG["id_cols"],
        resample=CFG["resample"],
        label_mode=CFG["label_mode"],
        label_delta_minutes=CFG["label_delta_minutes"],
        pos_dilate_back_steps=CFG["pos_dilate_back_steps"],
        seq_len=CFG["seq_len"], stride=CFG["stride"],
        val_holdout_days=CFG["val_holdout_days"],
        embargo_minutes=CFG["embargo_minutes"],
        min_pos_train=CFG["min_pos_train"], min_pos_holdout=CFG["min_pos_holdout"],
        target_train_frac=CFG["target_train_frac"], train_frac_weight=CFG["train_frac_weight"],
    ))

    # Holdout labels/timestamps
    y_hold = np.array([seqs_all[i].label for i in ho_idx], dtype=int)
    ts_vals = df2[CFG["ts_col"]].values.astype("datetime64[ns]")
    ho_end_ts = ts_vals[ho_end_rows]

    # Load TCN holdout probs
    probs_path = None
    for cand in CFG["probs_path_candidates"]:
        if Path(cand).exists():
            probs_path = cand; break
    if probs_path is None:
        raise FileNotFoundError("holdout_probs.csv not found in candidates: " + ", ".join(CFG["probs_path_candidates"]))
    tcn_df = pd.read_csv(probs_path)
    p_dl = tcn_df["p_hat"].to_numpy() if "p_hat" in tcn_df.columns else tcn_df.iloc[:, -1].to_numpy()

    # Truncate to min length if mismatch
    n = min(len(p_dl), len(y_hold))
    if len(p_dl) != len(y_hold):
        warnings.warn(f"Length mismatch: TCN probs {len(p_dl)} vs holdout windows {len(y_hold)} → truncating to {n}")
    p_dl = p_dl[:n]; y_hold = y_hold[:n]; ho_end_ts = ho_end_ts[:n]
    hold_rows = ho_end_rows[:n]
    df_hold = df2.iloc[hold_rows].copy()

    # Tabular probabilities
    p_tab, used_cols = predict_proba_tabular(df_hold, dict(ts_col=CFG["ts_col"], label_col=CFG["label_col"], id_cols=CFG["id_cols"]))
    p_tab = p_tab[:n]

    # Combine
    mode = args.mode
    if mode == "cascade":
        gate_th = CFG["gate_th"]
        mask = (p_dl >= gate_th).astype(float)
        p_ens = mask * p_tab
        print(f"[CASCADE] gate_th={gate_th:.3f} | suspicious windows={int(mask.sum())}/{len(mask)}")
        tag = "cascade"
    else:
        p_ens = (CFG["w_tab"] * p_tab + CFG["w_dl"] * p_dl).astype(float)
        tag = "soft"

    # Postprocess
    yb_hys = hysteresis_cooldown(p_ens, ho_end_ts, CFG["open_th"], CFG["close_th"], CFG["cooldown_min"])
    yb = k_consecutive_filter(yb_hys, CFG["post_k"])

    # Metrics
    ap_tab = average_precision_score(y_hold, p_tab)
    ap_dl  = average_precision_score(y_hold, p_dl)
    ap_ens = average_precision_score(y_hold, p_ens)

    # Threshold selection under precision floor
    prec, rec, thr = precision_recall_curve(y_hold, p_ens)
    thr = np.append(thr, 1.0)
    best_f1, best_t = -1.0, None
    for t in thr:
        yy = (p_ens >= t).astype(int)
        yy = k_consecutive_filter(yy, CFG["post_k"])
        P = precision_score(y_hold, yy, zero_division=0)
        R = recall_score(y_hold, yy, zero_division=0)
        if CFG["precision_floor"] is not None and P < CFG["precision_floor"]:
            continue
        f1v = (2*P*R/(P+R)) if (P+R)>0 else 0.0
        if f1v > best_f1:
            best_f1, best_t = f1v, t

    # Save plots/metrics
    save_eval_plots(y_hold, p_tab, p_dl, p_ens, outdir, tag)

    lines = []
    lines.append(f"Mode: {mode}")
    lines.append(f"AP_tab: {ap_tab:.6f}")
    lines.append(f"AP_dl: {ap_dl:.6f}")
    lines.append(f"AP_{tag}: {ap_ens:.6f}")
    if best_t is not None:
        yy = (p_ens >= best_t).astype(int)
        yy = k_consecutive_filter(yy, CFG["post_k"])
        P = precision_score(y_hold, yy, zero_division=0)
        R = recall_score(y_hold, yy, zero_division=0)
        F = f1_score(y_hold, yy, zero_division=0)
        lines.append(f"Best_threshold_under_precision_floor({CFG['precision_floor']}): {best_t:.6f}")
        lines.append(f"Precision@best: {P:.6f}")
        lines.append(f"Recall@best: {R:.6f}")
        lines.append(f"F1@best: {F:.6f}")
    else:
        lines.append("No threshold met the precision_floor constraint; consider relaxing the floor or adjust weights.")

    with open(outdir / f"metrics_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    dbg = pd.DataFrame({
        "y_true": y_hold,
        "p_tab": p_tab,
        "p_dl": p_dl,
        "p_ens": p_ens,
        CFG["ts_col"]: ho_end_ts.astype("datetime64[ns]")
    })
    dbg.to_csv(outdir / f"holdout_probs_{tag}.csv", index=False)

    print(f"[{mode.upper()}] Saved plots & metrics to {outdir}")

if __name__ == "__main__":
    main()
