"""
KAMP 소성가공 품질보증 AI — End-to-End 파이프라인 (라이트)
=======================================================
목적
----
- **검증 산출물(results/*) 전부 제거**하고, 모델링에 필요한 파일만 생성한다.
- 최종 산출물은 **dataset/** 폴더에 딱 2개:
  1) `final_prepared.csv` — *풀버전과 결과가 1:1로 동일*
  2) `train_features_{window}m.csv` — rolling-window 피처

실행 예시(한 줄)
----------------
python kamp_pipeline_lite.py --stage all --input "2. 소성가공 품질보증 AI 데이터셋.csv" --dataset-dir dataset --freq 5s --normalize-labels --drop-na-labels --window 60 --features-output dataset/train_features_60m.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# =====================================
# 공통 유틸
# =====================================

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def read_frame(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig")

def write_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")

def log(msg: str):
    print(f"[KAMP-LITE] {msg}")

# =====================================
# CLI
# =====================================

def build_parser():
    p = argparse.ArgumentParser(description="KAMP End-to-End Pipeline (Lite)")
    # 단계 선택
    p.add_argument("--stage", choices=["preprocess", "features", "all"], required=True,
                   help="실행 단계 선택: preprocess | features | all")
    # 전처리 옵션 (풀버전과 동일 로직 보장)
    p.add_argument("--input", default="2. 소성가공 품질보증 AI 데이터셋.csv", help="입력 CSV/Parquet 경로")
    p.add_argument("--dataset-dir", default="dataset", help="모델링용 최종 CSV 저장 폴더")
    p.add_argument("--label-col", default="passorfail", help="라벨 컬럼명")
    p.add_argument("--freq", default="5s", help="리샘플 간격(예: 5s, 1min). 내부에서 소문자 강제")
    p.add_argument("--gap-thr", type=float, default=10.0, help="누락 gap 임계(초)")
    p.add_argument("--normalize-labels", action="store_true", help="라벨 pass/fail/True/False → 1/0 정규화")
    p.add_argument("--drop-na-labels", action="store_true", help="정규화 후 라벨 NaN 행 제거")
    p.add_argument("--make-splits", action="store_true", help="dataset/train.csv,val.csv,test.csv 분할 생성")
    p.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test 비율 (합=1)")
    # 피처 옵션
    p.add_argument("--features-input", default="dataset/final_prepared.csv",
                   help="피처 생성 입력 CSV (기본: dataset/final_prepared.csv)")
    p.add_argument("--window", type=int, default=60, help="rolling window 크기(분 단위)")
    p.add_argument("--features-output", default=None,
                   help="피처 생성 결과 CSV (기본: dataset/train_features_{window}m.csv)")
    return p

# =====================================
# STEP0: 클렌징 (풀버전과 동일 동작)
# =====================================

def step0_clean_like_full(input_path: str | Path, label_col: str = "passorfail",
                          gap_thr: float = 10.0, force_grid: bool = True, freq: str = "5s"):
    """시간 정렬 → 중복 평균 병합 → 규칙격자 리샘플(ffill 1-step). (풀버전과 동일)
    반환: (df_all, df_labeled) — 둘 다 datetime 포함. 파일 저장 없음(라이트 정책).
    """
    log("STEP0: 시간 정렬 및 클렌징 (full 동작 복제)")
    df = read_frame(input_path)

    # datetime 파싱 (date 또는 datetime 열 지원)
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], format="%Y %m %d %H:%M:%S", errors="coerce")
        df = df.drop(columns=["date"])  # 원본 date 제거
    elif "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.drop(columns=["datetime"])  # 새로 넣기 위해 제거
    else:
        raise KeyError("입력 데이터에 'date' 또는 'datetime' 컬럼이 없습니다.")

    df.insert(0, "datetime", dt)
    df = df.sort_values("datetime").reset_index(drop=True)

    # 중복 timestamp 평균 병합(숫자열만)
    if df["datetime"].duplicated().any():
        df = df.groupby("datetime", as_index=False).mean(numeric_only=True)
        log("중복 timestamp → 평균 병합")

    # 누락 gap 검증(라이트는 파일 미저장)
    df["time_diff"] = df["datetime"].diff().dt.total_seconds()

    # 규칙 격자 보정 (freq 소문자 강제, ffill limit=1)
    df = df.set_index("datetime").sort_index()
    if force_grid:
        freq = str(freq).lower()
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
        df = df.reindex(full_idx)
        df = df.ffill(limit=1)
        df.index.name = "datetime"
        log(f"{freq} 격자 보정 + ffill(limit=1)")

    # 라벨 존재 기준 분기(풀버전과 동일)
    has_label = df[label_col].notna() if label_col in df.columns else pd.Series(True, index=df.index)
    df_labeled = df[has_label].reset_index()
    df_all = df.reset_index()
    return df_all, df_labeled

# =====================================
# STEP4: 그룹핑 피처 (풀버전과 동일 계산식 — 검증 산출 제거)
# =====================================

def step4_feature_grouping_like_full(df_labeled: pd.DataFrame, label_col: str = "passorfail") -> pd.DataFrame:
    log("STEP4: 그룹핑 및 피처 생성 (full 동작 복제)")
    groups = {
        "G1_ZoneTemp": ["EX1.Z1_PV", "EX1.Z2_PV", "EX1.Z3_PV", "EX1.Z4_PV"],
        "G2_HeadTemp": ["EX1.H1_PV", "EX1.H2_PV", "EX1.H3_PV", "EX1.H4_PV", "EX1.A1_PV", "EX1.A2_PV"],
        "G3_MeltTemp": ["EX2.MELT_TEMP", "EX3.MELT_TEMP", "EX4.MELT_TEMP", "EX5.MELT_TEMP"],
        "G4_PressureTorque": ["EX1.MELT_P_PV", "EX1.MD_PV", "EX1.MD_TQ"],
        "G5_CoolFlow": ["EX1.H2O_PV"],
    }
    df = df_labeled.copy()
    dt_series = pd.to_datetime(df["datetime"], errors="coerce") if "datetime" in df.columns else None
    num_df = df.select_dtypes(include=[np.number])
    feats = pd.DataFrame(index=num_df.index)
    for gname, cols in groups.items():
        existing = [c for c in cols if c in num_df.columns]
        if not existing:
            continue
        sub = num_df[existing]
        feats[f"{gname}_mean"] = sub.mean(axis=1)
        feats[f"{gname}_std"] = sub.std(axis=1, ddof=0)
        feats[f"{gname}_range"] = sub.max(axis=1) - sub.min(axis=1)
    if label_col in num_df.columns:
        feats[label_col] = num_df[label_col].values
    else:
        feats[label_col] = np.nan
    if dt_series is not None:
        feats.insert(0, "datetime", dt_series.values)
    return feats

# =====================================
# Dataset 저장 + (옵션) 분할 — 풀버전과 동일 동작
# =====================================

def normalize_labels_inplace(df: pd.DataFrame, label_col: str) -> None:
    if label_col in df.columns:
        df[label_col] = (
            df[label_col]
            .replace({"pass":1,"Pass":1,"PASS":1, True:1,
                      "fail":0,"Fail":0,"FAIL":0, False:0})
            .pipe(pd.to_numeric, errors="coerce")
        )

def save_dataset_like_full(feats: pd.DataFrame, dataset_dir: Path, normalize_labels: bool, drop_na_labels: bool, label_col: str):
    df = feats.copy()
    if normalize_labels:
        normalize_labels_inplace(df, label_col)
        log("라벨 정규화 완료 (pass/fail → 1/0)")
    if drop_na_labels and label_col in df.columns:
        n0 = len(df)
        df = df.dropna(subset=[label_col])
        log(f"라벨 NaN 제거: {n0 - len(df)}행 drop")
    write_df(df, dataset_dir / "final_prepared.csv")
    log("저장: dataset/final_prepared.csv")
    return df

def make_splits_like_full(df: pd.DataFrame, dataset_dir: Path, ratio_str: str = "0.8,0.1,0.1"):
    r = [float(x) for x in ratio_str.split(",")]
    assert abs(sum(r) - 1.0) < 1e-6, "--split 비율 합은 1이어야 함"
    n = len(df)
    n_train = int(n * r[0])
    n_val = int(n * r[1])
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    write_df(train, dataset_dir / "train.csv")
    write_df(val, dataset_dir / "val.csv")
    write_df(test, dataset_dir / "test.csv")
    log(f"splits → train:{len(train)}, val:{len(val)}, test:{len(test)}")

# =====================================
# Rolling-window 피처 (안전한 라벨 캐스팅 포함)
# =====================================

def _safe_numeric_cols(df):
    drop = {"datetime", "passorfail"}
    num_cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    return num_cols

def _time_index(df):
    if "datetime" not in df.columns:
        raise ValueError("`datetime` 컬럼이 필요함.")
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        df = df.dropna(subset=["datetime"])  # 깨진 날짜 drop
    df = df.sort_values("datetime").set_index("datetime")
    return df

def _rolling_features(df, cols, window_min):
    from scipy.stats import linregress
    w = f"{window_min}min"
    out = pd.DataFrame(index=df.index)
    agg_funcs = {"mean":"mean","std":"std","min":"min","max":"max","skew":"skew","kurt":"kurt"}
    for stat_name, func in agg_funcs.items():
        tmp = df[cols].rolling(w, min_periods=2).agg(func)
        tmp.columns = [f"{c}__{stat_name}__{window_min}m" for c in tmp.columns]
        out = out.join(tmp)
    rmax = df[cols].rolling(w, min_periods=2).max()
    rmin = df[cols].rolling(w, min_periods=2).min()
    rng = rmax - rmin; rng.columns = [f"{c}__range__{window_min}m" for c in cols]
    out = out.join(rng)
    first = df[cols].rolling(w, min_periods=2).apply(lambda x: x[0], raw=False)
    delta = df[cols] - first; delta.columns = [f"{c}__delta__{window_min}m" for c in cols]
    out = out.join(delta)
    def slope_over_time(y, times):
        if len(y) < 2: return np.nan
        t = (times.view("int64") // 10**9)
        return linregress(t, y).slope
    slope_df = pd.DataFrame(index=df.index)
    idx = df.index
    for c in cols:
        series = df[c]
        vals = []
        for i in range(len(series)):
            end_time = idx[i]
            start_time = end_time - pd.Timedelta(minutes=window_min)
            win = series.loc[start_time:end_time]
            vals.append(slope_over_time(win.values, win.index) if win.shape[0] >= 2 else np.nan)
        slope_df[c] = vals
    slope_df.columns = [f"{c}__slope__{window_min}m" for c in cols]
    out = out.join(slope_df)
    return out

def run_features(features_input: str | Path, window_min: int, features_output: str | None):
    inp = Path(features_input)
    if not inp.exists():
        raise FileNotFoundError(f"입력 파일 없음: {inp}")
    out_path = Path(features_output or f"dataset/train_features_{window_min}m.csv")
    base = pd.read_csv(inp)
    if "passorfail" not in base.columns:
        raise ValueError("`passorfail` 라벨 컬럼이 필요함.")
    df_idxed = _time_index(base)
    lab = df_idxed["passorfail"].replace({"pass":1,"Pass":1,"PASS":1, True:1,
                                          "fail":0,"Fail":0,"FAIL":0, False:0})
    lab = pd.to_numeric(lab, errors="coerce")
    mask = lab.notna()
    df_idxed = df_idxed.loc[mask]
    label = lab.loc[mask].astype(int)
    num_cols = _safe_numeric_cols(df_idxed.reset_index())
    if len(num_cols) == 0:
        raise ValueError("수치 피처 컬럼을 찾지 못함. 입력 CSV에 numeric 피처가 있는지 확인.")
    feats = _rolling_features(df_idxed, num_cols, window_min)
    out = feats.copy(); out["datetime"] = out.index; out = out.reset_index(drop=True)
    ydf = label.to_frame("passorfail").copy(); ydf["datetime"] = ydf.index; ydf = ydf.reset_index(drop=True)
    out = pd.merge_asof(out.sort_values("datetime"), ydf.sort_values("datetime"), on="datetime", direction="nearest")
    if out.isna().mean().mean() > 0.3:
        out = out.dropna()
    out.to_csv(out_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    log(f"✅ Saved features: {out_path}  (rows={len(out)}, cols={out.shape[1]})")

# =====================================
# MAIN
# =====================================

def main():
    args = build_parser().parse_args()
    dataset_dir = ensure_dir(args.dataset_dir)

    if args.stage in ("preprocess", "all"):
        df_all, df_labeled = step0_clean_like_full(
            input_path=args.input,
            label_col=args.label_col,
            gap_thr=args.gap_thr,
            force_grid=True,
            freq=args.freq,
        )
        feats = step4_feature_grouping_like_full(df_labeled, label_col=args.label_col)
        final = save_dataset_like_full(
            feats,
            dataset_dir=dataset_dir,
            normalize_labels=args.normalize_labels,
            drop_na_labels=args.drop_na_labels,
            label_col=args.label_col,
        )
        if args.make_splits:
            make_splits_like_full(final, dataset_dir, args.split)
        log("[stage=preprocess] 완료: dataset/final_prepared.csv")

    if args.stage in ("features", "all"):
        run_features(
            features_input=args.features_input,
            window_min=args.window,
            features_output=args.features_output,
        )
        log("[stage=features] 완료: rolling window features 저장")

    log("✅ LITE 파이프라인 종료")

if __name__ == "__main__":
    main()

