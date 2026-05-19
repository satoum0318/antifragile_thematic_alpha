# -*- coding: utf-8 -*-
"""
backtest_ic.py — シンプル forward-return / IC バックテスト

目的:
- JQuamtsScreeningBot.py / real_data_screening.py が出力する
  `screening_offline_*.csv` (スナップショット) を入力に、各銘柄の
  凍結価格キャッシュ (.jquants_cache_v2/frozen/prices/<code>.csv)
  と突き合わせて forward 21/63/126 営業日リターンを計算する。
- 各スコア (entry_score / fundamental_edge_score / total_score /
  piotroski_effective_score / return_252d 等) と forward リターン
  との Spearman IC、レーン別の勝率・平均リターン・最大DD を出力する。

使い方:
    python backtest_ic.py \
        --master output/reports/screening_offline_20260515_091919.csv \
        --asof 2026-05-15 \
        --horizons 21,63,126

複数の screening_offline_*.csv を渡すと、各日付ごとに集計しつつ
全体のプール IC も合算する。`--asof` を省略した場合はファイル名から
タイムスタンプを抽出する (例: screening_offline_YYYYMMDD_HHMMSS.csv)。

出力:
- `output/backtest/ic_summary.csv`            … スコア × ホライズンごとの IC
- `output/backtest/lane_stats.csv`            … レーン × ホライズンの勝率/平均/最大DD
- `output/backtest/per_stock_returns.csv`     … 銘柄 × ホライズン の forward return（明細）
- `output/backtest/run_meta.json`             … 実行メタ

NOTE:
- 「真の forward return」を測るためには現時点よりも `max(horizons)` 営業日以上
  古い `--asof` を指定すること（直近すぎると forward が欠損で計算されない）。
- 株式分割・配当再投資は調整しない（J-Quants v2 価格列が AdjustmentClose を
  含む場合は優先採用する）。
- スピアマン相関は scipy が無くても pandas.rank で計算可能なので外部依存は
  最小限。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd


CACHE_DIR = Path(".jquants_cache_v2")
FROZEN_PRICES_DIR = CACHE_DIR / "frozen" / "prices"
DEFAULT_OUT_DIR = Path("output") / "backtest"

DEFAULT_SCORE_COLUMNS = [
    "entry_score",
    "fundamental_edge_score",
    "total_score",
    "piotroski_effective_score",
    "piotroski_adjusted_score",
    "ps_vs_sector_pre",
    "return_252d",
    "momentum_6m_1m",
    "return_12m_1m",
    "ma200_timing_score",
    "safety_criteria_score",
    "spec_score",
]

DEFAULT_LANE_COLUMN = "candidate_lane"


def _parse_date(s: str) -> datetime.date:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"日付の形式が不明: {s}")


def _infer_asof_from_filename(path: Path) -> Optional[datetime.date]:
    m = re.search(r"screening_offline_(\d{8})_\d{6}\.csv", path.name)
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def _pick_close_column(df: pd.DataFrame) -> Optional[str]:
    candidates = (
        "AdjustmentClose",
        "AdjClose",
        "adjustment_close",
        "Close",
        "ClosePrice",
        "EndPrice",
        "close",
    )
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ("Date", "TradingDate", "date", "trade_date")
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def _load_prices_for_code(code: str) -> Optional[pd.DataFrame]:
    code = str(code).strip()
    p = FROZEN_PRICES_DIR / f"{code}.csv"
    if not p.exists():
        p2 = FROZEN_PRICES_DIR / f"{int(code):04d}.csv" if code.isdigit() else None
        if p2 is None or not p2.exists():
            return None
        p = p2
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if df.empty:
        return None
    cdate = _pick_date_column(df)
    cclose = _pick_close_column(df)
    if cdate is None or cclose is None:
        return None
    df = df[[cdate, cclose]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def _forward_return_business_days(
    prices: pd.DataFrame, asof: datetime.date, horizon: int
) -> Optional[float]:
    """asof より厳密に後 (>) の最初の営業日を起点とし、そこから horizon 営業日後の
    終値を用いて単純リターンを算出する。"""
    if prices is None or prices.empty:
        return None
    sub = prices[prices["date"] > asof]
    if sub.empty or len(sub) <= horizon:
        return None
    base = sub.iloc[0]["close"]
    target = sub.iloc[horizon - 1]["close"]  # horizon 営業日後の終値（=horizon-1 オフセット）
    if base is None or target is None or base <= 0:
        return None
    try:
        return float(target) / float(base) - 1.0
    except (TypeError, ZeroDivisionError):
        return None


def _spearman_ic(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], int]:
    """欠損を除いた上でスピアマン相関を返す。N も同時に返す。"""
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                       "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(df) < 5:
        return None, len(df)
    rx = df["x"].rank(method="average")
    ry = df["y"].rank(method="average")
    if rx.std(ddof=0) == 0 or ry.std(ddof=0) == 0:
        return None, len(df)
    val = float(rx.corr(ry, method="pearson"))
    if not np.isfinite(val):
        return None, len(df)
    return round(val, 4), len(df)


def _max_drawdown(returns: pd.Series) -> Optional[float]:
    """単純な等ウェイトポートフォリオを仮定し、累積リターン系列の最大DDを返す。
    入力は同一スナップショット時点の cross-section リターンなので、
    擬似的に「ソート順に逐次加算する」感覚で、ここではシンプルに
    `(min - mean)` を Drawdown 指標として返す（cross-section）。"""
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.min())


def _lane_stats_single(
    snap: pd.DataFrame,
    lane_col: str,
    horizon_col: str,
) -> pd.DataFrame:
    if lane_col not in snap.columns or horizon_col not in snap.columns:
        return pd.DataFrame()
    g = snap.groupby(lane_col)[horizon_col]
    rows: List[Dict[str, Any]] = []
    for lane, vals in g:
        v = pd.to_numeric(vals, errors="coerce").dropna()
        if v.empty:
            continue
        rows.append({
            "lane": str(lane),
            "horizon_col": horizon_col,
            "n": int(len(v)),
            "hit_rate": float((v > 0).mean()),
            "mean_return": float(v.mean()),
            "median_return": float(v.median()),
            "stdev": float(v.std(ddof=0)) if len(v) > 1 else 0.0,
            "worst": float(v.min()),
            "best": float(v.max()),
        })
    return pd.DataFrame(rows)


def run_single_snapshot(
    master_csv: Path,
    asof: Optional[datetime.date],
    horizons: List[int],
    score_cols: List[str],
    lane_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1つの screening_offline_*.csv について、forward returns / IC / lane_stats を計算する。

    戻り値:
      - per_stock_returns: 銘柄 × horizon 行の forward return 明細
      - ic_summary: スコア × horizon の IC
      - lane_stats: レーン × horizon の集計
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"master_csv not found: {master_csv}")
    df = pd.read_csv(master_csv, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "code" not in df.columns:
        raise ValueError("master_csv に code 列がありません (flatten 後の CSV を渡してください)")

    asof_eff = asof or _infer_asof_from_filename(master_csv) or datetime.date.today()

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        code = str(r.get("code") or "").strip()
        if not code:
            continue
        prices = _load_prices_for_code(code)
        if prices is None:
            continue
        rec: Dict[str, Any] = {
            "code": code,
            "name": r.get("name"),
            "asof": asof_eff.isoformat(),
            "snapshot": master_csv.name,
        }
        for h in horizons:
            rec[f"fwd_ret_{h}d"] = _forward_return_business_days(prices, asof_eff, h)
        rows.append(rec)

    fwd = pd.DataFrame(rows)
    if fwd.empty:
        return fwd, pd.DataFrame(), pd.DataFrame()

    merge_cols = [c for c in score_cols if c in df.columns]
    snap_cols = ["code"] + [lane_col] if lane_col in df.columns else ["code"]
    snap_cols += merge_cols
    snap = df[snap_cols].copy()
    snap["code"] = snap["code"].astype(str).str.strip()
    merged = fwd.merge(snap, on="code", how="left")

    ic_rows: List[Dict[str, Any]] = []
    for sc in merge_cols:
        for h in horizons:
            yname = f"fwd_ret_{h}d"
            ic, n = _spearman_ic(merged[sc], merged[yname])
            ic_rows.append({
                "asof": asof_eff.isoformat(),
                "snapshot": master_csv.name,
                "score": sc,
                "horizon_days": h,
                "n": n,
                "ic_spearman": ic,
            })
    ic_summary = pd.DataFrame(ic_rows)

    lane_rows: List[pd.DataFrame] = []
    if lane_col in merged.columns:
        for h in horizons:
            lane_rows.append(_lane_stats_single(merged, lane_col, f"fwd_ret_{h}d"))
    lane_stats = (
        pd.concat([x for x in lane_rows if not x.empty], ignore_index=True)
        if lane_rows
        else pd.DataFrame()
    )
    if not lane_stats.empty:
        lane_stats.insert(0, "snapshot", master_csv.name)
        lane_stats.insert(0, "asof", asof_eff.isoformat())

    return merged, ic_summary, lane_stats


def pool_ic_summary(ic_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not ic_frames:
        return pd.DataFrame()
    df = pd.concat(ic_frames, ignore_index=True)
    if df.empty:
        return df
    out = (
        df.dropna(subset=["ic_spearman"])
        .groupby(["score", "horizon_days"], as_index=False)
        .agg(
            n_snapshots=("ic_spearman", "size"),
            mean_ic=("ic_spearman", "mean"),
            median_ic=("ic_spearman", "median"),
            pos_ratio=("ic_spearman", lambda s: float((s > 0).mean())),
            total_n=("n", "sum"),
        )
    )
    out["mean_ic"] = out["mean_ic"].round(4)
    out["median_ic"] = out["median_ic"].round(4)
    out["pos_ratio"] = out["pos_ratio"].round(3)
    return out


def pool_lane_stats(lane_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not lane_frames:
        return pd.DataFrame()
    df = pd.concat([x for x in lane_frames if not x.empty], ignore_index=True)
    if df.empty:
        return df
    out = df.groupby(["lane", "horizon_col"], as_index=False).agg(
        n_snapshots=("n", "size"),
        mean_return=("mean_return", "mean"),
        median_return=("median_return", "mean"),
        hit_rate=("hit_rate", "mean"),
        worst=("worst", "min"),
        best=("best", "max"),
        total_n=("n", "sum"),
    )
    for c in ("mean_return", "median_return", "hit_rate", "worst", "best"):
        out[c] = out[c].round(4)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="screening_offline 系 CSV ベースのバックテスト/IC測定")
    ap.add_argument("--master", action="append", default=None,
                    help="screening_offline_*.csv のパス (複数指定可)")
    ap.add_argument("--asof", default=None,
                    help="マスタCSVの基準日 YYYY-MM-DD。複数 master 時はファイル名から推定。")
    ap.add_argument("--horizons", default="21,63,126",
                    help="forward リターンのホライズン (営業日, カンマ区切り)")
    ap.add_argument("--lane-col", default=DEFAULT_LANE_COLUMN,
                    help="レーン分類列 (default: candidate_lane)")
    ap.add_argument("--score-cols", default=None,
                    help="評価対象スコア列 (カンマ区切り)。省略時は既定一覧から実在列のみ採用")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT_DIR),
                    help="出力ディレクトリ (default: output/backtest)")
    args = ap.parse_args(argv)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if not horizons:
        print("[ERROR] horizons が空です", file=sys.stderr)
        return 2

    if args.score_cols:
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    else:
        score_cols = list(DEFAULT_SCORE_COLUMNS)

    if not args.master:
        # 直近 N 件を自動探索
        candidates: List[Path] = []
        for d in (Path("output") / "reports", Path("output")):
            if d.exists():
                candidates += sorted(d.glob("screening_offline_*.csv"))
        if not candidates:
            print("[ERROR] --master 未指定で、output 配下に screening_offline_*.csv が見つかりません。", file=sys.stderr)
            return 2
        masters = [candidates[-1]]
    else:
        masters = [Path(p) for p in args.master]

    asof_arg = _parse_date(args.asof) if args.asof else None

    all_fwd: List[pd.DataFrame] = []
    all_ic: List[pd.DataFrame] = []
    all_lane: List[pd.DataFrame] = []

    for m in masters:
        print(f"[INFO] processing {m.name} ...")
        fwd, ic, ln = run_single_snapshot(
            master_csv=m,
            asof=asof_arg,
            horizons=horizons,
            score_cols=score_cols,
            lane_col=args.lane_col,
        )
        all_fwd.append(fwd)
        all_ic.append(ic)
        all_lane.append(ln)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fwd_all = pd.concat([x for x in all_fwd if not x.empty], ignore_index=True) if all_fwd else pd.DataFrame()
    ic_all = pd.concat([x for x in all_ic if not x.empty], ignore_index=True) if all_ic else pd.DataFrame()
    lane_all = pd.concat([x for x in all_lane if not x.empty], ignore_index=True) if all_lane else pd.DataFrame()

    if not fwd_all.empty:
        fwd_all.to_csv(outdir / "per_stock_returns.csv", index=False, encoding="utf-8-sig")
    if not ic_all.empty:
        ic_all.to_csv(outdir / "ic_per_snapshot.csv", index=False, encoding="utf-8-sig")
        pooled_ic = pool_ic_summary(all_ic)
        if not pooled_ic.empty:
            pooled_ic.to_csv(outdir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    if not lane_all.empty:
        lane_all.to_csv(outdir / "lane_stats_per_snapshot.csv", index=False, encoding="utf-8-sig")
        pooled_lane = pool_lane_stats(all_lane)
        if not pooled_lane.empty:
            pooled_lane.to_csv(outdir / "lane_stats.csv", index=False, encoding="utf-8-sig")

    meta = {
        "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "masters": [str(m) for m in masters],
        "asof_arg": args.asof,
        "horizons": horizons,
        "score_cols_used": score_cols,
        "lane_col": args.lane_col,
        "frozen_prices_dir": str(FROZEN_PRICES_DIR),
        "n_codes_with_prices": int(fwd_all["code"].nunique()) if not fwd_all.empty else 0,
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not ic_all.empty:
        pooled = pool_ic_summary(all_ic)
        if not pooled.empty:
            print("\n=== IC summary (pooled) ===")
            print(pooled.to_string(index=False))
    if not lane_all.empty:
        pooled = pool_lane_stats(all_lane)
        if not pooled.empty:
            print("\n=== Lane stats (pooled) ===")
            print(pooled.to_string(index=False))

    print(f"\n[OK] artifacts written under: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
