# -*- coding: utf-8 -*-
"""Frozen screening scores and subsequent close-to-close gross returns.

Evaluate eligible A/B groups over 21/63/126/252 sessions by default. Standard
prices use a pinned cache snapshot, its trading calendar and split factors.
If no Standard snapshot exists, legacy frozen price CSVs remain supported.
The entry reference is the first close whose session opens after both the
information cutoff and report generation. Dividends and costs are excluded.
The return is a fixed-horizon diagnostic without external trading-bot exits.

Example
    python backtest_ic.py --master output_standard/history/RUN/screening_offline.csv
        --price-snapshot-id PRICE_SNAPSHOT --horizons 21,63,126,252

Each horizon needs horizon + 1 complete session closes. Dates come from the
sibling run_manifest.json, CSV context or a legacy filename. --asof supplies
a missing legacy baseline, and future prices remain missing until observed.
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
STANDARD_CACHE_DIR = Path(".jquants_cache_v2_standard")
JST = datetime.timezone(datetime.timedelta(hours=9))

DEFAULT_SCORE_COLUMNS = [
    "registration_a_score",
    "registration_b_score",
]

DEFAULT_LANE_COLUMN = "registration"


def _parse_date(s: str) -> datetime.date:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"日付の形式が不明: {s}")


def _infer_asof_from_filename(path: Path) -> Optional[datetime.date]:
    m = re.fullmatch(r"screening_offline_(?:r_)?(\d{8})_\d{6}(?:_\d{6}_[a-z0-9]+)?\.csv", path.name)
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
    return None


def _parse_timestamp(value: Any) -> Optional[datetime.datetime]:
    if value is None or pd.isna(value):
        return None
    parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=JST) if parsed.tzinfo is None else parsed.astimezone(JST)


def _screening_context(master_csv: Path, frame: pd.DataFrame, asof: Optional[datetime.date]) -> dict:
    manifest_path = master_csv.parent / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
    values = {}
    for key in ("price_asof_date", "information_cutoff", "generated_at", "next_trade_date", "run_id"):
        observed = frame[key].dropna().astype(str).unique() if key in frame else []
        if len(observed) > 1:
            raise ValueError("Mixed screening context: " + key)
        value = manifest.get(key)
        if value is not None and len(observed) and str(value) != observed[0]:
            raise ValueError("Manifest and screening CSV disagree: " + key)
        values[key] = value if value is not None else observed[0] if len(observed) else None
    price_day = _parse_date(values["price_asof_date"]) if values["price_asof_date"] else None
    baseline = asof or price_day or _infer_asof_from_filename(master_csv)
    if baseline is None:
        raise ValueError("Screening date unavailable; supply --asof for a legacy CSV")
    if price_day is not None and baseline < price_day:
        raise ValueError("--asof precedes the screening price date")
    timestamps = [_parse_timestamp(values[key]) for key in ("information_cutoff", "generated_at") if values[key]]
    return {"asof": baseline, "available_after": max(timestamps) if timestamps else None,
            "next_trade_date": _parse_date(values["next_trade_date"]) if values["next_trade_date"] else None,
            "run_id": values["run_id"]}


def _load_standard_prices(code: str, store: Any, snapshot: dict) -> Optional[pd.DataFrame]:
    import JQuamtsScreeningBot as bot

    item = snapshot.get("symbols", {}).get(code, {})
    reference = item.get("prices")
    if not reference:
        return None
    rows = bot.bundle_rows(store, reference)
    if not rows:
        return None
    api_code = str(rows[0].get("Code", ""))
    if bot.canonical_code(api_code) != code:
        raise ValueError("Price snapshot code mismatch: " + code)
    calendar_rows = bot.bundle_rows(store, snapshot.get("globals", {}).get("calendar"))
    sessions = bot.trading_sessions(calendar_rows)
    frame = bot.price_frame(rows, api_code, sessions, bot.date(snapshot.get("price_asof_date")))
    return pd.DataFrame({"date": frame.index.date, "close": frame["AC"].to_numpy()})


def _entry_reference_date(prices: pd.DataFrame, context: dict) -> Optional[datetime.date]:
    possible = prices.loc[prices["date"] > context["asof"], "date"]
    if context["next_trade_date"] is not None:
        possible = possible[possible >= context["next_trade_date"]]
    if context["available_after"] is not None:
        possible = possible[possible.map(lambda day: datetime.datetime.combine(day, datetime.time(9), JST)
                                        > context["available_after"])]
    return possible.iloc[0] if len(possible) else None


def _concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


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
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def _forward_return_business_days(
    prices: pd.DataFrame, asof: datetime.date, horizon: int,
    *, entry_date: Optional[datetime.date] = None,
) -> Optional[float]:
    """asof より厳密に後 (>) の最初の営業日を起点とし、そこから horizon 営業日後の
    終値を用いて単純リターンを算出する。"""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if prices is None or prices.empty:
        return None
    sub = prices[prices["date"] >= entry_date] if entry_date is not None else prices[prices["date"] > asof]
    if sub.empty or len(sub) <= horizon:
        return None
    base = sub.iloc[0]["close"]
    target = sub.iloc[horizon]["close"]
    window = pd.to_numeric(sub.iloc[:horizon + 1]["close"], errors="coerce")
    if not np.isfinite(window).all() or (window <= 0).any():
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
    *, price_store: Any = None, price_snapshot: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """1つの screening_offline_*.csv について、forward returns / IC / lane_stats を計算する。

    戻り値:
      - per_stock_returns: 銘柄 × horizon 行の forward return 明細
      - ic_summary: スコア × horizon の IC
      - lane_stats: レーン × horizon の集計
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"master_csv not found: {master_csv}")
    df = pd.read_csv(master_csv, encoding="utf-8-sig", dtype={"code": str})
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if "code" not in df.columns:
        raise ValueError("master_csv に code 列がありません (flatten 後の CSV を渡してください)")

    context = _screening_context(master_csv, df, asof)
    asof_eff = context["asof"]

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        code = str(r.get("code") or "").strip()
        if not code:
            continue
        prices = (_load_standard_prices(code, price_store, price_snapshot)
                  if price_snapshot is not None else _load_prices_for_code(code))
        if prices is None:
            continue
        entry_date = _entry_reference_date(prices, context)
        rec: Dict[str, Any] = {
            "code": code,
            "name": r.get("name"),
            "asof": asof_eff.isoformat(),
            "snapshot": master_csv.name,
            "run_id": context["run_id"],
            "entry_reference_date": entry_date.isoformat() if entry_date is not None else None,
            "available_after": context["available_after"].isoformat() if context["available_after"] else None,
        }
        for h in horizons:
            rec[f"fwd_ret_{h}d"] = (_forward_return_business_days(prices, asof_eff, h, entry_date=entry_date)
                                    if entry_date is not None else None)
        rows.append(rec)

    fwd = pd.DataFrame(rows)
    if fwd.empty:
        return fwd, pd.DataFrame(), pd.DataFrame()

    merge_cols = [c for c in score_cols if c in df.columns]
    snap_cols = ["code"] + ([lane_col] if lane_col in df.columns else []) + merge_cols
    eligible_columns = [f"registration_{route}_eligible" for route in ("a", "b")
                        if f"registration_{route}_eligible" in df.columns]
    snap_cols = list(dict.fromkeys(snap_cols + eligible_columns))
    snap = df[snap_cols].copy()
    snap["code"] = snap["code"].astype(str).str.strip()
    merged = fwd.merge(snap, on="code", how="left")

    ic_rows: List[Dict[str, Any]] = []
    for sc in merge_cols:
        scored = merged
        for route in ("a", "b"):
            flag = f"registration_{route}_eligible"
            if sc == f"registration_{route}_score" and flag in merged:
                scored = merged[merged[flag].astype(str).str.lower().eq("true")]
        for h in horizons:
            yname = f"fwd_ret_{h}d"
            ic, n = _spearman_ic(scored[sc], scored[yname])
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
    if lane_col == "registration" and eligible_columns:
        for route in ("a", "b"):
            flag = f"registration_{route}_eligible"
            if flag not in merged:
                continue
            eligible = merged[merged[flag].astype(str).str.lower().eq("true")].copy()
            eligible["registration"] = route
            for h in horizons:
                lane_rows.append(_lane_stats_single(eligible, "registration", f"fwd_ret_{h}d"))
    elif lane_col in merged.columns:
        for h in horizons:
            lane_rows.append(_lane_stats_single(merged, lane_col, f"fwd_ret_{h}d"))
    lane_stats = _concat_nonempty(lane_rows)
    if not lane_stats.empty:
        lane_stats.insert(0, "snapshot", master_csv.name)
        lane_stats.insert(0, "asof", asof_eff.isoformat())

    return merged, ic_summary, lane_stats


def pool_ic_summary(ic_frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not ic_frames:
        return pd.DataFrame()
    df = _concat_nonempty(ic_frames)
    if df.empty:
        return df
    df = df.dropna(subset=["ic_spearman"])
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["score", "horizon_days"], as_index=False)
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
    df = _concat_nonempty(lane_frames)
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
    ap.add_argument("--horizons", default="21,63,126,252",
                    help="forward リターンのホライズン (営業日, カンマ区切り)")
    ap.add_argument("--lane-col", default=DEFAULT_LANE_COLUMN,
                    help="レーン分類列。既定registrationは適格A/Bを別々に集計")
    ap.add_argument("--score-cols", default=None,
                    help="評価対象スコア列 (カンマ区切り)。省略時は既定一覧から実在列のみ採用")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT_DIR),
                    help="出力ディレクトリ (default: output/backtest)")
    ap.add_argument("--cache-dir", type=Path, default=STANDARD_CACHE_DIR,
                    help="将来価格を含むStandardキャッシュ")
    ap.add_argument("--price-snapshot-id", help="将来価格のsnapshot ID。省略時はlatestを固定")
    args = ap.parse_args(argv)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if not horizons or any(horizon < 1 for horizon in horizons):
        print("[ERROR] horizons は正の営業日数で指定してください", file=sys.stderr)
        return 2

    if args.score_cols:
        score_cols = [c.strip() for c in args.score_cols.split(",") if c.strip()]
    else:
        score_cols = list(DEFAULT_SCORE_COLUMNS)

    if not args.master:
        # Pin one immutable current-format run before considering old outputs.
        candidates: List[Path] = []
        current_history = Path("output_standard") / "history"
        candidates = sorted(current_history.glob("r_*/screening_offline.csv"))
        for d in (Path("output") / "reports", Path("output")) if not candidates else ():
            if d.exists():
                candidates += sorted(d.glob("screening_offline_*.csv"))
        if not candidates:
            print("[ERROR] screening CSV が見つかりません。--master を指定してください。", file=sys.stderr)
            return 2
        masters = [candidates[-1]]
    else:
        masters = [Path(p) for p in args.master]

    asof_arg = _parse_date(args.asof) if args.asof else None
    price_store, price_snapshot = None, None
    if args.price_snapshot_id is not None or (args.cache_dir / "latest_snapshot.json").is_file():
        import JQuamtsScreeningBot as bot

        price_store = bot.Store(args.cache_dir)
        price_snapshot = price_store.snapshot(args.price_snapshot_id)
        if not price_snapshot:
            raise ValueError("Price snapshot unavailable")

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
            price_store=price_store,
            price_snapshot=price_snapshot,
        )
        all_fwd.append(fwd)
        all_ic.append(ic)
        all_lane.append(ln)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fwd_all = _concat_nonempty(all_fwd)
    ic_all = _concat_nonempty(all_ic)
    lane_all = _concat_nonempty(all_lane)

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
        "price_snapshot_id": price_snapshot.get("snapshot_id") if price_snapshot else None,
        "price_snapshot_input_hash": bot.digest(price_snapshot) if price_snapshot else None,
        "price_cache_dir": str(args.cache_dir) if price_snapshot else None,
        "entry_basis": "close_of_first_session_opening_after_cutoff_and_report_generation",
        "horizon_basis": "elapsed_sessions_after_entry_reference_close",
        "return_basis": "split_adjusted_gross_close_to_close" if price_snapshot else "legacy_csv_close_to_close",
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
