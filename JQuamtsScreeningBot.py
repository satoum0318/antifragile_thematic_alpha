# -*- coding: utf-8 -*-
"""
J-Quants 収集→凍結キャッシュ→完全オフライン分析ワークフロー
- J-Quants API V2（x-api-key）を使用。キャッシュは .jquants_cache_v2（V1の .jquants_cache は混在させない）
- collect_all の日次「待機」は JQ_RPD 設定時のみ（未設定なら rpm 主制御で同一日内に待機しない）
- 収集の既定銘柄数は環境変数 JQ_DEFAULT_BUDGET（未設定時 800）、--budget で上書き（V2 では RPM も調整）
- モック不使用: オフライン時は“計算不能はNone”で返す（推定やランダムは行わない）
- 端末対話メニュー付き（引数未指定で起動するとメニュー表示）
- CLI対応:
    収集:   python script.py --phase collect --budget 800
    解析:   python script.py --phase analyze --top 10
    単銘柄: python script.py --phase single --code 8035
    全件:   python script.py --phase collect_all --budget 800
環境変数:
    JQUANTS_API_KEY または JQ_API_KEY（APIキー。未設定時は api.ini の DEFAULT.API_KEY）
    JQ_RPM=60        # 分あたりリクエスト上限の目安（公式プランに合わせて調整）
    JQ_RPD=          # 未設定なら日次セルフ上限なし（旧V1の800相当を自前で付けたい場合のみ数値を指定）
    JQ_DEFAULT_BUDGET=800  # 収集フェーズ・メニューの既定「1回あたり最大銘柄数」（未設定時は 800）

V2では date 指定で equities/bars/daily の全銘柄日次データ取得が可能なため、
日次更新は将来的に code 単位ではなく date 単位一括取得へ移行する。
初回移行では既存互換の code 単位取得を維持する。

追加（スクリーニングの「必須」フィルタを実装）
- 流動性フィルタ: avg_volume_30d >= MIN_AVG_VOLUME_30D かつ market_cap >= MIN_MARKET_CAP_JPY
- バリュエーション健全性: PS<=MAX_PS_DEFENSIVE を core 条件、PER>MAX_PER_CORE は satellite 扱い
- 収益安定性（営業利益）: 直近年で赤字を含まない + 直近が急落していない

フィルタは core/satellite/excluded の3分類としてレポート出力にも反映します。

スコアリング前提（重要）:
- EPS 成長率は当期・前期の**それぞれの発行済株式数**が揃っているときのみ計算（欠損時は None）。
- 決算系列が annual でない場合（フォールバック時）は sales_cagr および営業利益安定の判定は**非計算**（None / reason=non_annual_basis）。
- V2 では API キー認証のためトークン更新は使用しない。
- PEG / reference_peg は**参考列**（総合ランキングスコアの直接入力には用いない）。

現在のV2実装では、契約プランにより fins/details が403になる場合があります。
その場合は summary_only mode で動作し、流動資産・流動負債・売上総利益等を用いる一部のPiotroski項目は欠損扱いになります。
旧V1版と完全同等の財務スクリーニングには Premium 相当の fins/details アクセスが必要です。

必要: pandas, numpy, requests
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import copy
import signal
import logging
import datetime
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------
logger = logging.getLogger(__name__)
JQ_LOG_VERBOSE = os.getenv("JQ_LOG", "").lower() == "debug"
logger.setLevel(logging.DEBUG if JQ_LOG_VERBOSE else logging.INFO)
_log_stderr = logging.StreamHandler(sys.stderr)
_log_stderr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
_log_stderr.setLevel(logging.DEBUG if JQ_LOG_VERBOSE else logging.INFO)
logger.addHandler(_log_stderr)
logger.propagate = False

def _cli_stdout_utf8ish() -> bool:
    """Windows cp932 等で絵文字 print が UnicodeEncodeError になるのを避けるための判定。"""
    enc = (getattr(sys.stdout, "encoding", None) or "").upper()
    return "UTF" in enc

def _cli_print(msg: str, ascii_safe: str) -> None:
    """UTF-8 系コンソールでは msg、それ以外では ascii_safe（絵文字なし）を表示。"""
    print(msg if _cli_stdout_utf8ish() else ascii_safe)

# ------------------------------------------------------------
# 定数・パス
# ------------------------------------------------------------
JQUANTS_API_BASE = "https://api.jquants.com/v2"
CACHE_DIR = Path(".jquants_cache_v2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK_DAYS = 700
REPORTS_DIR = Path("output") / "reports"
COLLECTION_SKIPLIST_PATH = REPORTS_DIR / "collection_skiplist.json"
MISSING_FINANCIALS_CSV = REPORTS_DIR / "missing_financials_symbols.csv"
NON_STOCK_EXCLUDED_CSV = REPORTS_DIR / "non_stock_excluded_symbols.csv"
COLLECT_FILTER_EXCLUDED_CSV = REPORTS_DIR / "collectable_filter_excluded.csv"
SECTOR_NORM_AUDIT_CSV = REPORTS_DIR / "sector_normalization_audit.csv"

_CANDIDATE_LANE_SORT: Dict[str, int] = {
    "ma200_reclaim_core": 1,
    "bottom_reversal_core": 2,
    "weak_reclaim_watch": 4,
    "watch_fundamental_core": 5,
    "data_review_light": 6,  # rec_priority で reclaim 時は上位へ寄せる
    "extended_above_ma200": 7,
    "satellite_valuation": 8,
    "satellite_ps_only": 8,
    "data_review": 9,
    "cyclical_value_trap": 10,
    "core": 12,
    "excluded": 99,
}

_LANE_EXPORT_NAMES: Dict[str, str] = {
    "ma200_reclaim_core": "ma200_reclaim_core_candidates.csv",
    "bottom_reversal_core": "bottom_reversal_core_candidates.csv",
    "weak_reclaim_watch": "weak_reclaim_watch_candidates.csv",
    "watch_fundamental_core": "watch_fundamental_core_candidates.csv",
    "extended_above_ma200": "extended_above_ma200_candidates.csv",
    "cyclical_value_trap": "cyclical_value_traps.csv",
    "data_review": "data_review_candidates.csv",
    "data_review_light": "data_review_light_candidates.csv",
}


def _default_collect_budget_from_env() -> int:
    """インタラクティブ・CLI --budget 未指定・collect_all の日次デフォルトなどに使用。"""
    raw = os.getenv("JQ_DEFAULT_BUDGET", "800").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 800


DEFAULT_COLLECT_BUDGET = _default_collect_budget_from_env()

# summary_only での影響説明ログ（セッションにつき1回）
_SUMMARY_ONLY_MODE_INFO_LOGGED = False

# V2 legacy: 普通株で statements が空の銘柄をバッチ集計（銘柄ごと WARNING 抑制）
_V2_LEGACY_EMPTY_STOCK_CODES: List[str] = []


def reset_v2_legacy_batch_audit() -> None:
    global _V2_LEGACY_EMPTY_STOCK_CODES
    _V2_LEGACY_EMPTY_STOCK_CODES = []


def record_v2_legacy_empty_stock(log_code: str) -> None:
    c = (log_code or "").strip()
    if c and c not in _V2_LEGACY_EMPTY_STOCK_CODES:
        _V2_LEGACY_EMPTY_STOCK_CODES.append(c)


def flush_v2_legacy_batch_audit_loggers() -> None:
    global _V2_LEGACY_EMPTY_STOCK_CODES
    if not _V2_LEGACY_EMPTY_STOCK_CODES:
        return
    n = len(_V2_LEGACY_EMPTY_STOCK_CODES)
    sample = ",".join(_V2_LEGACY_EMPTY_STOCK_CODES[:24])
    logger.warning(
        "[WARN] stock financial statements missing: %s symbols (sample: %s)",
        n,
        sample,
    )
    _V2_LEGACY_EMPTY_STOCK_CODES = []

# ------------------------------------------------------------
# スクリーニング必須フィルタ（環境変数で上書き可）
# ------------------------------------------------------------
MIN_AVG_VOLUME_30D = int(os.getenv("MIN_AVG_VOLUME_30D", "50000"))
MIN_ADV_JPY_20D = int(os.getenv("MIN_ADV_JPY_20D", "300000000"))
MIN_MARKET_CAP_JPY = int(os.getenv("MIN_MARKET_CAP_JPY", "50000000000"))  # 50B JPY
MAX_PS_DEFENSIVE = float(os.getenv("MAX_PS_DEFENSIVE", "2.0"))
MAX_PER_CORE = float(os.getenv("MAX_PER_CORE", "60.0"))

# 収益安定性（営業利益）判定
OP_INCOME_YEARS = int(os.getenv("OP_INCOME_YEARS", "3"))                # 直近何年見るか（新しい年度順）
OP_INCOME_DROP_FLOOR = float(os.getenv("OP_INCOME_DROP_FLOOR", "0.6"))  # 直近営業利益が過去年中央値の何倍以上ならOK
EXCLUDE_OP_INCOME_DEFICIT = (os.getenv("EXCLUDE_OP_INCOME_DEFICIT", "1") != "0")  # 直近年に赤字があれば除外（デフォON）

# 200日線局面・大底レーン・ファンダエッジ（環境変数で上書き可）
MA200_CROSS_LOOKBACK_DAYS = int(os.getenv("MA200_CROSS_LOOKBACK_DAYS", "20"))
MA200_IDEAL_MAX_DISTANCE = float(os.getenv("MA200_IDEAL_MAX_DISTANCE", "0.08"))
MA200_EXTENDED_DISTANCE = float(os.getenv("MA200_EXTENDED_DISTANCE", "0.15"))
MA200_BELOW_MIN_RATIO = float(os.getenv("MA200_BELOW_MIN_RATIO", "0.75"))

BASING_MIN_REBOUND_FROM_LOW = float(os.getenv("BASING_MIN_REBOUND_FROM_LOW", "0.08"))
BASING_LOOKBACK_LOW_DAYS = int(os.getenv("BASING_LOOKBACK_LOW_DAYS", "120"))
RECENT_LOW_LOOKBACK_DAYS = int(os.getenv("RECENT_LOW_LOOKBACK_DAYS", "60"))
RECENT_LOW_NO_UPDATE_DAYS = int(os.getenv("RECENT_LOW_NO_UPDATE_DAYS", "10"))

MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY = float(os.getenv("MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY", "75"))
MIN_PIOTROSKI_CORE = int(os.getenv("MIN_PIOTROSKI_CORE", "6"))
MIN_PIOTROSKI_COVERAGE_CORE = float(os.getenv("MIN_PIOTROSKI_COVERAGE_CORE", "0.60"))

MAX_PS_VS_SECTOR_CORE = float(os.getenv("MAX_PS_VS_SECTOR_CORE", "1.10"))
MAX_CRITICAL_MISSING_CORE = int(os.getenv("MAX_CRITICAL_MISSING_CORE", "2"))
MAX_STATEMENT_STALENESS_DAYS_CORE = int(os.getenv("MAX_STATEMENT_STALENESS_DAYS_CORE", "270"))
STALE_STATEMENT_MEDIUM_DAYS = int(os.getenv("STALE_STATEMENT_MEDIUM_DAYS", "180"))

WATCH_FUNDAMENTAL_EDGE_MIN = float(os.getenv("WATCH_FUNDAMENTAL_EDGE_MIN", "60"))
MA200_RECLAIM_EDGE_MIN = float(os.getenv("MA200_RECLAIM_EDGE_MIN", "70"))
RECLAIM_CORE_MIN_FUNDAMENTAL = float(os.getenv("RECLAIM_CORE_MIN_FUNDAMENTAL", "70"))
WEAK_RECLAIM_MAX_FUNDAMENTAL = float(os.getenv("WEAK_RECLAIM_MAX_FUNDAMENTAL", "69"))
WEAK_RECLAIM_MIN_FUNDAMENTAL = float(os.getenv("WEAK_RECLAIM_MIN_FUNDAMENTAL", "60"))
EXTENDED_FUNDAMENTAL_EDGE_MIN = float(os.getenv("EXTENDED_FUNDAMENTAL_EDGE_MIN", "60"))

LANE_ENTRY_CAP: Dict[str, float] = {
    "ma200_reclaim_core": float(os.getenv("LANE_CAP_RECLAIM", "92")),
    "bottom_reversal_core": float(os.getenv("LANE_CAP_BOTTOM", "85")),
    "weak_reclaim_watch": float(os.getenv("LANE_CAP_WEAK_RECLAIM", "78")),
    "watch_fundamental_core": float(os.getenv("LANE_CAP_WATCH", "75")),
    "extended_above_ma200": float(os.getenv("LANE_CAP_EXTENDED", "65")),
    "data_review_light": float(os.getenv("LANE_CAP_DATA_REVIEW_LIGHT", "80")),
    "data_review": float(os.getenv("LANE_CAP_DATA_REVIEW", "70")),
    "data_review_severe": float(os.getenv("LANE_CAP_DATA_REVIEW_SEVERE", "55")),
    "cyclical_value_trap": float(os.getenv("LANE_CAP_CYCLICAL", "45")),
    "excluded": float(os.getenv("LANE_CAP_EXCLUDED", "30")),
    "satellite_valuation": float(os.getenv("LANE_CAP_SATELLITE", "72")),
    "satellite_ps_only": float(os.getenv("LANE_CAP_SATELLITE", "72")),
    "core": float(os.getenv("LANE_CAP_CORE_LEGACY", "88")),
}

# ------------------------------------------------------------
# ヘルパ
# ------------------------------------------------------------
def seconds_until_next_day(buffer_sec: int = 10) -> int:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    reset = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((reset - now).total_seconds()) + buffer_sec)

def build_prices_endpoint(stock_code: str, lookback_days: int = LOOKBACK_DAYS) -> str:
    """ログ用途・後方互換。実際の取得は fetch_prices_v2 を使用（equities/bars/daily）。候補の先頭を表示。"""
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = datetime.date.today().strftime("%Y-%m-%d")
    cands = api_code_candidates(stock_code)
    qc = cands[0] if cands else ""
    return f"equities/bars/daily?code={qc}&from={start}&to={end}"

def api_code_candidates(code: str) -> List[str]:
    """V2 API が受け付けるコード候補: 優先して内部4桁、次に4桁+'0'。zfill(5) は使わない。"""
    c = str(code).strip()
    m = re.search(r"\d{4}", c)
    if not m:
        digits = "".join(ch for ch in c if ch.isdigit())
        if len(digits) >= 4:
            c4 = digits[:4]
            return [c4, c4 + "0"]
        if digits:
            c4 = digits.zfill(4)[-4:]
            return [c4, c4 + "0"]
        return [c] if c else []
    c4 = m.group(0)
    return [c4, c4 + "0"]

def apply_v2_fins_summary_field_aliases(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    V2 fins/summary の略号列を、既存 build_financial_history が期待する V1 風キーへ付与する。
    （fins/details が使えないプランでも summary 単体で最低限動く）
    """
    if not row:
        return row
    out = dict(row)
    if out.get("DisclosedDate") is None and out.get("DiscDate") is not None:
        out["DisclosedDate"] = out.get("DiscDate")
    if out.get("TypeOfDocument") is None and out.get("DocType") is not None:
        out["TypeOfDocument"] = out.get("DocType")
    if out.get("CurrentPeriodType") is None and out.get("CurPerType") is not None:
        out["CurrentPeriodType"] = out.get("CurPerType")
    if out.get("CurrentPeriodEndDate") is None and out.get("CurPerEn") is not None:
        out["CurrentPeriodEndDate"] = out.get("CurPerEn")
    if out.get("CurrentFiscalYearEndDate") is None and out.get("CurFYEn") is not None:
        out["CurrentFiscalYearEndDate"] = out.get("CurFYEn")
    if out.get("NetSales") is None and out.get("Sales") is not None:
        out["NetSales"] = out.get("Sales")
    if out.get("OperatingIncome") is None and out.get("OP") is not None:
        out["OperatingIncome"] = out.get("OP")
    if out.get("NetIncomeLoss") is None and out.get("NP") is not None:
        out["NetIncomeLoss"] = out.get("NP")
    if out.get("TotalAssets") is None and out.get("TA") is not None:
        out["TotalAssets"] = out.get("TA")
    if out.get("EquityAttributableToOwnersOfParent") is None and out.get("Eq") is not None:
        out["EquityAttributableToOwnersOfParent"] = out.get("Eq")
    for _cfo_alt in ("CFO", "Cfo", "cfo"):
        if _cfo_alt in out and out.get(_cfo_alt) not in (None, "", "NA"):
            if out.get("NetCashProvidedByUsedInOperatingActivities") is None:
                out["NetCashProvidedByUsedInOperatingActivities"] = out.get(_cfo_alt)
            if out.get("CashFlowsFromOperatingActivities") is None:
                out["CashFlowsFromOperatingActivities"] = out.get(_cfo_alt)
            break
    cfo_cf = _non_null(
        out.get("NetCashProvidedByUsedInOperatingActivities"),
        out.get("CashFlowsFromOperatingActivities"),
        out.get("CFO"),
    )
    if cfo_cf is not None:
        out["NetCashProvidedByUsedInOperatingActivities"] = cfo_cf
        out["CashFlowsFromOperatingActivities"] = cfo_cf
    if out.get("CashAndCashEquivalents") is None and out.get("CashEq") is not None:
        out["CashAndCashEquivalents"] = out.get("CashEq")
    sh = _non_null(out.get("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"), out.get("ShOutFY"))
    if out.get("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock") is None and sh is not None:
        out["NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"] = sh
    cfo_unify = _pick_cfo_scalar_from_rows(out)
    if cfo_unify is not None:
        out["NetCashProvidedByUsedInOperatingActivities"] = cfo_unify
        out["CashFlowsFromOperatingActivities"] = cfo_unify
    return out

def _strip_slash(endpoint: str) -> str:
    return endpoint.strip().lstrip("/")

def _v2_resp_error_snippet(resp: requests.Response) -> str:
    try:
        return (resp.text or "")[:400]
    except Exception:
        return ""

def paginate_v2_endpoint(
    session: requests.Session,
    endpoint: str,
    params: Optional[Dict[str, Any]],
    *,
    max_pages: int = 1000,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """GET の data を連結。pyd 最初の応答ステータスを返す（200以外はページング中止）。"""
    ep = _strip_slash(endpoint)
    base_params = dict(params or {})
    all_rows: List[Dict[str, Any]] = []
    pagination_key: Optional[str] = None
    last_http = 200
    err_snippet = ""
    for page in range(max_pages):
        p = base_params.copy()
        if pagination_key:
            p["pagination_key"] = pagination_key
        url = f"{JQUANTS_API_BASE}/{ep}"
        try:
            resp = session.get(url, params=p, timeout=30)
        except requests.RequestException as e:
            logger.warning("paginate_v2_endpoint 通信失敗 %s: %s", url, e)
            return [], 599, str(e)
        last_http = resp.status_code
        if resp.status_code != 200:
            err_snippet = _v2_resp_error_snippet(resp)
            return [], resp.status_code, err_snippet
        try:
            body = resp.json()
        except Exception as e:
            logger.warning("paginate_v2_endpoint JSON decode失敗 %s: %s", url, e)
            return [], 598, str(e)
        data = body.get("data")
        if not isinstance(data, list):
            break
        all_rows.extend(data)
        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break
    return all_rows, last_http, err_snippet

def get_v2_all_pages(
    session: requests.Session,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    max_pages: int = 1000,
) -> List[Dict[str, Any]]:
    """
    GET {JQUANTS_API_BASE}/{endpoint}
    response['data'] を連結して返す。pagination_key がある限り続きを取得する。
    """
    ep = _strip_slash(endpoint)
    rows, status, err = paginate_v2_endpoint(session, ep, dict(params or {}), max_pages=max_pages)
    if status != 200:
        logger.warning("get_v2_all_pages HTTP %s %s %s", status, endpoint, err[:120] if err else "")
        return []
    return rows

def normalize_prices_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    V2 equities/bars/daily の短縮列名を既存内部標準列へ変換する。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {
        "O": "Open",
        "H": "High",
        "L": "Low",
        "C": "Close",
        "Vo": "Volume",
        "Va": "TurnoverValue",
        "AdjO": "AdjustmentOpen",
        "AdjH": "AdjustmentHigh",
        "AdjL": "AdjustmentLow",
        "AdjC": "AdjustmentClose",
        "AdjVo": "AdjustmentVolume",
    }
    colmap = {k: v for k, v in rename_map.items() if k in out.columns}
    if colmap:
        out = out.rename(columns=colmap)
    num_cols = [c for c in (
        "Open", "High", "Low", "Close", "Volume", "TurnoverValue",
        "AdjustmentOpen", "AdjustmentHigh", "AdjustmentLow", "AdjustmentClose", "AdjustmentVolume",
    ) if c in out.columns]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Date" in out.columns:
        out = out.sort_values("Date", ascending=True).reset_index(drop=True)
    return out

def normalize_equities_master_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    V2 equities/master を既存の Code, CompanyName, Sector33Name, MarketCode へ正規化する。
    J-Quants V2 では CoName / Mkt が返るためそれを優先する。
    """
    if df is None or df.empty:
        base_cols = ["Code", "CompanyName", "Sector33Name", "MarketCode", "CoNameEn", "Mkt", "MktNm"]
        return pd.DataFrame(columns=base_cols)

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        d = row.to_dict()
        code_cell = (
            _first_present(d, ("Code", "LocalCode", "code", "IssueCode", "CorporateCode"))
        )
        if code_cell is None or (isinstance(code_cell, float) and pd.isna(code_cell)):
            continue
        digits = "".join(ch for ch in str(code_cell) if ch.isdigit())
        if len(digits) >= 5:
            code_4 = digits[:4]
        elif len(digits) >= 4:
            code_4 = digits[:4]
        else:
            code_4 = digits.zfill(4)[-4:]
        company = _first_present(
            d,
            (
                "CoName",
                "CompanyName",
                "CompanyNameJapanese",
                "IssueName",
                "IssuerNameJa",
                "IssuerName",
            ),
        )
        co_en = _first_present(d, ("CoNameEn", "CompanyNameEnglish"))
        sector = _first_present(
            d,
            ("S33Nm", "Sector33Name", "Sector33EnglishName", "S17Nm", "Sector17Name"),
        )
        market = _first_present(
            d,
            ("Mkt", "MarketCode", "MarketDivision", "MarketSegment", "Market", "MarketName", "Section"),
        )
        mkt_nm = _first_present(d, ("MktNm", "MarketName", "Section"))
        rows.append({
            "Code": code_4,
            "CompanyName": company if company is not None and not (isinstance(company, float) and pd.isna(company)) else "",
            "Sector33Name": sector if sector is not None and not (isinstance(sector, float) and pd.isna(sector)) else "",
            "MarketCode": "" if market is None or (isinstance(market, float) and pd.isna(market)) else str(market),
            "CoNameEn": "" if co_en is None or (isinstance(co_en, float) and pd.isna(co_en)) else str(co_en),
            "Mkt": "" if market is None or (isinstance(market, float) and pd.isna(market)) else str(market),
            "MktNm": "" if mkt_nm is None or (isinstance(mkt_nm, float) and pd.isna(mkt_nm)) else str(mkt_nm),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["Code"], keep="first")
    return out

def _first_present(d: dict, keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None and not (isinstance(d[k], float) and pd.isna(d[k])):
            return d[k]
    return None

def _non_null(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        if v == "":
            continue
        return v
    return None

def _financial_scalar_absent(val: Any) -> bool:
    """0 は有効値として欠損扱いしない。None / "" / NaN のみ欠損。"""
    if val is None:
        return True
    if val == "":
        return True
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, np.integer)):
        return False
    if isinstance(val, (float, np.floating)):
        return bool(pd.isna(val))
    try:
        if pd.api.types.is_scalar(val) and pd.isna(val):
            return True
    except Exception:
        pass
    return False


def _canonical_internal_stock_code(
    code: str,
    valid_stock_codes: Optional[set[str]] = None,
) -> Optional[str]:
    """
    内部4桁コードへ正規化。曖昧・master不一致の場合は None。

    - 数字のみ抽出してから判定
    - ちょうど4桁: その文字列を候補（master がある場合は集合に存在する場合のみ採用）
    - 5桁かつ末尾が '0': 先頭4桁
    - 5桁かつ先頭が '0': 末尾4桁（例: 01301 -> 1301）
    - 5桁で上記以外: 先頭4桁・末尾4桁のどちらかが master にあればそれを採用（両方なら先に一致した方）
    - 6桁超: 先頭4桁・末尾4桁を master と照合
    - 4桁未満: zfill(4) を候補にし、master がある場合は集合に存在するときのみ採用
    """
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    if not digits:
        return None

    def _member(cand: str) -> bool:
        if valid_stock_codes is None:
            return True
        return cand in valid_stock_codes

    def _first_matching(cands: List[str]) -> Optional[str]:
        seen: set[str] = set()
        for cand in cands:
            if cand in seen:
                continue
            seen.add(cand)
            if _member(cand):
                return cand
        return None

    if len(digits) == 4:
        return digits if _member(digits) else None

    if len(digits) == 5:
        if digits.endswith("0"):
            cand = digits[:4]
            return cand if _member(cand) else None
        if digits.startswith("0"):
            cand = digits[-4:]
            return cand if _member(cand) else None
        cands = [digits[:4], digits[-4:]]
        if valid_stock_codes is None:
            return None
        return _first_matching(cands)

    if len(digits) > 5:
        if valid_stock_codes is None:
            return None
        cands = [digits[:4], digits[-4:]]
        return _first_matching(cands)

    # 1〜3桁
    cand = digits.zfill(4)
    return cand if _member(cand) else None


def _pick_cfo_scalar_from_rows(*rows: Optional[Dict[str, Any]]) -> Any:
    """top-level と FS 内から営業CF相当を1つ拾う（0は有効）。複数 dict を順に見る。"""
    keys = (
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromOperatingActivities",
        "OperatingCashFlow",
        "OCF",
        "ocf",
        "CFO",
        "Cfo",
        "cfo",
    )
    for row in rows:
        if not row or not isinstance(row, dict):
            continue
        fs = row.get("FS") if isinstance(row.get("FS"), dict) else None
        for src in (row, fs):
            if not isinstance(src, dict):
                continue
            for k in keys:
                if k not in src:
                    continue
                v = src[k]
                if v is None or v == "":
                    continue
                if isinstance(v, (float, np.floating)) and pd.isna(v):
                    continue
                if isinstance(v, (np.integer, np.floating)) and pd.isna(v):
                    continue
                if isinstance(v, str):
                    st = v.strip().replace(",", "")
                    if not st or st in ("-", "—", "–", "NaN", "nan", "None"):
                        continue
                    try:
                        fv = float(st)
                    except ValueError:
                        continue
                    if not math.isfinite(fv):
                        continue
                    return fv
                return v
    return None


def _valid_collectable_stock_codes(df: pd.DataFrame) -> set[str]:
    """collectable マスタ上の有効 Code（4桁・空白除去済み）。"""
    if df is None or df.empty or "Code" not in df.columns:
        return set()
    return {str(x).strip() for x in df["Code"].astype(str)}


def sanitize_pending_codes(codes: List[str], valid_stock_codes: set[str]) -> List[str]:
    """pending を内部4桁に正規化し、collectable master に存在するものだけ残す。"""
    out: List[str] = []
    seen: set[str] = set()
    dropped = 0
    for c in codes:
        cc = _canonical_internal_stock_code(str(c).strip(), valid_stock_codes)
        if cc is None:
            dropped += 1
            continue
        if cc in seen:
            continue
        seen.add(cc)
        out.append(cc)
    if dropped:
        logger.info(
            "[INFO] sanitize_pending: dropped %s codes (non-canonical / not in collectable master / duplicate)",
            dropped,
        )
    return out


def prune_stale_collection_sidecars(valid_stock_codes: set[str]) -> None:
    """master に存在しない skiplist / missing_financials CSV 行を削除（キーは4桁に統一）。"""
    blob = _load_skiplist_raw()
    skipped = blob.get("skipped") or {}
    new_skipped: Dict[str, Any] = {}
    rm_sl = 0
    for k, v in skipped.items():
        ck = _canonical_internal_stock_code(str(k).strip(), valid_stock_codes)
        if ck is not None and ck in valid_stock_codes:
            new_skipped[ck] = v
        else:
            rm_sl += 1
    blob["skipped"] = new_skipped
    if rm_sl:
        logger.info("[INFO] prune_stale_collection_sidecars: removed %s skiplist keys not in collectable master", rm_sl)
    _save_skiplist_raw(blob)

    if MISSING_FINANCIALS_CSV.exists():
        try:
            mdf = pd.read_csv(MISSING_FINANCIALS_CSV, encoding="utf-8-sig")
            if "code" in mdf.columns:
                def _cell(x: Any) -> str:
                    c = _canonical_internal_stock_code(str(x).strip(), valid_stock_codes)
                    return c if c else ""

                mdf["code"] = mdf["code"].astype(str).map(_cell)
                mdf = mdf[mdf["code"] != ""]
                before = len(mdf)
                mdf = mdf[mdf["code"].isin(valid_stock_codes)]
                if len(mdf) != before:
                    logger.info(
                        "[INFO] prune missing_financials_symbols: %s rows removed (stale codes)",
                        before - len(mdf),
                    )
                mdf.sort_values("code").to_csv(MISSING_FINANCIALS_CSV, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.warning("prune missing_financials_symbols.csv skip: %s", e)

def _audit_v2_legacy_statement_fields(
    legacy: List[Dict[str, Any]],
    log_code: str,
    *,
    financial_data_mode: Optional[str] = None,
) -> None:
    """convert 結果の最新行について主要キーを監査（変換後 legacy を参照）。空件は銘柄別 WARNING ではなくバッチ集計。"""
    if not legacy:
        record_v2_legacy_empty_stock(log_code)
        return
    latest = legacy[0]
    summary_only = financial_data_mode == "summary_only"
    groups: List[Tuple[str, Tuple[str, ...]]] = [
        ("TotalAssets", ("TotalAssets",)),
        ("EquityAttributableToOwnersOfParent", ("EquityAttributableToOwnersOfParent", "Equity", "NetAssets", "OwnersEquity")),
        ("CurrentAssets", ("CurrentAssets",)),
        ("CurrentLiabilities", ("CurrentLiabilities",)),
        ("NetCashProvidedByUsedInOperatingActivities", (
            "NetCashProvidedByUsedInOperatingActivities",
            "CashFlowsFromOperatingActivities",
            "OperatingCashFlow",
            "CFO",
            "Cfo",
            "cfo",
        )),
        ("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock", (
            "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
            "NumberOfIssuedAndOutstandingShares",
            "IssuedShares",
            "SharesOutstanding",
            "ShOutFY",
        )),
    ]
    missing_labels: List[str] = []
    for label, keys in groups:
        if all(_financial_scalar_absent(latest.get(k)) for k in keys):
            missing_labels.append(label)
    if summary_only:
        ocf_label = "NetCashProvidedByUsedInOperatingActivities"
        if ocf_label in missing_labels:
            cf_schema_keys = (
                "CFO", "CFF", "CFI", "OCF", "Cfo", "cfo", "ocf",
            )
            if any(k in latest for k in cf_schema_keys):
                missing_labels = [x for x in missing_labels if x != ocf_label]
                logger.debug(
                    "V2財務変換 [%s]: summary_only で CF スキーマキーはあるが最新行の値が null — "
                    "営業CFは API 上未開示の可能性（監査WARNING省略）",
                    log_code or "?",
                )
    if missing_labels:
        sample_keys = sorted(str(k) for k in latest.keys())[:48]
        summary_only_exempt = ("CurrentAssets", "CurrentLiabilities", "GrossProfit")
        if summary_only:
            severe = [x for x in missing_labels if x not in summary_only_exempt]
            if severe:
                logger.warning(
                    "V2財務変換 [%s]: 最新行で欠損の可能性がある主要キー: %s — V2のFSキー名・エイリアス不足の疑いあり。キー一覧(抜粋)=%s",
                    log_code or "?",
                    severe,
                    sample_keys,
                )
            elif missing_labels:
                logger.info(
                    "V2財務変換 [%s]: summary_only 想定内欠損のみ %s",
                    log_code or "?",
                    missing_labels,
                )
        else:
            logger.warning(
                "V2財務変換 [%s]: 最新行で欠損の可能性がある主要キー: %s — V2のFSキー名・エイリアス不足の疑いあり。キー一覧(抜粋)=%s",
                log_code or "?",
                missing_labels,
                sample_keys,
            )
    else:
        logger.debug("V2財務変換 [%s]: 主要勘定（TotalAssets/Equity/CA/CL/OCF/株数）は最新行で検出", log_code or "?")

    nd = latest.get("NetSales")
    ng = latest.get("GrossProfit")
    nn = latest.get("NetIncomeLoss")
    oi = latest.get("OperatingIncome")
    if summary_only:
        if _financial_scalar_absent(nd) and _financial_scalar_absent(oi):
            logger.warning(
                "V2財務変換 [%s]: summary_only だが NetSales/OperatingIncome が最新行で欠損",
                log_code or "?",
            )
        if _financial_scalar_absent(nn):
            logger.warning(
                "V2財務変換 [%s]: summary_only だが NetIncomeLoss が最新行で欠損",
                log_code or "?",
            )
    elif _financial_scalar_absent(nd) and _financial_scalar_absent(oi):
        logger.debug(
            "V2財務変換 [%s]: NetSales/OperatingIncome が最新行で未検出（四半期のみ等の可能性）",
            log_code or "?",
        )
    if _financial_scalar_absent(ng):
        logger.debug("V2財務変換 [%s]: GrossProfit が最新行で未検出", log_code or "?")
    if not summary_only and _financial_scalar_absent(nn):
        logger.debug("V2財務変換 [%s]: NetIncomeLoss が最新行で未検出", log_code or "?")

def _merge_fs_into_row(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    out = dict(row)
    fs = out.get("FS")
    if isinstance(fs, dict):
        for fk, fv in fs.items():
            if fk not in out or out[fk] is None:
                out[fk] = fv
    return out

def _norm_date_iso(val: Any) -> Optional[str]:
    dt = _parse_optional_date(val)
    return dt.isoformat() if dt is not None else None

def convert_v2_financials_to_legacy_statements(
    summary_rows: List[Dict[str, Any]],
    detail_rows: List[Dict[str, Any]],
    *,
    log_code: Optional[str] = None,
    financial_data_mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    V2 fins/summary + fins/details を既存コードが読む V1 風 statements list[dict] に変換する。
    """
    merged_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def merge_pair(srow: Dict[str, Any], drow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        s_flat = apply_v2_fins_summary_field_aliases(_merge_fs_into_row(srow))
        d_flat = apply_v2_fins_summary_field_aliases(_merge_fs_into_row(drow)) if drow else {}
        out: Dict[str, Any] = {**{k: v for k, v in s_flat.items() if k != "FS"}}
        for k, v in d_flat.items():
            if k == "FS":
                continue
            prev = out.get(k)
            out[k] = _non_null(prev, v)

        legacy_alias_sets: List[Tuple[str, Tuple[str, ...]]] = [
            ("NetSales", ("NetSales", "NetSalesIFRS", "Sales", "RevenueFromOperations")),
            ("OperatingIncome", ("OperatingIncome", "OperatingIncomeIFRS", "OperatingIncomeLoss")),
            ("NetIncomeLoss", ("NetIncomeLoss", "Profit", "ProfitAttributableToOwnersOfParent", "NetIncome")),
            ("NetCashProvidedByUsedInOperatingActivities", (
                "NetCashProvidedByUsedInOperatingActivities",
                "CashFlowsFromOperatingActivities",
                "OperatingCashFlow",
                "CFO",
                "Cfo",
                "cfo",
            )),
            ("TotalAssets", ("TotalAssets", "TA", "TotalAsset")),
            ("EquityAttributableToOwnersOfParent", (
                "EquityAttributableToOwnersOfParent",
                "Equity",
                "NetAssets",
                "OwnersEquity",
                "StockholdersEquity",
            )),
            ("CurrentAssets", ("CurrentAssets",)),
            ("CurrentLiabilities", ("CurrentLiabilities",)),
            ("GrossProfit", ("GrossProfit",)),
            ("CashAndCashEquivalents", (
                "CashAndCashEquivalents",
                "CashAndCashEquivalentsAtEndOfPeriod",
                "Cash",
                "CashEquivalents",
                "CashAndDeposits",
            )),
            ("NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock", (
                "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
                "NumberOfIssuedAndOutstandingShares",
                "IssuedShares",
            )),
            ("DisclosedDate", ("DisclosedDate", "DisclosureDate", "DisclosedAt")),
            ("CurrentPeriodEndDate", ("CurrentPeriodEndDate", "FiscalYearEnd", "PeriodEnd")),
            ("CurrentFiscalYearEndDate", ("CurrentFiscalYearEndDate",)),
            ("TypeOfDocument", ("TypeOfDocument", "DocumentType", "DocType", "ReportType")),
            ("CurrentPeriodType", ("CurrentPeriodType", "QuarterlyOrAnnual", "PeriodType")),
        ]
        for legacy_name, srcs in legacy_alias_sets:
            picked = _non_null(*(out.get(s) for s in srcs))
            if picked is not None:
                out[legacy_name] = picked
        ocf_val = _non_null(
            out.get("NetCashProvidedByUsedInOperatingActivities"),
            out.get("CashFlowsFromOperatingActivities"),
            out.get("CFO"),
            out.get("Cfo"),
            out.get("cfo"),
        )
        if ocf_val is not None:
            out["NetCashProvidedByUsedInOperatingActivities"] = ocf_val
            out["CashFlowsFromOperatingActivities"] = ocf_val
        cfo_final = _pick_cfo_scalar_from_rows(out, s_flat, d_flat)
        if cfo_final is not None:
            out["NetCashProvidedByUsedInOperatingActivities"] = cfo_final
            out["CashFlowsFromOperatingActivities"] = cfo_final
        if not out.get("TypeOfDocument") and isinstance(out.get("CurrentPeriodType"), str):
            out["TypeOfDocument"] = str(out["CurrentPeriodType"])
        return out

    detail_pool = list(detail_rows)
    used_detail_idx: set[int] = set()

    for s in summary_rows:
        dk = (_norm_date_iso(s.get("DisclosedDate")), _norm_date_iso(s.get("CurrentPeriodEndDate")))
        best_j: Optional[int] = None
        for j, dr in enumerate(detail_pool):
            if j in used_detail_idx:
                continue
            ddk = (_norm_date_iso(dr.get("DisclosedDate")), _norm_date_iso(dr.get("CurrentPeriodEndDate")))
            if ddk == dk and dk != (None, None):
                best_j = j
                break
        if best_j is None:
            for j, dr in enumerate(detail_pool):
                if j in used_detail_idx:
                    continue
                ddk = (_norm_date_iso(dr.get("DisclosedDate")), _norm_date_iso(dr.get("CurrentPeriodEndDate")))
                if ddk[0] == dk[0] or ddk[1] == dk[1]:
                    best_j = j
                    break
        dchosen = detail_pool[best_j] if best_j is not None else None
        if best_j is not None:
            used_detail_idx.add(best_j)
        one = merge_pair(s, dchosen)
        k1 = _norm_date_iso(one.get("DisclosedDate")) or ""
        k2 = _norm_date_iso(one.get("CurrentPeriodEndDate")) or ""
        k3 = str(one.get("TypeOfDocument") or one.get("CurrentPeriodType") or "")
        key = (k1, k2, k3)
        if key not in merged_by_key:
            merged_by_key[key] = one
        else:
            prev = merged_by_key[key]
            for kk, vv in one.items():
                if prev.get(kk) is None and vv is not None:
                    prev[kk] = vv

    for j, dr in enumerate(detail_pool):
        if j in used_detail_idx:
            continue
        one = merge_pair({}, dr)
        k1 = _norm_date_iso(one.get("DisclosedDate")) or ""
        k2 = _norm_date_iso(one.get("CurrentPeriodEndDate")) or ""
        k3 = str(one.get("TypeOfDocument") or one.get("CurrentPeriodType") or "")
        key = (k1, k2, k3)
        if key not in merged_by_key:
            merged_by_key[key] = one

    legacy = list(merged_by_key.values())
    legacy.sort(key=lambda r: _statement_sort_key(r), reverse=True)
    for row in legacy:
        ocf_inj = _pick_cfo_scalar_from_rows(row)
        if ocf_inj is not None:
            row["NetCashProvidedByUsedInOperatingActivities"] = ocf_inj
            row["CashFlowsFromOperatingActivities"] = ocf_inj
    if log_code:
        _audit_v2_legacy_statement_fields(
            legacy, log_code, financial_data_mode=financial_data_mode,
        )
    return legacy

def fetch_prices_v2_with_meta(
    session: requests.Session,
    code: str,
    lookback_days: int = LOOKBACK_DAYS,
    *,
    cache_name: Optional[str] = None,
    bypass_cache: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    today = datetime.date.today().strftime("%Y%m%d")
    cn = cache_name or f"prices_{code}"
    cache_file = CACHE_DIR / f"{cn}_{today}.csv"

    if cache_file.exists() and not bypass_cache:
        try:
            df = pd.read_csv(cache_file)
            if not df.empty:
                dfn = normalize_prices_v2(df)
                return dfn, {"http": 200, "rows": len(dfn), "from_cache": True, "transient": False}
        except Exception:
            pass

    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = datetime.date.today().strftime("%Y-%m-%d")
    rows: List[Dict[str, Any]] = []
    last_st = 200
    last_err = ""
    for qc in api_code_candidates(code):
        rows_try, st, err = paginate_v2_endpoint(session, "equities/bars/daily", {"code": qc, "from": start, "to": end})
        last_st = int(st)
        last_err = (err or "")[:400]
        if rows_try:
            rows = rows_try
            last_st = 200
            break
        if st != 200:
            logger.warning("equities/bars/daily HTTP %s code=%s", st, qc)
    if not rows:
        transient = last_st in (408, 409, 429, 500, 502, 503, 504) or last_st >= 520 or last_st == 599
        return pd.DataFrame(), {
            "http": last_st,
            "rows": 0,
            "err": last_err,
            "from_cache": False,
            "transient": transient,
        }
    df = normalize_prices_v2(pd.DataFrame(rows))
    try:
        df.to_csv(cache_file, index=False)
    except Exception:
        pass
    return df, {"http": 200, "rows": len(df), "from_cache": False, "transient": False}


def fetch_prices_v2(
    session: requests.Session,
    code: str,
    lookback_days: int = LOOKBACK_DAYS,
    *,
    cache_name: Optional[str] = None,
    bypass_cache: bool = False,
) -> pd.DataFrame:
    """
    銘柄ごとに日足を取得し、既存標準列へ正規化する（code 単位取得互換）。
    """
    df, _ = fetch_prices_v2_with_meta(
        session, code, lookback_days, cache_name=cache_name, bypass_cache=bypass_cache,
    )
    return df

def fetch_prices_by_date_v2_placeholder(session: requests.Session, _date: str) -> pd.DataFrame:
    """
    将来改善用プレースホルダ: date 指定で全銘柄日足を取得する経路。
    初回移行では collect/analyze の主経路には使わない。
    """
    raise NotImplementedError("date 単位一括取得は次フェーズで実装予定")

# ------------------------------------------------------------
# Graceful Shutdown
# ------------------------------------------------------------
class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        self._signal_received = False
        self._final_user_message_printed = False
        try:
            signal.signal(signal.SIGINT, self.exit_gracefully)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self.exit_gracefully)
        except Exception:
            pass

    def exit_gracefully(self, signum, frame):
        if self._signal_received:
            self.shutdown = True
            return
        self._signal_received = True
        self.shutdown = True
        _cli_print(
            f"\n⚠️ 中断シグナル受信: {signum}\n🛑 現在の処理を区切りで停止します",
            f"\n[中断] シグナル受信: {signum}\n現在の処理を区切りで停止します",
        )

    def print_safe_exit_once(self) -> None:
        if not self.shutdown:
            return
        if self._final_user_message_printed:
            return
        self._final_user_message_printed = True
        _cli_print("✅ 安全に終了しました", "[終了] 安全に終了しました")


graceful_shutdown = GracefulShutdown()


def _sleep_interruptible(total_seconds: float, chunk_seconds: float = 1.0) -> None:
    """長い time.sleep を分割し、graceful_shutdown 時に抜けられるようにする。"""
    if total_seconds <= 0:
        return
    end = time.time() + float(total_seconds)
    while time.time() < end:
        if graceful_shutdown.shutdown:
            return
        rem = end - time.time()
        time.sleep(min(chunk_seconds, rem) if rem > 0 else 0.0)


def _executor_shutdown_interrupt(ex: ThreadPoolExecutor, futs: List[Any]) -> None:
    """未着手の future を cancel し、プールを待たずに停止（ベストエフォート）。"""
    for f in futs:
        try:
            f.cancel()
        except Exception:
            pass
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)

# ------------------------------------------------------------
# レートリミッタ + 認証セッション
# ------------------------------------------------------------
class APIRateLimiter:
    """V2: 公式は主にリクエスト/分。JQ_RPD は任意のセルフ日次上限（未設定なら日次チェックなし）。"""
    def __init__(self, rpm: int = 60, rpd: Optional[int] = None):
        self.requests_per_minute = rpm
        self.requests_per_day = rpd  # None なら無制限（分間制御のみ）
        self.base_delay = 1.5
        self.request_timestamps: List[datetime.datetime] = []
        self.daily_count = 0
        self.last_reset = datetime.date.today()

    def wait_if_needed(self):
        now = datetime.datetime.now()
        if now.date() > self.last_reset:
            self.daily_count = 0
            self.last_reset = now.date()

        if self.requests_per_day is not None and self.daily_count >= self.requests_per_day:
            raise RuntimeError("日次レート制限到達（JQ_RPD）")

        one_minute_ago = now - datetime.timedelta(minutes=1)
        self.request_timestamps = [t for t in self.request_timestamps if t > one_minute_ago]

        if len(self.request_timestamps) >= self.requests_per_minute:
            wait = 61 - (now - min(self.request_timestamps)).total_seconds()
            if wait > 0:
                time.sleep(wait)

        time.sleep(self.base_delay)

    def mark(self):
        now = datetime.datetime.now()
        self.request_timestamps.append(now)
        if self.requests_per_day is not None:
            self.daily_count += 1

class AuthSession(requests.Session):
    """J-Quants V2 API キー＋レート制限対応 Session"""
    def __init__(self, limiter: APIRateLimiter, ini_file: str = "api.ini"):
        super().__init__()
        self.limiter = limiter
        self.ini_file = ini_file
        # fins/details: None=未試行、True=取得成功、False=403等でセッション中以降スキップ
        self.fins_details_available: Optional[bool] = None
        self.fins_details_disabled_reason: Optional[str] = None
        self.fins_details_last_status: Optional[int] = None

    def request(self, method, url, **kwargs):
        """送信直前に wait_if_needed。V2 は 401 でトークン更新しない。"""
        MAX = 5
        timeout = kwargs.pop("timeout", 30)

        for attempt in range(1, MAX + 1):
            self.limiter.wait_if_needed()
            try:
                resp = super().request(method, url, timeout=timeout, **kwargs)
            except requests.RequestException:
                if attempt == MAX:
                    raise
                time.sleep(1.5 * attempt)
                continue

            self.limiter.mark()

            if resp.status_code == 401:
                logger.warning("HTTP 401 Unauthorized（APIキーを確認してください）: %s", url)

            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt == MAX:
                    return resp
                time.sleep(min(2 ** attempt, 30))
                continue

            return resp

        raise RuntimeError(f"{method} {url} failed after {MAX} attempts")


def _resolve_api_ini_path(ini_rel: str) -> Optional[Path]:
    """api.ini を「スクリプト配置ディレクトリ」優先で解決する（cwd だけに依存しない）。"""
    name = (ini_rel or "api.ini").strip() or "api.ini"
    p = Path(name)
    if p.is_absolute():
        return p if p.is_file() else None
    script_dir = Path(__file__).resolve().parent
    for root in (script_dir, Path.cwd()):
        cand = (root / name).resolve()
        if cand.is_file():
            return cand
    return None


def _load_api_key_from_ini(ini_path: Path) -> str:
    """[DEFAULT] API_KEY を読む。interpolation 無効で % 等を安全に扱う。"""
    cfg = configparser.ConfigParser(interpolation=None)
    if not cfg.read(ini_path, encoding="utf-8"):
        return ""
    # has_section("DEFAULT") は stdlib の仕様で常に False のため使えない（[DEFAULT] は cfg["DEFAULT"] で読む）
    return (cfg["DEFAULT"].get("API_KEY") or "").strip()


_PLACEHOLDER_API_KEYS = frozenset(
    {
        "",
        "PASTE_YOUR_JQUANTS_V2_API_KEY_HERE",
    }
)


def get_authenticated_session_jquants(ini_file: str = "api.ini") -> requests.Session:
    api_key = (os.getenv("JQUANTS_API_KEY") or os.getenv("JQ_API_KEY") or "").strip()
    ini_resolved = _resolve_api_ini_path(ini_file)
    raw_ini_key = ""
    if not api_key and ini_resolved:
        raw_ini_key = _load_api_key_from_ini(ini_resolved)
        api_key = raw_ini_key
        if api_key in _PLACEHOLDER_API_KEYS:
            api_key = ""
    if not api_key:
        base = Path(__file__).resolve().parent
        hints: List[str] = [
            "APIキー未設定: 環境変数 JQUANTS_API_KEY / JQ_API_KEY、または api.ini の [DEFAULT] API_KEY を設定してください。",
        ]
        if ini_resolved is None:
            hints.append(
                f"api.ini が見つかりません（{base}、または現在のフォルダ {Path.cwd()} に {ini_file} を配置）"
            )
        elif raw_ini_key == "PASTE_YOUR_JQUANTS_V2_API_KEY_HERE":
            hints.append("api.ini の API_KEY がプレースホルダのままです。実キーに置き換えてください。")
        elif raw_ini_key == "":
            hints.append("api.ini はありますが [DEFAULT] API_KEY が空です。")
        raise RuntimeError(" ".join(hints))

    rpm = int(os.getenv("JQ_RPM", "60"))
    rpd_env = os.getenv("JQ_RPD")
    rpd = int(rpd_env) if rpd_env else None
    limiter = APIRateLimiter(rpm=rpm, rpd=rpd)
    ini_for_session = str(ini_resolved) if ini_resolved else ini_file
    session = AuthSession(limiter, ini_file=ini_for_session)
    session.headers.update({"x-api-key": api_key})
    _cli_print("✅ J-Quants API V2（x-api-key）", "[OK] J-Quants API V2 (x-api-key)")
    return session

# ------------------------------------------------------------
# 永続キャッシュ（凍結）
# ------------------------------------------------------------
class FrozenCache:
    BASE = CACHE_DIR / "frozen"
    def __init__(self):
        (self.BASE / "prices").mkdir(parents=True, exist_ok=True)
        (self.BASE / "statements").mkdir(parents=True, exist_ok=True)

    def prices_path(self, code: str) -> Path:
        return self.BASE / "prices" / f"{code}.csv"

    def stmts_path(self, code: str) -> Path:
        return self.BASE / "statements" / f"{code}.json"

    def save_prices(self, code: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        df.to_csv(self.prices_path(code), index=False)

    def load_prices(self, code: str) -> Optional[pd.DataFrame]:
        p = self.prices_path(code)
        if not p.exists():
            return None
        try:
            return pd.read_csv(p)
        except Exception:
            return None

    def save_statements(self, code: str, stmts: List[dict], financial_meta: Optional[Dict[str, Any]] = None) -> None:
        obj: Dict[str, Any] = {"statements": stmts}
        if financial_meta:
            for k, v in financial_meta.items():
                if k != "statements":
                    obj[k] = v
        self.stmts_path(code).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    def load_statement_bundle(self, code: str) -> Tuple[List[dict], Dict[str, Any]]:
        p = self.stmts_path(code)
        if not p.exists():
            return [], {}
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return [], {}
        stmts_raw = j.pop("statements", [])
        stmts = stmts_raw if isinstance(stmts_raw, list) else []
        meta = dict(j)
        if stmts and "financial_data_mode" not in meta:
            meta.setdefault("financial_data_mode", "legacy_frozen_statement_cache")
            meta.setdefault("fins_details_available", None)
            meta.setdefault("fins_details_status", None)
            meta.setdefault("fins_details_error", None)
        return stmts, meta

    def load_statements(self, code: str) -> List[dict]:
        stmts, _ = self.load_statement_bundle(code)
        return stmts

    def has_prices(self, code: str) -> bool:
        return self.prices_path(code).exists()

    def has_all(self, code: str, max_age_days: Optional[int] = None) -> bool:
        p1, p2 = self.prices_path(code), self.stmts_path(code)
        try:
            st1, st2 = p1.stat(), p2.stat()
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning("FrozenCache.has_all: stat失敗 %s", e)
            return False
        if max_age_days is None:
            return True
        try:
            now = time.time()
            age_days_prices = (now - st1.st_mtime) / 86400.0
            age_days_stmts = (now - st2.st_mtime) / 86400.0
            oldest = max(age_days_prices, age_days_stmts)
            return oldest <= max_age_days
        except OSError as e:
            logger.warning("FrozenCache.has_all: 鮮度判定失敗 %s", e)
            return False

# ------------------------------------------------------------
# セクター平均・銘柄リスト
# ------------------------------------------------------------
class DynamicSectorAverages:
    SECTOR_MEDIANS = {
        "電気機器": {"ca_ratio": 0.62, "cl_ratio": 0.38, "gpm": 0.31},
        "半導体":   {"ca_ratio": 0.55, "cl_ratio": 0.42, "gpm": 0.39},
        "銀行":     {"ca_ratio": 0.28, "cl_ratio": 0.90, "gpm": 0.20},
        "情報・通信業": {"ca_ratio": 0.57, "cl_ratio": 0.32, "gpm": 0.34},
        "サービス": {"ca_ratio": 0.60, "cl_ratio": 0.35, "gpm": 0.29},
        "化学":     {"ca_ratio": 0.58, "cl_ratio": 0.37, "gpm": 0.27},
        "小売": {"ca_ratio": 0.58, "cl_ratio": 0.38, "gpm": 0.26},
        "卸売": {"ca_ratio": 0.55, "cl_ratio": 0.42, "gpm": 0.18},
        "建設": {"ca_ratio": 0.62, "cl_ratio": 0.38, "gpm": 0.20},
        "陸運": {"ca_ratio": 0.50, "cl_ratio": 0.45, "gpm": 0.15},
        "海運": {"ca_ratio": 0.45, "cl_ratio": 0.48, "gpm": 0.22},
        "空運": {"ca_ratio": 0.48, "cl_ratio": 0.44, "gpm": 0.18},
        "電気・ガス": {"ca_ratio": 0.35, "cl_ratio": 0.55, "gpm": 0.25},
        "食品": {"ca_ratio": 0.55, "cl_ratio": 0.35, "gpm": 0.28},
        "機械": {"ca_ratio": 0.58, "cl_ratio": 0.38, "gpm": 0.28},
        "自動車": {"ca_ratio": 0.55, "cl_ratio": 0.40, "gpm": 0.20},
        "医薬品": {"ca_ratio": 0.58, "cl_ratio": 0.32, "gpm": 0.65},
        "商社": {"ca_ratio": 0.52, "cl_ratio": 0.38, "gpm": 0.15},
        "保険": {"ca_ratio": 0.22, "cl_ratio": 0.85, "gpm": 0.22},
        "証券": {"ca_ratio": 0.35, "cl_ratio": 0.62, "gpm": 0.35},
        "不動産": {"ca_ratio": 0.55, "cl_ratio": 0.40, "gpm": 0.40},
        "ゲーム": {"ca_ratio": 0.60, "cl_ratio": 0.35, "gpm": 0.55},
        "その他":   {"ca_ratio": 0.60, "cl_ratio": 0.40, "gpm": 0.25},
    }

    def __init__(self, session: requests.Session):
        self.session = session
        self.sector_cache: dict = {}
        self.cache_timestamp: Optional[float] = None
        self.cache_duration = 3600

    @staticmethod
    def normalize_sector(sector: str) -> str:
        s = (sector or "").strip()
        if not s:
            return "その他"
        # J-Quants Sector33Name（および近い表記）→ 社内セクターキー
        if "銀行" in s:
            return "銀行"
        if "保険" in s:
            return "保険"
        if "証券" in s or "商品先物" in s:
            return "証券"
        if "不動産" in s:
            return "不動産"
        if "小売業" in s or s == "小売":
            return "小売"
        if "卸売業" in s or s == "卸売":
            return "卸売"
        if "建設業" in s:
            return "建設"
        if "陸運業" in s or s == "陸運":
            return "陸運"
        if "海運業" in s or s == "海運":
            return "海運"
        if "空運業" in s or s == "空運":
            return "空運"
        if "電気・ガス業" in s or "ガス業" in s:
            return "電気・ガス"
        if "食料品" in s or (s.startswith("食品") and len(s) <= 4):
            return "食品"
        if "輸送用機器" in s or "自動車" in s:
            return "自動車"
        if s == "機械" or "機械器具" in s:
            return "機械"
        if "化学" in s:
            return "化学"
        if "電気機器" in s:
            return "電気機器"
        if "半導体" in s:
            return "半導体"
        if "情報" in s or "通信" in s:
            return "情報・通信業"
        if "サービス業" in s or (s.endswith("サービス") and "情報" not in s):
            return "サービス"
        if "医薬品" in s or "製薬" in s:
            return "医薬品"
        if "卸売" in s and "小売" not in s:
            return "卸売"
        if "運輸" in s and "機器" not in s:
            return "陸運"
        known = DynamicSectorAverages.SECTOR_MEDIANS.keys()
        if s in known:
            return s
        return "その他"

    @staticmethod
    def get_sector_static(stock_code: str) -> str:
        sector_mapping = {
            '7203': '自動車','7267':'自動車','7269':'自動車','7270':'自動車','7261':'自動車','7202':'自動車','7211':'自動車',
            '8035':'半導体','6861':'半導体','6594':'半導体','6503':'半導体','6723':'半導体','6752':'半導体','6981':'半導体',
            '6758':'電気機器','6501':'電気機器','6954':'電気機器','6702':'電気機器','6976':'電気機器',
            '8306':'銀行','8316':'銀行','8411':'銀行','8331':'銀行','8354':'銀行','8393':'銀行',
            '9984':'情報・通信業','9432':'情報・通信業','9433':'情報・通信業','4689':'情報・通信業','3659':'情報・通信業','4751':'情報・通信業',
            '4568':'医薬品','4519':'医薬品','4523':'医薬品','4503':'医薬品','4506':'医薬品','4507':'医薬品',
            '8058':'商社','8031':'商社','2768':'商社','8002':'商社','8001':'商社','8053':'商社',
            '9983':'小売','3382':'小売','8267':'小売','3086':'小売','3099':'小売','8233':'小売',
            '4661':'サービス','9602':'サービス','2432':'サービス','4324':'サービス','6178':'サービス',
            '7974':'ゲーム','9684':'ゲーム','7832':'ゲーム','3765':'ゲーム',
            '4901':'化学','4452':'化学','4063':'化学','4005':'化学','4188':'化学','4183':'化学',
            '5401':'鉄鋼','5411':'鉄鋼','5406':'鉄鋼','5423':'鉄鋼',
            '8801':'不動産','8802':'不動産','3289':'不動産','1928':'不動産',
            '2914':'食品','2502':'食品','2269':'食品','2503':'食品','2801':'食品','2871':'食品',
            '6367':'機械','6473':'機械','6326':'機械',
            '9020':'運輸','9021':'運輸','9022':'運輸','9101':'運輸','9104':'運輸','9107':'運輸',
            '8473':'証券','8601':'証券','8604':'証券',
            '5020':'エネルギー','1605':'エネルギー','1662':'エネルギー',
            '1801':'建設','1802':'建設','1803':'建設','1928':'建設',
        }
        return sector_mapping.get(stock_code, "その他")

    def is_cache_valid(self) -> bool:
        if not self.cache_timestamp:
            return False
        return (time.time() - self.cache_timestamp) < self.cache_duration

    @staticmethod
    def default_sector_average(sector: str) -> dict:
        defaults = {
            '自動車': {'ps': 0.8},
            '半導体': {'ps': 4.5},
            '電気機器': {'ps': 1.8},
            '銀行': {'ps': 2.5},
            '情報・通信業': {'ps': 1.2},
            '医薬品': {'ps': 3.8},
            '商社': {'ps': 0.4},
            '小売': {'ps': 0.8},
            '卸売': {'ps': 0.35},
            '建設': {'ps': 0.45},
            '陸運': {'ps': 0.55},
            '海運': {'ps': 0.5},
            '空運': {'ps': 0.85},
            '電気・ガス': {'ps': 0.6},
            '食品': {'ps': 0.9},
            '機械': {'ps': 1.1},
            'サービス': {'ps': 2.2},
            'ゲーム': {'ps': 3.5},
            '化学': {'ps': 1.0},
            '鉄鋼': {'ps': 0.35},
            '不動産': {'ps': 0.9},
            'エネルギー': {'ps': 0.5},
            '証券': {'ps': 1.8},
            'その他': {'ps': 1.5},
        }
        default = defaults.get(sector, defaults['その他'])
        return {
            **default,
            'sample_count': 0,
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': 'static_default'
        }

    def get_default_sector_average(self, sector: str) -> dict:
        return self.default_sector_average(sector)

    def get_sector_averages(self, force_refresh: bool = False) -> dict:
        if not force_refresh and self.is_cache_valid() and self.sector_cache:
            _cli_print("📊 セクター平均: メモリキャッシュ", "[セクター平均] メモリキャッシュ")
            return self.sector_cache

        cache_file = CACHE_DIR / "sector_averages.json"
        if cache_file.exists() and not force_refresh:
            try:
                j = json.loads(cache_file.read_text(encoding="utf-8"))
                if time.time() - j.get("timestamp", 0) <= 86400:
                    self.sector_cache = j.get("data", {})
                    self.cache_timestamp = time.time()
                    _cli_print("📊 セクター平均: ファイルキャッシュ", "[セクター平均] ファイルキャッシュ")
                    return self.sector_cache
            except Exception:
                pass

        _cli_print("📊 セクター平均: 静的デフォルト", "[セクター平均] 静的デフォルト")
        sectors = [
            '自動車', '半導体', '電気機器', '銀行', '情報・通信業', '医薬品', '商社',
            '小売', '卸売', '建設', '陸運', '海運', '空運', '電気・ガス', '食品', '機械',
            'サービス', 'ゲーム', '化学', '鉄鋼', '不動産', 'エネルギー', '証券', 'その他',
        ]
        data = {s: self.default_sector_average(s) for s in sectors}
        cache_file.write_text(json.dumps({"timestamp": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
        self.sector_cache = data
        self.cache_timestamp = time.time()
        return data

    def get_fallback_stock_list_v2(self) -> list[dict]:
        return [
            {"Code":"7203","CompanyName":"トヨタ自動車","Sector33Name":"輸送用機器","MarketCode":"111"},
            {"Code":"8306","CompanyName":"三菱UFJフィナンシャルG","Sector33Name":"銀行業","MarketCode":"111"},
            {"Code":"6758","CompanyName":"ソニーG","Sector33Name":"電気機器","MarketCode":"111"},
            {"Code":"9984","CompanyName":"ソフトバンクG","Sector33Name":"情報・通信業","MarketCode":"111"},
            {"Code":"8035","CompanyName":"東京エレクトロン","Sector33Name":"電気機器","MarketCode":"111"},
        ]

    def get_stock_list_v2(self, force_refresh: bool = False) -> pd.DataFrame:
        try:
            today = datetime.date.today().strftime("%Y%m%d")
            # CoName/Mkt 正規化後のマスタ（旧キャッシュは列不足のため別名）
            cache_file = CACHE_DIR / f"sector_stock_list_v2norm_{today}.csv"
            if cache_file.exists() and not force_refresh:
                df_cached = pd.read_csv(cache_file)
                return enhance_stock_list_with_sectors(ensure_instrument_type_column(df_cached))

            _cli_print("📋 銘柄リスト取得…", "[銘柄リスト] 取得中…")
            rows = get_v2_all_pages(self.session, "equities/master", {})
            df = normalize_equities_master_v2(pd.DataFrame(rows))
            if not df.empty:
                if "Code" in df.columns:
                    df["Code"] = df["Code"].astype(str).str.strip()
                    df = df.dropna(subset=["Code"])
                    df = df[df["Code"].str.match(r"^\d{4}$", na=False)].drop_duplicates("Code")
                df = ensure_instrument_type_column(df)
                df = enhance_stock_list_with_sectors(df)
                df.to_csv(cache_file, index=False)
                return df

            fb = pd.DataFrame(self.get_fallback_stock_list_v2())
            fb = ensure_instrument_type_column(fb)
            fb = enhance_stock_list_with_sectors(fb)
            fb.to_csv(cache_file, index=False)
            return fb
        except Exception:
            fb = pd.DataFrame(self.get_fallback_stock_list_v2())
            fb = ensure_instrument_type_column(fb)
            fb = enhance_stock_list_with_sectors(fb)
            return fb

    def calculate_sector_averages_from_cache(self, max_samples_per_sector: int = 100) -> dict:
        try:
            tasks = build_offline_analysis_tasks(self.session)
            if not tasks:
                _cli_print("📊 セクター平均計算: キャッシュデータが不足しています", "[セクター平均計算] キャッシュデータが不足しています")
                return {}

            _cli_print(f"📊 セクター平均計算: {len(tasks)}銘柄から計算中...", f"[セクター平均計算] {len(tasks)}銘柄から計算中...")
            results = []
            max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
            task_slice = tasks[: max_samples_per_sector * 20]
            ex = ThreadPoolExecutor(max_workers=max_workers)
            futs = [
                ex.submit(
                    analyze_single_stock_complete_v3,
                    self.session, {}, code, name, market, sector,
                    offline=True
                )
                for (code, name, market, sector) in task_slice
            ]
            try:
                for i, fut in enumerate(as_completed(futs), 1):
                    if graceful_shutdown.shutdown:
                        logger.info("shutdown requested; stopping sector average loop")
                        break
                    res = fut.result()
                    if res.get("success") and res.get("ps_ratio") is not None:
                        results.append(res)
                    if i % 100 == 0:
                        _cli_print(
                            f"  ⏱ {i}/{len(futs)} 完了 (有効データ={len(results)})",
                            f"  [{i}/{len(futs)}] 完了 (有効データ={len(results)})",
                        )
            finally:
                _executor_shutdown_interrupt(ex, futs)

            if graceful_shutdown.shutdown and len(results) < len(futs):
                _cli_print(
                    "🛑 セクター平均計算を中断しました（結果は未反映の可能性があります）",
                    "[中断] セクター平均計算を打ち切り",
                )

            if not results:
                _cli_print("📊 セクター平均計算: 有効なデータがありません", "[セクター平均計算] 有効なデータがありません")
                return {}

            df = pd.DataFrame([
                {
                    "sector": self.normalize_sector(r.get("sector_name") or ""),
                    "ps": r.get("ps_ratio"),
                    "per": r.get("per"),
                }
                for r in results
            ])

            sector_stats = {}
            for sector in df["sector"].unique():
                sector_df = df[df["sector"] == sector]
                if len(sector_df) < 3:
                    continue

                ps_values = sector_df["ps"].dropna()
                per_values = sector_df["per"].dropna()

                sector_stats[sector] = {
                    "ps": float(ps_values.median()) if len(ps_values) > 0 else None,
                    "per": float(per_values.median()) if len(per_values) > 0 else None,
                    "sample_count": len(sector_df),
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_source": "calculated_from_cache"
                }

            _cli_print(f"📊 セクター平均計算完了: {len(sector_stats)}セクター", f"[セクター平均計算完了] {len(sector_stats)}セクター")
            return sector_stats

        except Exception as e:
            _cli_print(f"⚠️ セクター平均計算エラー: {e}", f"[警告] セクター平均計算エラー: {e}")
            import traceback
            traceback.print_exc()
            return {}

# ------------------------------------------------------------
# FinancialDataManager（最小）
# ------------------------------------------------------------
class FinancialDataManager:
    def __init__(self, session: requests.Session):
        self.session = session
        self.base_url = JQUANTS_API_BASE
        self.cache_dir = CACHE_DIR
        self._last_financial_fetch_meta: Dict[str, Any] = {}

    def get_stock_list_v2(self, force_refresh: bool = False) -> pd.DataFrame:
        helper = DynamicSectorAverages(self.session)
        return helper.get_stock_list_v2(force_refresh=force_refresh)

    def get_last_financial_fetch_meta(self) -> Dict[str, Any]:
        return dict(self._last_financial_fetch_meta)

    def fetch_financials_v2_as_statements(self, code: str, force_refresh: bool = False) -> List[dict]:
        """
        fins/summary と（利用可能なら）fins/details を取得し、既存の build_financial_history が読める
        V1 互換 statements list[dict] に変換する。
        診断キーは self._last_financial_fetch_meta に格納する。
        """
        global _SUMMARY_ONLY_MODE_INFO_LOGGED
        self._last_financial_fetch_meta = {}
        cache_key = f"v2_fins_legacy_statements_{code}"
        fpath = self.cache_dir / f"{cache_key}.json"
        ttl_sec = 12 * 3600
        if not force_refresh and fpath.exists():
            try:
                mtime = datetime.datetime.fromtimestamp(fpath.stat().st_mtime)
                if (datetime.datetime.now() - mtime).total_seconds() < ttl_sec:
                    j = json.loads(fpath.read_text(encoding="utf-8"))
                    stmts = j.get("statements", [])
                    meta = {k: v for k, v in j.items() if k != "statements"}
                    self._last_financial_fetch_meta = meta
                    if stmts:
                        return stmts
            except Exception:
                pass

        session = self.session
        summary_rows: List[Dict[str, Any]] = []
        summary_http = 200
        summary_err = ""
        for qc in api_code_candidates(code):
            sr, st, err_snip = paginate_v2_endpoint(session, "fins/summary", {"code": qc})
            summary_http = int(st)
            summary_err = (err_snip or "")[:400]
            if sr:
                summary_rows = sr
                summary_http = 200
                break
            if st != 200:
                logger.warning("fins/summary HTTP %s code=%s", st, qc)

        detail_rows: List[Dict[str, Any]] = []
        ds: Optional[int] = None
        der = ""
        fds = getattr(session, "fins_details_available", None)

        if fds is False:
            ds = getattr(session, "fins_details_last_status", None)
            der = getattr(session, "fins_details_disabled_reason", "") or ""
        else:
            for qc in api_code_candidates(code):
                dr, dst, der = paginate_v2_endpoint(session, "fins/details", {"code": qc})
                ds = int(dst) if dst is not None else None
                session.fins_details_last_status = dst
                if dst == 403:
                    session.fins_details_available = False
                    session.fins_details_disabled_reason = der or "HTTP 403"
                    detail_rows = []
                    break
                if dr:
                    detail_rows = dr
                    session.fins_details_available = True
                    break

        detail_ok = bool(detail_rows)
        summary_ok = bool(summary_rows)
        financial_data_mode = "summary_plus_details" if (detail_ok and summary_ok) else "summary_only"
        fins_details_available = detail_ok
        meta_out: Dict[str, Any] = {
            "financial_data_mode": financial_data_mode,
            "fins_details_available": fins_details_available,
            "fins_details_status": ds,
            "fins_details_error": der[:500] if der else "",
            "api_status_summary": summary_http,
            "api_error_summary": summary_err,
            "summary_rows": len(summary_rows),
            "detail_rows": len(detail_rows),
        }
        self._last_financial_fetch_meta = meta_out

        if financial_data_mode == "summary_only" and summary_ok and not _SUMMARY_ONLY_MODE_INFO_LOGGED:
            logger.info(
                "[INFO] fins/details unavailable; running in summary_only mode. Piotroski coverage may be lower. "
                "Affected: CurrentAssets, CurrentLiabilities, GrossProfit, gross_profit_margin, current_ratio_up, gpm_up."
            )
            _SUMMARY_ONLY_MODE_INFO_LOGGED = True

        stmts = convert_v2_financials_to_legacy_statements(
            summary_rows,
            detail_rows,
            log_code=str(code).strip(),
            financial_data_mode=financial_data_mode,
        )
        try:
            blob = {"statements": stmts}
            blob.update(meta_out)
            fpath.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return stmts

    def fetch_statements(self, code: str, force_refresh: bool = False) -> List[dict]:
        """既存インタフェース維持。内部は V2 summary+details → legacy statements。"""
        return self.fetch_financials_v2_as_statements(code, force_refresh=force_refresh)

    def _fill_missing_fields(self, fin: dict) -> dict:
        """診断・セクター比較用の欠損補完。Piotroski/バリュエーション本計算には流さないこと（analyze では raw_fin を使用）。"""
        cur, prev = fin.get("current", {}), fin.get("previous", {})
        imputed: dict[str, str] = {}

        for fld in ("current_assets", "current_liabilities", "gross_profit_margin", "shares_outstanding"):
            if cur.get(fld) is None and prev.get(fld) is not None:
                cur[fld] = prev.get(fld)
                imputed[fld] = "previous_period"

        sector = DynamicSectorAverages.normalize_sector(fin.get("sector", "その他"))
        med = DynamicSectorAverages.SECTOR_MEDIANS.get(sector, DynamicSectorAverages.SECTOR_MEDIANS["その他"])
        ca_ratio = med.get("ca_ratio")
        cl_ratio = med.get("cl_ratio")
        gpm_med  = med.get("gpm")

        if (cur.get("current_assets") is None and cur.get("total_assets") and ca_ratio):
            cur["current_assets"] = cur["total_assets"] * ca_ratio
            imputed["current_assets"] = "sector_ratio"

        if (cur.get("current_liabilities") is None and cur.get("total_assets") and cur.get("equity") and cl_ratio):
            cur["current_liabilities"] = (cur["total_assets"] - cur["equity"]) * cl_ratio
            imputed["current_liabilities"] = "sector_ratio"

        if cur.get("gross_profit_margin") is None and gpm_med:
            cur["gross_profit_margin"] = gpm_med * 0.95
            imputed["gross_profit_margin"] = "sector_median_discounted"

        fin["current"] = cur
        fin["previous"] = prev
        for k, v in cur.items():
            fin[f"current_{k}"] = v
        fin["_imputation"] = {
            "has_imputation": bool(imputed),
            "field_count": len(imputed),
            "fields": imputed,
        }
        return fin

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def enhance_stock_list_with_sectors(df: pd.DataFrame) -> pd.DataFrame:
    if "Code" not in df.columns:
        return df
    if "Sector33Name" not in df.columns:
        df["Sector33Name"] = df["Code"].astype(str).map(DynamicSectorAverages.get_sector_static).fillna("その他")
    if "MarketCode" not in df.columns:
        df["MarketCode"] = ""
    if "CompanyName" not in df.columns:
        df["CompanyName"] = ""
    if "CoNameEn" not in df.columns:
        df["CoNameEn"] = ""
    if "Mkt" not in df.columns:
        df["Mkt"] = df["MarketCode"].astype(str) if "MarketCode" in df.columns else ""
    if "MktNm" not in df.columns:
        df["MktNm"] = ""
    base_cols = ["Code", "CompanyName", "Sector33Name", "MarketCode"]
    extra_ordered = ("instrument_type", "CoNameEn", "Mkt", "MktNm")
    extra_cols = [c for c in extra_ordered if c in df.columns]
    return df[base_cols + extra_cols]

# ------------------------------------------------------------
# 銘柄名フィルタ（全銘柄収集・一覧用。--phase single は通さない）
# ------------------------------------------------------------
# J-Quants V2 equities/master の Mkt（実データ確認）:
# 0109=ETF・PRO 等、0111/0112/0113=プライム・スタンダード・グロース、0105=東京 PRO 市場（株式）
_JQ_MKT_ETF_OR_FUND_SEGMENT = frozenset({"0109"})
_JQ_MKT_LISTED_EQUITY_SEGMENT = frozenset({"0105", "0111", "0112", "0113"})


def _canonical_jq_segment(val: Any) -> str:
    """CSV が 109→109.0 等になり得るため 4 桁セグメントに正規化する。"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        iv = int(float(str(val).strip()))
    except (ValueError, TypeError):
        s = str(val).strip()
        digs = "".join(ch for ch in s if ch.isdigit())
        if len(digs) >= 4:
            return digs[-4:].zfill(4)
        if digs:
            return digs.zfill(4)[-4:]
        return s
    if 0 <= iv <= 9999:
        return f"{iv:04d}"
    return str(iv)

_FUND_NAME_KEYWORDS = (
    "ＥＴＦ", "ETF", "ETFs", "etf",
    "上場投信", "上場投資信託", "インデックスファンド", "連動型上場投信",
    "投資信託", "上場インデックス", "投信",
    "ETN", "ＥＴＮ", "etn",
    "REIT", "ＲＥＩＴ", "reit", "リート", "投資法人",
    "インデックス", "指数", "TOPIX", "日経225", "連動",
    "ベア", "ブル", "レバレッジ", "ダブルインバース", "インバース",
    "アセットマネジメント",
    "先物", "債券", "国債", "米国債", "金", "原油", "商品",
)

# ETF 専用レーン向け（本スクリプトでは未スコア。将来: AUM推移・出来高・信託報酬・NAV乖離・トラッキングエラー・原資産・
# レバ/インバース・200日線押し目・52週調整・RSI など。J-Quants 未取得は推定せず None。）
_ETF_LANE_FUTURE_NOTE = "etf_lane_future_metrics"

_REIT_NAME_KEYWORDS = ("REIT", "ＲＥＩＴ", "reit", "リート", "投資法人")
_ETN_NAME_KEYWORDS = ("ETN", "ＥＴＮ", "etn")
_ETF_NAME_KEYWORDS = (
    "ＥＴＦ", "ETF", "ETFs", "etf",
    "上場投信", "上場投資信託", "インデックスファンド", "連動型上場投信",
    "上場インデックス",
    "インデックス", "指数", "TOPIX", "日経225", "連動",
)
_FUND_NON_ETF_KEYWORDS = (
    "投資信託", "投信", "投資信託", "ベア", "ブル", "レバレッジ",
    "ダブルインバース", "インバース", "先物", "債券", "国債", "米国債", "金", "原油", "商品",
)


def check_company_name_validity(company_name: str) -> Tuple[bool, str]:
    if company_name is None or not str(company_name).strip():
        return False, "会社名空欄"
    cn = str(company_name)
    etf_keywords = [
        "ＥＴＦ", "ETF", "ETFs", "etf", "上場投信", "上場投資信託", "インデックスファンド", "連動型上場投信",
        "上場インデックス", "TOPIX", "日経225", "投資法人", "リート", "REIT", "ＲＥＩＴ",
    ]
    if any(k in cn for k in etf_keywords):
        return False, "ETF/投信/REIT"
    fund_company_keywords = ["アセットマネジメント", "投信"]
    if any(k in cn for k in fund_company_keywords):
        return False, "投信会社商品"
    return True, "OK"


def _fund_name_blob_excludes_jp(blob: str) -> bool:
    if not blob:
        return False
    b_lo = blob
    if any(k in b_lo for k in _FUND_NAME_KEYWORDS):
        return True
    blo = b_lo.lower()
    for k in ("etf", "etn", "reit", "topix"):
        if k in blo:
            return True
    return False


def _mktnm_suggests_equity_listing(mkt_nm: str) -> bool:
    """Mkt 列の欠損・将来揺れ時の保険（ETF 市場0109は除外済みであること）。"""
    n = (mkt_nm or "").strip()
    if not n:
        return False
    return any(
        k in n
        for k in (
            "プライム",
            "スタンダード",
            "グロース",
            "TOKYO PRO MARKET",
            "Pro Market",
            "プロマーケット",
        )
    )


def _fund_code_band_fallback(code4: str) -> bool:
    """個別株財務スクリーニングには不適切なコード帯（ETF/投信・指数連動等が多い）。"""
    try:
        n = int(str(code4).strip()[:4])
    except ValueError:
        return False
    if 1300 <= n <= 1399:
        return True
    if 1450 <= n <= 1499:
        return True
    if 1550 <= n <= 1699:
        return True
    if 2000 <= n <= 2099:
        return True
    if 2230 <= n <= 2869:
        return True
    return False


def classify_instrument_from_master(row: Dict[str, Any]) -> str:
    """
    V2 equities/master 相当の1行（正規化列）から instrument_type を返す。
    stock / etf / etn / reit / fund / fund_like / unknown

    判定順（コード帯は最後）:
    1) Mkt で ETF 市場(0109) → etf
    2) Mkt / MktNm で上場株式と判定 → stock
    3) 名称キーワード
    4) コード帯フォールバック → fund_like
    5) その他 stock
    """
    code_cell = row.get("Code")
    digits = "".join(ch for ch in str(code_cell or "") if ch.isdigit())
    if len(digits) >= 4:
        code4 = digits[:4]
    else:
        code4 = ""

    cn = str(row.get("CompanyName") or "").strip()
    co_en = str(row.get("CoNameEn") or "").strip()
    mkt_raw = row.get("Mkt") if row.get("Mkt") not in (None, "") else row.get("MarketCode")
    mkt = _canonical_jq_segment(mkt_raw)
    mkt_nm = str(row.get("MktNm") or "").strip()
    if not cn:
        return "unknown"

    if not code4.isdigit() or len(code4) != 4:
        return "unknown"

    blob = f"{cn} {co_en} {mkt_nm}"

    # 1) ETF・上場投信市場（0109）
    if mkt in _JQ_MKT_ETF_OR_FUND_SEGMENT:
        return "etf"

    # 2) 上場株式セグメント（0105 PRO / プライム・スタンダード・グロース）— コード帯より先
    if mkt in _JQ_MKT_LISTED_EQUITY_SEGMENT:
        return "stock"
    if mkt not in _JQ_MKT_ETF_OR_FUND_SEGMENT and _mktnm_suggests_equity_listing(mkt_nm):
        return "stock"

    # 3) 名称（REIT→ETN→投信系→ETF）
    if any(k in blob for k in _REIT_NAME_KEYWORDS):
        return "reit"
    if any(k in blob for k in _ETN_NAME_KEYWORDS):
        return "etn"
    if _fund_name_blob_excludes_jp(blob) or any(k in blob for k in _FUND_NON_ETF_KEYWORDS):
        if any(k in blob for k in _ETF_NAME_KEYWORDS) or any(k in blob.lower() for k in ("etf", "etfn")):
            return "etf"
        return "fund"
    if any(k in blob for k in _ETF_NAME_KEYWORDS):
        return "etf"

    # 4) コード帯（ETF/投信が多い帯のみ。上記で stock を確定できない場合）
    if _fund_code_band_fallback(code4):
        return "fund_like"

    return "stock"


def ensure_instrument_type_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["instrument_type"] = out.apply(
        lambda r: classify_instrument_from_master(r.to_dict()),
        axis=1,
    )
    return out


def _save_etf_candidates_unscored_csv(df_non_stock: pd.DataFrame) -> None:
    """ETF/fund レーン検討用（未スコア）。個別株ランキングには混ぜない。"""
    if df_non_stock is None or df_non_stock.empty:
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ("Code", "CompanyName", "instrument_type", "Sector33Name", "MarketCode") if c in df_non_stock.columns]
    if len(cols) < 3:
        return
    path = REPORTS_DIR / "etf_candidates_unscored.csv"
    try:
        df_non_stock[cols].sort_values("Code").to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        logger.warning("etf_candidates_unscored 保存スキップ: %s", e)


def _master_row_for_reports(r: Any) -> Dict[str, str]:
    if r is None:
        return {
            "company_name": "", "sector": "", "market": "", "mkt": "", "mkt_nm": "",
            "instrument_type": "",
        }
    d = r if isinstance(r, dict) else r.to_dict()
    return {
        "company_name": str(d.get("CompanyName") or ""),
        "sector": str(d.get("Sector33Name") or ""),
        "market": str(d.get("MarketCode") or ""),
        "mkt": str(d.get("Mkt") or d.get("MarketCode") or ""),
        "mkt_nm": str(d.get("MktNm") or ""),
        "instrument_type": str(d.get("instrument_type") or ""),
    }


def _reports_upsert_csv(path: Path, row: dict, key_col: str = "code") -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(path)
    kval = str(row.get(key_col, "")).strip()
    if path.exists():
        try:
            df_old = pd.read_csv(path, encoding="utf-8-sig")
            if key_col in df_old.columns:
                df_old = df_old[df_old[key_col].astype(str).str.strip() != kval]
            df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
        except Exception:
            df_new = pd.DataFrame([row])
    else:
        df_new = pd.DataFrame([row])
    df_new.to_csv(path, index=False, encoding="utf-8-sig")


def _load_skiplist_raw() -> dict:
    if not COLLECTION_SKIPLIST_PATH.exists():
        return {"skipped": {}}
    try:
        return json.loads(COLLECTION_SKIPLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"skipped": {}}


def _save_skiplist_raw(blob: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTION_SKIPLIST_PATH.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")


def _skiplist_add_entry(code: str, reason: str, **extra: Any) -> None:
    blob = _load_skiplist_raw()
    blob.setdefault("skipped", {})
    ck = _canonical_internal_stock_code(str(code).strip(), None)
    if ck is None:
        dig = "".join(ch for ch in str(code).strip() if ch.isdigit())
        if len(dig) >= 4:
            ck = dig[:4]
        else:
            logger.warning("skiplist 見送り: コードを正規化できません: %s", code)
            return
    blob["skipped"][ck] = {
        "reason": reason,
        "last_attempt_at": datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    _save_skiplist_raw(blob)


_NON_STOCK_NAME_SUBSTRINGS_EXTRA = (
    "インフラファンド", "優先出資", "受益証券", "ＥＴＦ", "ETF", "ETN", "ＥＴＮ",
    "REIT", "ＲＥＩＴ", "リート", "投資法人", "上場投信", "上場投資信託", "投資信託", "投信",
)


def _non_stock_keyword_hit(company_name: str) -> Optional[str]:
    cn = company_name or ""
    for sub in _NON_STOCK_NAME_SUBSTRINGS_EXTRA:
        if sub in cn:
            return f"name_keyword:{sub}"
    return None


def _non_stock_market_hit(row: dict) -> Optional[str]:
    seg = _canonical_jq_segment(row.get("Mkt") if row.get("Mkt") not in (None, "") else row.get("MarketCode"))
    if seg in _JQ_MKT_ETF_OR_FUND_SEGMENT:
        return "mkt_segment_etf_fund"
    mnm = str(row.get("MktNm") or "")
    for k in ("ＥＴＦ", "ETF", "上場投信", "投信", "リート", "REIT"):
        if k in mnm:
            return f"mktnm:{k}"
    return None


def _collect_hard_exclusion_reason(row: dict) -> Optional[str]:
    r = _non_stock_keyword_hit(str(row.get("CompanyName") or ""))
    if r:
        return r
    r2 = _non_stock_market_hit(row)
    if r2:
        return r2
    it = str(row.get("instrument_type") or "").strip()
    if it and it != "stock":
        return f"instrument_type:{it}"
    return None


def _record_permanent_missing_financials(res: dict, master_row: Optional[Any]) -> None:
    code = str(res.get("code", "")).strip()
    m = _master_row_for_reports(master_row)
    _skiplist_add_entry(code, "permanent_missing_financials", detail=res.get("reason", ""))
    row = {
        "code": code,
        "company_name": m["company_name"],
        "sector": m["sector"],
        "market": m["market"],
        "instrument_type": m["instrument_type"],
        "reason": res.get("reason", ""),
        "last_attempt_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "price_rows": res.get("price_rows", 0),
        "summary_rows": res.get("summary_rows", 0),
        "detail_rows": res.get("detail_rows", 0),
        "financial_data_mode": res.get("financial_data_mode", ""),
        "api_status_summary": res.get("api_status_summary", ""),
        "api_status_details": res.get("api_status_details", ""),
    }
    _reports_upsert_csv(MISSING_FINANCIALS_CSV, row, "code")


def _record_non_stock_excluded(res: dict, master_row: Optional[Any]) -> None:
    code = str(res.get("code", "")).strip()
    m = _master_row_for_reports(master_row)
    _skiplist_add_entry(code, "non_stock_or_fund_like", detail=res.get("reason", ""))
    row = {
        "code": code,
        "company_name": m["company_name"],
        "sector": m["sector"],
        "market": m["market"],
        "mkt": m["mkt"],
        "mkt_nm": m["mkt_nm"],
        "instrument_type": m["instrument_type"],
        "reason": res.get("reason", ""),
    }
    _reports_upsert_csv(NON_STOCK_EXCLUDED_CSV, row, "code")


def write_sector_normalization_audit_csv(session: requests.Session) -> Optional[Path]:
    """マスタの収集対象銘柄について Sector33 → 正規化セクターの監査CSVを出す。"""
    try:
        fdm = FinancialDataManager(session)
        df = filter_collectable_equities_df(fdm.get_stock_list_v2(force_refresh=False))
        if df is None or df.empty:
            return None
        rows: List[dict] = []
        for _, r in df.iterrows():
            o = str(r.get("Sector33Name") or "")
            n = DynamicSectorAverages.normalize_sector(o)
            rows.append({
                "code": str(r.get("Code") or "").strip(),
                "company_name": str(r.get("CompanyName") or ""),
                "original_sector33": o,
                "normalized_sector": n,
                "market": str(r.get("MarketCode") or ""),
                "reason_if_other": (o or "empty_sector33") if n == "その他" else "",
            })
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        p = SECTOR_NORM_AUDIT_CSV
        pd.DataFrame(rows).sort_values("code").to_csv(p, index=False, encoding="utf-8-sig")
        return p
    except Exception as e:
        logger.warning("sector_normalization_audit 出力スキップ: %s", e)
        return None


def filter_collectable_equities_df(df: pd.DataFrame) -> pd.DataFrame:
    """instrument_type が stock のみ収集対象。非株は etf_candidates_unscored に別途保存する。"""
    _ = _ETF_LANE_FUTURE_NOTE  # メタ情報用（削除されないように参照のみ）
    if df is None or df.empty:
        return df
    if "CompanyName" not in df.columns:
        return df
    df = ensure_instrument_type_column(df)

    try:
        m_non_stock = df["instrument_type"].astype(str).str.strip() != "stock"
        _save_etf_candidates_unscored_csv(df.loc[m_non_stock].copy())
    except Exception:
        pass

    n_in = len(df)
    keep_rows: List[int] = []
    removed_empty_name = 0
    removed_unknown = 0
    removed_fund_like = 0
    excluded_audit: List[Dict[str, str]] = []

    for idx, r in df.iterrows():
        cn_raw = str(r.get("CompanyName", "") or "").strip()
        rd = r.to_dict()
        code_s = str(r.get("Code", "")).strip()
        if not cn_raw:
            removed_empty_name += 1
            continue
        it = str(r.get("instrument_type") or "").strip() or classify_instrument_from_master(rd)
        if it == "unknown":
            removed_unknown += 1
            continue
        if it != "stock":
            removed_fund_like += 1
            continue
        hard = _collect_hard_exclusion_reason(rd)
        if hard:
            removed_fund_like += 1
            excluded_audit.append({
                "code": code_s,
                "company_name": cn_raw,
                "sector": str(r.get("Sector33Name") or ""),
                "market": str(r.get("MarketCode") or ""),
                "reason": hard,
            })
            continue
        ok_name, reason_nm = check_company_name_validity(cn_raw)
        if not ok_name:
            removed_fund_like += 1
            excluded_audit.append({
                "code": code_s,
                "company_name": cn_raw,
                "sector": str(r.get("Sector33Name") or ""),
                "market": str(r.get("MarketCode") or ""),
                "reason": reason_nm or "name_filter",
            })
            continue
        keep_rows.append(idx)

    out = df.loc[keep_rows].copy() if keep_rows else pd.DataFrame(columns=df.columns)

    if excluded_audit:
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(excluded_audit).sort_values("code").to_csv(
                COLLECT_FILTER_EXCLUDED_CSV, index=False, encoding="utf-8-sig",
            )
        except Exception as e:
            logger.warning("collectable_filter_excluded CSV 失敗: %s", e)

    logger.info(
        "[INFO] collectable filter: input=%s output=%s removed_fund_like=%s removed_empty_name=%s removed_unknown=%s",
        n_in,
        len(out),
        removed_fund_like,
        removed_empty_name,
        removed_unknown,
    )

    return out.reset_index(drop=True)

# ------------------------------------------------------------
# テクニカル
# ------------------------------------------------------------
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if np.isfinite(v) else 50.0

def calculate_adx_and_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float, float, float]:
    if min(len(high), len(low), len(close)) < period + 5:
        return 20.0, 20.0, 20.0

    df = pd.DataFrame({"high": high, "low": low, "close": close}).dropna()
    if len(df) < period + 1:
        return 20.0, 20.0, 20.0

    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
    minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    def clamp(x, lo, hi):
        return float(max(lo, min(hi, x))) if np.isfinite(x) else float(lo)

    return clamp(adx.iloc[-1], 5, 80), clamp(plus_di.iloc[-1], 5, 95), clamp(minus_di.iloc[-1], 5, 95)

def calculate_moving_averages(prices: pd.Series, periods: Tuple[int, ...] = (25, 75, 200)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in periods:
        if len(prices) >= p:
            ma = prices.rolling(window=p).mean().iloc[-1]
            out[f"ma_{p}"] = float(ma) if np.isfinite(ma) else float(prices.iloc[-1])
        elif len(prices):
            out[f"ma_{p}"] = float(prices.iloc[-1])
        else:
            out[f"ma_{p}"] = None
    return out

def calculate_volatility(prices: pd.Series, period: int = 20) -> Tuple[Optional[float], Optional[float]]:
    if len(prices) < max(5, period):
        return None, None
    try:
        returns = prices.pct_change(fill_method=None).dropna()
    except TypeError:
        returns = prices.pct_change().dropna()
    cur = returns.tail(period).std() * np.sqrt(252) if len(returns) >= period else returns.std() * np.sqrt(252)
    avg = returns.std() * np.sqrt(252)
    return float(cur), float(avg)

# ------------------------------------------------------------
# 長期投資向け: 最大DD / 売上CAGR / 安全基準
# ------------------------------------------------------------
def calculate_max_drawdown(prices: pd.Series, lookback_days: Optional[int] = None) -> Optional[float]:
    if prices is None or len(prices) < 2:
        return None
    try:
        prices_series = prices.copy()
        if lookback_days is not None and lookback_days > 1:
            prices_series = prices_series.tail(lookback_days)
        if len(prices_series) < 2:
            return None

        prices_sorted = prices_series.sort_index() if hasattr(prices_series.index, "is_monotonic_increasing") else prices_series
        cumulative_max = prices_sorted.cummax()
        drawdowns = (prices_sorted - cumulative_max) / cumulative_max
        max_dd = float(drawdowns.min())
        return max_dd if np.isfinite(max_dd) else None
    except Exception:
        return None

def _pick_numeric_field(record: dict, keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in record and record[key] not in (None, "", "NA"):
            try:
                return float(record[key])
            except Exception:
                continue
    return None

def _fiscal_year_from_statement(record: dict) -> int:
    for ky in ("fiscalYear", "FiscalYear", "period", "CurrentFiscalYearEndDate", "DisclosedDate"):
        value = str(record.get(ky) or "")
        match = re.findall(r"\d{4}", value)
        if match:
            return int(match[0])
    return -1

def _parse_optional_date(value: Any) -> Optional[datetime.date]:
    if value in (None, "", "NA"):
        return None
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10:
        text = text[:10]
    text = text.replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(\d{4})(\d{2})(\d{2})", text)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None

def _statement_period_type(record: dict) -> str:
    parts = [
        str(record.get(k) or "")
        for k in (
            "TypeOfDocument", "DocumentType", "ReportType",
            "CurrentPeriodType", "CurrentFiscalYearEndDate", "CurrentPeriodEndDate"
        )
    ]
    text = " ".join(parts).lower()
    if any(x in text for x in ("q1", "1q", "第1四半期")):
        return "q1"
    if any(x in text for x in ("q2", "2q", "第2四半期", "半期", "interim")):
        return "q2"
    if any(x in text for x in ("q3", "3q", "第3四半期")):
        return "q3"
    if any(x in text for x in ("annual", "fy", "通期", "本決算", "年度")):
        return "annual"
    return "unknown"

def _statement_disclosed_date(record: dict) -> Optional[datetime.date]:
    for key in ("DisclosedDate", "DisclosureDate", "PublishedDate", "Date"):
        dt = _parse_optional_date(record.get(key))
        if dt is not None:
            return dt
    return None

def _statement_period_end_date(record: dict) -> Optional[datetime.date]:
    for key in ("CurrentPeriodEndDate", "CurrentFiscalYearEndDate", "FiscalYearEnd", "PeriodEnd", "DisclosedDate"):
        dt = _parse_optional_date(record.get(key))
        if dt is not None:
            return dt
    return None

def _statement_sort_key(record: dict) -> tuple:
    disclosed = _statement_disclosed_date(record) or datetime.date.min
    period_end = _statement_period_end_date(record) or datetime.date.min
    return (disclosed, period_end, _fiscal_year_from_statement(record))

def build_financial_history_from_statements(
    stmts: List[dict],
    max_years: int = 5,
    as_of_date: Optional[datetime.date] = None,
    statement_basis: str = "annual",
) -> Tuple[List[dict], str]:
    if not stmts:
        return [], statement_basis
    filtered_stmts = []
    for stmt in stmts:
        disclosed = _statement_disclosed_date(stmt)
        if as_of_date is not None and disclosed is not None and disclosed > as_of_date:
            continue
        filtered_stmts.append(stmt)

    statement_basis_used: str
    if statement_basis == "annual":
        annual_only = [s for s in filtered_stmts if _statement_period_type(s) == "annual"]
        if annual_only:
            source_stmts = annual_only
            statement_basis_used = "annual"
        else:
            source_stmts = filtered_stmts if as_of_date is not None else stmts
            statement_basis_used = "fallback_primary_type"
    else:
        source_stmts = filtered_stmts if as_of_date is not None else stmts
        statement_basis_used = statement_basis

    if not source_stmts:
        return [], statement_basis_used
    sorted_stmts = sorted(source_stmts, key=_statement_sort_key, reverse=True)

    primary_type = "unknown"
    for stmt in sorted_stmts:
        candidate_type = _statement_period_type(stmt)
        if candidate_type != "unknown":
            primary_type = candidate_type
            break

    comparable = [
        stmt for stmt in sorted_stmts
        if primary_type == "unknown" or _statement_period_type(stmt) == primary_type
    ]
    if comparable:
        sorted_stmts = comparable

    history: list[dict] = []
    seen_keys: set[tuple] = set()
    for stmt in sorted_stmts:
        fiscal_year = _fiscal_year_from_statement(stmt)
        period_type = _statement_period_type(stmt)
        period_end = _statement_period_end_date(stmt)
        dedupe_key = (fiscal_year, period_type, period_end)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        rec = {
            "fiscal_year": fiscal_year,
            "revenue": _pick_numeric_field(stmt, ["NetSales", "Revenue", "OperatingRevenue"]),
            "operating_income": _pick_numeric_field(stmt, ["OperatingIncome", "OperatingIncomeLoss", "OperatingProfit"]),
            "net_income": _pick_numeric_field(stmt, ["NetIncomeLoss", "Profit", "ProfitAttributableToOwnersOfParent", "NetIncome"]),
            "operating_cash_flow": _pick_numeric_field(stmt, ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromOperatingActivities"]),
            "total_assets": _pick_numeric_field(stmt, ["TotalAssets"]),
            "equity": _pick_numeric_field(stmt, ["EquityAttributableToOwnersOfParent", "Equity", "NetAssets"]),
            "current_assets": _pick_numeric_field(stmt, ["CurrentAssets"]),
            "current_liabilities": _pick_numeric_field(stmt, ["CurrentLiabilities"]),
            "gross_profit_margin": None,
            "disclosed_date": _statement_disclosed_date(stmt).isoformat() if _statement_disclosed_date(stmt) else None,
            "period_end_date": period_end.isoformat() if period_end else None,
            "statement_type": period_type,
            "shares_outstanding": _pick_numeric_field(
                stmt,
                [
                    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
                    "NumberOfIssuedAndOutstandingShares",
                ],
            ),
        }

        cash_and_equivalents = _pick_numeric_field(
            stmt,
            [
                "CashAndCashEquivalents",
                "CashAndCashEquivalentsAtEndOfPeriod",
                "Cash",
                "CashEquivalents",
                "CashAndDeposits",
            ],
        )
        if cash_and_equivalents is None:
            cash_and_equivalents = rec["current_assets"]
        rec["cash_and_equivalents"] = cash_and_equivalents

        if rec["total_assets"] is not None and rec["total_assets"] > 0 and rec["equity"] is not None:
            rec["equity_ratio"] = rec["equity"] / rec["total_assets"]
        else:
            rec["equity_ratio"] = None

        gross_profit = _pick_numeric_field(stmt, ["GrossProfit"])
        if (
            rec["revenue"] is not None
            and rec["revenue"] != 0
            and gross_profit is not None
        ):
            rec["gross_profit_margin"] = gross_profit / rec["revenue"]

        history.append(rec)
        if len(history) >= max_years:
            break
    return history, statement_basis_used

def calculate_liquidity_metrics(
    close: pd.Series,
    volume: Optional[pd.Series],
) -> dict:
    out = {
        "avg_volume_30d": None,
        "adv_jpy_20d": None,
        "adv_jpy_60d": None,
        "traded_days_60d": None,
    }
    if not isinstance(volume, pd.Series) or volume.empty or close is None or len(close) == 0:
        return out
    try:
        vol = volume.astype(float).clip(lower=0)
        px = close.astype(float)
        out["avg_volume_30d"] = int(vol.tail(30).mean()) if len(vol.tail(30)) else None
        traded_value = (px * vol).replace([np.inf, -np.inf], np.nan)
        adv20 = traded_value.tail(20).mean()
        adv60 = traded_value.tail(60).mean()
        out["adv_jpy_20d"] = float(adv20) if pd.notna(adv20) else None
        out["adv_jpy_60d"] = float(adv60) if pd.notna(adv60) else None
        out["traded_days_60d"] = int((vol.tail(60) > 0).sum())
        return out
    except Exception:
        return out

def calculate_medium_term_momentum(prices: pd.Series) -> dict:
    out = {
        "return_21d": None,
        "return_63d": None,
        "return_126d": None,
        "return_252d": None,
        "momentum_6m_1m": None,
        "momentum_6m_3m": None,
        "momentum_3m_1m": None,
    }
    if prices is None or len(prices) < 2:
        return out
    series = prices.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return out

    def _ret(lookback: int) -> Optional[float]:
        if len(series) <= lookback:
            return None
        base = series.iloc[-lookback - 1]
        last = series.iloc[-1]
        if base <= 0:
            return None
        return float(last / base - 1.0)

    def _window_ret(long_lookback: int, short_lookback: int) -> Optional[float]:
        if len(series) <= long_lookback or len(series) <= short_lookback:
            return None
        start = series.iloc[-long_lookback - 1]
        end = series.iloc[-short_lookback - 1]
        if start <= 0 or end <= 0:
            return None
        return float(end / start - 1.0)

    out["return_21d"] = _ret(21)
    out["return_63d"] = _ret(63)
    out["return_126d"] = _ret(126)
    out["return_252d"] = _ret(252)
    out["momentum_6m_1m"] = _window_ret(126, 21)
    out["momentum_6m_3m"] = _window_ret(126, 63)
    out["momentum_3m_1m"] = _window_ret(63, 21)
    return out


def _recent_ma200_cross_up(close: pd.Series, lookback: int) -> bool:
    if close is None or len(close) < 200 + lookback + 2:
        return False
    s = close.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 200 + lookback + 2:
        return False
    ma200 = s.rolling(200, min_periods=200).mean()
    for k in range(1, lookback + 1):
        if k + 1 > len(s):
            break
        i = len(s) - k
        i_prev = i - 1
        c0 = float(s.iloc[i_prev])
        c1 = float(s.iloc[i])
        m0 = float(ma200.iloc[i_prev])
        m1 = float(ma200.iloc[i])
        if not (np.isfinite(c0) and np.isfinite(c1) and np.isfinite(m0) and np.isfinite(m1)):
            continue
        if c0 < m0 and c1 >= m1:
            return True
    return False


def _new_low_in_last_n_days(close: pd.Series, window: int, days_back: int) -> Optional[bool]:
    if close is None or len(close) < window + days_back + 1:
        return None
    s = close.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < window + days_back + 1:
        return None
    for k in range(days_back):
        i = len(s) - 1 - k
        if i < window:
            return None
        prior = s.iloc[i - window : i]
        if len(prior) < window:
            return None
        if float(s.iloc[i]) <= float(prior.min()):
            return True
    return False


def _ma_slope_ratio(ma_series: pd.Series, lookback: int) -> Optional[float]:
    if ma_series is None or len(ma_series) < lookback + 2:
        return None
    last = float(ma_series.iloc[-1])
    prev = float(ma_series.iloc[-1 - lookback])
    if not (np.isfinite(last) and np.isfinite(prev)) or prev == 0:
        return None
    return (last - prev) / abs(prev)


def _adx_compare_recent(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    """直近終値のADXと、5本前までを除いた終値のADX（低下傾向の判定用）。"""
    adx_now: Optional[float] = None
    adx_prev: Optional[float] = None
    try:
        if len(close) >= 30:
            adx_now, _, _ = calculate_adx_and_di(high, low, close)
        if len(close) >= 35:
            adx_prev, _, _ = calculate_adx_and_di(high.iloc[:-5], low.iloc[:-5], close.iloc[:-5])
    except Exception:
        return None, None
    return adx_now, adx_prev


def evaluate_ma200_entry_state(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    *,
    adx: Optional[float] = None,
    plus_di: Optional[float] = None,
    minus_di: Optional[float] = None,
) -> dict[str, Any]:
    """
    200日線を「単純除外」ではなく局面分類に使用。計算不能は None / False 相当で ma200_unknown へ。
    """
    empty = {
        "ma200_state": "ma200_unknown",
        "ma200_timing_score": 0.0,
        "ma200_risk_penalty": 0.0,
        "ma200_reason": "insufficient_price_history",
        "distance_from_ma200": None,
        "crossed_above_ma200_recently": False,
        "below_ma200_basing": False,
        "below_ma200_downtrend": False,
        "above_ma200_extended": False,
        "recent_60d_low_update": None,
    }
    if close is None or len(close) < 200:
        return empty

    s = close.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    hi = high.reindex(s.index).astype(float).replace([np.inf, -np.inf], np.nan)
    lo = low.reindex(s.index).astype(float).replace([np.inf, -np.inf], np.nan)
    if len(s) < 200:
        return empty

    current_price = float(s.iloc[-1])
    ma200_s = s.rolling(200, min_periods=200).mean()
    ma25_s = s.rolling(25, min_periods=25).mean()
    ma200 = float(ma200_s.iloc[-1])
    ma25 = float(ma25_s.iloc[-1]) if len(s) >= 25 and np.isfinite(ma25_s.iloc[-1]) else None
    if not np.isfinite(ma200) or ma200 <= 0 or not np.isfinite(current_price):
        out = dict(empty)
        out["ma200_reason"] = "ma200_not_finite"
        return out

    distance_from_ma200 = float(current_price / ma200 - 1.0)
    crossed = _recent_ma200_cross_up(s, MA200_CROSS_LOOKBACK_DAYS)
    mom = calculate_medium_term_momentum(s)
    ret_20 = mom.get("return_21d")
    ma25_slope = _ma_slope_ratio(ma25_s, 20) if len(s) >= 45 else None

    adx_now, adx_prev = _adx_compare_recent(hi, lo, s)
    adx_use = adx if (adx is not None and np.isfinite(adx)) else adx_now
    adx_falling = (
        adx_use is not None
        and adx_prev is not None
        and np.isfinite(adx_use)
        and np.isfinite(adx_prev)
        and float(adx_use) < float(adx_prev)
    )
    adx_high = bool(adx_use is not None and np.isfinite(adx_use) and float(adx_use) > 35.0)

    ratio_pm = None
    if (
        plus_di is not None
        and minus_di is not None
        and np.isfinite(plus_di)
        and np.isfinite(minus_di)
        and float(plus_di) > 0
    ):
        ratio_pm = float(minus_di) / float(plus_di)

    new_60_low = _new_low_in_last_n_days(s, RECENT_LOW_LOOKBACK_DAYS, RECENT_LOW_NO_UPDATE_DAYS)
    low120 = float(s.tail(min(BASING_LOOKBACK_LOW_DAYS, len(s))).min())
    low60 = float(s.tail(min(RECENT_LOW_LOOKBACK_DAYS, len(s))).min())
    rebound_from_120 = (current_price / low120 - 1.0) if low120 > 0 else None

    cond_below = current_price < ma200
    cond_above = current_price > ma200

    downtrend = False
    dt_reasons: list[str] = []
    if ma25 is not None and np.isfinite(ma25):
        if current_price < ma25 and ma25_slope is not None and ma25_slope < 0:
            downtrend = True
            dt_reasons.append("price_below_ma25_and_ma25_slope_negative")
    if new_60_low is True:
        downtrend = True
        dt_reasons.append("new_60d_low_recent")
    if ret_20 is not None and np.isfinite(ret_20) and ret_20 < 0:
        downtrend = True
        dt_reasons.append("return_20d_negative")
    if (
        ratio_pm is not None
        and adx_use is not None
        and np.isfinite(adx_use)
        and ratio_pm > 1.3
        and float(adx_use) >= 20.0
    ):
        downtrend = True
        dt_reasons.append("minus_di_dominant_adx_strong")
    if distance_from_ma200 < (MA200_BELOW_MIN_RATIO - 1.0):
        downtrend = True
        dt_reasons.append("deep_below_ma200")

    basing = False
    bs_reasons: list[str] = []
    if cond_below and not downtrend:
        ok_ratio = distance_from_ma200 >= (MA200_BELOW_MIN_RATIO - 1.0)
        ok_ma25 = (ma25 is not None and current_price > ma25) or (
            ma25_slope is not None and ma25_slope > 0
        )
        ok_ret = ret_20 is not None and np.isfinite(ret_20) and ret_20 > 0
        ok_no_60 = new_60_low is False
        ok_rebound = (
            rebound_from_120 is not None
            and np.isfinite(rebound_from_120)
            and rebound_from_120 >= BASING_MIN_REBOUND_FROM_LOW
        )
        ok_adx = (not adx_high) or adx_falling
        ok_di = ratio_pm is None or ratio_pm <= 1.2
        ok_struct = low60 > low120 * 1.0001
        basing = bool(
            ok_ratio
            and ok_ma25
            and ok_ret
            and ok_no_60
            and ok_rebound
            and ok_adx
            and ok_di
            and ok_struct
        )
        if basing:
            bs_reasons.append("basing_pattern")

    ma200_state = "ma200_unknown"
    reason = ""
    timing = 35.0
    risk_pen = 0.0
    above_ext = False
    below_base_flag = False
    below_dt_flag = False

    if cond_above:
        if crossed:
            ma200_state = "ma200_reclaim"
            reason = "cross_up_recent_below_to_above_ma200"
            ideal_lo = 0.0
            ideal_hi = MA200_IDEAL_MAX_DISTANCE
            if ideal_lo <= distance_from_ma200 <= ideal_hi:
                timing = 95.0
            elif distance_from_ma200 > MA200_EXTENDED_DISTANCE:
                timing = 55.0
                risk_pen += 8.0
            else:
                timing = 78.0
        elif distance_from_ma200 > MA200_EXTENDED_DISTANCE:
            ma200_state = "above_ma200_extended"
            above_ext = True
            reason = "extended_above_ma200"
            timing = 40.0
            risk_pen = 12.0 + min(18.0, max(0.0, distance_from_ma200 - MA200_EXTENDED_DISTANCE) * 60.0)
        else:
            ma200_state = "above_ma200_near"
            reason = "above_ma200_not_recent_cross"
            timing = 70.0
            if distance_from_ma200 > MA200_IDEAL_MAX_DISTANCE:
                risk_pen += 4.0
    elif cond_below:
        below_dt_flag = bool(downtrend)
        below_base_flag = bool(basing)
        if downtrend:
            ma200_state = "below_ma200_downtrend"
            reason = ";".join(dt_reasons) if dt_reasons else "below_ma200_downtrend"
            timing = 15.0
            risk_pen = 20.0
        elif basing:
            ma200_state = "below_ma200_basing"
            reason = ";".join(bs_reasons) if bs_reasons else "below_ma200_basing"
            timing = 55.0
            risk_pen = 6.0
        else:
            ma200_state = "below_ma200_downtrend"
            reason = "below_ma200_unconfirmed_not_basing"
            below_dt_flag = True
            timing = 22.0
            risk_pen = 14.0

    return {
        "ma200_state": ma200_state,
        "ma200_timing_score": round(float(timing), 2),
        "ma200_risk_penalty": round(float(risk_pen), 2),
        "ma200_reason": reason,
        "distance_from_ma200": round(float(distance_from_ma200), 5) if np.isfinite(distance_from_ma200) else None,
        "crossed_above_ma200_recently": bool(crossed),
        "below_ma200_basing": bool(below_base_flag),
        "below_ma200_downtrend": bool(below_dt_flag),
        "above_ma200_extended": bool(above_ext),
        "recent_60d_low_update": new_60_low,
    }


def _ps_vs_sector_ratio(ps: Optional[float], sector_ps_benchmark: Optional[float], sector_med: Optional[float]) -> Optional[float]:
    med = sector_ps_benchmark
    if med is None or (isinstance(med, float) and (not np.isfinite(med) or med <= 0)):
        med = sector_med
    if ps is None or med is None or (not np.isfinite(ps)) or (not np.isfinite(med)) or med <= 0:
        return None
    return float(ps) / float(med)


def compute_fundamental_edge_score(
    *,
    ps_ratio: Optional[float],
    ps_vs_sector: Optional[float],
    reference_peg: Optional[float],
    eps_growth_rate: Optional[float],
    piot: dict,
    peg_quality: dict,
    op_income_stable: Optional[bool],
    sales_cagr: Optional[float],
    per: Optional[float],
    critical_missing_count: int,
    statement_basis_used: Optional[str],
    statement_staleness_days: Optional[float],
) -> float:
    """ファンダ優先の 0〜100 スコア（total_score とは別軸）。"""
    s = 0.0
    if ps_ratio is not None and np.isfinite(ps_ratio) and ps_ratio > 0:
        if ps_ratio <= 0.35:
            s += 18.0
        elif ps_ratio <= 0.7:
            s += 15.0
        elif ps_ratio <= 1.2:
            s += 11.0
        elif ps_ratio <= MAX_PS_DEFENSIVE:
            s += 7.0
        else:
            s += 3.0

    if ps_vs_sector is not None and np.isfinite(ps_vs_sector):
        if ps_vs_sector <= 0.55:
            s += 22.0
        elif ps_vs_sector <= 0.8:
            s += 18.0
        elif ps_vs_sector <= 1.0:
            s += 14.0
        elif ps_vs_sector <= MAX_PS_VS_SECTOR_CORE:
            s += 9.0
        else:
            s += 3.0

    pt = peg_quality.get("peg_trusted")
    pw = str(peg_quality.get("peg_warning") or "")
    if pt is True and pw == "ok":
        s += 14.0
    elif pt is True and pw == "expensive_or_moderate":
        s += 8.0
    elif pt == "caution":
        s += 5.0
    else:
        s += 1.0

    eff = piot.get("piotroski_effective_score")
    cov = piot.get("piotroski_coverage_ratio")
    if eff is not None and np.isfinite(eff):
        w = 22.0
        if cov is not None and np.isfinite(cov):
            w *= 0.65 + 0.35 * min(1.0, float(cov) / 0.85)
        s += min(22.0, float(eff) / 9.0 * w)
    if op_income_stable is True:
        s += 9.0
    elif op_income_stable is None:
        s += 2.0

    if sales_cagr is not None and np.isfinite(sales_cagr):
        if sales_cagr >= 0.15:
            s += 6.0
        elif sales_cagr >= 0.08:
            s += 4.0
        elif sales_cagr >= 0.0:
            s += 2.5
    if eps_growth_rate is not None and np.isfinite(eps_growth_rate):
        eg = float(eps_growth_rate)
        if 5.0 <= eg <= 45.0:
            s += 5.0
        elif eg > 45.0 and eg <= 80.0:
            s += 2.0

    # 低PS・セクター割安だけが効いていて実体（Piotroski）が弱い場合の切り上げ抑制
    adj_q = piot.get("piotroski_adjusted_score")
    val_screen_heavy = False
    if ps_vs_sector is not None and np.isfinite(ps_vs_sector) and float(ps_vs_sector) <= 0.72:
        val_screen_heavy = True
    if ps_ratio is not None and np.isfinite(ps_ratio) and float(ps_ratio) <= 0.65:
        val_screen_heavy = True
    if val_screen_heavy and adj_q is not None and np.isfinite(float(adj_q)) and float(adj_q) < 6.0:
        if pt is not True or pw != "ok":
            s -= 14.0

    if per is not None and np.isfinite(per):
        if per > 120:
            s -= 6.0
        elif per > 90:
            s -= 3.0

    if critical_missing_count > 0:
        s -= min(18.0, 3.5 * max(0, critical_missing_count - 1))
    if (statement_basis_used or "") != "annual":
        s -= 10.0
    if statement_staleness_days is not None and np.isfinite(statement_staleness_days):
        if statement_staleness_days > float(MAX_STATEMENT_STALENESS_DAYS_CORE):
            s -= 10.0
        elif statement_staleness_days > 200:
            s -= 4.0
    if piot.get("piotroski_confidence_low"):
        s -= 6.0

    raw_adj = piot.get("piotroski_raw_score")
    if raw_adj is None:
        raw_adj = piot.get("score")
    adj_adj = piot.get("piotroski_adjusted_score")
    if adj_adj is not None and np.isfinite(float(adj_adj)) and float(adj_adj) < 5.0:
        s -= 15.0
    if raw_adj is not None and np.isfinite(float(raw_adj)) and int(float(raw_adj)) <= 3:
        s -= 12.0

    return float(round(max(0.0, min(100.0, s)), 2))


def compute_entry_score(fundamental_edge: float, ma_eval: dict) -> float:
    st = str(ma_eval.get("ma200_state") or "ma200_unknown")
    bonuses = {
        "ma200_reclaim": 15.0,
        "below_ma200_basing": 8.0,
        "above_ma200_near": 5.0,
        "above_ma200_extended": -10.0,
        "below_ma200_downtrend": -25.0,
        "ma200_unknown": -5.0,
    }
    out = float(fundamental_edge) + bonuses.get(st, -5.0)
    dist = ma_eval.get("distance_from_ma200")
    if dist is not None and np.isfinite(dist):
        if st in ("ma200_reclaim", "above_ma200_near", "above_ma200_extended") and dist > MA200_IDEAL_MAX_DISTANCE:
            out -= min(14.0, max(0.0, (float(dist) - MA200_IDEAL_MAX_DISTANCE) * 65.0))
    return float(round(max(0.0, min(100.0, out)), 2))


def _piot_raw_int(piot: Optional[dict]) -> Optional[int]:
    if not piot:
        return None
    v = piot.get("piotroski_raw_score")
    if v is None:
        v = piot.get("score")
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _piot_adj_float(piot: Optional[dict]) -> Optional[float]:
    if not piot:
        return None
    v = piot.get("piotroski_adjusted_score")
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_data_review_meta(
    *,
    critical_missing_count: int,
    statement_basis_used: Optional[str],
    fallback_basis_flag: bool,
    piot_coverage: Optional[float],
    statement_staleness_days: Optional[float],
    financial_data_mode: Optional[str],
    fins_details_available: Optional[bool],
    sector_normalized: str,
    sector33_raw: Optional[str],
    piot_adjusted: Optional[float],
    piot_raw: Optional[float],
) -> dict[str, Any]:
    reasons: list[str] = []
    levels: list[int] = []

    def add(reason: str, lvl: int) -> None:
        if reason not in reasons:
            reasons.append(reason)
        levels.append(lvl)

    if (statement_basis_used or "") != "annual" or fallback_basis_flag:
        add("statement_basis_fallback", 1)
    if critical_missing_count > MAX_CRITICAL_MISSING_CORE:
        add("critical_missing_too_many", 2)
    if (
        piot_coverage is not None
        and np.isfinite(float(piot_coverage))
        and float(piot_coverage) < MIN_PIOTROSKI_COVERAGE_CORE
    ):
        add("piotroski_coverage_low", 1)
    fdm = (financial_data_mode or "").strip().lower()
    if fdm == "summary_only":
        add("summary_only_financials", 1)
    elif fins_details_available is False and fdm and "summary" in fdm:
        add("summary_only_financials", 1)
    sn = (sector33_raw or "").strip()
    if sector_normalized == "その他" and (not sn):
        add("sector_unknown", 1)
    if piot_adjusted is not None and np.isfinite(float(piot_adjusted)) and float(piot_adjusted) < 4.0:
        add("piotroski_too_low", 2)
    if statement_staleness_days is not None and np.isfinite(float(statement_staleness_days)):
        sd = float(statement_staleness_days)
        if sd > float(MAX_STATEMENT_STALENESS_DAYS_CORE):
            add("stale_statement", 2)
        elif sd > float(STALE_STATEMENT_MEDIUM_DAYS):
            add("stale_statement", 1)
    if piot_raw is not None and np.isfinite(float(piot_raw)) and int(float(piot_raw)) <= 3:
        add("piotroski_raw_weak", 1)

    if not reasons:
        return {
            "has_issues": False,
            "data_review_level": "",
            "data_review_reason": "",
            "data_review_reasons": [],
        }

    mx = max(levels)
    if set(reasons) <= {"summary_only_financials"} and (
        piot_coverage is None
        or (
            np.isfinite(float(piot_coverage))
            and float(piot_coverage) >= MIN_PIOTROSKI_COVERAGE_CORE
        )
    ):
        final_level = "light"
    elif mx >= 2:
        final_level = "severe"
    elif mx >= 1:
        final_level = "medium"
    else:
        final_level = "light"

    return {
        "has_issues": True,
        "data_review_level": final_level,
        "data_review_reason": ",".join(reasons),
        "data_review_reasons": reasons,
    }


def cap_entry_score(raw: float, lane: str, dr_meta: dict) -> float:
    cap = float(LANE_ENTRY_CAP.get(lane, 88.0))
    if lane == "data_review":
        dl = dr_meta.get("data_review_level")
        if dl == "severe":
            cap = min(cap, float(LANE_ENTRY_CAP["data_review_severe"]))
        elif dl == "medium":
            cap = min(cap, float(LANE_ENTRY_CAP["data_review"]))
    return float(round(min(float(raw), cap), 2))


def assign_entry_candidate_lane(
    *,
    fundamental_edge: float,
    ma_eval: dict,
    peg_quality: dict,
    eps_growth_rate: Optional[float],
    op_income_stable: Optional[bool],
    op_income_yoy: Optional[float],
    piot: dict,
    ret_21d: Optional[float],
    dr_meta: dict,
    valuation_satellite_candidate: bool,
    ps_only_satellite_candidate: bool,
    buy_timing_gate: bool,
    downtrend_rejection_gate: bool,
) -> str:
    """推奨レーン（CSV・MD 用）。data_review_* は dr_meta と整合させる。"""
    pw = str(peg_quality.get("peg_warning") or "")
    peg_extreme = pw in ("extremely_low_possible_oneoff", "very_low_check_cyclical")
    peg_reclaim_bad = pw in ("extremely_low_possible_oneoff", "eps_growth_too_high_oneoff_risk")
    eps_spike = eps_growth_rate is not None and np.isfinite(eps_growth_rate) and float(eps_growth_rate) > 80.0
    op_bad = op_income_stable is False or (
        op_income_yoy is not None and np.isfinite(op_income_yoy) and float(op_income_yoy) < 0
    )
    cyclical_trap = (peg_extreme or eps_spike) and op_bad

    raw_i = _piot_raw_int(piot)
    adj_f = _piot_adj_float(piot)
    cov = piot.get("piotroski_coverage_ratio")
    cov_f = float(cov) if cov is not None and np.isfinite(float(cov)) else None

    st = str(ma_eval.get("ma200_state") or "")
    r60 = ma_eval.get("recent_60d_low_update")

    dr_lvl = str(dr_meta.get("data_review_level") or "")
    dr_bad = bool(dr_meta.get("has_issues")) and dr_lvl in ("medium", "severe")
    dr_light_only = bool(dr_meta.get("has_issues")) and dr_lvl == "light"

    reclaim_quality = (
        op_income_stable is True
        and not peg_reclaim_bad
        and adj_f is not None
        and adj_f >= 6.0
        and cov_f is not None
        and cov_f >= MIN_PIOTROSKI_COVERAGE_CORE
        and raw_i is not None
        and raw_i > 3
    )

    bottom_quality = (
        reclaim_quality
        and adj_f is not None
        and adj_f >= 6.5
        and ret_21d is not None
        and np.isfinite(ret_21d)
        and float(ret_21d) > 0
        and r60 is False
    )

    if cyclical_trap:
        return "cyclical_value_trap"

    if adj_f is not None and np.isfinite(adj_f) and float(adj_f) < 4.0:
        return "data_review"

    if dr_bad:
        return "data_review"

    if downtrend_rejection_gate:
        return "excluded"

    if raw_i is not None and raw_i <= 3:
        return "data_review"

    if (
        st == "ma200_reclaim"
        and (not downtrend_rejection_gate)
        and reclaim_quality
        and fundamental_edge >= RECLAIM_CORE_MIN_FUNDAMENTAL
        and (not dr_meta.get("has_issues"))
    ):
        return "ma200_reclaim_core"

    if (
        st == "ma200_reclaim"
        and (not downtrend_rejection_gate)
        and reclaim_quality
        and fundamental_edge >= RECLAIM_CORE_MIN_FUNDAMENTAL
        and dr_light_only
    ):
        return "data_review_light"

    if (
        st == "ma200_reclaim"
        and (not downtrend_rejection_gate)
        and reclaim_quality
        and WEAK_RECLAIM_MIN_FUNDAMENTAL <= fundamental_edge < RECLAIM_CORE_MIN_FUNDAMENTAL
        and (not dr_bad)
    ):
        return "weak_reclaim_watch"

    if (
        fundamental_edge >= MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY
        and st == "below_ma200_basing"
        and ma_eval.get("below_ma200_downtrend") is not True
        and op_income_stable is True
        and (not downtrend_rejection_gate)
        and bottom_quality
    ):
        return "bottom_reversal_core"

    if (
        st == "above_ma200_extended"
        and fundamental_edge >= EXTENDED_FUNDAMENTAL_EDGE_MIN
        and raw_i is not None
        and raw_i > 3
        and adj_f is not None
        and adj_f >= 5.0
        and (not dr_bad)
    ):
        return "extended_above_ma200"

    if (
        fundamental_edge >= WATCH_FUNDAMENTAL_EDGE_MIN
        and (not buy_timing_gate)
        and st not in ("above_ma200_extended",)
        and raw_i is not None
        and raw_i > 3
        and (not dr_bad)
    ):
        return "watch_fundamental_core"

    if dr_light_only:
        return "data_review_light"

    if valuation_satellite_candidate or ps_only_satellite_candidate:
        return "satellite_valuation"

    return "excluded"


def compute_sales_cagr(history: List[dict], years: int = 3) -> Optional[float]:
    if not history or len(history) <= years:
        return None
    latest = history[0].get("revenue")
    past = history[years].get("revenue") if len(history) > years else None
    if latest is None or past is None or past <= 0 or latest <= 0:
        return None
    try:
        return (latest / past) ** (1 / years) - 1
    except (ZeroDivisionError, OverflowError):
        return None

def evaluate_operating_income_stability(history: List[dict],
                                        years: int = OP_INCOME_YEARS,
                                        drop_floor: float = OP_INCOME_DROP_FLOOR,
                                        exclude_deficit: bool = EXCLUDE_OP_INCOME_DEFICIT) -> dict:
    out = {"stable": None, "reason": "insufficient", "years_checked": years, "latest": None, "median_prev": None}
    if not history:
        return out

    op: list[Optional[float]] = []
    for rec in history[:max(1, years)]:
        v = rec.get("operating_income")
        op.append(None if v is None else float(v))

    if len(op) < 1 or op[0] is None:
        out["reason"] = "latest_missing"
        return out

    out["latest"] = op[0]

    if any(v is None for v in op):
        out["reason"] = "missing_in_window"
        out["stable"] = None
        return out

    if exclude_deficit and any(v <= 0 for v in op):
        out["stable"] = False
        out["reason"] = "deficit_in_window"
        return out

    prev = op[1:] if len(op) > 1 else []
    if prev:
        med_prev = float(np.median(prev))
        out["median_prev"] = med_prev
        if med_prev > 0 and op[0] < med_prev * float(drop_floor):
            out["stable"] = False
            out["reason"] = f"drop_below_floor({drop_floor})"
            return out

    out["stable"] = True
    out["reason"] = "ok"
    return out

def calculate_safety_criteria_v1(
    ps_ratio: Optional[float],
    cash_and_equivalents: Optional[float],
    market_cap: Optional[float],
    operating_cash_flow: Optional[float],
    equity_ratio: Optional[float],
    sales_cagr: Optional[float],
    max_drawdown: Optional[float],
) -> dict:
    criteria = {
        "ps_under_1": False,
        "cash_rich": False,
        "positive_ocf": False,
        "equity_ratio_50plus": False,
        "equity_ratio_70plus": False,
        "growth_potential": False,
        "no_speculative_drop": False,
    }
    scores: dict[str, float] = {}
    total_score = 0.0
    max_score = 100.0

    if ps_ratio is not None and ps_ratio < 1.0:
        criteria["ps_under_1"] = True
        scores["ps_under_1"] = 25.0
        total_score += 25.0
    else:
        scores["ps_under_1"] = 0.0

    if (cash_and_equivalents is not None and market_cap is not None and market_cap > 0 and cash_and_equivalents > market_cap):
        criteria["cash_rich"] = True
        scores["cash_rich"] = 20.0
        total_score += 20.0
    else:
        scores["cash_rich"] = 0.0

    if operating_cash_flow is not None and operating_cash_flow > 0:
        criteria["positive_ocf"] = True
        scores["positive_ocf"] = 20.0
        total_score += 20.0
    else:
        scores["positive_ocf"] = 0.0

    if equity_ratio is not None and equity_ratio >= 0.5:
        criteria["equity_ratio_50plus"] = True
        scores["equity_ratio_50plus"] = 15.0
        total_score += 15.0
    else:
        scores["equity_ratio_50plus"] = 0.0

    if equity_ratio is not None and equity_ratio >= 0.7:
        criteria["equity_ratio_70plus"] = True
        scores["equity_ratio_70plus"] = 10.0
        total_score += 10.0
    else:
        scores["equity_ratio_70plus"] = 0.0

    if sales_cagr is not None and sales_cagr > 0:
        criteria["growth_potential"] = True
        scores["growth_potential"] = 5.0
        total_score += 5.0
    else:
        scores["growth_potential"] = 0.0

    if max_drawdown is not None:
        if max_drawdown > -0.8:
            criteria["no_speculative_drop"] = True
            scores["no_speculative_drop"] = 5.0
            total_score += 5.0
        else:
            scores["no_speculative_drop"] = 0.0
    else:
        scores["no_speculative_drop"] = 0.0

    required_conditions_met = (
        criteria["ps_under_1"] and
        criteria["positive_ocf"] and
        criteria["equity_ratio_50plus"] and
        criteria["no_speculative_drop"]
    )

    return {
        "criteria": criteria,
        "scores": scores,
        "total_score": round(total_score, 1),
        "max_score": max_score,
        "required_conditions_met": required_conditions_met,
        "ps_ratio": ps_ratio,
        "cash_and_equivalents": cash_and_equivalents,
        "equity_ratio": equity_ratio,
        "sales_cagr": sales_cagr,
        "max_drawdown": max_drawdown,
    }

# ------------------------------------------------------------
# Piotroski（実データのみ）
# ------------------------------------------------------------
def calculate_piotroski_real(fin: dict) -> dict:
    def parse_num(x) -> Optional[float]:
        if x in (None, "", "NA") or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
            return None
        try:
            if isinstance(x, str):
                x = x.replace(",", "").strip()
            return float(x)
        except (TypeError, ValueError):
            return None

    def ratio_num(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None or b == 0:
            return None
        return a / b

    cur = fin.get("current") or {}
    prev = fin.get("previous") or {}

    ni_c = parse_num(cur.get("net_income"))
    ni_p = parse_num(prev.get("net_income"))
    ocf_c = parse_num(cur.get("operating_cash_flow"))
    ocf_p = parse_num(prev.get("operating_cash_flow"))
    rev_c = parse_num(cur.get("revenue"))
    rev_p = parse_num(prev.get("revenue"))
    ta_c = parse_num(cur.get("total_assets"))
    ta_p = parse_num(prev.get("total_assets"))
    eq_c = parse_num(cur.get("equity"))
    eq_p = parse_num(prev.get("equity"))
    ca_c = parse_num(cur.get("current_assets"))
    ca_p = parse_num(prev.get("current_assets"))
    cl_c = parse_num(cur.get("current_liabilities"))
    cl_p = parse_num(prev.get("current_liabilities"))
    sh_c = parse_num(cur.get("shares_outstanding"))
    sh_p = parse_num(prev.get("shares_outstanding"))
    gpm_c = parse_num(cur.get("gross_profit_margin"))
    gpm_p = parse_num(prev.get("gross_profit_margin"))

    comp: dict[str, Optional[bool]] = {}
    comp["positive_net_income"] = None if ni_c is None else (ni_c > 0)
    comp["positive_ocf"] = None if ocf_c is None else (ocf_c > 0)
    if ocf_c is None or ni_c is None:
        comp["ocf_gt_ni"] = None
    else:
        comp["ocf_gt_ni"] = ocf_c > ni_c

    roa_c = ratio_num(ni_c, ta_c)
    roa_p = ratio_num(ni_p, ta_p)
    if roa_c is None or roa_p is None:
        comp["roa_up"] = None
    else:
        comp["roa_up"] = roa_c > roa_p

    om_c = ratio_num(ocf_c, rev_c)
    om_p = ratio_num(ocf_p, rev_p)
    if om_c is None or om_p is None:
        comp["ocf_margin_up"] = None
    else:
        comp["ocf_margin_up"] = om_c > om_p

    cr_c = ratio_num(ca_c, cl_c)
    cr_p = ratio_num(ca_p, cl_p)
    if cr_c is None or cr_p is None:
        comp["current_ratio_up"] = None
    else:
        comp["current_ratio_up"] = cr_c > cr_p

    if sh_c is None or sh_p is None:
        comp["shares_down"] = None
    else:
        comp["shares_down"] = sh_c < sh_p

    if gpm_c is None or gpm_p is None:
        comp["gpm_up"] = None
    else:
        comp["gpm_up"] = gpm_c > gpm_p

    def leverage(ta: Optional[float], eq: Optional[float]) -> Optional[float]:
        if ta is None or eq is None or ta == 0:
            return None
        return (ta - eq) / ta

    lv_c = leverage(ta_c, eq_c)
    lv_p = leverage(ta_p, eq_p)
    if lv_c is None or lv_p is None:
        comp["leverage_down"] = None
    else:
        comp["leverage_down"] = lv_c < lv_p

    possible_items = 9
    observed_items = sum(1 for v in comp.values() if v is not None)
    observed_score = sum(1 for v in comp.values() if v is True)
    coverage_ratio = (observed_items / possible_items) if possible_items else None

    if observed_items == 0:
        piotroski_adjusted_score = None
    else:
        piotroski_adjusted_score = round(float(observed_score) / float(observed_items) * 9.0, 3)
    conf_mult = 1.0
    if coverage_ratio is not None and np.isfinite(coverage_ratio):
        conf_mult = 0.55 + 0.45 * min(1.0, float(coverage_ratio) / 0.80)
    piotroski_effective_score = (
        None if piotroski_adjusted_score is None else round(piotroski_adjusted_score * conf_mult, 3)
    )
    piotroski_quality_positive = False
    if piotroski_adjusted_score is not None:
        piotroski_quality_positive = bool(
            observed_score >= MIN_PIOTROSKI_CORE
            or piotroski_adjusted_score >= 6.5
        )
    piotroski_confidence_low = (
        coverage_ratio is not None
        and np.isfinite(coverage_ratio)
        and float(coverage_ratio) < MIN_PIOTROSKI_COVERAGE_CORE
    )

    base_eval = (
        "優秀" if observed_score >= 7 else "良好" if observed_score >= 5 else "普通" if observed_score >= 3 else "注意"
    )
    if observed_items < possible_items:
        evaluation = f"{base_eval}・要注意(coverage_low)"
    else:
        evaluation = base_eval

    return {
        "score": observed_score,
        "piotroski_raw_score": observed_score,
        "piotroski_available_items": observed_items,
        "piotroski_coverage_ratio": coverage_ratio,
        "piotroski_adjusted_score": piotroski_adjusted_score,
        "piotroski_effective_score": piotroski_effective_score,
        "piotroski_quality_positive": piotroski_quality_positive,
        "piotroski_confidence_low": piotroski_confidence_low,
        "details": comp,
        "evaluation": evaluation,
        "mode": "real",
        "observed_items": observed_items,
        "possible_items": possible_items,
        "coverage_ratio": coverage_ratio,
    }

# ------------------------------------------------------------
# バリュエーション（モックなし）
# ------------------------------------------------------------
def calculate_ps_ratio(current_price: Optional[float], revenue_per_share: Optional[float]=None,
                       market_cap: Optional[float]=None, revenue: Optional[float]=None) -> Optional[float]:
    try:
        ps: Optional[float] = None
        if current_price is not None and current_price > 0:
            if revenue_per_share is not None and revenue_per_share > 0:
                ps = float(current_price) / float(revenue_per_share)
        if ps is None:
            if (
                market_cap is not None and market_cap > 0
                and revenue is not None and revenue > 0
            ):
                ps = float(market_cap) / float(revenue)
        if ps is None or not np.isfinite(ps):
            return None
        if ps <= 0 or ps >= 1000:
            return None
        return ps
    except Exception:
        return None

def calculate_peg_ratio(per: Optional[float], eps_growth_rate_pct: Optional[float]) -> Optional[float]:
    """PEG=PER/EPS成長率。0や極小値は欠損扱い（成長率異常・データ不備の可能性）"""
    try:
        if per is None or eps_growth_rate_pct is None:
            return None
        if per <= 0 or eps_growth_rate_pct <= 0:
            return None
        peg = float(per) / float(eps_growth_rate_pct)
        if peg <= 0 or not np.isfinite(peg):
            return None
        if peg < 0.01:
            return None
        return peg
    except Exception:
        return None

def evaluate_peg_quality(
    reference_peg: Optional[float],
    eps_growth_rate: Optional[float],
) -> dict:
    """PEG の信頼区間と警告理由（総合スコア直結はせず、ファンダエッジ・レーン判定に使用）。"""
    def _bad_num(x: Optional[float]) -> bool:
        return x is None or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x)))

    out: dict[str, Any] = {"peg_trusted": False, "peg_warning": "missing"}
    if _bad_num(reference_peg):
        return out
    rp = float(reference_peg)
    if rp <= 0:
        out["peg_warning"] = "non_positive"
        return out
    if rp < 0.10:
        out["peg_warning"] = "extremely_low_possible_oneoff"
        return out
    if 0.10 <= rp < 0.30:
        if _bad_num(eps_growth_rate):
            out["peg_warning"] = "eps_growth_missing"
            return out
        out["peg_trusted"] = "caution"
        out["peg_warning"] = "very_low_check_cyclical"
        return out
    if _bad_num(eps_growth_rate):
        out["peg_warning"] = "eps_growth_missing"
        return out
    eg = float(eps_growth_rate)
    if eg > 80.0:
        out["peg_warning"] = "eps_growth_too_high_oneoff_risk"
        return out
    if 0.30 <= rp <= 1.20:
        out["peg_trusted"] = True
        out["peg_warning"] = "ok"
        return out
    out["peg_trusted"] = True
    out["peg_warning"] = "expensive_or_moderate"
    return out


def estimate_eps_growth_rate(net_income_current: Optional[float],
                             net_income_previous: Optional[float],
                             shares_outstanding_current: Optional[float],
                             shares_outstanding_previous: Optional[float]) -> Optional[float]:
    """YoY EPS 成長率(%)。当期・前期で**別々の発行済株式数**で EPS を定義。前期株数欠損時は計算しない。"""
    try:
        if net_income_current is None or net_income_previous is None:
            return None
        if shares_outstanding_current is None or shares_outstanding_previous is None:
            return None
        if shares_outstanding_current <= 0 or shares_outstanding_previous <= 0:
            return None
        eps_cur = float(net_income_current) / float(shares_outstanding_current)
        eps_prev = float(net_income_previous) / float(shares_outstanding_previous)
        if eps_prev <= 0:
            return None
        return (eps_cur / eps_prev - 1.0) * 100.0
    except Exception:
        return None

def calculate_valuation_metrics_ps_peg(current_price: Optional[float],
                                       net_income_current: Optional[float],
                                       net_income_previous: Optional[float],
                                       revenue_current: Optional[float],
                                       shares_outstanding: Optional[float],
                                       shares_outstanding_previous: Optional[float] = None) -> dict:
    rps = None
    if (
        revenue_current is not None
        and shares_outstanding is not None
        and shares_outstanding > 0
    ):
        rps = float(revenue_current) / float(shares_outstanding)

    per = None
    if (
        net_income_current is not None
        and shares_outstanding is not None
        and shares_outstanding > 0
    ):
        eps = float(net_income_current) / float(shares_outstanding)
        if eps > 0 and current_price is not None and current_price > 0:
            per = float(current_price) / eps

    eps_growth = estimate_eps_growth_rate(
        net_income_current,
        net_income_previous,
        shares_outstanding,
        shares_outstanding_previous,
    )
    ps_ratio = calculate_ps_ratio(current_price, revenue_per_share=rps)
    peg_ratio = calculate_peg_ratio(per, eps_growth)
    return {
        "revenue_per_share": rps,
        "per": per,
        "ps_ratio": ps_ratio,
        "eps_growth_rate": eps_growth,
        "peg_ratio": peg_ratio,
        "reference_peg": peg_ratio,
        "peg_trusted": False
    }

# ------------------------------------------------------------
# 安全性・投機性
# ------------------------------------------------------------
def calculate_safety_score_v3(
    margin_ratio: float = None,
    short_selling_change_rate: float = None,
    yoy_eps_growth: float = None,
    dividend_status: str = None,
    avg_volume: int = None,
    avg_trading_value: float = None,
    stagnant_days_after_spike: int = None,
    current_volatility: float = None,
    average_volatility: float = None,
    below_ma25: bool = False,
    below_ma75: bool = False
) -> dict:
    safety_score = 0.0
    details = {}
    max_total_score = 25.0
    w = {'margin_ratio':4.0,'short_selling':4.0,'earnings_stability':3.5,'dividend_stability':3.0,
         'liquidity':2.5,'momentum_stability':2.5,'volatility_stability':2.5,'technical_strength':3.0}

    if margin_ratio is None:
        margin_score = 0.0
    else:
        margin_score = w['margin_ratio'] * (1.0 if margin_ratio<=3 else 0.8 if margin_ratio<=5 else 0.6 if margin_ratio<=10 else 0.3 if margin_ratio<=20 else 0)
    if short_selling_change_rate is None:
        short_score = 0.0
    else:
        short_score  = w['short_selling'] * (1.0 if short_selling_change_rate<=5 else 0.8 if short_selling_change_rate<=15 else 0.5 if short_selling_change_rate<=30 else 0.2 if short_selling_change_rate<=50 else 0)
    safety_score += margin_score + short_score
    details['信用安全性'] = (
        f"不明 (0.0)" if margin_ratio is None else f"{margin_ratio:.1f}倍 ({margin_score:.1f})"
    )
    details['空売り安全性'] = (
        f"不明 (0.0)" if short_selling_change_rate is None else f"{short_selling_change_rate:.1f}% ({short_score:.1f})"
    )

    if yoy_eps_growth is None:
        eps_score = 0.0
    else:
        eps_score = w['earnings_stability'] * (1.0 if yoy_eps_growth>=20 else 0.8 if yoy_eps_growth>=10 else 0.7 if yoy_eps_growth>=0 else 0.4 if yoy_eps_growth>=-10 else 0.2 if yoy_eps_growth>=-20 else 0)
    if dividend_status in (None, ""):
        div_score = 0.0
    else:
        div_score = w['dividend_stability'] * (1.0 if dividend_status=='増配' else 0.8 if dividend_status=='維持' else 0.3 if dividend_status=='未定' else 0.1 if dividend_status=='減配' else 0)
    safety_score += eps_score + div_score
    details['業績安定性'] = (
        f"不明 (0.0)" if yoy_eps_growth is None else f"EPS成長率{yoy_eps_growth:.1f}% ({eps_score:.1f})"
    )
    details['配当安定性'] = f"{dividend_status or '不明'} (0.0)" if dividend_status in (None, "") else f"{dividend_status} ({div_score:.1f})"

    liq_observed = (avg_volume is not None) or (
        avg_trading_value is not None and np.isfinite(avg_trading_value)
    )
    if not liq_observed:
        volume_score = 0.0
        vm = 0.0
    else:
        if avg_volume is None:
            vm = 0.0
        else:
            vm = 1.0 if avg_volume>=500000 else 0.8 if avg_volume>=200000 else 0.6 if avg_volume>=100000 else 0.3 if avg_volume>=50000 else 0
        volume_score = w['liquidity'] * vm
        if avg_trading_value is not None and np.isfinite(avg_trading_value):
            adv_score = 1.0 if avg_trading_value >= 1_000_000_000 else 0.8 if avg_trading_value >= 500_000_000 else 0.6 if avg_trading_value >= 300_000_000 else 0.3 if avg_trading_value >= 100_000_000 else 0.0
            volume_score = w['liquidity'] * max(vm, adv_score)
    safety_score += volume_score
    if not liq_observed:
        liq_note = '不明'
    else:
        liq_note = ''
        if avg_volume is not None:
            liq_note = f'{avg_volume:,}株'
        if avg_trading_value is not None and np.isfinite(avg_trading_value):
            liq_note += (' / ' if liq_note else '') + f'ADV{avg_trading_value/1e6:.0f}MJPY'
        if not liq_note:
            liq_note = 'データあり'
    details['流動性'] = f"{liq_note} ({volume_score:.1f})" if liq_observed else f"不明 (0.0)"

    if stagnant_days_after_spike is None:
        stagnant_score = 0.0
    else:
        stagnant_score = w['momentum_stability'] * (1.0 if stagnant_days_after_spike==0 else 0.8 if stagnant_days_after_spike<=2 else 0.5 if stagnant_days_after_spike<=4 else 0.2 if stagnant_days_after_spike<=6 else 0)
    if current_volatility is not None and average_volatility not in (None, 0):
        vr = current_volatility / average_volatility
        vol_score = w['volatility_stability'] * (1.0 if vr<=1.2 else 0.8 if vr<=1.5 else 0.5 if vr<=2.0 else 0.2 if vr<=2.5 else 0)
        vol_note = f"{vr:.1f}倍"
    else:
        vol_score = 0.0
        vol_note = "不明 (0.0)"
    safety_score += stagnant_score + vol_score
    details['モメンタム安定性'] = (
        f"不明 (0.0)" if stagnant_days_after_spike is None else f"{stagnant_days_after_spike}日 ({stagnant_score:.1f})"
    )
    details['ボラティリティ安定性'] = f"{vol_note} ({vol_score:.1f})"

    if not below_ma25 and not below_ma75:
        tech_score = w['technical_strength']
        tech_note = "25日・75日線上方"
    elif not below_ma25 or not below_ma75:
        tech_score = w['technical_strength'] * 0.5
        tech_note = "一部移動平均線上方"
    else:
        tech_score = 0.0
        tech_note = "25日・75日線下方"
    safety_score += tech_score
    details['テクニカル強さ'] = f"{tech_note} ({tech_score:.1f})"

    possible_items = 8
    observed_items = sum([
        margin_ratio is not None,
        short_selling_change_rate is not None,
        yoy_eps_growth is not None,
        dividend_status not in (None, ""),
        liq_observed,
        stagnant_days_after_spike is not None,
        current_volatility is not None and average_volatility not in (None, 0),
        True,
    ])
    coverage_ratio = observed_items / possible_items if possible_items else None

    ratio = safety_score / max_total_score
    level = "🟢 非常に安全" if ratio>=0.8 else "🔵 安全" if ratio>=0.6 else "🟡 普通" if ratio>=0.4 else "🟠 やや危険" if ratio>=0.2 else "🔴 危険"
    return {
        "total_score": round(safety_score,1),
        "max_score": max_total_score,
        "safety_level": level,
        "details": details,
        "observed_items": observed_items,
        "possible_items": possible_items,
        "coverage_ratio": coverage_ratio,
    }

def detect_speculative_manipulation_v2(
    margin_ratio: float | None = None,
    short_selling_change_rate: float | None = None,
    yoy_eps_growth: float | None = None,
    dividend_status: str | None = None,
    avg_volume: int | None = None,
    stagnant_days_after_spike: int | None = None,
    current_volatility: float | None = None,
    average_volatility: float | None = None,
    below_ma25: bool = False,
    below_ma75: bool = False,
    current_price: float | None = 1000.0,
    mas: dict | None = None,
    stock_code: str | None = None
) -> dict:
    # 名称は歴史的互換のため維持。入力欠損が多い場合はテクニカル寄りの検出に偏る（返却の model_scope を参照）。
    score = 0
    flags = []; risks = []
    if margin_ratio is not None:
        if margin_ratio >= 50: score += 25; flags.append(f"🚨 信用倍率異常高: {margin_ratio:.1f}倍")
        elif margin_ratio >= 20: score += 15; flags.append(f"⚠️ 信用倍率高: {margin_ratio:.1f}倍")
        elif margin_ratio >= 10: score += 8; risks.append(f"信用倍率やや高: {margin_ratio:.1f}倍")
    if short_selling_change_rate is not None:
        if short_selling_change_rate >= 100: score += 20; flags.append(f"🚨 空売り残急増: +{short_selling_change_rate:.1f}%")
        elif short_selling_change_rate >= 50: score += 12; flags.append(f"⚠️ 空売り残増加: +{short_selling_change_rate:.1f}%")
        elif short_selling_change_rate >= 25: score += 6; risks.append(f"空売り残やや増加: +{short_selling_change_rate:.1f}%")
    if stagnant_days_after_spike is not None:
        if stagnant_days_after_spike >= 5: score += 15; flags.append(f"📉 急騰後の横ばい: {stagnant_days_after_spike}日")
        elif stagnant_days_after_spike >= 3: score += 8; risks.append(f"横ばい傾向: {stagnant_days_after_spike}日")
    if current_volatility is not None and average_volatility not in (None, 0):
        vr = current_volatility / average_volatility
        if vr >= 3.0: score += 20; flags.append(f"🚨 ボラティリティ異常: {vr:.1f}倍")
        elif vr >= 2.0: score += 12; flags.append(f"⚠️ ボラティリティ高: {vr:.1f}倍")
        elif vr >= 1.5: score += 6; risks.append(f"ボラティリティやや高: {vr:.1f}倍")
    if below_ma25 and below_ma75: score += 8; flags.append("⚠️ 25・75日線の両方割れ")
    elif below_ma25 or below_ma75: score += 4; risks.append("移動平均線の一部割れ")
    if avg_volume is not None and avg_volume < 30000: score += 8; flags.append(f"⚠️ 流動性低: {avg_volume:,}株/日")
    if dividend_status in {"未定","減配"}: score += 6; risks.append(f"配当{dividend_status}")
    if yoy_eps_growth is not None and yoy_eps_growth < -30: score += 8; flags.append(f"⚠️ EPS急減: {yoy_eps_growth:.1f}%")

    level = "🔴 極めて投機的" if score>=70 else "🟠 高い" if score>=50 else "🟡 やや高い" if score>=30 else "🟢 低い"
    return {
        "score": score,
        "level": level,
        "warning_flags": flags,
        "risk_factors": risks,
        "max_score": 100,
        "model_scope": "technical_only",
        "scope_note": "需給・信用残などの外部データ未接続時は、価格・出来高・ボラ・移動平均に基づく簡易プロキシです。",
    }

# ------------------------------------------------------------
# 単銘柄フル分析（offline対応、モックなし）
# ------------------------------------------------------------
def analyze_single_stock_complete_v3(session: requests.Session,
                                     sector_averages: dict,
                                     code: str,
                                     name: str = "",
                                     market: str = "",
                                     sector_hint: str | None = None,
                                     *,
                                     offline: bool = False,
                                     instrument_type: str | None = None,
                                     ) -> dict:
    try:
        fdm = FinancialDataManager(session)
        sector33_raw = (sector_hint or "").strip() or None
        sector_src = sector33_raw or DynamicSectorAverages.get_sector_static(code)
        sector = DynamicSectorAverages.normalize_sector(str(sector_src or ""))
        fc = FrozenCache()
        fin_meta_src: Dict[str, Any] = {}
        inst_t = (instrument_type or "stock").strip() or "stock"
        skip_financial_lane = inst_t != "stock"

        # 価格
        if offline:
            price_df = fc.load_prices(code)
        else:
            price_df = fetch_prices_v2(session, code, cache_name=f"prices_{code}")
        if price_df is None or price_df.empty:
            return {
                "stock_code": code,
                "company_name": name,
                "sector_name": sector,
                "success": False,
                "error": "price_missing",
                "instrument_type": inst_t,
            }

        def _col(df, *cands):
            lc = {c.lower(): c for c in df.columns}
            for c in cands:
                for k, v in lc.items():
                    if k == c.lower():
                        return v
            return None

        c_close = _col(price_df, "Close","ClosePrice","EndPrice","AdjustmentClose","AdjClose")
        c_high  = _col(price_df, "High","HighPrice")
        c_low   = _col(price_df, "Low","LowPrice")
        c_vol   = _col(price_df, "Volume","TradingVolume")
        c_date  = _col(price_df, "Date","TradingDate")
        if c_date:
            price_df = price_df.sort_values(c_date)

        close = price_df[c_close].astype(float) if c_close in price_df.columns else pd.Series([], dtype=float)
        high  = price_df[c_high].astype(float)  if c_high  in price_df.columns else close
        low   = price_df[c_low].astype(float)   if c_low   in price_df.columns else close
        vol_s = price_df[c_vol].astype(float)   if c_vol   in price_df.columns else None

        current_price = float(close.iloc[-1]) if len(close) else None
        mas = calculate_moving_averages(close) if len(close) else {}
        rsi = float(calculate_rsi(close)) if len(close) else None
        adx, plus_di, minus_di = calculate_adx_and_di(high, low, close) if len(close) else (None, None, None)
        ma200_eval = evaluate_ma200_entry_state(
            close, high, low, adx=adx, plus_di=plus_di, minus_di=minus_di,
        ) if len(close) else {
            "ma200_state": "ma200_unknown",
            "ma200_timing_score": 0.0,
            "ma200_risk_penalty": 0.0,
            "ma200_reason": "insufficient_price_history",
            "distance_from_ma200": None,
            "crossed_above_ma200_recently": False,
            "below_ma200_basing": False,
            "below_ma200_downtrend": False,
            "above_ma200_extended": False,
            "recent_60d_low_update": None,
        }
        cur_vol, avg_vol = calculate_volatility(close) if len(close) else (None, None)
        momentum = calculate_medium_term_momentum(close) if len(close) else {}

        below_ma25 = bool(current_price is not None and mas.get("ma_25") is not None and current_price < mas["ma_25"])
        below_ma75 = bool(current_price is not None and mas.get("ma_75") is not None and current_price < mas["ma_75"])
        below_ma200 = bool(current_price is not None and mas.get("ma_200") is not None and current_price < mas["ma_200"])
        latest_price_date = None
        if c_date and c_date in price_df.columns and len(price_df[c_date]):
            latest_price_date = _parse_optional_date(price_df[c_date].iloc[-1])
        liquidity = calculate_liquidity_metrics(close, vol_s)
        avg_volume = liquidity.get("avg_volume_30d")
        adv_jpy_20d = liquidity.get("adv_jpy_20d")
        adv_jpy_60d = liquidity.get("adv_jpy_60d")
        traded_days_60d = liquidity.get("traded_days_60d")

        # 財務（ETF/ETN/投信等は個別株用の fins を取得しない。※未取得は None、推定しない）
        _skip_fin_meta = {
            "financial_data_mode": "skipped_non_stock_instrument",
            "fins_details_available": False,
            "fins_details_status": None,
            "fins_details_error": "",
        }
        if skip_financial_lane:
            stmts = []
            fin_meta_src = dict(_skip_fin_meta)
        elif offline:
            stmts, fin_meta_src = fc.load_statement_bundle(code)
        else:
            stmts = fdm.fetch_statements(code)
            fin_meta_src = fdm.get_last_financial_fetch_meta()

        financial_history, statement_basis_used = build_financial_history_from_statements(
            stmts if isinstance(stmts, list) else [],
            max_years=5,
            as_of_date=latest_price_date,
            statement_basis="annual",
        )
        annual_quality_ok = statement_basis_used == "annual"
        cur_fin = financial_history[0].copy() if financial_history else {}
        prv_fin = financial_history[1].copy() if len(financial_history) > 1 else {}
        raw_fin = {"current": cur_fin.copy(), "previous": prv_fin.copy(), "current_price": current_price, "sector": sector}
        imputed_fin = fdm._fill_missing_fields(copy.deepcopy(raw_fin))
        imputation = imputed_fin.get("_imputation", {}) if isinstance(imputed_fin, dict) else {}

        sector_benchmark = (
            sector_averages.get(sector)
            or sector_averages.get(sector_src or "")
            or DynamicSectorAverages.default_sector_average(sector)
        ) if isinstance(sector_averages, dict) else DynamicSectorAverages.default_sector_average(sector)

        # 指標
        piot = calculate_piotroski_real(raw_fin)
        val = calculate_valuation_metrics_ps_peg(
            current_price=current_price,
            net_income_current=raw_fin["current"].get("net_income"),
            net_income_previous=raw_fin["previous"].get("net_income"),
            revenue_current=raw_fin["current"].get("revenue"),
            shares_outstanding=raw_fin["current"].get("shares_outstanding"),
            shares_outstanding_previous=raw_fin["previous"].get("shares_outstanding"),
        )
        peg_quality = evaluate_peg_quality(val.get("reference_peg"), val.get("eps_growth_rate"))
        val["peg_trusted"] = peg_quality.get("peg_trusted")
        val["peg_warning"] = peg_quality.get("peg_warning")
        eps_growth_for_scoring = val.get("eps_growth_rate")
        safety = calculate_safety_score_v3(
            yoy_eps_growth=eps_growth_for_scoring,
            avg_volume=avg_volume,
            avg_trading_value=adv_jpy_20d,
            current_volatility=cur_vol, average_volatility=avg_vol,
            below_ma25=below_ma25, below_ma75=below_ma75
        )
        spec = detect_speculative_manipulation_v2(
            yoy_eps_growth=eps_growth_for_scoring,
            avg_volume=avg_volume,
            current_volatility=cur_vol, average_volatility=avg_vol,
            below_ma25=below_ma25, below_ma75=below_ma75,
            current_price=current_price, mas=mas, stock_code=code
        )

        shares_outstanding = raw_fin["current"].get("shares_outstanding")
        market_cap = None
        if current_price is not None and shares_outstanding is not None and shares_outstanding > 0:
            market_cap = current_price * shares_outstanding

        max_dd = calculate_max_drawdown(close, lookback_days=LOOKBACK_DAYS) if len(close) > 0 else None
        if annual_quality_ok and financial_history:
            sales_cagr = compute_sales_cagr(financial_history, years=3)
        else:
            sales_cagr = None
        cash_eq = raw_fin["current"].get("cash_and_equivalents")
        equity_ratio = raw_fin["current"].get("equity_ratio")
        if equity_ratio is None:
            ta = raw_fin["current"].get("total_assets")
            eq = raw_fin["current"].get("equity")
            if ta not in (None, 0) and eq is not None:
                equity_ratio = eq / ta

        safety_criteria = calculate_safety_criteria_v1(
            ps_ratio=val.get("ps_ratio"),
            cash_and_equivalents=cash_eq,
            market_cap=market_cap,
            operating_cash_flow=raw_fin["current"].get("operating_cash_flow"),
            equity_ratio=equity_ratio,
            sales_cagr=sales_cagr,
            max_drawdown=max_dd,
        )

        # ★必須フィルタ
        liquidity_ok = bool(
            avg_volume is not None and avg_volume >= MIN_AVG_VOLUME_30D and
            adv_jpy_20d is not None and adv_jpy_20d >= MIN_ADV_JPY_20D
        )
        market_cap_ok = (market_cap is not None and market_cap >= MIN_MARKET_CAP_JPY)

        ps_ratio = val.get("ps_ratio")
        per = val.get("per")

        ps_available = (ps_ratio is not None and np.isfinite(ps_ratio))
        per_available = (per is not None and np.isfinite(per))
        valuation_available = bool(ps_available and per_available)

        rc, rp = raw_fin["current"], raw_fin["previous"]
        valuation_input_complete = (
            current_price is not None
            and rc.get("shares_outstanding") not in (None, 0)
            and rc.get("revenue") is not None
            and rc.get("net_income") is not None
            and rp.get("net_income") is not None
            and rp.get("shares_outstanding") is not None
        )
        eps_growth_input_complete = bool(
            rc.get("net_income") is not None
            and rp.get("net_income") is not None
            and rc.get("shares_outstanding") is not None
            and rc.get("shares_outstanding") > 0
            and rp.get("shares_outstanding") is not None
            and rp.get("shares_outstanding") > 0
        )
        piotroski_input_complete = bool(
            rc.get("net_income") is not None
            and rc.get("operating_cash_flow") is not None
            and rc.get("revenue") is not None
            and rc.get("total_assets") is not None
            and rc.get("equity") is not None
            and rc.get("current_assets") is not None
            and rc.get("current_liabilities") is not None
            and rc.get("gross_profit_margin") is not None
            and rp.get("net_income") is not None
            and rp.get("total_assets") is not None
            and rp.get("equity") is not None
            and rp.get("operating_cash_flow") is not None
            and rp.get("revenue") is not None
            and rp.get("current_assets") is not None
            and rp.get("current_liabilities") is not None
            and rp.get("gross_profit_margin") is not None
            and rp.get("shares_outstanding") is not None
        )

        defensive_ps_ok = (ps_ratio is not None and np.isfinite(ps_ratio) and ps_ratio <= MAX_PS_DEFENSIVE)
        per_satellite = (per is not None and np.isfinite(per) and per > MAX_PER_CORE)
        per_core_ok = (per is not None and np.isfinite(per) and per <= MAX_PER_CORE)

        sector_ps_benchmark = sector_benchmark.get("ps") if isinstance(sector_benchmark, dict) else None
        try:
            _sb_ps = float(sector_ps_benchmark) if sector_ps_benchmark is not None else float("nan")
        except (TypeError, ValueError):
            _sb_ps = float("nan")
        if np.isfinite(_sb_ps) and _sb_ps > 0:
            ps_satellite_limit = float(max(MAX_PS_DEFENSIVE, _sb_ps * 1.25))
        else:
            ps_satellite_limit = float(max(MAX_PS_DEFENSIVE, 3.0))

        if annual_quality_ok:
            op_income_eval = evaluate_operating_income_stability(
                financial_history,
                years=OP_INCOME_YEARS,
                drop_floor=OP_INCOME_DROP_FLOOR,
                exclude_deficit=EXCLUDE_OP_INCOME_DEFICIT
            )
        else:
            latest_op = financial_history[0].get("operating_income") if financial_history else None
            op_income_eval = {
                "stable": None,
                "reason": "non_annual_basis",
                "years_checked": OP_INCOME_YEARS,
                "latest": latest_op,
                "median_prev": None,
            }
        op_income_stable = (op_income_eval.get("stable") is True)

        statement_type = financial_history[0].get("statement_type") if financial_history else None
        statement_disclosed_date = financial_history[0].get("disclosed_date") if financial_history else None
        statement_staleness_days = None
        if latest_price_date and statement_disclosed_date:
            disclosed_dt = _parse_optional_date(statement_disclosed_date)
            if disclosed_dt:
                statement_staleness_days = (latest_price_date - disclosed_dt).days

        critical_missing = sum(
            value is None for value in (
                val.get("ps_ratio"),
                val.get("per"),
                piot.get("score") if isinstance(piot, dict) else None,
                sales_cagr,
                adv_jpy_20d,
                statement_disclosed_date,
            )
        )
        fin_diag = {
            "financial_data_mode": fin_meta_src.get("financial_data_mode"),
            "fins_details_available": fin_meta_src.get("fins_details_available"),
            "fins_details_status": fin_meta_src.get("fins_details_status"),
            "fins_details_error": fin_meta_src.get("fins_details_error"),
            "instrument_type": inst_t,
        }
        diagnostics = {
            **fin_diag,
            "latest_price_date": latest_price_date.isoformat() if latest_price_date else None,
            "statement_disclosed_date": statement_disclosed_date,
            "statement_type": statement_type,
            "statement_staleness_days": statement_staleness_days,
            "statement_basis_used": statement_basis_used,
            "fallback_basis_flag": (statement_basis_used == "fallback_primary_type"),
            "annual_financial_compare_ok": annual_quality_ok,
            "piotroski_basis_note": (
                None if annual_quality_ok
                else "non_annual_basis: 年次と四半期が混在しうるため、前年比較系の解釈に注意"
            ),
            "valuation_input_complete": valuation_input_complete,
            "eps_growth_input_complete": eps_growth_input_complete,
            "piotroski_input_complete": piotroski_input_complete,
            "growth_proxy_detached_from_scoring": False,
            "imputation": imputation,
            "critical_missing_count": critical_missing,
            "sector33_name_raw": sector33_raw,
            "raw_financial_fields": {
                "has_current_shares": raw_fin["current"].get("shares_outstanding") is not None,
                "has_previous_net_income": raw_fin["previous"].get("net_income") is not None,
                "has_current_assets": raw_fin["current"].get("current_assets") is not None,
                "has_current_liabilities": raw_fin["current"].get("current_liabilities") is not None,
                "has_gross_profit_margin": raw_fin["current"].get("gross_profit_margin") is not None,
            },
        }

        base_ok = bool(liquidity_ok and market_cap_ok and op_income_stable)
        core_candidate = bool(base_ok and valuation_available and defensive_ps_ok and per_core_ok)
        valuation_satellite_candidate = bool(base_ok and valuation_available and (not core_candidate))
        ps_only_satellite_candidate = bool(
            base_ok
            and ps_available
            and (not per_available)
            and np.isfinite(ps_ratio)
            and (ps_ratio <= ps_satellite_limit)
        )
        satellite_candidate = bool(valuation_satellite_candidate or ps_only_satellite_candidate)
        excluded_candidate = bool((not core_candidate) and (not satellite_candidate))
        if core_candidate:
            valuation_lane = "core"
        elif valuation_satellite_candidate:
            valuation_lane = "satellite_valuation"
        elif ps_only_satellite_candidate:
            valuation_lane = "satellite_ps_only"
        else:
            valuation_lane = "excluded"

        op_income_yoy_pct = None
        try:
            _opc = raw_fin["current"].get("operating_income")
            _opp = raw_fin["previous"].get("operating_income")
            if _opc is not None and _opp is not None and float(_opp) != 0:
                op_income_yoy_pct = (float(_opc) / float(_opp) - 1.0) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            op_income_yoy_pct = None

        ps_vs_sector_pre = _ps_vs_sector_ratio(ps_ratio, sector_ps_benchmark, None)
        _op_stable_tri = op_income_eval.get("stable")
        fundamental_edge_score = compute_fundamental_edge_score(
            ps_ratio=ps_ratio,
            ps_vs_sector=ps_vs_sector_pre,
            reference_peg=val.get("reference_peg"),
            eps_growth_rate=val.get("eps_growth_rate"),
            piot=piot if isinstance(piot, dict) else {},
            peg_quality=peg_quality,
            op_income_stable=_op_stable_tri if _op_stable_tri is not None else None,
            sales_cagr=sales_cagr,
            per=per,
            critical_missing_count=critical_missing,
            statement_basis_used=statement_basis_used,
            statement_staleness_days=float(statement_staleness_days) if statement_staleness_days is not None else None,
        )
        _pr_meta = piot.get("piotroski_raw_score")
        if _pr_meta is None:
            _pr_meta = piot.get("score")
        dr_meta = compute_data_review_meta(
            critical_missing_count=critical_missing,
            statement_basis_used=statement_basis_used,
            fallback_basis_flag=(statement_basis_used == "fallback_primary_type"),
            piot_coverage=piot.get("piotroski_coverage_ratio") if isinstance(piot, dict) else None,
            statement_staleness_days=float(statement_staleness_days) if statement_staleness_days is not None else None,
            financial_data_mode=fin_diag.get("financial_data_mode"),
            fins_details_available=fin_diag.get("fins_details_available"),
            sector_normalized=sector,
            sector33_raw=sector33_raw,
            piot_adjusted=piot.get("piotroski_adjusted_score") if isinstance(piot, dict) else None,
            piot_raw=float(_pr_meta) if _pr_meta is not None and np.isfinite(float(_pr_meta)) else None,
        )
        buy_timing_gate = bool(
            (str(ma200_eval.get("ma200_state")) == "ma200_reclaim")
            or (
                str(ma200_eval.get("ma200_state")) == "below_ma200_basing"
                and fundamental_edge_score >= MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY
            )
            or (str(ma200_eval.get("ma200_state")) == "above_ma200_near")
        )
        downtrend_rejection_gate = bool(ma200_eval.get("below_ma200_downtrend"))
        _entry_raw = compute_entry_score(fundamental_edge_score, ma200_eval)
        entry_candidate_lane = assign_entry_candidate_lane(
            fundamental_edge=fundamental_edge_score,
            ma_eval=ma200_eval,
            peg_quality=peg_quality,
            eps_growth_rate=val.get("eps_growth_rate"),
            op_income_stable=_op_stable_tri if _op_stable_tri is not None else None,
            op_income_yoy=op_income_yoy_pct,
            piot=piot if isinstance(piot, dict) else {},
            ret_21d=momentum.get("return_21d"),
            dr_meta=dr_meta,
            valuation_satellite_candidate=valuation_satellite_candidate,
            ps_only_satellite_candidate=ps_only_satellite_candidate,
            buy_timing_gate=buy_timing_gate,
            downtrend_rejection_gate=downtrend_rejection_gate,
        )
        entry_score = cap_entry_score(_entry_raw, entry_candidate_lane, dr_meta)

        filter_details = {
            "liquidity_ok": liquidity_ok,
            "market_cap_ok": market_cap_ok,
            "ps_available": ps_available,
            "per_available": per_available,
            "valuation_available": valuation_available,
            "defensive_ps_ok": defensive_ps_ok,
            "per_satellite": per_satellite,
            "per_core_ok": per_core_ok,
            "op_income_stable": op_income_stable,
            "op_income_reason": op_income_eval.get("reason"),
            "base_ok": base_ok,
            "core_candidate": core_candidate,
            "satellite_candidate": satellite_candidate,
            "valuation_satellite_candidate": valuation_satellite_candidate,
            "ps_only_satellite_candidate": ps_only_satellite_candidate,
            "ps_satellite_limit": ps_satellite_limit,
            "valuation_lane": valuation_lane,
            "candidate_lane": valuation_lane,
            "entry_candidate_lane": entry_candidate_lane,
            "fundamental_edge_score": fundamental_edge_score,
            "entry_score": entry_score,
            "buy_timing_gate": buy_timing_gate,
            "downtrend_rejection_gate": downtrend_rejection_gate,
            "data_review_reason": dr_meta.get("data_review_reason"),
            "data_review_level": dr_meta.get("data_review_level"),
            "excluded_candidate": excluded_candidate,
            "thresholds": {
                "MIN_AVG_VOLUME_30D": MIN_AVG_VOLUME_30D,
                "MIN_ADV_JPY_20D": MIN_ADV_JPY_20D,
                "MIN_MARKET_CAP_JPY": MIN_MARKET_CAP_JPY,
                "MAX_PS_DEFENSIVE": MAX_PS_DEFENSIVE,
                "MAX_PER_CORE": MAX_PER_CORE,
                "OP_INCOME_YEARS": OP_INCOME_YEARS,
                "OP_INCOME_DROP_FLOOR": OP_INCOME_DROP_FLOOR,
                "EXCLUDE_OP_INCOME_DEFICIT": EXCLUDE_OP_INCOME_DEFICIT,
            }
        }

        return {
            "stock_code": code, "company_name": name, "sector_name": sector,
            "current_price": current_price, "mas": mas, "rsi": rsi, "adx": adx,
            "plus_di": plus_di, "minus_di": minus_di,
            "volatility": cur_vol, "avg_volatility": avg_vol,
            "below_ma25": below_ma25, "below_ma75": below_ma75, "below_ma200": below_ma200,
            "return_21d": momentum.get("return_21d"),
            "return_63d": momentum.get("return_63d"),
            "return_126d": momentum.get("return_126d"),
            "return_252d": momentum.get("return_252d"),
            "momentum_6m_1m": momentum.get("momentum_6m_1m"),
            "momentum_6m_3m": momentum.get("momentum_6m_3m"),
            "momentum_3m_1m": momentum.get("momentum_3m_1m"),
            "piotroski": piot,
            "ps_ratio": ps_ratio,
            "peg_ratio": val.get("peg_ratio"),
            "reference_peg": val.get("reference_peg"),
            "peg_trusted": val.get("peg_trusted"),
            "per": per,
            "revenue_per_share": val.get("revenue_per_share"),
            "safety": safety, "speculation": spec, "success": True,
            "avg_volume_30d": avg_volume,
            "adv_jpy_20d": adv_jpy_20d,
            "adv_jpy_60d": adv_jpy_60d,
            "traded_days_60d": traded_days_60d,
            "financial_history": financial_history,
            "market_cap": market_cap,
            "shares_outstanding": shares_outstanding,
            "max_drawdown": max_dd,
            "sales_cagr": sales_cagr,
            "safety_criteria": safety_criteria,
            "sector_benchmark": sector_benchmark,
            "diagnostics": diagnostics,
            "filters": filter_details,
            "candidate_lane": entry_candidate_lane,
            "valuation_lane": valuation_lane,
            "fundamental_edge_score": fundamental_edge_score,
            "entry_score": entry_score,
            "ma200_evaluation": ma200_eval,
            "peg_warning": peg_quality.get("peg_warning"),
            "peg_quality": peg_quality,
            "buy_timing_gate": buy_timing_gate,
            "downtrend_rejection_gate": downtrend_rejection_gate,
            "op_income_yoy_pct": op_income_yoy_pct,
            "ps_vs_sector_pre": ps_vs_sector_pre,
            "ps_satellite_limit": ps_satellite_limit,
            "financial_data_mode": fin_diag.get("financial_data_mode"),
            "fins_details_available": fin_diag.get("fins_details_available"),
            "fins_details_status": fin_diag.get("fins_details_status"),
            "fins_details_error": fin_diag.get("fins_details_error"),
            "instrument_type": inst_t,
            "data_review_reason": dr_meta.get("data_review_reason"),
            "data_review_level": dr_meta.get("data_review_level"),
        }
    except Exception as e:
        return {
            "stock_code": code,
            "company_name": name,
            "sector_name": sector_hint or "その他",
            "error": f"{e}",
            "success": False,
            "instrument_type": instrument_type or "stock",
        }

# ------------------------------------------------------------
# 収集（単体/全体）
# ------------------------------------------------------------
def collect_one_code_result(
    session: requests.Session,
    code: str,
    master_row: Optional[dict] = None,
    *,
    force_refresh: bool = False,
    instrument_type: str = "stock",
) -> Dict[str, Any]:
    fc = FrozenCache()
    code = str(code).strip()
    out: Dict[str, Any] = {
        "ok": False,
        "status": "transient_error",
        "code": code,
        "reason": "",
        "price_rows": 0,
        "summary_rows": 0,
        "detail_rows": 0,
        "saved_prices": False,
        "saved_statements": False,
        "financial_data_mode": "",
        "api_status_summary": "",
        "api_status_details": "",
    }
    try:
        if master_row:
            hr = _collect_hard_exclusion_reason(master_row)
            if hr:
                return {
                    **out,
                    "status": "non_stock_or_fund_like",
                    "reason": hr,
                }

        if instrument_type != "stock":
            df_p, pmeta = fetch_prices_v2_with_meta(
                session, code, cache_name=f"prices_{code}", bypass_cache=force_refresh,
            )
            out["price_rows"] = int(pmeta.get("rows") or 0)
            if df_p is not None and not df_p.empty:
                fc.save_prices(code, df_p)
                out["saved_prices"] = True
            if fc.has_prices(code):
                return {**out, "ok": True, "status": "success", "reason": ""}
            ph = int(pmeta.get("http", 200) or 200)
            if ph in (401, 403):
                return {**out, "status": "auth_or_permission_error", "reason": f"price_http_{ph}"}
            return {**out, "status": "transient_error", "reason": f"price_http_{ph}"}

        df_p, pmeta = fetch_prices_v2_with_meta(
            session, code, cache_name=f"prices_{code}", bypass_cache=force_refresh,
        )
        out["price_rows"] = int(pmeta.get("rows") or 0)
        if df_p is not None and not df_p.empty:
            fc.save_prices(code, df_p)
            out["saved_prices"] = True
        else:
            ph = int(pmeta.get("http", 200) or 200)
            if ph in (401, 403):
                return {**out, "status": "auth_or_permission_error", "reason": f"price_http_{ph}"}
            if bool(pmeta.get("transient")):
                return {**out, "status": "transient_error", "reason": f"price_transient_http_{ph}"}

        fdm = FinancialDataManager(session)
        stmts = fdm.fetch_statements(code, force_refresh=force_refresh)
        meta = fdm.get_last_financial_fetch_meta()
        sum_st = int(meta.get("api_status_summary", 200) or 200)
        n_sum = int(meta.get("summary_rows", 0) or 0)
        n_det = int(meta.get("detail_rows", 0) or 0)
        out["summary_rows"] = n_sum
        out["detail_rows"] = n_det
        out["financial_data_mode"] = str(meta.get("financial_data_mode") or "")
        out["api_status_summary"] = sum_st
        ds = meta.get("fins_details_status")
        out["api_status_details"] = str(ds) if ds is not None else ""

        if sum_st in (401, 403):
            return {**out, "status": "auth_or_permission_error", "reason": "fins_summary_auth"}
        if sum_st != 200:
            return {**out, "status": "transient_error", "reason": f"fins_summary_http_{sum_st}"}
        if n_sum == 0:
            return {**out, "status": "permanent_missing_financials", "reason": "fins_summary_empty_http200"}
        if not stmts:
            return {**out, "status": "permanent_missing_financials", "reason": "statements_empty_after_convert"}

        fc.save_statements(code, stmts, financial_meta=meta)
        out["saved_statements"] = True

        if fc.has_all(code):
            return {**out, "ok": True, "status": "success", "reason": ""}
        if not fc.has_prices(code):
            return {**out, "status": "transient_error", "reason": "price_missing_after_fin_save"}
        return {**out, "status": "transient_error", "reason": "cache_incomplete_after_save"}

    except RuntimeError as e:
        if "日次レート制限到達" in str(e):
            raise
        return {**out, "status": "transient_error", "reason": str(e)}
    except requests.RequestException as e:
        return {**out, "status": "transient_error", "reason": f"request:{e!s}"}
    except Exception as e:
        logger.warning("collect_one_code_result 例外 code=%s: %s", code, e)
        return {**out, "status": "transient_error", "reason": str(e)}


def collect_one_code(
    session: requests.Session,
    code: str,
    name: str = "",
    *,
    force_refresh: bool = False,
    instrument_type: str = "stock",
) -> bool:
    return bool(
        collect_one_code_result(session, code, None, force_refresh=force_refresh, instrument_type=instrument_type).get(
            "ok"
        )
    )

PENDING_FILE = CACHE_DIR / "pending_codes.json"

def _save_pending(codes: list[str]) -> None:
    PENDING_FILE.write_text(json.dumps({"codes": codes}, ensure_ascii=False), encoding="utf-8")

def _load_pending(
    df: pd.DataFrame,
    *,
    force_full: bool = False,
    refresh_days: Optional[int] = None,
    ignore_skiplist: bool = False,
) -> list[str]:
    fc = FrozenCache()
    allowed = {str(c).strip() for c in df["Code"].astype(str)}
    skip_set: set[str] = set()
    if not ignore_skiplist:
        for k in _load_skiplist_raw().get("skipped", {}).keys():
            cc = _canonical_internal_stock_code(str(k).strip(), allowed)
            if cc:
                skip_set.add(cc)

    def _intersect_allow(codes_in: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        dropped = 0
        for c in codes_in:
            cc = _canonical_internal_stock_code(str(c).strip(), allowed)
            if cc is None or cc not in allowed:
                dropped += 1
                continue
            if cc in seen:
                dropped += 1
                continue
            seen.add(cc)
            out.append(cc)
        if dropped:
            logger.info(
                "[INFO] pending_codes dropped %s stale/duplicate entries (not in collectable master or non-canonical code)",
                dropped,
            )
        return out

    def _skip_skipped(codes_in: List[str]) -> List[str]:
        return [c for c in codes_in if c not in skip_set]

    if force_full:
        codes = _skip_skipped(_intersect_allow([str(c).strip() for c in df["Code"].astype(str)]))
        _save_pending(codes)
        return codes

    if refresh_days is not None:
        codes_raw = [
            str(c).strip()
            for c in df["Code"].astype(str)
            if str(c).strip() in allowed and not fc.has_all(str(c).strip(), max_age_days=refresh_days)
        ]
        codes = _skip_skipped(_intersect_allow(codes_raw))
        _save_pending(codes)
        return codes

    if PENDING_FILE.exists():
        try:
            raw_pending = json.loads(PENDING_FILE.read_text(encoding="utf-8")).get("codes", [])
            filt = _skip_skipped(_intersect_allow([str(c) for c in raw_pending]))
            _save_pending(filt)
            return filt
        except Exception:
            pass

    codes = _skip_skipped(
        _intersect_allow([str(c).strip() for c in df["Code"].astype(str) if not fc.has_all(str(c).strip())])
    )
    _save_pending(codes)
    return codes

def collect_all_daemon(session: requests.Session,
                       daily_budget: Optional[int] = None,
                       refresh_days: Optional[int] = None,
                       force_full: bool = False,
                       reset_pending: bool = False) -> None:
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = filter_collectable_equities_df(df)
    valid_set = _valid_collectable_stock_codes(df)

    if reset_pending and force_full and COLLECTION_SKIPLIST_PATH.exists():
        try:
            COLLECTION_SKIPLIST_PATH.unlink()
        except OSError:
            pass

    if reset_pending and PENDING_FILE.exists():
        try:
            PENDING_FILE.unlink()
        except Exception:
            pass

    if not (reset_pending and force_full):
        prune_stale_collection_sidecars(valid_set)

    ignore_skiplist = bool(force_full)
    pending = _load_pending(df, force_full=force_full, refresh_days=refresh_days, ignore_skiplist=ignore_skiplist)
    if not pending:
        _cli_print("📦 すでに全件取得済み", "[収集] すでに全件取得済み")
        write_sector_normalization_audit_csv(session)
        return

    code_to_row = {str(r.get("Code", "")).strip(): r.to_dict() for _, r in df.iterrows()}

    if daily_budget is None:
        rpd_env = os.getenv("JQ_RPD")
        if rpd_env:
            rpd = int(rpd_env)
            daily_budget = max(1, min(len(pending), max(1, rpd // 2 - 5)))
        else:
            daily_budget = min(len(pending), DEFAULT_COLLECT_BUDGET)

    mode = "強制再収集" if force_full else (f"{refresh_days}日超のみ再収集" if refresh_days is not None else "未取得のみ")
    _cli_print(
        f"▶ 全自動収集開始  残り{len(pending)}銘柄  本バッチ上限={daily_budget}銘柄（V2はrpm制御・budgetは取得件数上限）  モード={mode}",
        f"[収集開始] 残り{len(pending)}銘柄  batch_limit={daily_budget}  mode={mode}",
    )

    while pending:
        if graceful_shutdown.shutdown:
            logger.info("shutdown requested; stopping collect_all_daemon")
            break
        pending_before = len(pending)
        taken = 0
        start = time.time()
        stats = {
            "success": 0,
            "permanent_missing_financials": 0,
            "non_stock_or_fund_like": 0,
            "transient_error": 0,
            "auth_or_permission_error": 0,
        }
        reset_v2_legacy_batch_audit()
        batch_attempted_codes: List[str] = []
        batch_attempt_statuses: List[str] = []
        try:
            for code in list(pending):
                if graceful_shutdown.shutdown:
                    logger.info("shutdown requested; stopping current collect batch")
                    break
                if taken >= daily_budget:
                    break
                c_norm = _canonical_internal_stock_code(str(code).strip(), valid_set)
                if c_norm is None:
                    logger.warning(
                        "pending から無効コードを除去: %r（正規化不能または collectable master に不在）",
                        code,
                    )
                    pc = str(code).strip()
                    for alt in list(pending):
                        if str(alt).strip() == pc:
                            pending.remove(alt)
                    _save_pending(pending)
                    continue
                batch_attempted_codes.append(c_norm)
                mr = code_to_row.get(c_norm)
                res = collect_one_code_result(
                    session,
                    c_norm,
                    mr,
                    force_refresh=(force_full or refresh_days is not None),
                    instrument_type="stock",
                )
                stt = str(res.get("status") or "")
                if stt == "success":
                    if c_norm in pending:
                        pending.remove(c_norm)
                    stats["success"] += 1
                elif stt == "permanent_missing_financials":
                    if c_norm in pending:
                        pending.remove(c_norm)
                    _record_permanent_missing_financials(res, mr)
                    stats["permanent_missing_financials"] += 1
                elif stt == "non_stock_or_fund_like":
                    if c_norm in pending:
                        pending.remove(c_norm)
                    _record_non_stock_excluded(res, mr)
                    stats["non_stock_or_fund_like"] += 1
                elif stt == "auth_or_permission_error":
                    stats["auth_or_permission_error"] += 1
                else:
                    stats["transient_error"] += 1
                batch_attempt_statuses.append(stt)
                _save_pending(pending)
                taken += 1
                if taken % 20 == 0 or taken == daily_budget:
                    elapsed = time.time() - start
                    _cli_print(
                        "  ⏱ 本日 {}/{} 件 | pending_before={} | success={} | permanent_skip={} | "
                        "non_stock={} | transient={} | auth={} | pending_after={} | 経過{}s".format(
                            taken,
                            daily_budget,
                            pending_before,
                            stats["success"],
                            stats["permanent_missing_financials"],
                            stats["non_stock_or_fund_like"],
                            stats["transient_error"],
                            stats["auth_or_permission_error"],
                            len(pending),
                            int(elapsed),
                        ),
                        "  [進捗] {}/{} pending {} ok {} skip {}".format(
                            taken, daily_budget, len(pending), stats["success"], stats["permanent_missing_financials"],
                        ),
                    )
                    sys.stdout.flush()
        except RuntimeError as e:
            if "日次レート制限到達" in str(e):
                pass
            else:
                raise

        flush_v2_legacy_batch_audit_loggers()
        pending_after = len(pending)
        summary_lines = (
            f"収集バッチ終了: tried={taken}, success={stats['success']}, "
            f"permanent_missing_financials={stats['permanent_missing_financials']}, "
            f"non_stock_or_fund_like={stats['non_stock_or_fund_like']}, transient_error={stats['transient_error']}, "
            f"auth_or_permission_error={stats['auth_or_permission_error']}, "
            f"pending_before={pending_before}, pending_after={pending_after}"
        )
        _cli_print("📦 " + summary_lines, "[収集] " + summary_lines)
        batch_bad = (
            taken >= 20
            and stats["success"] == 0
            and (stats["permanent_missing_financials"] / max(taken, 1)) > 0.8
        )
        if batch_bad:
            logger.warning(
                "[WARN] no successful collection in this batch; likely stale pending or invalid stock universe. "
                "transient_error / auth_or_permission_error の銘柄は pending に残し、skiplist には追加しません。"
            )
            for bc, stt in zip(batch_attempted_codes, batch_attempt_statuses):
                if stt in ("transient_error", "auth_or_permission_error"):
                    continue
                if stt in ("permanent_missing_financials", "non_stock_or_fund_like", "success"):
                    continue
                if bc not in code_to_row:
                    if bc in pending:
                        pending.remove(bc)
                    sk = _load_skiplist_raw().get("skipped") or {}
                    if bc not in sk:
                        _skiplist_add_entry(
                            bc,
                            "not_in_collectable_master",
                            detail="batch_anomaly_cleanup",
                        )
            _save_pending(pending)
            break
        if not pending:
            _cli_print("✅ 全銘柄の凍結収集が完了", "[OK] 全銘柄の凍結収集が完了")
            break

        if os.getenv("JQ_RPD"):
            wait_sec = seconds_until_next_day()
            h, rem = divmod(wait_sec, 3600)
            m, s = divmod(rem, 60)
            _cli_print(f"⏳ JQ_RPD 設定のため日が変わるまで待機: {h}h{m}m{s}s", f"[待機] JQ_RPD: {h}h{m}m{s}s")
            _sleep_interruptible(wait_sec)
            if graceful_shutdown.shutdown:
                break
        else:
            if pending_after >= pending_before and taken > 0:
                _cli_print(
                    "⛔ 進捗なしのため停止。missing_financials_symbols.csv と collection_skiplist.json を確認してください。",
                    "[停止] 進捗なし: skiplist / missing_financials を確認",
                )
                break
            _cli_print(
                "📌 JQ_RPD 未設定のため日をまたがないで次バッチへ（同一日内の継続）。",
                "[情報] JQ_RPD 未設定: 継続収集",
            )

    write_sector_normalization_audit_csv(session)

def collect_batch(
    session: requests.Session,
    max_codes: int,
    *,
    force_refresh: bool = False,
) -> dict:
    reset_v2_legacy_batch_audit()
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = filter_collectable_equities_df(df)
    valid_cb = _valid_collectable_stock_codes(df)
    prune_stale_collection_sidecars(valid_cb)
    fc = FrozenCache()
    code_to_row = {str(r.get("Code", "")).strip(): r.to_dict() for _, r in df.iterrows()}

    skip_set: set[str] = set()
    for k in _load_skiplist_raw().get("skipped", {}).keys():
        cc = _canonical_internal_stock_code(str(k).strip(), valid_cb)
        if cc:
            skip_set.add(cc)
    if force_refresh:
        pending = [
            str(c).strip()
            for c in df["Code"].astype(str)
            if str(c).strip() not in skip_set
        ][:max_codes]
    else:
        pending = [
            str(c).strip()
            for c in df["Code"].astype(str)
            if not fc.has_all(str(c).strip()) and str(c).strip() not in skip_set
        ]
        pending = pending[:max_codes]
    picked = pending
    ok = 0
    fail = 0
    stats = {
        "success": 0,
        "permanent_missing_financials": 0,
        "non_stock_or_fund_like": 0,
        "transient_error": 0,
        "auth_or_permission_error": 0,
    }
    start = time.time()

    for i, code in enumerate(picked, 1):
        if graceful_shutdown.shutdown:
            logger.info("shutdown requested; stopping collect_batch")
            break
        cc = _canonical_internal_stock_code(str(code).strip(), valid_cb)
        if cc is None:
            fail += 1
            continue
        mr = code_to_row.get(cc)
        res = collect_one_code_result(
            session,
            cc,
            mr,
            force_refresh=force_refresh,
            instrument_type="stock",
        )
        stt = str(res.get("status") or "")
        if stt == "success":
            ok += 1
            stats["success"] += 1
        elif stt == "permanent_missing_financials":
            _record_permanent_missing_financials(res, mr)
            fail += 1
            stats["permanent_missing_financials"] += 1
        elif stt == "non_stock_or_fund_like":
            _record_non_stock_excluded(res, mr)
            fail += 1
            stats["non_stock_or_fund_like"] += 1
        elif stt == "auth_or_permission_error":
            fail += 1
            stats["auth_or_permission_error"] += 1
        else:
            fail += 1
            stats["transient_error"] += 1
        if i % 20 == 0 or i == len(picked):
            elapsed = time.time() - start
            _cli_print(
                f"  ⏱ {i}/{len(picked)} 収集中 (OK={ok} FAIL={fail}) 経過{elapsed:.0f}s",
                f"  [{i}/{len(picked)}] 収集中 (OK={ok} FAIL={fail}) 経過{elapsed:.0f}s",
            )
            sys.stdout.flush()

    flush_v2_legacy_batch_audit_loggers()
    return {
        "tried": len(picked),
        "ok": ok,
        "fail": fail,
        **stats,
    }

# ------------------------------------------------------------
# オフライン分析タスク生成 / 銘柄名取得
# ------------------------------------------------------------
def build_offline_analysis_tasks(session: requests.Session) -> list[tuple[str, str, str, str | None]]:
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
    df_list = filter_collectable_equities_df(df_list)
    fc = FrozenCache()

    df_list = df_list.copy()
    df_list["Code"] = df_list["Code"].astype(str)
    mask = df_list["Code"].apply(lambda c: fc.has_all(c))
    rows = df_list[mask][["Code", "CompanyName", "MarketCode", "Sector33Name"]]

    tasks: list[tuple[str, str, str, str | None]] = []
    for row in rows.itertuples(index=False):
        code   = str(row.Code)
        name   = str(getattr(row, "CompanyName", "") or "")
        market = str(getattr(row, "MarketCode", "") or "")
        sector = str(getattr(row, "Sector33Name", "") or "") or None
        tasks.append((code, name, market, sector))
    return tasks

def lookup_company_name(session: requests.Session, code: str) -> str:
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
    df_list = df_list.copy()
    df_list["Code"] = df_list["Code"].astype(str)
    hit = df_list[df_list["Code"] == str(code)]
    if not hit.empty:
        return str(hit.iloc[0].get("CompanyName") or "")
    return ""


def lookup_equity_name_and_instrument(session: requests.Session, code: str) -> Tuple[str, str]:
    """マスタから銘柄名と instrument_type を返す（single 用。名称空でも拒否しない）。"""
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
    df_list = df_list.copy()
    df_list["Code"] = df_list["Code"].astype(str).str.strip()
    hit = df_list[df_list["Code"] == str(code).strip()]
    if hit.empty:
        return "", classify_instrument_from_master(
            {"Code": code, "CompanyName": "", "CoNameEn": "", "Mkt": "", "MktNm": ""}
        )
    row = hit.iloc[0]
    name = str(row.get("CompanyName") or "")
    it = row.get("instrument_type")
    if it is None or (isinstance(it, float) and pd.isna(it)) or str(it).strip() == "":
        it = classify_instrument_from_master(row.to_dict())
    return name, str(it)

# ------------------------------------------------------------
# レポート出力（flatten / csv / md）
# ------------------------------------------------------------
def _safe_bool(x) -> bool:
    return bool(x) if x is not None else False

def _extract_filters(d: dict) -> dict:
    f = d.get("filters") or {}
    return {
        "liquidity_ok": _safe_bool(f.get("liquidity_ok")),
        "market_cap_ok": _safe_bool(f.get("market_cap_ok")),
        "defensive_ps_ok": _safe_bool(f.get("defensive_ps_ok")),
        "ps_available": _safe_bool(f.get("ps_available")),
        "per_available": _safe_bool(f.get("per_available")),
        "valuation_available": _safe_bool(f.get("valuation_available")),
        "per_satellite": _safe_bool(f.get("per_satellite")),
        "op_income_stable": _safe_bool(f.get("op_income_stable")),
        "op_income_reason": f.get("op_income_reason"),
        "base_ok": _safe_bool(f.get("base_ok")),
        "core_candidate": _safe_bool(f.get("core_candidate")),
        "satellite_candidate": _safe_bool(f.get("satellite_candidate")),
        "valuation_satellite_candidate": _safe_bool(f.get("valuation_satellite_candidate")),
        "ps_only_satellite_candidate": _safe_bool(f.get("ps_only_satellite_candidate")),
        "ps_satellite_limit": f.get("ps_satellite_limit"),
        "valuation_lane": f.get("valuation_lane") or f.get("candidate_lane"),
        "entry_candidate_lane": f.get("entry_candidate_lane"),
        "fundamental_edge_score": f.get("fundamental_edge_score"),
        "entry_score": f.get("entry_score"),
        "buy_timing_gate": _safe_bool(f.get("buy_timing_gate")),
        "downtrend_rejection_gate": _safe_bool(f.get("downtrend_rejection_gate")),
        "data_review_reason": f.get("data_review_reason"),
        "data_review_level": f.get("data_review_level"),
        "candidate_lane": d.get("candidate_lane")
        or f.get("entry_candidate_lane")
        or f.get("candidate_lane"),
        "excluded_candidate": _safe_bool(f.get("excluded_candidate")),
    }

def _flatten_result(d: dict) -> dict:
    pio = d.get("piotroski") or {}
    saf = d.get("safety") or {}
    spc = d.get("speculation") or {}
    safety_criteria = d.get("safety_criteria") or {}
    criteria = safety_criteria.get("criteria", {}) if isinstance(safety_criteria, dict) else {}
    diagnostics = d.get("diagnostics") or {}
    imputation = diagnostics.get("imputation", {}) if isinstance(diagnostics, dict) else {}
    sector_benchmark = d.get("sector_benchmark") or {}
    flt = _extract_filters(d)

    return {
        "code": d.get("stock_code"),
        "name": d.get("company_name"),
        "instrument_type": d.get("instrument_type")
        or (diagnostics.get("instrument_type") if isinstance(diagnostics, dict) else None),
        "sector": d.get("sector_name"),
        "price": d.get("current_price"),
        "ps": d.get("ps_ratio"),
        "reference_peg": d.get("reference_peg") if d.get("reference_peg") is not None else d.get("peg_ratio"),
        "peg_trusted": d.get("peg_trusted"),
        "per": d.get("per"),
        "rsi": d.get("rsi"),
        "adx": d.get("adx"),
        "below_ma200": d.get("below_ma200"),
        "return_21d": d.get("return_21d"),
        "return_63d": d.get("return_63d"),
        "return_126d": d.get("return_126d"),
        "return_252d": d.get("return_252d"),
        "momentum_6m_1m": d.get("momentum_6m_1m"),
        "momentum_6m_3m": d.get("momentum_6m_3m"),
        "momentum_3m_1m": d.get("momentum_3m_1m"),
        "piot": pio.get("score"),
        "piot_eval": pio.get("evaluation"),
        "piot_observed_items": pio.get("observed_items"),
        "piot_coverage_ratio": pio.get("coverage_ratio"),
        "piotroski_raw_score": pio.get("piotroski_raw_score"),
        "piotroski_available_items": pio.get("piotroski_available_items"),
        "piotroski_coverage_ratio": pio.get("piotroski_coverage_ratio"),
        "piotroski_adjusted_score": pio.get("piotroski_adjusted_score"),
        "piotroski_effective_score": pio.get("piotroski_effective_score"),
        "safety": saf.get("total_score"),
        "safety_level": saf.get("safety_level"),
        "safety_coverage_ratio": saf.get("coverage_ratio"),
        "safety_observed_items": saf.get("observed_items"),
        "spec_score": spc.get("score"),
        "spec_level": spc.get("level"),
        "safety_criteria_score": safety_criteria.get("total_score") if isinstance(safety_criteria, dict) else None,
        "ps_under_1": criteria.get("ps_under_1", False),
        "cash_rich": criteria.get("cash_rich", False),
        "positive_ocf": criteria.get("positive_ocf", False),
        "equity_ratio_50plus": criteria.get("equity_ratio_50plus", False),
        "equity_ratio_70plus": criteria.get("equity_ratio_70plus", False),
        "growth_potential": criteria.get("growth_potential", False),
        "no_speculative_drop": criteria.get("no_speculative_drop", False),
        "equity_ratio": safety_criteria.get("equity_ratio") if isinstance(safety_criteria, dict) else None,
        "max_drawdown": safety_criteria.get("max_drawdown") if isinstance(safety_criteria, dict) else None,
        "sales_cagr": safety_criteria.get("sales_cagr") if isinstance(safety_criteria, dict) else None,
        "market_cap": d.get("market_cap"),
        "avg_volume_30d": d.get("avg_volume_30d"),
        "adv_jpy_20d": d.get("adv_jpy_20d"),
        "adv_jpy_60d": d.get("adv_jpy_60d"),
        "traded_days_60d": d.get("traded_days_60d"),
        "sector_ps_benchmark": sector_benchmark.get("ps"),
        "sector_benchmark_source": sector_benchmark.get("data_source"),
        "statement_type": diagnostics.get("statement_type"),
        "statement_basis_used": diagnostics.get("statement_basis_used"),
        "fallback_basis_flag": diagnostics.get("fallback_basis_flag"),
        "annual_financial_compare_ok": diagnostics.get("annual_financial_compare_ok"),
        "piotroski_basis_note": diagnostics.get("piotroski_basis_note"),
        "valuation_input_complete": diagnostics.get("valuation_input_complete"),
        "eps_growth_input_complete": diagnostics.get("eps_growth_input_complete"),
        "piotroski_input_complete": diagnostics.get("piotroski_input_complete"),
        "growth_proxy_detached_from_scoring": diagnostics.get("growth_proxy_detached_from_scoring"),
        "statement_disclosed_date": diagnostics.get("statement_disclosed_date"),
        "statement_staleness_days": diagnostics.get("statement_staleness_days"),
        "price_asof_date": diagnostics.get("latest_price_date"),
        "has_imputation": imputation.get("has_imputation"),
        "imputed_field_count": imputation.get("field_count"),
        "imputed_fields": ",".join(sorted((imputation.get("fields") or {}).keys())),
        "critical_missing_count": diagnostics.get("critical_missing_count"),
        "financial_data_mode": diagnostics.get("financial_data_mode"),
        "fins_details_available": diagnostics.get("fins_details_available"),
        "fins_details_status": diagnostics.get("fins_details_status"),
        "fins_details_error": diagnostics.get("fins_details_error"),
        "fundamental_edge_score": d.get("fundamental_edge_score"),
        "entry_score": d.get("entry_score"),
        "data_review_reason": flt.get("data_review_reason") or d.get("data_review_reason"),
        "data_review_level": flt.get("data_review_level") or d.get("data_review_level"),
        "ps_vs_sector_pre": d.get("ps_vs_sector_pre"),
        "peg_warning": d.get("peg_warning"),
        "op_income_yoy_pct": d.get("op_income_yoy_pct"),
        "valuation_lane": d.get("valuation_lane") or flt.get("valuation_lane") or flt.get("candidate_lane"),
        "ma200_state": (d.get("ma200_evaluation") or {}).get("ma200_state"),
        "ma200_timing_score": (d.get("ma200_evaluation") or {}).get("ma200_timing_score"),
        "ma200_risk_penalty": (d.get("ma200_evaluation") or {}).get("ma200_risk_penalty"),
        "ma200_reason": (d.get("ma200_evaluation") or {}).get("ma200_reason"),
        "distance_from_ma200": (d.get("ma200_evaluation") or {}).get("distance_from_ma200"),
        "crossed_above_ma200_recently": (d.get("ma200_evaluation") or {}).get("crossed_above_ma200_recently"),
        "below_ma200_basing": (d.get("ma200_evaluation") or {}).get("below_ma200_basing"),
        "below_ma200_downtrend": (d.get("ma200_evaluation") or {}).get("below_ma200_downtrend"),
        "above_ma200_extended": (d.get("ma200_evaluation") or {}).get("above_ma200_extended"),
        **flt,
        "ok": d.get("success"),
        "error": d.get("error"),
    }

def _report_file_base(path: Path) -> str:
    """レポートファイルの正規化ベース名。top_recommended_core と top_recommended_* を同一グループに"""
    stem = path.stem
    m = re.match(r"^(.+)_\d{8}_\d{6}$", stem)
    base = m.group(1) if m else stem
    if base.endswith("_core"):
        return base[:-5]
    return base

def _cleanup_old_files_by_ext(outdir: Path, ext: str) -> int:
    """指定拡張子のファイルを種類ごとに最新1件だけ残し、古いものを削除。削除件数を返す"""
    files = list(outdir.glob(f"*{ext}"))
    if not files:
        return 0
    by_base: Dict[str, list] = {}
    for f in files:
        base = _report_file_base(f)
        by_base.setdefault(base, []).append(f)
    deleted = 0
    for base, flist in by_base.items():
        if len(flist) <= 1:
            continue
        latest = max(flist, key=lambda p: p.stat().st_mtime)
        for f in flist:
            if f != latest:
                try:
                    f.unlink()
                    deleted += 1
                except OSError as e:
                    logger.warning("古いファイル削除失敗: %s (%s)", f, e)
    return deleted

def cleanup_old_report_files(outdir: Path) -> None:
    """各レポート種類ごとに最新1件だけ残し、古いCSV/MD/JSONを削除してストレージを節約する"""
    outdir = Path(outdir)
    if not outdir.exists():
        return
    total = 0
    total += _cleanup_old_files_by_ext(outdir, ".csv")
    total += _cleanup_old_files_by_ext(outdir, ".md")
    total += _cleanup_old_files_by_ext(outdir, ".json")
    if total > 0:
        _cli_print(f"🗑️ 古いレポート {total}件を削除しました ({outdir})", f"[削除] 古いレポート {total}件を削除しました ({outdir})")

def write_candidate_sets(flat: pd.DataFrame, outdir: Path, timestamp: Optional[str] = None) -> list[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    ok = flat[flat["ok"] == True].copy()
    if ok.empty:
        return []

    core = ok[ok["core_candidate"] == True].copy()
    sat = ok[ok["satellite_candidate"] == True].copy()
    if "valuation_satellite_candidate" in ok.columns:
        sat_val = ok[ok["valuation_satellite_candidate"] == True].copy()
    else:
        sat_val = pd.DataFrame()
    if "ps_only_satellite_candidate" in ok.columns:
        sat_ps = ok[ok["ps_only_satellite_candidate"] == True].copy()
    else:
        sat_ps = pd.DataFrame()
    if "excluded_candidate" in ok.columns:
        exc = ok[ok["excluded_candidate"] == True].copy()
    else:
        exc = ok[(ok["base_ok"] != True) | (ok["op_income_stable"] != True) | (ok["liquidity_ok"] != True) | (ok["market_cap_ok"] != True)].copy()

    # 固定名で上書き（ストレージ節約）
    p_core = outdir / "core_candidates.csv"
    p_sat  = outdir / "satellite_candidates.csv"
    p_sat_val = outdir / "satellite_valuation_candidates.csv"
    p_sat_ps = outdir / "satellite_ps_only_candidates.csv"
    p_exc  = outdir / "excluded.csv"
    core.to_csv(p_core, index=False, encoding="utf-8-sig")
    sat.to_csv(p_sat, index=False, encoding="utf-8-sig")
    sat_val.to_csv(p_sat_val, index=False, encoding="utf-8-sig")
    sat_ps.to_csv(p_sat_ps, index=False, encoding="utf-8-sig")
    exc.to_csv(p_exc, index=False, encoding="utf-8-sig")

    summary = {
        "total_ok": int(len(ok)),
        "core": int(len(core)),
        "satellite": int(len(sat)),
        "satellite_valuation": int(len(sat_val)),
        "satellite_ps_only": int(len(sat_ps)),
        "excluded": int(len(exc)),
        "thresholds": {
            "MIN_AVG_VOLUME_30D": MIN_AVG_VOLUME_30D,
            "MIN_ADV_JPY_20D": MIN_ADV_JPY_20D,
            "MIN_MARKET_CAP_JPY": MIN_MARKET_CAP_JPY,
            "MAX_PS_DEFENSIVE": MAX_PS_DEFENSIVE,
            "MAX_PER_CORE": MAX_PER_CORE,
            "OP_INCOME_YEARS": OP_INCOME_YEARS,
            "OP_INCOME_DROP_FLOOR": OP_INCOME_DROP_FLOOR,
            "EXCLUDE_OP_INCOME_DEFICIT": EXCLUDE_OP_INCOME_DEFICIT,
        }
    }
    p_sum = outdir / "filter_summary.json"
    p_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return [p_core, p_sat, p_sat_val, p_sat_ps, p_exc, p_sum]


def write_ma200_lane_csvs(flat: pd.DataFrame, outdir: Path) -> list[Path]:
    """MA200・ファンダレーン別の候補CSV（推奨順位は candidate_lane ベース）。"""
    outdir.mkdir(parents=True, exist_ok=True)
    ok = flat[flat["ok"] == True].copy()
    if ok.empty or "candidate_lane" not in ok.columns:
        return []
    outs: list[Path] = []
    for lane, fname in _LANE_EXPORT_NAMES.items():
        sub = ok[ok["candidate_lane"].astype(str) == lane].copy()
        sort_cols = [c for c in ("entry_score", "fundamental_edge_score", "total_score") if c in sub.columns]
        if sort_cols:
            sub = sub.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))
        p = outdir / fname
        sub.to_csv(p, index=False, encoding="utf-8-sig")
        outs.append(p)
    return outs

def write_reports(flat: pd.DataFrame, outdir: Path, topn: int = 10, timestamp: Optional[str] = None) -> list[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    ok = flat[flat["ok"] == True].copy()
    if ok.empty:
        return []

    # 固定名で上書き（ストレージ節約）
    for c in ["safety","piot","spec_score","per","reference_peg","ps","rsi","adx","safety_criteria_score","market_cap","avg_volume_30d"]:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    core = ok[ok["core_candidate"] == True].copy()
    sat = ok[ok["satellite_candidate"] == True].copy()
    ranked_core = _build_ranked(core) if not core.empty else core
    ranked_sat = _build_ranked(sat) if not sat.empty else sat

    outs: list[Path] = []

    if not ranked_core.empty:
        p1 = outdir / "top_recommended_core.csv"
        _rc_sort = ["rec_priority", "rec_secondary", "total_score"]
        _rc_sort = [c for c in _rc_sort if c in ranked_core.columns]
        ranked_core.sort_values(by=_rc_sort, ascending=[True] + [False] * (len(_rc_sort) - 1)).head(topn).to_csv(
            p1, index=False, encoding="utf-8-sig",
        )
        outs.append(p1)

        p2 = outdir / "top_safety_core.csv"
        ranked_core.sort_values(by=["safety_score_scaled","resilience_score","financial_score"], ascending=[False,False,False]).head(topn).to_csv(p2, index=False, encoding="utf-8-sig")
        outs.append(p2)

        p3 = outdir / "top_speculative_core.csv"
        ranked_core.sort_values(by=["spec_score"], ascending=False).head(topn).to_csv(p3, index=False, encoding="utf-8-sig")
        outs.append(p3)

        p4 = outdir / "top_piotroski_core.csv"
        ranked_core.sort_values(by=["piot","financial_score","total_score"], ascending=[False,False,False]).head(topn).to_csv(p4, index=False, encoding="utf-8-sig")
        outs.append(p4)

        if "safety_criteria_score" in ranked_core.columns:
            p5 = outdir / "top_safe_long_term_core.csv"
            ranked_core.sort_values(by=["resilience_score","safety_criteria_score"], ascending=[False,False]).head(topn).to_csv(p5, index=False, encoding="utf-8-sig")
            outs.append(p5)

    if not ranked_sat.empty:
        p6 = outdir / "top_recommended_satellite.csv"
        _rs_sort = ["rec_priority", "rec_secondary", "total_score"]
        _rs_sort = [c for c in _rs_sort if c in ranked_sat.columns]
        ranked_sat.sort_values(by=_rs_sort, ascending=[True] + [False] * (len(_rs_sort) - 1)).head(topn).to_csv(
            p6, index=False, encoding="utf-8-sig",
        )
        outs.append(p6)

    return outs

def write_markdown_report(flat: pd.DataFrame, outdir: Path, topn: int = 10, timestamp: Optional[str] = None) -> Optional[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    ok = flat[flat["ok"] == True].copy()
    core = ok[ok["core_candidate"] == True].copy()
    if core.empty:
        return None

    for c in ["safety","piot","spec_score","ps","per","market_cap","avg_volume_30d"]:
        if c in core.columns:
            core[c] = pd.to_numeric(core[c], errors="coerce")

    ranked_md = _build_ranked(core)
    _md_sort = [c for c in ["rec_priority", "rec_secondary", "total_score"] if c in ranked_md.columns]
    if len(_md_sort) >= 2:
        rec = ranked_md.sort_values(by=_md_sort, ascending=[True] + [False] * (len(_md_sort) - 1)).head(topn)
    else:
        rec = ranked_md.sort_values(by=["total_score"], ascending=[False]).head(topn)

    lines = ["# おすすめトップテン（Core候補）", ""]
    lines.append("**注:** PEG/reference_peg は参考列であり、総合スコア・安全性・投機性判定には未使用です。")
    lines.append("**注:** statement_basis_used が fallback_primary_type の銘柄は通期比較が取れず、順位に軽微なデータ品質ペナルティを加算します。")
    lines.append("**注:** PS-only satellite は PER 欠損だが PS と基礎品質で残した候補です。PEG/reference_peg は参考列のみでスコア未使用。fallback_primary_type は軽微ペナルティ対象です。")
    lines.append(
        "**注:** `legacy_total` は従来の総合スコア（バリュエーション・財務・安全性等の合成）です。"
        "`entry_timing_score` は買いタイミング評価であり、銘柄品質そのものではありません。"
        "最終判断は candidate_lane、fundamental_edge、data_review_reason を併用します。"
    )
    lines.append("")
    if timestamp:
        lines.append(f"**生成日時:** {timestamp.replace('_', ' ')}")
        lines.append("")
    lines.append(
        f"**フィルタ:** avg_volume_30d>={MIN_AVG_VOLUME_30D}, ADV20>={MIN_ADV_JPY_20D:,}JPY, "
        f"market_cap>={MIN_MARKET_CAP_JPY:,}JPY, PS<={MAX_PS_DEFENSIVE}, PER<={MAX_PER_CORE}, "
        f"営業利益安定（急落floor={OP_INCOME_DROP_FLOOR}）"
    )
    lines.append("")
    for _, r in rec.iterrows():
        mc = r.get("market_cap")
        mc_str = f"{mc/1e9:.1f}B" if pd.notna(mc) else "N/A"
        vol = r.get("avg_volume_30d")
        vol_str = f"{int(vol):,}" if pd.notna(vol) else "N/A"
        rp = r.get("reference_peg")
        if rp is None or (isinstance(rp, float) and pd.isna(rp)):
            peg_str = "N/A"
        else:
            peg_str = f"{float(rp):.2f}"
        fe = r.get("fundamental_edge_score")
        es = r.get("entry_score")
        fe_s = f"{float(fe):.1f}" if fe is not None and pd.notna(fe) else "N/A"
        es_s = f"{float(es):.1f}" if es is not None and pd.notna(es) else "N/A"
        lane = r.get("candidate_lane", "")
        dr_r = r.get("data_review_reason") or ""
        lines.append(
            f"- **{r['code']} {r['name']}** | lane `{lane}` | legacy_total {r['total_score']:.1f} | "
            f"fundamental_edge {fe_s} | entry_timing_score {es_s} | 財務 {r['financial_score']:.1f} | 仕手 {r['spec_score']} | "
            f"PER {r['per']} | PS {r['ps']} | refPEG {peg_str} | data_review `{dr_r}` | 時価総額 {mc_str} | 出来高(30d) {vol_str}"
        )

    p = outdir / f"report_core_top{topn}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p

# ==== 投資助言レポート生成 ====
def _grade_from_score(s: float) -> str:
    if s >= 85: return "A+"
    if s >= 75: return "A"
    if s >= 65: return "B+"
    if s >= 55: return "B"
    return "C"

def _val_score_from_ps_vs_sector(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x): return 0.0
    if x <= 0.6: return 10.0
    if x <= 0.9: return 8.0
    if x <= 1.2: return 6.0
    if x <= 1.6: return 3.0
    if x <= 2.0: return 1.0
    return 0.0

def _val_score_from_peg(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x): return 0.0
    if x <= 0: return 0.0
    if x <= 0.7: return 10.0
    if x <= 1.0: return 8.0
    if x <= 1.5: return 6.0
    if x <= 2.0: return 3.0
    if x <= 3.0: return 1.0
    return 0.0

def _quality_score_from_piotroski(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    x = float(x)
    if x >= 8: return 14.0
    if x >= 7: return 12.0
    if x >= 6: return 10.0
    if x >= 5: return 7.0
    if x >= 4: return 4.0
    return 0.0

def _growth_score_from_sales_cagr(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    if x >= 0.18: return 8.0
    if x >= 0.12: return 6.5
    if x >= 0.06: return 5.0
    if x >= 0.00: return 3.0
    if x >= -0.05: return 1.0
    return 0.0

def _resilience_score_from_drawdown(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    if x >= -0.15: return 8.0
    if x >= -0.25: return 6.0
    if x >= -0.35: return 4.0
    if x >= -0.45: return 2.0
    return 0.0

def _momentum_score_from_returns(
    ret_21d: Optional[float],
    ret_63d: Optional[float],
    ret_126d: Optional[float],
    momentum_6m_1m: Optional[float],
    momentum_6m_3m: Optional[float],
    momentum_3m_1m: Optional[float],
    below_ma200: Optional[bool],
) -> float:
    score = 0.0
    if momentum_6m_1m is not None and np.isfinite(momentum_6m_1m):
        if 0.08 <= momentum_6m_1m <= 0.45: score += 6.0
        elif 0.00 <= momentum_6m_1m < 0.08 or 0.45 < momentum_6m_1m <= 0.65: score += 3.0
        elif -0.08 <= momentum_6m_1m < 0.00: score += 1.0
    if momentum_6m_3m is not None and np.isfinite(momentum_6m_3m):
        if 0.03 <= momentum_6m_3m <= 0.25: score += 4.0
        elif 0.00 <= momentum_6m_3m < 0.03 or 0.25 < momentum_6m_3m <= 0.40: score += 2.0
    if momentum_3m_1m is not None and np.isfinite(momentum_3m_1m):
        if -0.05 <= momentum_3m_1m <= 0.18: score += 2.0
        elif 0.18 < momentum_3m_1m <= 0.30: score += 1.0
    if ret_63d is not None and np.isfinite(ret_63d):
        if 0.05 <= ret_63d <= 0.30: score += 4.0
        elif 0.00 <= ret_63d < 0.05 or 0.30 < ret_63d <= 0.45: score += 2.0
        elif -0.05 <= ret_63d < 0.00: score += 0.5
    if ret_126d is not None and np.isfinite(ret_126d):
        if 0.10 <= ret_126d <= 0.50: score += 4.0
        elif 0.00 <= ret_126d < 0.10 or 0.50 < ret_126d <= 0.70: score += 2.0
        elif -0.08 <= ret_126d < 0.00: score += 0.5
    if ret_21d is not None and np.isfinite(ret_21d):
        if -0.08 <= ret_21d <= 0.12:
            score += 2.0
        elif -0.15 <= ret_21d < -0.08 or 0.12 < ret_21d <= 0.20:
            score += 0.5
        elif ret_21d > 0.20:
            score -= 1.0
    return score

def _data_quality_penalty(imputed_field_count: Optional[float], critical_missing_count: Optional[float], statement_staleness_days: Optional[float]) -> float:
    penalty = 0.0
    if imputed_field_count is not None and np.isfinite(imputed_field_count):
        penalty += min(float(imputed_field_count), 4.0) * 1.5
    if critical_missing_count is not None and np.isfinite(critical_missing_count):
        penalty += min(float(critical_missing_count), 4.0) * 1.5
    if statement_staleness_days is not None and np.isfinite(statement_staleness_days):
        if statement_staleness_days > 540:
            penalty += 6.0
        elif statement_staleness_days > 365:
            penalty += 3.0
    return min(penalty, 12.0)

def _build_ranked(flat: pd.DataFrame) -> pd.DataFrame:
    df = flat.copy()
    if "reference_peg" not in df.columns and "peg" in df.columns:
        df["reference_peg"] = df["peg"]
    if "candidate_lane" not in df.columns:
        if "valuation_lane" in df.columns:
            df["candidate_lane"] = df["valuation_lane"]
        else:
            df["candidate_lane"] = "excluded"
    for _c in ("fundamental_edge_score", "entry_score"):
        if _c not in df.columns:
            df[_c] = np.nan
    for c in [
        "ps", "reference_peg", "per", "rsi", "adx", "piot", "safety", "spec_score", "below_ma200",
        "return_21d", "return_63d", "return_126d", "return_252d",
        "momentum_6m_1m", "momentum_6m_3m", "momentum_3m_1m", "safety_criteria_score",
        "max_drawdown", "sales_cagr", "adv_jpy_20d", "adv_jpy_60d", "sector_ps_benchmark",
        "imputed_field_count", "critical_missing_count", "statement_staleness_days"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    sec_med = df.groupby("sector")["ps"].median()

    def _ps_vs_sector(row):
        ps = row.get("ps")
        med = row.get("sector_ps_benchmark")
        if pd.isna(med) or med <= 0:
            med = sec_med.get(row.get("sector"), np.nan)
        if pd.isna(ps) or pd.isna(med) or med <= 0:
            return np.nan
        return float(ps) / float(med)

    df["ps_vs_sector"] = df.apply(_ps_vs_sector, axis=1)

    df["valuation_ps"] = df["ps_vs_sector"].apply(_val_score_from_ps_vs_sector)
    df["valuation_score"] = df["valuation_ps"]                     # PEGは参考指標のためスコアに含めない（0-10）

    df["quality_piotroski"] = df["piot"].apply(_quality_score_from_piotroski)
    df["quality_op_income"] = np.where(df.get("op_income_stable", False) == True, 8.0, 0.0)
    df["quality_growth"] = df["sales_cagr"].apply(_growth_score_from_sales_cagr)
    df["financial_score"] = df["quality_piotroski"] + df["quality_op_income"] + df["quality_growth"]  # 0-30

    df["resilience_safety_criteria"] = df["safety_criteria_score"].fillna(0.0).clip(lower=0, upper=100) * (12.0 / 100.0)
    df["resilience_drawdown"] = df["max_drawdown"].apply(_resilience_score_from_drawdown)
    df["resilience_score"] = df["resilience_safety_criteria"] + df["resilience_drawdown"]  # 0-20

    df["technical_score"] = [
        _momentum_score_from_returns(r21, r63, r126, m61, m63, m31, False)
        for r21, r63, r126, m61, m63, m31 in zip(
            df["return_21d"], df["return_63d"], df["return_126d"],
            df["momentum_6m_1m"], df["momentum_6m_3m"], df["momentum_3m_1m"],
        )
    ]  # 0-22（200日線単純ペナルティは廃止）

    safety_base = df["safety"].fillna(0.0).clip(lower=0, upper=25.0)
    adv_bonus = np.where(df["adv_jpy_20d"] >= 1_000_000_000, 2.0, np.where(df["adv_jpy_20d"] >= 500_000_000, 1.0, 0.0))
    df["safety_score_scaled"] = (safety_base * (10.0 / 25.0) + adv_bonus).clip(lower=0, upper=12.0)  # 0-12

    df["spec_penalty"] = df["spec_score"].fillna(0.0).clip(lower=0, upper=100) * (8.0 / 100.0)  # 0-8
    df["per_penalty"] = np.where(df["per"].notna() & (df["per"] > MAX_PER_CORE), 4.0, 0.0)
    if "fallback_basis_flag" in df.columns:
        fb_series = df["fallback_basis_flag"].apply(
            lambda x: (x is True) or (str(x).strip().lower() == "true")
        )
    else:
        fb_series = pd.Series(False, index=df.index)
    if "ps_only_satellite_candidate" in df.columns:
        ps_only_series = df["ps_only_satellite_candidate"].apply(
            lambda x: (x is True) or (str(x).strip().lower() == "true")
        )
    else:
        ps_only_series = pd.Series(False, index=df.index)
    _penalties: list[float] = []
    for imputed, missing, stale, fb, ps_only in zip(
        df["imputed_field_count"],
        df["critical_missing_count"],
        df["statement_staleness_days"],
        fb_series,
        ps_only_series,
    ):
        pen = _data_quality_penalty(imputed, missing, stale) + (2.0 if fb else 0.0)
        if ps_only:
            pen = max(0.0, pen - 1.5)
        _penalties.append(pen)
    df["data_penalty"] = _penalties

    df["total_score"] = (
        df["valuation_score"] +
        df["financial_score"] +
        df["resilience_score"] +
        df["technical_score"] +
        df["safety_score_scaled"] -
        df["spec_penalty"] -
        df["per_penalty"] -
        df["data_penalty"]
    )
    df["total_score"] = df["total_score"].clip(lower=0, upper=100)
    df["grade"] = df["total_score"].apply(_grade_from_score)
    df["pio_disp"] = df["piot"].fillna(0).astype(int).astype(str) + "/9"
    df["candidate_lane_sort"] = df["candidate_lane"].astype(str).map(
        lambda x: _CANDIDATE_LANE_SORT.get(str(x).strip(), 50)
    )

    lane = df["candidate_lane"].astype(str)
    ma_state = df["ma200_state"].astype(str) if "ma200_state" in df.columns else pd.Series("", index=df.index, dtype=str)
    dr_light_reclaim = (lane == "data_review_light") & (ma_state == "ma200_reclaim")
    dr_light_other = (lane == "data_review_light") & ~dr_light_reclaim
    df["rec_priority"] = np.select(
        [
            lane == "ma200_reclaim_core",
            lane == "bottom_reversal_core",
            dr_light_reclaim,
            lane == "watch_fundamental_core",
            lane == "weak_reclaim_watch",
            lane == "extended_above_ma200",
            lane == "data_review",
            dr_light_other,
            lane == "cyclical_value_trap",
            lane.isin(["satellite_valuation", "satellite_ps_only"]),
        ],
        [10, 20, 30, 40, 45, 50, 60, 65, 70, 80],
        default=200 + df["candidate_lane_sort"].fillna(50).astype(int),
    ).astype(int)

    ent = pd.to_numeric(df["entry_score"], errors="coerce")
    fed = pd.to_numeric(df["fundamental_edge_score"], errors="coerce")
    df["rec_secondary"] = ent
    df.loc[df["rec_priority"] == 40, "rec_secondary"] = fed.loc[df["rec_priority"] == 40]

    return df

def write_investment_advice_report(flat: pd.DataFrame, outdir: Path,
                                   topn: int = 15, details_n: int = 30, timestamp: Optional[str] = None) -> list[Path]:
    outdir.mkdir(exist_ok=True, parents=True)

    ok = flat[flat["ok"] == True].copy()
    core = ok[ok["core_candidate"] == True].copy()
    sat = ok[ok["satellite_candidate"] == True].copy()
    if core.empty:
        return []

    ranked = _build_ranked(core)

    _sort_cols2 = [c for c in ["rec_priority", "rec_secondary", "total_score"] if c in ranked.columns]
    if len(_sort_cols2) < 2:
        _sort_cols2 = [c for c in ["candidate_lane_sort", "entry_score", "fundamental_edge_score", "total_score"] if c in ranked.columns]
    ranked = ranked.sort_values(
        by=_sort_cols2,
        ascending=[True] + [False] * (len(_sort_cols2) - 1),
    )

    now = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
    n = len(ranked)
    avg_score = ranked["total_score"].mean()
    grade_counts = ranked["grade"].value_counts().reindex(["A+","A","B+","B","C"]).fillna(0).astype(int)

    ps_avg = ranked["ps_vs_sector"].mean(skipna=True)
    ps_med = ranked["ps_vs_sector"].median(skipna=True)
    ps_min = ranked["ps_vs_sector"].min(skipna=True)
    ps_max = ranked["ps_vs_sector"].max(skipna=True)

    if "reference_peg" in ranked.columns:
        refpeg = ranked["reference_peg"]
    elif "peg" in ranked.columns:
        refpeg = ranked["peg"]
    else:
        refpeg = pd.Series(np.nan, index=ranked.index)
    peg_avg = refpeg.mean(skipna=True)
    peg_med = refpeg.median(skipna=True)
    peg_min = refpeg.min(skipna=True)
    peg_max = refpeg.max(skipna=True)

    top = ranked.head(topn)

    lines: list[str] = []
    lines.append("# 🏆 PS・PEGレシオ対応 投資銘柄スクリーニング レポート（Core候補）")
    lines.append("")
    lines.append("**注:** PEG/reference_peg は参考列であり、総合スコア・安全性・投機性判定には未使用です。")
    lines.append("**注:** statement_basis_used が fallback_primary_type の銘柄は通期比較が取れず、順位に軽微なデータ品質ペナルティを加算します。")
    lines.append("**注:** PS-only satellite は PER 欠損だが PS と基礎品質で残した候補です。PEG/reference_peg は参考列のみでスコア未使用。fallback_primary_type は軽微ペナルティ対象です。")
    lines.append(
        "**注:** 推奨順は rec_priority（例: ma200_reclaim_core→bottom→data_review_light+reclaim→watch…）を最優先し、"
        "次に entry_timing_score（タイミング）または watch の fundamental_edge を参照します。"
        "`legacy_total` は旧総合スコア、`entry_timing_score` は買いタイミング評価です（品質そのものではありません）。"
        "`bottom_reversal_core` は200日線下の逆張り候補です。"
    )
    lines.append("")
    lines.append(f"**📅 生成日時:** {now}")
    lines.append(f"**📊 分析対象(Core):** {n}銘柄")
    lines.append("")
    lines.append("## ✅ 必須フィルタ（Core前提）")
    lines.append("")
    lines.append(f"- avg_volume_30d >= {MIN_AVG_VOLUME_30D:,}")
    lines.append(f"- adv_jpy_20d >= {MIN_ADV_JPY_20D:,} JPY")
    lines.append(f"- market_cap >= {MIN_MARKET_CAP_JPY:,} JPY")
    lines.append(f"- 営業利益安定（直近{OP_INCOME_YEARS}年・赤字除外={EXCLUDE_OP_INCOME_DEFICIT}・急落floor={OP_INCOME_DROP_FLOOR}）")
    lines.append(f"- PS <= {MAX_PS_DEFENSIVE}")
    lines.append(f"- PER <= {MAX_PER_CORE}（超はSatellite扱い）")
    lines.append("")
    lines.append("## 📋 エグゼクティブサマリー")
    lines.append("")
    lines.append(f"- **平均投資スコア:** {avg_score:.1f}点")
    lines.append("- **グレード分布:**")
    lines.append(f"  - A+: {grade_counts['A+']}銘柄")
    lines.append(f"  - A: {grade_counts['A']}銘柄")
    lines.append(f"  - B+: {grade_counts['B+']}銘柄")
    lines.append(f"  - B: {grade_counts['B']}銘柄")
    lines.append(f"  - C: {grade_counts['C']}銘柄")
    lines.append("")
    lines.append("## 💰 バリュエーション分析")
    lines.append("")
    lines.append("### PSレシオ（セクター比）")
    lines.append(f"- 平均: {ps_avg:.2f}")
    lines.append(f"- 中央値: {ps_med:.2f}")
    lines.append(f"- 最小: {ps_min:.2f}")
    lines.append(f"- 最大: {ps_max:.2f}")
    lines.append("")
    lines.append("### PEGレシオ（参考・スコア未使用）")
    lines.append(f"- 平均: {peg_avg:.2f}")
    lines.append(f"- 中央値: {peg_med:.2f}")
    lines.append(f"- 最小: {peg_min:.2f}")
    lines.append(f"- 最大: {peg_max:.2f}")
    lines.append("")
    lines.append(f"## 🏆 投資推奨 Top{topn}銘柄（Core、lane優先・legacy_total / entry_timing_score は別軸）")
    lines.append("")
    lines.append("| 順位 | 銘柄 | 名 | G | legacy_total | lane | MA200局面 | fundamental_edge | entry_timing_score | dr_reason | dr_lvl | d200 | 上抜け | basing | down | peg_w | p_adj | p_cov | PS/防 | refPEG | pio |")
    lines.append("|---|---|---|---|---:|---|---|---|---:|---|---|---|---:|---:|---|---|---|---|---|---:|---:|---|")
    for i, r in enumerate(top.itertuples(index=False), 1):
        ps_vs = 0 if pd.isna(r.ps_vs_sector) else r.ps_vs_sector
        psp = getattr(r, "ps_vs_sector_pre", np.nan)
        psp_c = "N/A" if psp is None or (isinstance(psp, float) and pd.isna(psp)) else f"{float(psp):.2f}"
        rpeg = getattr(r, "reference_peg", None)
        if rpeg is None or (isinstance(rpeg, float) and pd.isna(rpeg)):
            rpeg = getattr(r, "peg", np.nan)
        if rpeg is None or (isinstance(rpeg, float) and pd.isna(rpeg)):
            peg_cell = "N/A"
        else:
            peg_cell = f"{float(rpeg):.2f}"
        lane = getattr(r, "candidate_lane", "")
        ms = getattr(r, "ma200_state", "")
        fe = getattr(r, "fundamental_edge_score", np.nan)
        es = getattr(r, "entry_score", np.nan)
        dm = getattr(r, "distance_from_ma200", np.nan)
        xr = getattr(r, "crossed_above_ma200_recently", "")
        bs = getattr(r, "below_ma200_basing", "")
        dn = getattr(r, "below_ma200_downtrend", "")
        pwg = getattr(r, "peg_warning", "")
        pad = getattr(r, "piotroski_adjusted_score", np.nan)
        pcv = getattr(r, "piotroski_coverage_ratio", np.nan)
        drr = getattr(r, "data_review_reason", "") or ""
        drl = getattr(r, "data_review_level", "") or ""
        fe_s = "N/A" if fe is None or (isinstance(fe, float) and pd.isna(fe)) else f"{float(fe):.1f}"
        es_s = "N/A" if es is None or (isinstance(es, float) and pd.isna(es)) else f"{float(es):.1f}"
        dm_s = "N/A" if dm is None or (isinstance(dm, float) and pd.isna(dm)) else f"{float(dm):.3f}"
        pad_s = "N/A" if pad is None or (isinstance(pad, float) and pd.isna(pad)) else f"{float(pad):.2f}"
        pcv_s = "N/A" if pcv is None or (isinstance(pcv, float) and pd.isna(pcv)) else f"{float(pcv):.2f}"
        lines.append(
            f"| {i} | {r.code} | {r.name} | {r.grade} | {r.total_score:.1f} | {lane} | {ms} | {fe_s} | {es_s} | {drr} | {drl} | {dm_s} | {xr} | {bs} | {dn} | {pwg} | {pad_s} | {pcv_s} | {psp_c} | {peg_cell} | {r.pio_disp} |"
        )
    lines.append("")
    lines.append(f"## 📊 詳細分析（上位{details_n}銘柄）")
    lines.append("")
    detail_df = ranked.head(details_n)
    for i, r in enumerate(detail_df.itertuples(index=False), 1):
        lines.append(f"### {i}. [{r.code}] {r.name} ({r.sector})")
        lines.append(f"**legacy_total（旧総合）:** {r.total_score:.1f}点 | **グレード:** {r.grade}")
        _lane = getattr(r, "candidate_lane", "")
        _ms = getattr(r, "ma200_state", "")
        _fe = getattr(r, "fundamental_edge_score", np.nan)
        _es = getattr(r, "entry_score", np.nan)
        _drr = getattr(r, "data_review_reason", "") or ""
        _drl = getattr(r, "data_review_level", "") or ""
        _fe_s = "N/A" if _fe is None or (isinstance(_fe, float) and pd.isna(_fe)) else f"{float(_fe):.1f}"
        _es_s = "N/A" if _es is None or (isinstance(_es, float) and pd.isna(_es)) else f"{float(_es):.1f}"
        lines.append(
            f"**レーン:** {_lane} | **MA200局面:** {_ms} | **fundamental_edge:** {_fe_s} | **entry_timing_score:** {_es_s}"
        )
        if _drr or _drl:
            lines.append(f"**データ要レビュー:** `{_drr}` （level: {_drl}）")
        lines.append("**スコア内訳:**")
        lines.append(f"- バリュエーション: {r.valuation_score:.1f}点")
        lines.append(f"- 財務健全性: {r.financial_score:.1f}点")
        lines.append(f"- レジリエンス: {r.resilience_score:.1f}点")
        lines.append(f"- モメンタム: {r.technical_score:.1f}点")
        lines.append(f"- 安全性: {r.safety_score_scaled:.1f}点")
        lines.append(f"- 仕手株ペナルティ: {r.spec_penalty:.1f}点")
        lines.append(f"- PERペナルティ: {r.per_penalty:.1f}点")
        lines.append(f"- データ品質ペナルティ: {r.data_penalty:.1f}点")
        lines.append("")

    if not sat.empty:
        lines.append("## 🛰 Satellite候補（参考）")
        lines.append("")
        lines.append(f"- 件数: {len(sat)}（base_okだがPS/PER条件でcore外）")
        lines.append("")

    p_csv = outdir / "ranked_with_scores_core.csv"
    p_md  = outdir / "report_investment_advice_core.md"
    p_csv.write_text(ranked.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")
    p_md.write_text("\n".join(lines), encoding="utf-8")
    return [p_csv, p_md]

# ------------------------------------------------------------
# ★ 追加: master CSVから一括レポート生成（外部モジュール不要）
# ------------------------------------------------------------
def _infer_timestamp_from_master_csv(path: Path) -> str:
    m = re.search(r"screening_offline_(\d{8}_\d{6})\.csv", path.name)
    if m:
        return m.group(1)
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_reports_from_master_csv(
    master_csv_path: str | Path,
    reports_dir: str | Path = REPORTS_DIR,
    topn: int = 30,
) -> list[Path]:
    master_csv_path = Path(master_csv_path)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(exist_ok=True, parents=True)

    if not master_csv_path.exists():
        raise FileNotFoundError(f"master_csv_path not found: {master_csv_path}")

    flat = pd.read_csv(master_csv_path, encoding="utf-8-sig")
    ts = _infer_timestamp_from_master_csv(master_csv_path)

    outputs: list[Path] = []
    outputs.append(master_csv_path)

    outputs += write_candidate_sets(flat, reports_dir, timestamp=ts)
    outputs += write_ma200_lane_csvs(flat, reports_dir)
    outputs += write_reports(flat, reports_dir, topn=topn, timestamp=ts)

    p_md = write_markdown_report(flat, reports_dir, topn=min(10, topn), timestamp=ts)
    if p_md:
        outputs.append(p_md)

    outputs += write_investment_advice_report(flat, reports_dir, topn=max(15, min(topn, 30)), details_n=max(30, topn), timestamp=ts)
    # 全レポート出力後に古いCSVを削除（output/ と output/reports/ の両方、ストレージ節約）
    cleanup_old_report_files(reports_dir)
    if reports_dir.parent != reports_dir:
        cleanup_old_report_files(reports_dir.parent)
    return outputs

# ------------------------------------------------------------
# インタラクティブUI / CLI
# ------------------------------------------------------------
def run_interactive():
    session = get_authenticated_session_jquants()
    sector_avgs = DynamicSectorAverages(session).get_sector_averages()
    outdir = REPORTS_DIR
    outdir.mkdir(exist_ok=True, parents=True)

    while True:
        if graceful_shutdown.shutdown:
            logger.info("shutdown requested; stopping menu loop")
            break
        print("=== メニュー ===")
        print("1) 収集（価格+財務を凍結保存）")
        print("2) オフライン一括分析（core/satellite/excluded 出力）")
        print("3) 単銘柄分析（キャッシュ使用）")
        print("4) セクター平均を更新（キャッシュから計算）")
        print("5) 全銘柄ゆっくり収集（自動待機・再開可）")
        print("6) 鮮度で取り直し収集（例: 7日より古いものだけ）")
        print("7) 全銘柄“強制”再収集（pending初期化＋当日再取得）")
        print("q) 終了")
        try:
            choice = input("選択: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            graceful_shutdown.shutdown = True
            _cli_print("\n🛑 中断しました", "\n[中断] 終了しました")
            break

        if choice == "1":
            budget = input(f"本日収集する銘柄数（推奨既定{DEFAULT_COLLECT_BUDGET}、Enter で既定）: ").strip()
            budget = int(budget) if budget.isdigit() else DEFAULT_COLLECT_BUDGET
            s = collect_batch(session, budget)
            _cli_print(f"📦 収集: tried={s['tried']} ok={s['ok']} fail={s['fail']}", f"[収集] tried={s['tried']} ok={s['ok']} fail={s['fail']}")

        elif choice == "2":
            tasks = build_offline_analysis_tasks(session)
            if not tasks:
                print("キャッシュ不足。先に収集を実行してください。")
                continue

            results = []
            max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
            ex = ThreadPoolExecutor(max_workers=max_workers)
            futs = [
                ex.submit(
                    analyze_single_stock_complete_v3,
                    session, sector_avgs, code, name, market, sector,
                    offline=True
                )
                for (code, name, market, sector) in tasks
            ]
            try:
                for i, fut in enumerate(as_completed(futs), 1):
                    if graceful_shutdown.shutdown:
                        logger.info("shutdown requested; stopping analyze loop")
                        break
                    results.append(fut.result())
                    if i % 200 == 0 or i == len(futs):
                        ok_cnt = sum(1 for r in results if r.get("success"))
                        _cli_print(
                            f"  ⏱ {i}/{len(futs)} 完了 (OK={ok_cnt})",
                            f"  [{i}/{len(futs)}] 完了 (OK={ok_cnt})",
                        )
            finally:
                _executor_shutdown_interrupt(ex, futs)

            if graceful_shutdown.shutdown and len(results) < len(futs):
                _cli_print(
                    "🛑 中断のため一括分析を打ち切りました（CSVは保存していません）",
                    "[中断] 一括分析を打ち切り",
                )
                continue

            flat = pd.DataFrame([_flatten_result(r) for r in results])
            master = outdir / "screening_offline.csv"
            flat.to_csv(master, index=False, encoding="utf-8-sig")

            outputs = generate_reports_from_master_csv(master, outdir, topn=30)
            _cli_print(f"✅ 出力: {master}", f"[OK] 出力: {master}")
            _cli_print(f"✅ 出力先: {outdir}", f"[OK] 出力先: {outdir}")
            print("=== 生成物 ===")
            for p in outputs:
                print(f"  - {p}")

        elif choice == "3":
            code = input("銘柄コード4桁: ").strip()
            name, inst = lookup_equity_name_and_instrument(session, code)
            res = analyze_single_stock_complete_v3(
                session, sector_avgs, code, name=name, offline=True, instrument_type=inst,
            )
            df = pd.DataFrame([_flatten_result(res)])
            fp = outdir / f"single_{code}.csv"
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            cleanup_old_report_files(outdir)
            cleanup_old_report_files(outdir.parent)
            _cli_print(f"✅ 出力: {fp}", f"[OK] 出力: {fp}")
            print("注: PEG/reference_peg は参考列であり、総合スコア・安全性・投機性判定には未使用です。")

        elif choice == "4":
            _cli_print("📊 セクター平均をキャッシュから計算中...", "[セクター平均] キャッシュから計算中...")
            sector_avgs_obj = DynamicSectorAverages(session)
            updated_avgs = sector_avgs_obj.calculate_sector_averages_from_cache()
            if updated_avgs:
                cache_file = CACHE_DIR / "sector_averages.json"
                sectors = [
            '自動車', '半導体', '電気機器', '銀行', '情報・通信業', '医薬品', '商社',
            '小売', '卸売', '建設', '陸運', '海運', '空運', '電気・ガス', '食品', '機械',
            'サービス', 'ゲーム', '化学', '鉄鋼', '不動産', 'エネルギー', '証券', 'その他',
        ]
                data = {}
                for sector in sectors:
                    if sector in updated_avgs:
                        data[sector] = updated_avgs[sector]
                    else:
                        data[sector] = sector_avgs_obj.get_default_sector_average(sector)
                cache_file.write_text(json.dumps({"timestamp": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
                sector_avgs_obj.sector_cache = data
                sector_avgs_obj.cache_timestamp = time.time()
                sector_avgs = data
                _cli_print(f"✅ セクター平均を更新しました（{len(updated_avgs)}セクター）", f"[OK] セクター平均を更新しました（{len(updated_avgs)}セクター）")
                for sector, stats in updated_avgs.items():
                    ps_val = stats.get('ps', None)
                    per_val = stats.get('per', None)
                    sample_count = stats.get('sample_count', 0)
                    ps_str = f"{ps_val:.2f}" if ps_val is not None else "N/A"
                    per_str = f"{per_val:.2f}" if per_val is not None else "N/A"
                    print(f"  {sector}: PS={ps_str}, PER={per_str}, サンプル数={sample_count}")
                p_audit = write_sector_normalization_audit_csv(session)
                if p_audit:
                    _cli_print(f"📄 セクター正規化監査: {p_audit}", f"[監査] sector_normalization {p_audit}")
            else:
                _cli_print(
                    "⚠️ セクター平均の計算に失敗しました。キャッシュデータが不足している可能性があります。",
                    "[警告] セクター平均の計算に失敗しました。キャッシュデータが不足している可能性があります。",
                )

        elif choice == "5":
            budget = input(f"1日あたりの最大収集銘柄数（既定{DEFAULT_COLLECT_BUDGET}）: ").strip()
            budget = int(budget) if budget.isdigit() else DEFAULT_COLLECT_BUDGET
            collect_all_daemon(session, daily_budget=budget)

        elif choice == "6":
            days = input("何日より古ければ取り直すか（日数。例: 7）: ").strip()
            days = int(days) if days.isdigit() else 7
            budget = input(f"1日あたりの最大収集銘柄数（既定{DEFAULT_COLLECT_BUDGET}）: ").strip()
            budget = int(budget) if budget.isdigit() else DEFAULT_COLLECT_BUDGET
            collect_all_daemon(session, daily_budget=budget, refresh_days=days, reset_pending=True)

        elif choice == "7":
            budget = input(f"1日あたりの最大収集銘柄数（既定{DEFAULT_COLLECT_BUDGET}）: ").strip()
            budget = int(budget) if budget.isdigit() else DEFAULT_COLLECT_BUDGET
            collect_all_daemon(session, daily_budget=budget, force_full=True, reset_pending=True)

        elif choice == "q":
            break

        else:
            print("無効な選択")

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",
                        choices=["collect","collect_all","analyze","single","interactive"],
                        default="interactive")
    parser.add_argument("--code")
    parser.add_argument("--budget", type=int, default=DEFAULT_COLLECT_BUDGET)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--reset-pending", action="store_true")
    parser.add_argument("--refresh-days", type=int)
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--force", action="store_true", help="--force-full と同等（skiplist 無視・メニュー7相当）")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="collect のみ: キャッシュをバイパスし、collectable 先頭 N 件を再取得（--budget で件数指定）",
    )
    args = parser.parse_args()

    if args.phase == "interactive":
        try:
            run_interactive()
        except KeyboardInterrupt:
            graceful_shutdown.shutdown = True
            _cli_print("\n🛑 中断しました", "\n[中断] 終了しました")
        return 130 if graceful_shutdown.shutdown else 0

    args.force_full = bool(args.force_full or getattr(args, "force", False))

    try:
        session = get_authenticated_session_jquants()
        sector_avgs = DynamicSectorAverages(session).get_sector_averages()
        outdir = REPORTS_DIR
        outdir.mkdir(exist_ok=True, parents=True)

        if args.phase == "collect_all":
            collect_all_daemon(session,
                               daily_budget=args.budget,
                               refresh_days=args.refresh_days,
                               force_full=args.force_full,
                               reset_pending=args.reset_pending)
            return 130 if graceful_shutdown.shutdown else 0

        if args.phase == "collect":
            s = collect_batch(session, args.budget, force_refresh=bool(args.force_refresh))
            detail = (
                f"success={s.get('success', s['ok'])} permanent_missing_financials={s.get('permanent_missing_financials', 0)} "
                f"non_stock_or_fund_like={s.get('non_stock_or_fund_like', 0)} transient_error={s.get('transient_error', 0)} "
                f"auth_or_permission_error={s.get('auth_or_permission_error', 0)}"
            )
            _cli_print(
                f"📦 収集: tried={s['tried']} ok={s['ok']} fail={s['fail']} | {detail}",
                f"[収集] tried={s['tried']} ok={s['ok']} fail={s['fail']} | {detail}",
            )
            return 130 if graceful_shutdown.shutdown else 0

        if args.phase == "single":
            if not args.code:
                raise SystemExit("--code 必須")
            name, inst = lookup_equity_name_and_instrument(session, args.code)
            res = analyze_single_stock_complete_v3(
                session, sector_avgs, args.code, name=name, offline=True, instrument_type=inst,
            )
            df = pd.DataFrame([_flatten_result(res)])
            fp = outdir / f"single_{args.code}.csv"
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            cleanup_old_report_files(outdir)
            cleanup_old_report_files(outdir.parent)
            _cli_print(f"✅ 単銘柄出力: {fp}", f"[OK] 単銘柄出力: {fp}")
            print("注: PEG/reference_peg は参考列であり、総合スコア・安全性・投機性判定には未使用です。")
            return 130 if graceful_shutdown.shutdown else 0

        if args.phase == "analyze":
            tasks = build_offline_analysis_tasks(session)
            if not tasks:
                print("キャッシュ不足。先に --phase collect か collect_all を実行してください。")
                return 0

            results = []
            max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
            ex = ThreadPoolExecutor(max_workers=max_workers)
            futs = [
                ex.submit(
                    analyze_single_stock_complete_v3,
                    session, sector_avgs, code, name, market, sector,
                    offline=True
                )
                for (code, name, market, sector) in tasks
            ]
            try:
                for i, fut in enumerate(as_completed(futs), 1):
                    if graceful_shutdown.shutdown:
                        logger.info("shutdown requested; stopping analyze loop")
                        break
                    results.append(fut.result())
                    if i % 200 == 0 or i == len(futs):
                        ok_cnt = sum(1 for r in results if r.get("success"))
                        _cli_print(
                            f"  ⏱ {i}/{len(futs)} 完了 (OK={ok_cnt})",
                            f"  [{i}/{len(futs)}] 完了 (OK={ok_cnt})",
                        )
            finally:
                _executor_shutdown_interrupt(ex, futs)

            if graceful_shutdown.shutdown and len(results) < len(futs):
                _cli_print(
                    "🛑 中断のため一括分析を打ち切りました（CSVは保存していません）",
                    "[中断] 一括分析を打ち切り",
                )
                return 130

            flat = pd.DataFrame([_flatten_result(r) for r in results])
            master = outdir / "screening_offline.csv"
            flat.to_csv(master, index=False, encoding="utf-8-sig")

            outputs = generate_reports_from_master_csv(master, outdir, topn=max(10, args.top))
            _cli_print(f"✅ オフライン分析出力: {master}", f"[OK] オフライン分析出力: {master}")
            _cli_print(f"✅ 出力先: {outdir}", f"[OK] 出力先: {outdir}")
            print("=== 生成物 ===")
            for p in outputs:
                print(f"  - {p}")
            return 130 if graceful_shutdown.shutdown else 0

    except KeyboardInterrupt:
        graceful_shutdown.shutdown = True
        _cli_print("\n🛑 中断しました", "\n[中断] 終了しました")
        return 130

    return 0


def run_canonical_code_selftest() -> None:
    """_canonical_internal_stock_code の期待ケース検証（手動: --selftest-canonical）。"""
    v1301 = {"1301"}
    v0130 = {"0130"}
    v7203 = {"7203"}
    cases: List[Tuple[str, Optional[set[str]], Optional[str]]] = [
        ("7203", None, "7203"),
        ("72030", None, "7203"),
        ("13010", v1301, "1301"),
        ("01301", v1301, "1301"),
        ("0130", v0130, "0130"),
        ("0130", v1301, None),
        ("130", v1301, None),
        ("130", v0130, "0130"),
        ("12345", None, None),
        ("12345", v7203, None),
        ("720300", v7203, "7203"),
    ]
    print("=== _canonical_internal_stock_code selftest ===")
    for s, v, exp in cases:
        got = _canonical_internal_stock_code(s, v)
        ok = "OK" if got == exp else "FAIL"
        print(f"  {s!r} valid={v!r} -> {got!r} expect {exp!r} [{ok}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest-canonical":
        run_canonical_code_selftest()
        raise SystemExit(0)
    _main_exit = 0
    try:
        _main_exit = int(main() or 0)
    except KeyboardInterrupt:
        graceful_shutdown.shutdown = True
        _cli_print("\n🛑 中断しました", "\n[中断] 終了しました")
        _main_exit = 130
    finally:
        graceful_shutdown.print_safe_exit_once()
    raise SystemExit(_main_exit)
