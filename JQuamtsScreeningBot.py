# -*- coding: utf-8 -*-
"""
J-Quants 収集→凍結キャッシュ→完全オフライン分析ワークフロー
- 4日運用: 収集(価格+財務)を日次800req内で進め、4日目にオフライン一括分析
- モック不使用: オフライン時は“計算不能はNone”で返す（推定やランダムは行わない）
- 端末対話メニュー付き（引数未指定で起動するとメニュー表示）
- CLI対応:
    収集:   python script.py --phase collect --budget 380
    解析:   python script.py --phase analyze --top 10
    単銘柄: python script.py --phase single --code 8035
    全件:   python script.py --phase collect_all --budget 380
環境変数:
    JQ_RPM=50  JQ_RPD=800  # 必要なら調整

追加（スクリーニングの「必須」フィルタを実装）
- 流動性フィルタ: avg_volume_30d >= MIN_AVG_VOLUME_30D かつ market_cap >= MIN_MARKET_CAP_JPY
- バリュエーション健全性: PS<=MAX_PS_DEFENSIVE を core 条件、PER>MAX_PER_CORE は satellite 扱い
- 収益安定性（営業利益）: 直近年で赤字を含まない + 直近が急落していない

フィルタは core/satellite/excluded の3分類としてレポート出力にも反映します。

必要: pandas, numpy, requests
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
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
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

# ------------------------------------------------------------
# 定数・パス
# ------------------------------------------------------------
JQUANTS_API_BASE = "https://api.jquants.com/v1"
CACHE_DIR = Path(".jquants_cache")
CACHE_DIR.mkdir(exist_ok=True)
LOOKBACK_DAYS = 700
REPORTS_DIR = Path("output") / "reports"

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
OP_INCOME_DROP_FLOOR = float(os.getenv("OP_INCOME_DROP_FLOOR", "0.3"))  # 直近営業利益が過去年中央値の何倍以上ならOK
EXCLUDE_OP_INCOME_DEFICIT = (os.getenv("EXCLUDE_OP_INCOME_DEFICIT", "1") != "0")  # 直近年に赤字があれば除外（デフォON）

# ------------------------------------------------------------
# ヘルパ
# ------------------------------------------------------------
def seconds_until_next_day(buffer_sec: int = 10) -> int:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    reset = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((reset - now).total_seconds()) + buffer_sec)

def build_prices_endpoint(stock_code: str, lookback_days: int = LOOKBACK_DAYS) -> str:
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return f"prices/daily_quotes?code={stock_code}&from={start}"

# ------------------------------------------------------------
# Graceful Shutdown
# ------------------------------------------------------------
class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        try:
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
        except Exception:
            pass

    def exit_gracefully(self, signum, frame):
        print(f"\n⚠️ 中断シグナル受信: {signum}\n🛑 安全に終了します")
        self.shutdown = True
        sys.exit(130)

graceful_shutdown = GracefulShutdown()

# ------------------------------------------------------------
# レートリミッタ + 認証セッション
# ------------------------------------------------------------
class APIRateLimiter:
    """J-Quants 50 req/min, 800 req/day を想定"""
    def __init__(self, rpm: int = 50, rpd: int = 800):
        self.requests_per_minute = rpm
        self.requests_per_day = rpd
        self.base_delay = 1.5
        self.request_timestamps: List[datetime.datetime] = []
        self.daily_count = 0
        self.last_reset = datetime.date.today()

    def wait_if_needed(self):
        now = datetime.datetime.now()
        if now.date() > self.last_reset:
            self.daily_count = 0
            self.last_reset = now.date()

        if self.daily_count >= self.requests_per_day:
            raise RuntimeError("日次レート制限到達")

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
        self.daily_count += 1

class AuthSession(requests.Session):
    """J-Quants 認証＋レート制限対応Session"""
    def __init__(self, limiter: APIRateLimiter, ini_file: str = "api.ini"):
        super().__init__()
        self.limiter = limiter
        self.ini_file = ini_file

    def request(self, method, url, **kwargs):
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

            # 401: idToken refresh
            if resp.status_code == 401 and attempt == 1:
                _refresh_id_token(self, ini_file=self.ini_file)
                continue

            # Retryable
            if resp.status_code in (429,) or resp.status_code >= 500:
                if attempt == MAX:
                    return resp
                time.sleep(min(2 ** attempt, 30))
                continue

            self.limiter.mark()
            return resp

        raise RuntimeError(f"{method} {url} failed after {MAX} attempts")

def get_authenticated_session_jquants(ini_file: str = "api.ini") -> requests.Session:
    token_cache = CACHE_DIR / "access_token.json"
    rpm = int(os.getenv("JQ_RPM", "50"))
    rpd = int(os.getenv("JQ_RPD", "800"))
    limiter = APIRateLimiter(rpm=rpm, rpd=rpd)
    session = AuthSession(limiter, ini_file=ini_file)

    if token_cache.exists():
        try:
            cached = json.loads(token_cache.read_text(encoding="utf-8"))
            exp = datetime.datetime.strptime(cached["expires_at"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
            if datetime.datetime.now(datetime.timezone.utc) < exp:
                session.headers.update({"Authorization": f"Bearer {cached['token']}"})
                print("✅ キャッシュidTokenを使用")
                return session
        except Exception:
            pass

    print("🔑 認証開始…")
    _refresh_id_token(session, ini_file=ini_file)
    print("✅ 認証成功")
    return session

def _refresh_id_token(session: requests.Session, ini_file: str = "api.ini") -> str:
    config = configparser.ConfigParser()
    config.read(ini_file, encoding="utf-8")

    email = (config["DEFAULT"].get("MAIL_ADDRESS") or
             config["DEFAULT"].get("mail_address") or
             config["DEFAULT"].get("email"))
    password = (config["DEFAULT"].get("PASSWORD") or
                config["DEFAULT"].get("password"))

    if not (email and password):
        raise RuntimeError("メールアドレス／パスワード未設定(api.ini)")

    auth_payload = {"mailaddress": email, "password": password}
    res = requests.post(f"{JQUANTS_API_BASE}/token/auth_user", json=auth_payload, timeout=20)
    res.raise_for_status()
    refresh_token = res.json().get("refreshToken")
    if not refresh_token:
        raise RuntimeError("refreshToken取得失敗")

    tok_res = requests.post(f"{JQUANTS_API_BASE}/token/auth_refresh?refreshtoken={refresh_token}", timeout=20)
    tok_res.raise_for_status()
    id_token = tok_res.json().get("idToken")
    if not id_token:
        raise RuntimeError("idToken取得失敗")

    session.headers.update({"Authorization": f"Bearer {id_token}"})
    expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=23)
    (CACHE_DIR / "access_token.json").write_text(
        json.dumps({"token": id_token, "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False),
        encoding="utf-8"
    )
    return id_token

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

    def save_statements(self, code: str, stmts: List[dict]) -> None:
        self.stmts_path(code).write_text(json.dumps({"statements": stmts}, ensure_ascii=False), encoding="utf-8")

    def load_statements(self, code: str) -> List[dict]:
        p = self.stmts_path(code)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("statements", [])
        except Exception:
            return []

    def has_all(self, code: str, max_age_days: Optional[int] = None) -> bool:
        p1, p2 = self.prices_path(code), self.stmts_path(code)
        if not p1.exists() or not p2.exists():
            return False
        if max_age_days is None:
            return True
        try:
            now = time.time()
            age_days_prices = (now - p1.stat().st_mtime) / 86400.0
            age_days_stmts  = (now - p2.stat().st_mtime) / 86400.0
            oldest = max(age_days_prices, age_days_stmts)
            return oldest <= max_age_days
        except Exception:
            return True

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
        if "銀行" in s:
            return "銀行"
        if "情報" in s or "通信" in s:
            return "情報・通信業"
        if "電気機器" in s:
            return "電気機器"
        if "輸送用機器" in s or "自動車" in s:
            return "自動車"
        if "サービス" in s:
            return "サービス"
        if "化学" in s:
            return "化学"
        return s if s in DynamicSectorAverages.SECTOR_MEDIANS else "その他"

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
            '自動車': {'ps': 0.8, 'peg': 1.2, 'eps_growth': 8.5},
            '半導体': {'ps': 4.5, 'peg': 1.8, 'eps_growth': 12.2},
            '電気機器': {'ps': 1.8, 'peg': 1.5, 'eps_growth': 12.3},
            '銀行': {'ps': 2.5, 'peg': 0.8, 'eps_growth': 10.6},
            '情報・通信業': {'ps': 1.2, 'peg': 1.3, 'eps_growth': 11.2},
            '医薬品': {'ps': 3.8, 'peg': 1.6, 'eps_growth': 10.5},
            '商社': {'ps': 0.4, 'peg': 0.9, 'eps_growth': 10.2},
            '小売': {'ps': 0.8, 'peg': 1.4, 'eps_growth': 11.1},
            'サービス': {'ps': 2.2, 'peg': 1.7, 'eps_growth': 12.1},
            'ゲーム': {'ps': 3.5, 'peg': 1.4, 'eps_growth': 12.3},
            '化学': {'ps': 1.0, 'peg': 1.4, 'eps_growth': 9.1},
            'その他': {'ps': 1.5, 'peg': 1.5, 'eps_growth': 10.0},
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
            print("📊 セクター平均: メモリキャッシュ")
            return self.sector_cache

        cache_file = CACHE_DIR / "sector_averages.json"
        if cache_file.exists() and not force_refresh:
            try:
                j = json.loads(cache_file.read_text(encoding="utf-8"))
                if time.time() - j.get("timestamp", 0) <= 86400:
                    self.sector_cache = j.get("data", {})
                    self.cache_timestamp = time.time()
                    print("📊 セクター平均: ファイルキャッシュ")
                    return self.sector_cache
            except Exception:
                pass

        print("📊 セクター平均: 静的デフォルト")
        sectors = ['自動車','半導体','電気機器','銀行','情報・通信業','医薬品','商社','小売','サービス','ゲーム','化学','その他']
        data = {s: self.default_sector_average(s) for s in sectors}
        cache_file.write_text(json.dumps({"timestamp": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
        self.sector_cache = data
        self.cache_timestamp = time.time()
        return data

    def load_or_download_data_v2(self, endpoint: str, cache_name: str, bypass_cache: bool = False) -> pd.DataFrame:
        """当日CSVキャッシュ→API→CSV保存。bypass_cache=True なら当日キャッシュを無視して取り直す。"""
        try:
            today = datetime.date.today().strftime("%Y%m%d")
            cache_file = CACHE_DIR / f"{cache_name}_{today}.csv"

            if cache_file.exists() and not bypass_cache:
                try:
                    df = pd.read_csv(cache_file)
                    if not df.empty:
                        return df
                except Exception:
                    pass

            url = f"{JQUANTS_API_BASE}/{endpoint}"
            res = self.session.get(url, timeout=30)
            if res.status_code != 200:
                return pd.DataFrame()

            response_data = res.json()
            keys = ["info", "daily_quotes", "statements", "data", "results", "items", "companies", "stocks"]
            data = None
            for k in keys:
                if k in response_data and response_data[k]:
                    data = response_data[k]
                    break
            if data is None:
                data = response_data

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                try:
                    df.to_csv(cache_file, index=False)
                except Exception:
                    pass
                return df

            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

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
            cache_file = CACHE_DIR / f"sector_stock_list_{today}.csv"
            if cache_file.exists() and not force_refresh:
                return pd.read_csv(cache_file)

            print("📋 銘柄リスト取得…")
            df = self.load_or_download_data_v2("listed/info", "sector_listed_info")
            if not df.empty:
                if "Code" in df.columns:
                    df["Code"] = df["Code"].astype(str).str.extract(r"(\d{4})", expand=False)
                    df = df.dropna(subset=["Code"])
                    df = df[df["Code"].str.isdigit()].drop_duplicates("Code")
                df = enhance_stock_list_with_sectors(df)
                df.to_csv(cache_file, index=False)
                return df

            fb = pd.DataFrame(self.get_fallback_stock_list_v2())
            fb = enhance_stock_list_with_sectors(fb)
            fb.to_csv(cache_file, index=False)
            return fb
        except Exception:
            fb = pd.DataFrame(self.get_fallback_stock_list_v2())
            fb = enhance_stock_list_with_sectors(fb)
            return fb

    def calculate_sector_averages_from_cache(self, max_samples_per_sector: int = 100) -> dict:
        try:
            tasks = build_offline_analysis_tasks(self.session)
            if not tasks:
                print("📊 セクター平均計算: キャッシュデータが不足しています")
                return {}

            print(f"📊 セクター平均計算: {len(tasks)}銘柄から計算中...")
            results = []
            max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(
                        analyze_single_stock_complete_v3,
                        self.session, {}, code, name, market, sector,
                        offline=True
                    ) for (code, name, market, sector) in tasks[:max_samples_per_sector * 20]
                ]
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    if res.get("success") and res.get("ps_ratio") is not None:
                        results.append(res)
                    if i % 100 == 0:
                        print(f"  ⏱ {i}/{len(futs)} 完了 (有効データ={len(results)})")

            if not results:
                print("📊 セクター平均計算: 有効なデータがありません")
                return {}

            df = pd.DataFrame([
                {
                    "sector": self.normalize_sector(r.get("sector_name") or ""),
                    "ps": r.get("ps_ratio"),
                    "peg": r.get("peg_ratio"),
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
                peg_values = sector_df["peg"].dropna()
                per_values = sector_df["per"].dropna()

                sector_stats[sector] = {
                    "ps": float(ps_values.median()) if len(ps_values) > 0 else None,
                    "peg": float(peg_values.median()) if len(peg_values) > 0 else None,
                    "per": float(per_values.median()) if len(per_values) > 0 else None,
                    "sample_count": len(sector_df),
                    "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_source": "calculated_from_cache"
                }

            print(f"📊 セクター平均計算完了: {len(sector_stats)}セクター")
            return sector_stats

        except Exception as e:
            print(f"⚠️ セクター平均計算エラー: {e}")
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

    def get_stock_list_v2(self, force_refresh: bool = False) -> pd.DataFrame:
        helper = DynamicSectorAverages(self.session)
        return helper.get_stock_list_v2(force_refresh=force_refresh)

    def load_or_download_data_v2(self, endpoint: str, cache_name: str) -> pd.DataFrame:
        helper = DynamicSectorAverages(self.session)
        return helper.load_or_download_data_v2(endpoint, cache_name)

    def _load_json_cached(self, endpoint: str, cache_name: str, ttl_hours: int = 24) -> dict:
        f = self.cache_dir / f"{cache_name}.json"
        if f.exists():
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            if (datetime.datetime.now() - mtime).total_seconds() < ttl_hours * 3600:
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass

        url = f"{self.base_url}/{endpoint}"
        try:
            res = self.session.get(url, timeout=30)
            if res.status_code == 200:
                data = res.json()
                try:
                    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                return data
            return {}
        except Exception:
            return {}

    def fetch_statements(self, code: str, force_refresh: bool = False) -> List[dict]:
        cache_key = f"fins_statements_{code}"
        if not force_refresh:
            cached = self._load_json_cached(f"fins/statements?code={code}", cache_key, ttl_hours=12)
            if cached and cached.get("statements"):
                return cached["statements"]

        url = f"{self.base_url}/fins/statements?code={code}"
        for attempt in range(1, 6):
            resp = self.session.get(url, timeout=30)
            status = resp.status_code
            try:
                data = resp.json()
                stmts = data.get("statements", [])
            except Exception:
                stmts = []
                data = {}

            if status == 200:
                try:
                    (self.cache_dir / f"{cache_key}.json").write_text(
                        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8"
                    )
                except Exception:
                    pass
                return stmts

            if status in (402, 403):
                return []

            if status == 429:
                time.sleep(2 ** attempt)
                continue

            if status >= 500:
                time.sleep(1.5 * attempt)
                continue

        return []

    def _fill_missing_fields(self, fin: dict) -> dict:
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
    return df[["Code", "CompanyName", "Sector33Name", "MarketCode"]]

# ------------------------------------------------------------
# 銘柄名フィルタ
# ------------------------------------------------------------
def check_company_name_validity(company_name: str) -> Tuple[bool, str]:
    if not company_name:
        return True, "OK"
    etf_keywords = [
        'ＥＴＦ','ETF','上場投信','インデックスファンド','連動型上場投信',
        '上場インデックス','TOPIX','日経225','投資法人','リート','REIT'
    ]
    if any(k in company_name for k in etf_keywords):
        return False, "ETF/投信"
    fund_company_keywords = ['アセットマネジメント','投信']
    if any(k in company_name for k in fund_company_keywords):
        return False, "投信会社商品"
    return True, "OK"

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
) -> List[dict]:
    if not stmts:
        return []
    filtered_stmts = []
    for stmt in stmts:
        disclosed = _statement_disclosed_date(stmt)
        if as_of_date is not None and disclosed is not None and disclosed > as_of_date:
            continue
        filtered_stmts.append(stmt)
    source_stmts = filtered_stmts if as_of_date is not None else stmts
    if not source_stmts:
        return []
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
        if rec["revenue"] and gross_profit and rec["revenue"] != 0:
            rec["gross_profit_margin"] = gross_profit / rec["revenue"]

        history.append(rec)
        if len(history) >= max_years:
            break
    return history

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

def compute_sales_cagr(history: List[dict], years: int = 3) -> Optional[float]:
    if not history or len(history) <= years:
        return None
    latest = history[0].get("revenue")
    past = history[years].get("revenue") if len(history) > years else None
    if not latest or not past or past <= 0 or latest <= 0:
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
        scores["no_speculative_drop"] = 2.5
        total_score += 2.5

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
    def nz(x, default=0.0):
        if x in (None, "", "NA") or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
            return default
        try:
            if isinstance(x, str):
                x = x.replace(",", "").strip()
            return float(x)
        except (TypeError, ValueError):
            return default
    def safe_ratio(a, b):
        a, b = nz(a, 0.0), nz(b, 0.0)
        return a / b if b != 0 else 0.0

    cur  = {k: nz(v, None) for k, v in fin.get("current", {}).items()}
    prev = {k: nz(v, None) for k, v in fin.get("previous", {}).items()}

    comp = {}
    comp["positive_net_income"] = nz(cur.get("net_income")) > 0
    comp["positive_ocf"] = nz(cur.get("operating_cash_flow")) > 0
    comp["ocf_gt_ni"] = nz(cur.get("operating_cash_flow")) > nz(cur.get("net_income"))
    comp["roa_up"] = safe_ratio(cur.get("net_income"), cur.get("total_assets")) > safe_ratio(prev.get("net_income"), prev.get("total_assets"))
    comp["ocf_margin_up"] = safe_ratio(cur.get("operating_cash_flow"), cur.get("revenue")) > safe_ratio(prev.get("operating_cash_flow"), prev.get("revenue"))
    comp["current_ratio_up"] = safe_ratio(cur.get("current_assets"), cur.get("current_liabilities")) > safe_ratio(prev.get("current_assets"), prev.get("current_liabilities"))
    comp["shares_down"] = nz(cur.get("shares_outstanding")) < nz(prev.get("shares_outstanding"))
    comp["gpm_up"] = nz(cur.get("gross_profit_margin")) > nz(prev.get("gross_profit_margin"))
    lev_cur  = safe_ratio(nz(cur.get("total_assets")) - nz(cur.get("equity")), nz(cur.get("total_assets")))
    lev_prev = safe_ratio(nz(prev.get("total_assets")) - nz(prev.get("equity")), nz(prev.get("total_assets")))
    comp["leverage_down"] = lev_cur < lev_prev

    score = int(sum(bool(v) for v in comp.values()))
    evaluation = ("優秀" if score >= 7 else "良好" if score >= 5 else "普通" if score >= 3 else "注意")
    return {"score": score, "details": comp, "evaluation": evaluation, "mode": "real"}

# ------------------------------------------------------------
# バリュエーション（モックなし）
# ------------------------------------------------------------
def calculate_ps_ratio(current_price: Optional[float], revenue_per_share: Optional[float]=None,
                       market_cap: Optional[float]=None, revenue: Optional[float]=None) -> Optional[float]:
    try:
        if current_price and revenue_per_share and revenue_per_share > 0:
            return float(current_price) / float(revenue_per_share)
        if market_cap and revenue and revenue > 0:
            return float(market_cap) / float(revenue)
        return None
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

def estimate_eps_growth_rate(net_income_current: Optional[float],
                             net_income_previous: Optional[float],
                             shares_outstanding: Optional[float]) -> Optional[float]:
    try:
        if not all(v is not None for v in [net_income_current, net_income_previous, shares_outstanding]):
            return None
        if shares_outstanding <= 0:
            return None
        eps_cur = float(net_income_current) / float(shares_outstanding)
        eps_prev = float(net_income_previous) / float(shares_outstanding)
        if eps_prev <= 0:
            return None
        return (eps_cur / eps_prev - 1.0) * 100.0
    except Exception:
        return None

def calculate_valuation_metrics_ps_peg(current_price: Optional[float],
                                       net_income_current: Optional[float],
                                       net_income_previous: Optional[float],
                                       revenue_current: Optional[float],
                                       shares_outstanding: Optional[float]) -> dict:
    rps = None
    if revenue_current and shares_outstanding and shares_outstanding > 0:
        rps = revenue_current / shares_outstanding

    per = None
    if net_income_current and shares_outstanding and shares_outstanding > 0:
        eps = net_income_current / shares_outstanding
        if eps > 0 and current_price and current_price > 0:
            per = current_price / eps

    eps_growth = estimate_eps_growth_rate(net_income_current, net_income_previous, shares_outstanding)
    ps_ratio = calculate_ps_ratio(current_price, revenue_per_share=rps)
    peg_ratio = calculate_peg_ratio(per, eps_growth)
    return {
        "revenue_per_share": rps,
        "per": per,
        "ps_ratio": ps_ratio,
        "eps_growth_rate": eps_growth,
        "peg_ratio": peg_ratio,
        "peg_trusted": (peg_ratio is not None)
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

    margin_score = w['margin_ratio'] * (0.6 if margin_ratio is None else 1.0 if margin_ratio<=3 else 0.8 if margin_ratio<=5 else 0.6 if margin_ratio<=10 else 0.3 if margin_ratio<=20 else 0)
    short_score  = w['short_selling'] * (0.6 if short_selling_change_rate is None else 1.0 if short_selling_change_rate<=5 else 0.8 if short_selling_change_rate<=15 else 0.5 if short_selling_change_rate<=30 else 0.2 if short_selling_change_rate<=50 else 0)
    safety_score += margin_score + short_score
    details['信用安全性'] = f"{'不明' if margin_ratio is None else f'{margin_ratio:.1f}倍'} ({margin_score:.1f})"
    details['空売り安全性'] = f"{'不明' if short_selling_change_rate is None else f'{short_selling_change_rate:.1f}%'} ({short_score:.1f})"

    eps_score = w['earnings_stability'] * (0.5 if yoy_eps_growth is None else 1.0 if yoy_eps_growth>=20 else 0.8 if yoy_eps_growth>=10 else 0.7 if yoy_eps_growth>=0 else 0.4 if yoy_eps_growth>=-10 else 0.2 if yoy_eps_growth>=-20 else 0)
    div_score = w['dividend_stability'] * (0.5 if not dividend_status else 1.0 if dividend_status=='増配' else 0.8 if dividend_status=='維持' else 0.3 if dividend_status=='未定' else 0.1 if dividend_status=='減配' else 0)
    safety_score += eps_score + div_score
    details['業績安定性'] = f"{'不明' if yoy_eps_growth is None else f'EPS成長率{yoy_eps_growth:.1f}%'} ({eps_score:.1f})"
    details['配当安定性'] = f"{dividend_status or '不明'} ({div_score:.1f})"

    volume_score = w['liquidity'] * (0.5 if avg_volume is None else 1.0 if avg_volume>=500000 else 0.8 if avg_volume>=200000 else 0.6 if avg_volume>=100000 else 0.3 if avg_volume>=50000 else 0)
    if avg_trading_value is not None and np.isfinite(avg_trading_value):
        adv_score = 1.0 if avg_trading_value >= 1_000_000_000 else 0.8 if avg_trading_value >= 500_000_000 else 0.6 if avg_trading_value >= 300_000_000 else 0.3 if avg_trading_value >= 100_000_000 else 0.0
        volume_score = w['liquidity'] * max(volume_score / max(w['liquidity'], 1e-9), adv_score)
    safety_score += volume_score
    liq_note = '不明'
    if avg_volume is not None:
        liq_note = f'{avg_volume:,}株'
    if avg_trading_value is not None and np.isfinite(avg_trading_value):
        liq_note += f' / ADV{avg_trading_value/1e6:.0f}MJPY'
    details['流動性'] = f"{liq_note} ({volume_score:.1f})"

    stagnant_score = w['momentum_stability'] * (0.6 if stagnant_days_after_spike is None else 1.0 if stagnant_days_after_spike==0 else 0.8 if stagnant_days_after_spike<=2 else 0.5 if stagnant_days_after_spike<=4 else 0.2 if stagnant_days_after_spike<=6 else 0)
    if current_volatility is not None and average_volatility not in (None, 0):
        vr = current_volatility / average_volatility
        vol_score = w['volatility_stability'] * (1.0 if vr<=1.2 else 0.8 if vr<=1.5 else 0.5 if vr<=2.0 else 0.2 if vr<=2.5 else 0)
        vol_note = f"{vr:.1f}倍"
    else:
        vol_score = w['volatility_stability'] * 0.6
        vol_note = "不明"
    safety_score += stagnant_score + vol_score
    details['モメンタム安定性'] = f"{'不明' if stagnant_days_after_spike is None else f'{stagnant_days_after_spike}日'} ({stagnant_score:.1f})"
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

    ratio = safety_score / max_total_score
    level = "🟢 非常に安全" if ratio>=0.8 else "🔵 安全" if ratio>=0.6 else "🟡 普通" if ratio>=0.4 else "🟠 やや危険" if ratio>=0.2 else "🔴 危険"
    return {"total_score": round(safety_score,1), "max_score": max_total_score, "safety_level": level, "details": details}

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
    return {"score": score, "level": level, "warning_flags": flags, "risk_factors": risks, "max_score": 100}

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
                                     offline: bool = False) -> dict:
    try:
        fdm = FinancialDataManager(session)
        sector_raw = sector_hint or DynamicSectorAverages.get_sector_static(code)
        sector = DynamicSectorAverages.normalize_sector(sector_raw)
        fc = FrozenCache()

        # 価格
        if offline:
            price_df = fc.load_prices(code)
        else:
            price_df = fdm.load_or_download_data_v2(build_prices_endpoint(code), f"prices_{code}")
        if price_df is None or price_df.empty:
            return {"stock_code": code, "company_name": name, "sector_name": sector, "success": False, "error": "price_missing"}

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

        # 財務
        if offline:
            stmts = fc.load_statements(code)
        else:
            stmts = fdm.fetch_statements(code)

        financial_history = build_financial_history_from_statements(
            stmts if isinstance(stmts, list) else [],
            max_years=5,
            as_of_date=latest_price_date,
        )
        cur_fin = financial_history[0].copy() if financial_history else {}
        prv_fin = financial_history[1].copy() if len(financial_history) > 1 else {}
        raw_fin = {"current": cur_fin.copy(), "previous": prv_fin.copy(), "current_price": current_price, "sector": sector}
        fin = fdm._fill_missing_fields(copy.deepcopy(raw_fin))
        imputation = fin.get("_imputation", {}) if isinstance(fin, dict) else {}

        sector_benchmark = (
            sector_averages.get(sector)
            or sector_averages.get(sector_raw or "")
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
        )
        safety = calculate_safety_score_v3(
            yoy_eps_growth=val.get("eps_growth_rate"),
            avg_volume=avg_volume,
            avg_trading_value=adv_jpy_20d,
            current_volatility=cur_vol, average_volatility=avg_vol,
            below_ma25=below_ma25, below_ma75=below_ma75
        )
        spec = detect_speculative_manipulation_v2(
            yoy_eps_growth=val.get("eps_growth_rate"),
            avg_volume=avg_volume,
            current_volatility=cur_vol, average_volatility=avg_vol,
            below_ma25=below_ma25, below_ma75=below_ma75,
            current_price=current_price, mas=mas, stock_code=code
        )

        shares_outstanding = raw_fin["current"].get("shares_outstanding")
        market_cap = None
        if current_price is not None and shares_outstanding not in (None, 0):
            market_cap = current_price * shares_outstanding

        max_dd = calculate_max_drawdown(close, lookback_days=LOOKBACK_DAYS) if len(close) > 0 else None
        sales_cagr = compute_sales_cagr(financial_history, years=3) if financial_history else None
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

        defensive_ps_ok = (ps_ratio is not None and np.isfinite(ps_ratio) and ps_ratio <= MAX_PS_DEFENSIVE)
        per_satellite = (per is not None and np.isfinite(per) and per > MAX_PER_CORE)
        per_core_ok = (per is not None and np.isfinite(per) and per <= MAX_PER_CORE)

        op_income_eval = evaluate_operating_income_stability(
            financial_history,
            years=OP_INCOME_YEARS,
            drop_floor=OP_INCOME_DROP_FLOOR,
            exclude_deficit=EXCLUDE_OP_INCOME_DEFICIT
        )
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
        diagnostics = {
            "latest_price_date": latest_price_date.isoformat() if latest_price_date else None,
            "statement_disclosed_date": statement_disclosed_date,
            "statement_type": statement_type,
            "statement_staleness_days": statement_staleness_days,
            "imputation": imputation,
            "critical_missing_count": critical_missing,
            "raw_financial_fields": {
                "has_current_shares": raw_fin["current"].get("shares_outstanding") is not None,
                "has_previous_net_income": raw_fin["previous"].get("net_income") is not None,
                "has_current_assets": raw_fin["current"].get("current_assets") is not None,
                "has_current_liabilities": raw_fin["current"].get("current_liabilities") is not None,
                "has_gross_profit_margin": raw_fin["current"].get("gross_profit_margin") is not None,
            },
        }

        base_ok = bool(liquidity_ok and market_cap_ok and op_income_stable)
        core_candidate = bool(base_ok and defensive_ps_ok and per_core_ok)
        satellite_candidate = bool(base_ok and (not core_candidate))

        filter_details = {
            "liquidity_ok": liquidity_ok,
            "market_cap_ok": market_cap_ok,
            "defensive_ps_ok": defensive_ps_ok,
            "per_satellite": per_satellite,
            "per_core_ok": per_core_ok,
            "op_income_stable": op_income_stable,
            "op_income_reason": op_income_eval.get("reason"),
            "base_ok": base_ok,
            "core_candidate": core_candidate,
            "satellite_candidate": satellite_candidate,
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
            "ps_ratio": ps_ratio, "peg_ratio": val.get("peg_ratio"), "per": per,
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
        }
    except Exception as e:
        return {"stock_code": code, "company_name": name, "sector_name": sector_hint or "その他", "error": f"{e}", "success": False}

# ------------------------------------------------------------
# 収集（単体/全体）
# ------------------------------------------------------------
def collect_one_code(session: requests.Session, code: str, name: str = "", *, force_refresh: bool = False) -> bool:
    fc = FrozenCache()
    helper = DynamicSectorAverages(session)
    try:
        # 価格
        price_df = helper.load_or_download_data_v2(
            build_prices_endpoint(code),
            f"prices_{code}",
            bypass_cache=force_refresh
        )
        if price_df is not None and not price_df.empty:
            fc.save_prices(code, price_df)

        # 財務
        fdm = FinancialDataManager(session)
        stmts = fdm.fetch_statements(code, force_refresh=force_refresh)
        if stmts:
            fc.save_statements(code, stmts)

        return fc.has_all(code)
    except RuntimeError as e:
        if "日次レート制限到達" in str(e):
            raise
        return False
    except Exception:
        return False

PENDING_FILE = CACHE_DIR / "pending_codes.json"

def _save_pending(codes: list[str]) -> None:
    PENDING_FILE.write_text(json.dumps({"codes": codes}, ensure_ascii=False), encoding="utf-8")

def _load_pending(df: pd.DataFrame, *, force_full: bool = False, refresh_days: Optional[int] = None) -> list[str]:
    fc = FrozenCache()
    if force_full:
        codes = [str(c) for c in df["Code"].astype(str)]
        _save_pending(codes)
        return codes

    if refresh_days is not None:
        codes = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c), max_age_days=refresh_days)]
        _save_pending(codes)
        return codes

    if PENDING_FILE.exists():
        try:
            return json.loads(PENDING_FILE.read_text(encoding="utf-8")).get("codes", [])
        except Exception:
            pass

    codes = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c))]
    _save_pending(codes)
    return codes

def collect_all_daemon(session: requests.Session,
                       daily_budget: Optional[int] = None,
                       refresh_days: Optional[int] = None,
                       force_full: bool = False,
                       reset_pending: bool = False) -> None:
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = df[df.apply(lambda r: check_company_name_validity(str(r.get("CompanyName","")))[0], axis=1)].reset_index(drop=True)

    if reset_pending and PENDING_FILE.exists():
        try:
            PENDING_FILE.unlink()
        except Exception:
            pass

    pending = _load_pending(df, force_full=force_full, refresh_days=refresh_days)
    if not pending:
        print("📦 すでに全件取得済み")
        return

    if daily_budget is None:
        rpd = int(os.getenv("JQ_RPD", "800"))
        daily_budget = max(1, min(len(pending), rpd // 2 - 5))

    mode = "強制再収集" if force_full else (f"{refresh_days}日超のみ再収集" if refresh_days is not None else "未取得のみ")
    print(f"▶ 全自動収集開始  残り{len(pending)}銘柄  日次上限目安={daily_budget}銘柄/日  モード={mode}")

    while pending:
        taken = 0
        start = time.time()
        try:
            for code in list(pending):
                if taken >= daily_budget:
                    break
                ok = collect_one_code(session, code, force_refresh=(force_full or refresh_days is not None))
                if ok:
                    pending.remove(code)
                    _save_pending(pending)
                taken += 1
                if taken % 20 == 0 or taken == daily_budget:
                    elapsed = time.time() - start
                    print(f"  ⏱ 本日 {taken}/{daily_budget} 件  残り{len(pending)}  経過{int(elapsed)}s", flush=True)
        except RuntimeError as e:
            if "日次レート制限到達" in str(e):
                pass
            else:
                raise

        print(f"📦 今日の収集バッチ終了: {taken}件  残り{len(pending)}件")
        if not pending:
            print("✅ 全銘柄の凍結収集が完了")
            break

        wait_sec = seconds_until_next_day()
        h, rem = divmod(wait_sec, 3600)
        m, s = divmod(rem, 60)
        print(f"⏳ 日次上限回復待ち: {h}h{m}m{s}s 待機")
        time.sleep(wait_sec)

def collect_batch(session: requests.Session, max_codes: int) -> dict:
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = df[df.apply(lambda r: check_company_name_validity(str(r.get("CompanyName","")))[0], axis=1)].reset_index(drop=True)
    fc = FrozenCache()

    pending = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c))]
    picked  = pending[:max_codes]
    ok = 0
    fail = 0
    start = time.time()

    for i, code in enumerate(picked, 1):
        ok_flag = collect_one_code(session, code)
        if ok_flag:
            ok += 1
        else:
            fail += 1
        if i % 20 == 0 or i == len(picked):
            elapsed = time.time() - start
            print(f"  ⏱ {i}/{len(picked)} 収集中 (OK={ok} FAIL={fail}) 経過{elapsed:.0f}s", flush=True)

    return {"tried": len(picked), "ok": ok, "fail": fail}

# ------------------------------------------------------------
# オフライン分析タスク生成 / 銘柄名取得
# ------------------------------------------------------------
def build_offline_analysis_tasks(session: requests.Session) -> list[tuple[str, str, str, str | None]]:
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
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
        "per_satellite": _safe_bool(f.get("per_satellite")),
        "op_income_stable": _safe_bool(f.get("op_income_stable")),
        "op_income_reason": f.get("op_income_reason"),
        "base_ok": _safe_bool(f.get("base_ok")),
        "core_candidate": _safe_bool(f.get("core_candidate")),
        "satellite_candidate": _safe_bool(f.get("satellite_candidate")),
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
        "sector": d.get("sector_name"),
        "price": d.get("current_price"),
        "ps": d.get("ps_ratio"),
        "peg": d.get("peg_ratio"),
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
        "safety": saf.get("total_score"),
        "safety_level": saf.get("safety_level"),
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
        "sector_peg_benchmark": sector_benchmark.get("peg"),
        "sector_benchmark_source": sector_benchmark.get("data_source"),
        "statement_type": diagnostics.get("statement_type"),
        "statement_disclosed_date": diagnostics.get("statement_disclosed_date"),
        "statement_staleness_days": diagnostics.get("statement_staleness_days"),
        "price_asof_date": diagnostics.get("latest_price_date"),
        "has_imputation": imputation.get("has_imputation"),
        "imputed_field_count": imputation.get("field_count"),
        "imputed_fields": ",".join(sorted((imputation.get("fields") or {}).keys())),
        "critical_missing_count": diagnostics.get("critical_missing_count"),
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
        print(f"🗑️ 古いレポート {total}件を削除しました ({outdir})")

def write_candidate_sets(flat: pd.DataFrame, outdir: Path, timestamp: Optional[str] = None) -> list[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    ok = flat[flat["ok"] == True].copy()
    if ok.empty:
        return []

    core = ok[ok["core_candidate"] == True].copy()
    sat = ok[ok["satellite_candidate"] == True].copy()
    exc = ok[(ok["base_ok"] != True) | (ok["op_income_stable"] != True) | (ok["liquidity_ok"] != True) | (ok["market_cap_ok"] != True)].copy()

    # 固定名で上書き（ストレージ節約）
    p_core = outdir / "core_candidates.csv"
    p_sat  = outdir / "satellite_candidates.csv"
    p_exc  = outdir / "excluded.csv"
    core.to_csv(p_core, index=False, encoding="utf-8-sig")
    sat.to_csv(p_sat, index=False, encoding="utf-8-sig")
    exc.to_csv(p_exc, index=False, encoding="utf-8-sig")

    summary = {
        "total_ok": int(len(ok)),
        "core": int(len(core)),
        "satellite": int(len(sat)),
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
    return [p_core, p_sat, p_exc, p_sum]

def write_reports(flat: pd.DataFrame, outdir: Path, topn: int = 10, timestamp: Optional[str] = None) -> list[Path]:
    outdir.mkdir(exist_ok=True, parents=True)
    ok = flat[flat["ok"] == True].copy()
    if ok.empty:
        return []

    # 固定名で上書き（ストレージ節約）
    for c in ["safety","piot","spec_score","per","peg","ps","rsi","adx","safety_criteria_score","market_cap","avg_volume_30d"]:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    core = ok[ok["core_candidate"] == True].copy()
    sat = ok[ok["satellite_candidate"] == True].copy()
    ranked_core = _build_ranked(core) if not core.empty else core
    ranked_sat = _build_ranked(sat) if not sat.empty else sat

    outs: list[Path] = []

    if not ranked_core.empty:
        p1 = outdir / "top_recommended_core.csv"
        ranked_core.sort_values(by=["total_score"], ascending=[False]).head(topn).to_csv(p1, index=False, encoding="utf-8-sig")
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
        ranked_sat.sort_values(by=["total_score"], ascending=[False]).head(topn).to_csv(p6, index=False, encoding="utf-8-sig")
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

    rec = _build_ranked(core).sort_values(by=["total_score"], ascending=[False]).head(topn)

    lines = ["# おすすめトップテン（Core候補）", ""]
    if timestamp:
        lines.append(f"**生成日時:** {timestamp.replace('_', ' ')}")
        lines.append("")
    lines.append(f"**フィルタ:** avg_volume_30d>={MIN_AVG_VOLUME_30D}, ADV20>={MIN_ADV_JPY_20D:,}JPY, market_cap>={MIN_MARKET_CAP_JPY:,}JPY, PS<={MAX_PS_DEFENSIVE}, PER<={MAX_PER_CORE}, 営業利益安定")
    lines.append("")
    for _, r in rec.iterrows():
        mc = r.get("market_cap")
        mc_str = f"{mc/1e9:.1f}B" if pd.notna(mc) else "N/A"
        vol = r.get("avg_volume_30d")
        vol_str = f"{int(vol):,}" if pd.notna(vol) else "N/A"
        lines.append(
            f"- **{r['code']} {r['name']}** | 総合 {r['total_score']:.1f} | 財務 {r['financial_score']:.1f} | 仕手 {r['spec_score']} | "
            f"PER {r['per']} | PS {r['ps']} | 時価総額 {mc_str} | 出来高(30d) {vol_str}"
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
    if below_ma200 is True:
        score = max(0.0, score - 3.0)
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
    for c in [
        "ps", "peg", "per", "rsi", "adx", "piot", "safety", "spec_score", "below_ma200",
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
    df["valuation_peg"] = df["peg"].apply(_val_score_from_peg)
    df["valuation_score"] = df["valuation_ps"] + df["valuation_peg"]                     # 0-20

    df["quality_piotroski"] = df["piot"].apply(_quality_score_from_piotroski)
    df["quality_op_income"] = np.where(df.get("op_income_stable", False) == True, 8.0, 0.0)
    df["quality_growth"] = df["sales_cagr"].apply(_growth_score_from_sales_cagr)
    df["financial_score"] = df["quality_piotroski"] + df["quality_op_income"] + df["quality_growth"]  # 0-30

    df["resilience_safety_criteria"] = df["safety_criteria_score"].fillna(0.0).clip(lower=0, upper=100) * (12.0 / 100.0)
    df["resilience_drawdown"] = df["max_drawdown"].apply(_resilience_score_from_drawdown)
    df["resilience_score"] = df["resilience_safety_criteria"] + df["resilience_drawdown"]  # 0-20

    bm200 = df.get("below_ma200", pd.Series([np.nan] * len(df)))
    df["technical_score"] = [
        _momentum_score_from_returns(r21, r63, r126, m61, m63, m31, (False if pd.isna(x) else bool(x)))
        for r21, r63, r126, m61, m63, m31, x in zip(
            df["return_21d"], df["return_63d"], df["return_126d"],
            df["momentum_6m_1m"], df["momentum_6m_3m"], df["momentum_3m_1m"], bm200
        )
    ]  # 0-22

    safety_base = df["safety"].fillna(0.0).clip(lower=0, upper=25.0)
    adv_bonus = np.where(df["adv_jpy_20d"] >= 1_000_000_000, 2.0, np.where(df["adv_jpy_20d"] >= 500_000_000, 1.0, 0.0))
    df["safety_score_scaled"] = (safety_base * (10.0 / 25.0) + adv_bonus).clip(lower=0, upper=12.0)  # 0-12

    df["spec_penalty"] = df["spec_score"].fillna(0.0).clip(lower=0, upper=100) * (8.0 / 100.0)  # 0-8
    df["per_penalty"] = np.where(df["per"].notna() & (df["per"] > MAX_PER_CORE), 4.0, 0.0)
    df["data_penalty"] = [
        _data_quality_penalty(imputed, missing, stale)
        for imputed, missing, stale in zip(
            df["imputed_field_count"],
            df["critical_missing_count"],
            df["statement_staleness_days"],
        )
    ]

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

    now = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
    n = len(ranked)
    avg_score = ranked["total_score"].mean()
    grade_counts = ranked["grade"].value_counts().reindex(["A+","A","B+","B","C"]).fillna(0).astype(int)

    ps_avg = ranked["ps_vs_sector"].mean(skipna=True)
    ps_med = ranked["ps_vs_sector"].median(skipna=True)
    ps_min = ranked["ps_vs_sector"].min(skipna=True)
    ps_max = ranked["ps_vs_sector"].max(skipna=True)

    peg_avg = ranked["peg"].mean(skipna=True)
    peg_med = ranked["peg"].median(skipna=True)
    peg_min = ranked["peg"].min(skipna=True)
    peg_max = ranked["peg"].max(skipna=True)

    top = ranked.sort_values("total_score", ascending=False).head(topn)

    lines: list[str] = []
    lines.append("# 🏆 PS・PEGレシオ対応 投資銘柄スクリーニング レポート（Core候補）")
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
    lines.append("### PEGレシオ")
    lines.append(f"- 平均: {peg_avg:.2f}")
    lines.append(f"- 中央値: {peg_med:.2f}")
    lines.append(f"- 最小: {peg_min:.2f}")
    lines.append(f"- 最大: {peg_max:.2f}")
    lines.append("")
    lines.append(f"## 🏆 投資推奨 Top{topn}銘柄（Core）")
    lines.append("")
    lines.append("| 順位 | 銘柄コード | 銘柄名 | グレード | スコア | セクター | PS比 | PEG | ピオトロスキー |")
    lines.append("|------|------------|--------|----------|--------|----------|------|-----|---------------|")
    for i, r in enumerate(top.itertuples(index=False), 1):
        ps_vs = 0 if pd.isna(r.ps_vs_sector) else r.ps_vs_sector
        peg = 0 if pd.isna(r.peg) else r.peg
        lines.append(f"| {i} | {r.code} | {r.name} | {r.grade} | {r.total_score:.1f} | {r.sector} | "
                     f"{ps_vs:.2f} | {peg:.2f} | {r.pio_disp} |")
    lines.append("")
    lines.append(f"## 📊 詳細分析（上位{details_n}銘柄）")
    lines.append("")
    detail_df = ranked.sort_values("total_score", ascending=False).head(details_n)
    for i, r in enumerate(detail_df.itertuples(index=False), 1):
        lines.append(f"### {i}. [{r.code}] {r.name} ({r.sector})")
        lines.append(f"**総合スコア:** {r.total_score:.1f}点 | **グレード:** {r.grade}")
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
        print("=== メニュー ===")
        print("1) 収集（価格+財務を凍結保存）")
        print("2) オフライン一括分析（core/satellite/excluded 出力）")
        print("3) 単銘柄分析（キャッシュ使用）")
        print("4) セクター平均を更新（キャッシュから計算）")
        print("5) 全銘柄ゆっくり収集（自動待機・再開可）")
        print("6) 鮮度で取り直し収集（例: 7日より古いものだけ）")
        print("7) 全銘柄“強制”再収集（pending初期化＋当日再取得）")
        print("q) 終了")
        choice = input("選択: ").strip().lower()

        if choice == "1":
            budget = input("本日収集する銘柄数（推奨380）: ").strip()
            budget = int(budget) if budget.isdigit() else 380
            s = collect_batch(session, budget)
            print(f"📦 収集: tried={s['tried']} ok={s['ok']} fail={s['fail']}")

        elif choice == "2":
            tasks = build_offline_analysis_tasks(session)
            if not tasks:
                print("キャッシュ不足。先に収集を実行してください。")
                continue

            results = []
            max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(
                    analyze_single_stock_complete_v3,
                    session, sector_avgs, code, name, market, sector,
                    offline=True
                ) for (code, name, market, sector) in tasks]
                for i, fut in enumerate(as_completed(futs), 1):
                    results.append(fut.result())
                    if i % 200 == 0 or i == len(futs):
                        ok_cnt = sum(1 for r in results if r.get("success"))
                        print(f"  ⏱ {i}/{len(futs)} 完了 (OK={ok_cnt})")

            flat = pd.DataFrame([_flatten_result(r) for r in results])
            master = outdir / "screening_offline.csv"
            flat.to_csv(master, index=False, encoding="utf-8-sig")

            outputs = generate_reports_from_master_csv(master, outdir, topn=30)
            print(f"✅ 出力: {master}")
            print(f"✅ 出力先: {outdir}")
            print("=== 生成物 ===")
            for p in outputs:
                print(f"  - {p}")

        elif choice == "3":
            code = input("銘柄コード4桁: ").strip()
            name = lookup_company_name(session, code)
            res = analyze_single_stock_complete_v3(session, sector_avgs, code, name=name, offline=True)
            df = pd.DataFrame([_flatten_result(res)])
            fp = outdir / f"single_{code}.csv"
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            cleanup_old_report_files(outdir)
            cleanup_old_report_files(outdir.parent)
            print(f"✅ 出力: {fp}")

        elif choice == "4":
            print("📊 セクター平均をキャッシュから計算中...")
            sector_avgs_obj = DynamicSectorAverages(session)
            updated_avgs = sector_avgs_obj.calculate_sector_averages_from_cache()
            if updated_avgs:
                cache_file = CACHE_DIR / "sector_averages.json"
                sectors = ['自動車','半導体','電気機器','銀行','情報・通信業','医薬品','商社','小売','サービス','ゲーム','化学','その他']
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
                print(f"✅ セクター平均を更新しました（{len(updated_avgs)}セクター）")
                for sector, stats in updated_avgs.items():
                    ps_val = stats.get('ps', None)
                    peg_val = stats.get('peg', None)
                    sample_count = stats.get('sample_count', 0)
                    ps_str = f"{ps_val:.2f}" if ps_val is not None else "N/A"
                    peg_str = f"{peg_val:.2f}" if peg_val is not None else "N/A"
                    print(f"  {sector}: PS={ps_str}, PEG={peg_str}, サンプル数={sample_count}")
            else:
                print("⚠️ セクター平均の計算に失敗しました。キャッシュデータが不足している可能性があります。")

        elif choice == "5":
            budget = input("1日あたりの最大収集銘柄数（既定380）: ").strip()
            budget = int(budget) if budget.isdigit() else 380
            collect_all_daemon(session, daily_budget=budget)

        elif choice == "6":
            days = input("何日より古ければ取り直すか（日数。例: 7）: ").strip()
            days = int(days) if days.isdigit() else 7
            budget = input("1日あたりの最大収集銘柄数（既定380）: ").strip()
            budget = int(budget) if budget.isdigit() else 380
            collect_all_daemon(session, daily_budget=budget, refresh_days=days, reset_pending=True)

        elif choice == "7":
            budget = input("1日あたりの最大収集銘柄数（既定380）: ").strip()
            budget = int(budget) if budget.isdigit() else 380
            collect_all_daemon(session, daily_budget=budget, force_full=True, reset_pending=True)

        elif choice == "q":
            break

        else:
            print("無効な選択")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",
                        choices=["collect","collect_all","analyze","single","interactive"],
                        default="interactive")
    parser.add_argument("--code")
    parser.add_argument("--budget", type=int, default=380)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--reset-pending", action="store_true")
    parser.add_argument("--refresh-days", type=int)
    parser.add_argument("--force-full", action="store_true")
    args = parser.parse_args()

    if args.phase == "interactive":
        run_interactive()
        return

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
        return

    if args.phase == "collect":
        s = collect_batch(session, args.budget)
        print(f"📦 収集: tried={s['tried']} ok={s['ok']} fail={s['fail']}")
        return

    if args.phase == "single":
        if not args.code:
            raise SystemExit("--code 必須")
        name = lookup_company_name(session, args.code)
        res = analyze_single_stock_complete_v3(session, sector_avgs, args.code, name=name, offline=True)
        df = pd.DataFrame([_flatten_result(res)])
        fp = outdir / f"single_{args.code}.csv"
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        cleanup_old_report_files(outdir)
        cleanup_old_report_files(outdir.parent)
        print(f"✅ 単銘柄出力: {fp}")
        return

    if args.phase == "analyze":
        tasks = build_offline_analysis_tasks(session)
        if not tasks:
            print("キャッシュ不足。先に --phase collect か collect_all を実行してください。")
            return

        results = []
        max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(
                analyze_single_stock_complete_v3,
                session, sector_avgs, code, name, market, sector,
                offline=True
            ) for (code, name, market, sector) in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 200 == 0 or i == len(futs):
                    ok_cnt = sum(1 for r in results if r.get("success"))
                    print(f"  ⏱ {i}/{len(futs)} 完了 (OK={ok_cnt})")

        flat = pd.DataFrame([_flatten_result(r) for r in results])
        master = outdir / "screening_offline.csv"
        flat.to_csv(master, index=False, encoding="utf-8-sig")

        outputs = generate_reports_from_master_csv(master, outdir, topn=max(10, args.top))
        print(f"✅ オフライン分析出力: {master}")
        print(f"✅ 出力先: {outdir}")
        print("=== 生成物 ===")
        for p in outputs:
            print(f"  - {p}")

if __name__ == "__main__":
    main()
