# -*- coding: utf-8 -*-
"""
J-Quants 収集→凍結キャッシュ→完全オフライン分析ワークフロー
- 4日運用: 収集(価格+財務)を日次800req内で進め、4日目にオフライン一括分析
- モック不使用: オフライン時は“計算不能はNone”で返す（推定やランダムは行わない）
- 端末対話メニュー付き（引数未指定で起動するとメニュー表示）
- CLI対応:
    収集:   python script.py --phase collect --budget 380
    分析:   python script.py --phase analyze --top 10
    単銘柄: python script.py --phase single --code 8035
環境変数:
    JQ_RPM=50  JQ_RPD=800  # 必要なら調整
必要: pandas, numpy, requests
"""

import os
import re
import sys
import json
import time
import math
import signal
import logging
import datetime
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Iterable, Sequence
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from math import isfinite
import yaml

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
THEME_CONFIG_PATH = Path("config") / "theme_tags.yaml"

AGING_DX_PROFILE_NAME = "aging_dx_alpha"
AGING_DX_THEME_FILTER = {"AGING", "NURSING_CARE", "HOME_MEDICAL", "MEDICAL_DX", "SENIOR_LIFE"}
AGING_DX_PRIORITIZED_BUSINESS_MODELS = {"COST_REDUCTION_INFRA", "DATA_PLATFORM"}
BUSINESS_MODEL_WEIGHTS = {
    "COST_REDUCTION_INFRA": 2.0,
    "DATA_PLATFORM": 2.0,
    "SERVICE_PROVIDER": 1.0,
    "POLICY_DEPENDENT": -2.0,
    "ROBOTICS_CORE": 1.5,
    "OTHER": 0.0,
}
AGING_DX_MIN_PIOTROSKI = 7
AGING_DX_MIN_POLICY_STRESS = 2
AGING_DX_MIN_MARKET_CAP = 100 * 1e8    # 100億円
AGING_DX_MAX_MARKET_CAP = 3000 * 1e8   # 3000億円
AGING_DX_MIN_DAILY_TRADING_VALUE = 0.12 * 1e9  # 0.12億円
AGING_DX_MAX_PS_RATIO = 10.0
AGING_DX_SCORE_WEIGHTS = {
    "f_score": 0.35,
    "growth": 0.25,
    "policy": 0.25,
    "moat": 0.15,
}
AGING_DX_OUTPUT_COLUMNS = [
    "code",
    "name",
    "theme_tags",
    "business_model",
    "policy_stress_score",
    "sales_CAGR_3y",
    "F_score",
    "ps_ratio",
    "per",
    "peg",
    "market_cap",
    "avg_trading_value",
    "total_score",
]

# ヘルパ（先頭のimport群の下あたり）
def seconds_until_next_day(buffer_sec: int = 10) -> int:
    now = datetime.datetime.now()
    tomorrow = now + datetime.timedelta(days=1)
    reset = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((reset - now).total_seconds()) + buffer_sec)

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
        print(f"[警告] 銘柄 {code} の収集でエラー: {e}")
        return False
    except Exception as e:
        print(f"[警告] 銘柄 {code} の収集で予期しないエラー: {type(e).__name__}: {e}")
        return False


PENDING_FILE = CACHE_DIR / "pending_codes.json"

def _load_pending(df: pd.DataFrame, *, force_full: bool = False, refresh_days: Optional[int] = None) -> list[str]:
    fc = FrozenCache()
    # 明示指定があれば優先
    if force_full:
        codes = [str(c) for c in df["Code"].astype(str)]
        _save_pending(codes)
        return codes
    if refresh_days is not None:
        codes = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c), max_age_days=refresh_days)]
        _save_pending(codes)
        return codes
    # 既存pendingがあれば継続
    if PENDING_FILE.exists():
        try:
            return json.loads(PENDING_FILE.read_text(encoding="utf-8")).get("codes", [])
        except Exception:
            pass
    # 通常初期化（未取得のみ）
    codes = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c))]
    _save_pending(codes)
    return codes


def _save_pending(codes: list[str]) -> None:
    PENDING_FILE.write_text(json.dumps({"codes": codes}, ensure_ascii=False), encoding="utf-8")

def collect_all_daemon(session: requests.Session,
                       daily_budget: Optional[int] = None,
                       refresh_days: Optional[int] = None,
                       force_full: bool = False,
                       reset_pending: bool = False) -> None:
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = df[df.apply(lambda r: check_company_name_validity(str(r.get("CompanyName","")))[0], axis=1)].reset_index(drop=True)

    # pending 初期化オプション
    if reset_pending and PENDING_FILE.exists():
        try:
            PENDING_FILE.unlink()
        except Exception:
            pass

    pending = _load_pending(df, force_full=force_full, refresh_days=refresh_days)
    if not pending:
        print("📦 すでに全件取得済み"); return

    # 1銘柄=価格+財務で概ね2リク。日次800→余裕をみて 380/日
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
            print("[OK] 全銘柄の凍結収集が完了"); break

        wait_sec = seconds_until_next_day()
        h, rem = divmod(wait_sec, 3600)
        m, s = divmod(rem, 60)
        print(f"⏳ 日次上限回復待ち: {h}h{m}m{s}s 待機")
        time.sleep(wait_sec)


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
        print(f"\n[警告] 中断シグナル受信: {signum}\n[停止] 安全に終了します")
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
        self.errs = 0

    def wait_if_needed(self):
        now = datetime.datetime.now()
        if now.date() > self.last_reset:
            self.daily_count = 0
            self.last_reset = now.date()
        if self.daily_count >= self.requests_per_day:
            error_msg = f"日次レート制限到達: {self.daily_count}/{self.requests_per_day} リクエスト使用済み"
            print(f"[エラー] {error_msg}")
            raise RuntimeError(error_msg)
        one_minute_ago = now - datetime.timedelta(minutes=1)
        self.request_timestamps = [t for t in self.request_timestamps if t > one_minute_ago]
        if len(self.request_timestamps) >= self.requests_per_minute:
            wait = 61 - (now - min(self.request_timestamps)).total_seconds()
            if wait > 0:
                print(f"[待機] 分間レート制限: {wait:.1f}秒待機中...")
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
            except requests.RequestException as e:
                if attempt == MAX:
                    print(f"[エラー] リクエスト失敗: {method} {url} - {type(e).__name__}: {e}")
                    raise
                print(f"[警告] リクエストエラー (試行 {attempt}/{MAX}): {type(e).__name__}: {e}")
                time.sleep(1.5 * attempt)
                continue

            # 401→トークン更新
            if resp.status_code == 401 and attempt == 1:
                print(f"[警告] 認証エラー (401) 検出。トークンを更新します...")
                try:
                    _refresh_id_token(self, ini_file=self.ini_file)
                    print("[OK] トークン更新成功")
                    continue
                except Exception as e:
                    error_msg = f"idToken refresh failed: {e}"
                    print(f"[エラー] {error_msg}")
                    raise RuntimeError(error_msg) from e

            # レート or サーバ
            if resp.status_code in (429,) or resp.status_code >= 500:
                error_body = ""
                try:
                    error_body = resp.text[:200]
                except:
                    pass
                if attempt == MAX:
                    print(f"[エラー] APIエラー: {resp.status_code} - {error_body}")
                    return resp
                print(f"[警告] APIエラー (試行 {attempt}/{MAX}): {resp.status_code} - {error_body}")
                time.sleep(min(2 ** attempt, 30))
                continue

            # その他のエラーステータス
            if resp.status_code >= 400:
                error_body = ""
                try:
                    error_body = resp.text[:200]
                except:
                    pass
                print(f"[警告] HTTP {resp.status_code}: {error_body}")

            self.limiter.mark()
            return resp
        raise RuntimeError(f"{method} {url} failed after {MAX} attempts")

def get_authenticated_session_jquants(ini_file="api.ini") -> requests.Session:
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
                print("[OK] キャッシュidTokenを使用")
                return session
        except Exception:
            pass

    print("[認証] 認証開始…")
    _refresh_id_token(session, ini_file=ini_file)
    print("[OK] 認証成功")
    return session

def _refresh_id_token(session: requests.Session, ini_file="api.ini") -> str:
    config = configparser.ConfigParser()
    if not Path(ini_file).exists():
        raise RuntimeError(f"設定ファイルが見つかりません: {ini_file}")
    config.read(ini_file, encoding="utf-8")
    email = (config["DEFAULT"].get("MAIL_ADDRESS") or
             config["DEFAULT"].get("mail_address") or
             config["DEFAULT"].get("email"))
    password = (config["DEFAULT"].get("PASSWORD") or
                config["DEFAULT"].get("password"))
    if not (email and password):
        raise RuntimeError("メールアドレス／パスワード未設定")

    try:
        auth_payload = {"mailaddress": email, "password": password}
        res = requests.post(f"{JQUANTS_API_BASE}/token/auth_user", json=auth_payload, timeout=20)
        if res.status_code != 200:
            print(f"[エラー] 認証リクエスト失敗: {res.status_code} - {res.text[:200]}")
        res.raise_for_status()
        refresh_token = res.json().get("refreshToken")
        if not refresh_token:
            raise RuntimeError("refreshToken取得失敗")

        tok_res = requests.post(f"{JQUANTS_API_BASE}/token/auth_refresh?refreshtoken={refresh_token}", timeout=20)
        if tok_res.status_code != 200:
            print(f"[エラー] トークン更新リクエスト失敗: {tok_res.status_code} - {tok_res.text[:200]}")
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
    except requests.RequestException as e:
        raise RuntimeError(f"認証API接続エラー: {type(e).__name__}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"認証処理エラー: {type(e).__name__}: {e}") from e

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
        "情報・通信業":{"ca_ratio": 0.57, "cl_ratio": 0.32, "gpm": 0.34},
        "サービス": {"ca_ratio": 0.60, "cl_ratio": 0.35, "gpm": 0.29},
        "化学":     {"ca_ratio": 0.58, "cl_ratio": 0.37, "gpm": 0.27},
        "その他":   {"ca_ratio": 0.60, "cl_ratio": 0.40, "gpm": 0.25},
    }
    def __init__(self, session: requests.Session):
        self.session = session
        self.sector_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 3600

    @staticmethod
    def get_sector_static(stock_code: str) -> str:
        sector_mapping = {
            '7203': '自動車','7267':'自動車','7269':'自動車','7270':'自動車','7261':'自動車','7202':'自動車','7211':'自動車',
            '8035':'半導体','6861':'半導体','6594':'半導体','6503':'半導体','6723':'半導体','6752':'半導体','6981':'半導体',
            '6758':'エレクトロニクス','6501':'エレクトロニクス','6954':'エレクトロニクス','6702':'エレクトロニクス','6976':'エレクトロニクス',
            '8306':'銀行','8316':'銀行','8411':'銀行','8331':'銀行','8354':'銀行','8393':'銀行',
            '9984':'通信','9432':'通信','9433':'通信','4689':'通信','3659':'通信','4751':'通信',
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

    def is_cache_valid(self):
        if not self.cache_timestamp:
            return False
        return (time.time() - self.cache_timestamp) < self.cache_duration

    def get_default_sector_average(self, sector):
        defaults = {
            '自動車': {'ps': 0.8, 'peg': 1.2, 'eps_growth': 8.5},
            '半導体': {'ps': 4.5, 'peg': 1.8, 'eps_growth': 12.2},
            'エレクトロニクス': {'ps': 1.8, 'peg': 1.5, 'eps_growth': 12.3},
            '銀行': {'ps': 2.5, 'peg': 0.8, 'eps_growth': 10.6},
            '通信': {'ps': 1.2, 'peg': 1.3, 'eps_growth': 11.2},
            '医薬品': {'ps': 3.8, 'peg': 1.6, 'eps_growth': 10.5},
            '商社': {'ps': 0.4, 'peg': 0.9, 'eps_growth': 10.2},
            '小売': {'ps': 0.8, 'peg': 1.4, 'eps_growth': 11.1},
            'サービス': {'ps': 2.2, 'peg': 1.7, 'eps_growth': 12.1},
            'ゲーム': {'ps': 3.5, 'peg': 1.4, 'eps_growth': 12.3},
            '化学': {'ps': 1.0, 'peg': 1.4, 'eps_growth': 9.1},
            'その他': {'ps': 1.5, 'peg': 1.5, 'eps_growth': 10.0}
        }
        default = defaults.get(sector, defaults['その他'])
        return {
            **default,
            'sample_count': 0,
            'last_updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': 'static_default'
        }

    def get_sector_averages(self, force_refresh=False):
        if not force_refresh and self.is_cache_valid() and self.sector_cache:
            print("[分析] セクター平均: メモリキャッシュ")
            return self.sector_cache
        cache_file = CACHE_DIR / "sector_averages.json"
        if cache_file.exists() and not force_refresh:
            try:
                j = json.loads(cache_file.read_text(encoding="utf-8"))
                if time.time() - j.get("timestamp", 0) <= 86400:
                    self.sector_cache = j.get("data", {})
                    self.cache_timestamp = time.time()
                    print("[分析] セクター平均: ファイルキャッシュ")
                    return self.sector_cache
            except Exception:
                pass
        print("[分析] セクター平均: 静的デフォルト")
        sectors = ['自動車','半導体','エレクトロニクス','銀行','通信','医薬品','商社','小売','サービス','ゲーム','化学','その他']
        data = {s: self.get_default_sector_average(s) for s in sectors}
        cache_file.write_text(json.dumps({"timestamp": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
        self.sector_cache = data
        self.cache_timestamp = time.time()
        return data

    def load_or_download_data_v2(self, endpoint, cache_name, bypass_cache: bool = False):
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


    def get_fallback_stock_list_v2(self):
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

    def load_or_download_data_v2(self, endpoint, cache_name):
        helper = DynamicSectorAverages(self.session)
        return helper.load_or_download_data_v2(endpoint, cache_name)

    def _load_json_cached(self, endpoint: str, cache_name: str, ttl_hours: int = 24):
        f = self.cache_dir / f"{cache_name}.json"
        if f.exists():
            mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            if (datetime.datetime.now() - mtime).total_seconds() < ttl_hours * 3600:
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        return json.load(fp)
                except Exception:
                    pass
        url = f"{self.base_url}/{endpoint}"
        try:
            res = self.session.get(url, timeout=30)
            if res.status_code == 200:
                data = res.json()
                with open(f, "w", encoding="utf-8") as fp:
                    json.dump(data, fp, ensure_ascii=False)
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
                    with open(self.cache_dir / f"{cache_key}.json", "w", encoding="utf-8") as fp:
                        json.dump(data, fp, ensure_ascii=False, separators=(",", ":"))
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
        # セクタ中央値での軽微な補完のみ（モック生成はしない）
        to_f = lambda v: float(v) if v not in (None, "", "NA") else None
        cur, prev = fin["current"], fin["previous"]
        for fld in ("current_assets", "current_liabilities", "gross_profit_margin", "shares_outstanding"):
            if cur.get(fld) is None and prev.get(fld) is not None:
                cur[fld] = prev[fld]
        sector = fin.get("sector", "その他")
        med = DynamicSectorAverages.SECTOR_MEDIANS.get(sector, DynamicSectorAverages.SECTOR_MEDIANS["その他"])
        ca_ratio = med.get("ca_ratio")
        cl_ratio = med.get("cl_ratio")
        gpm_med  = med.get("gpm")
        if (cur.get("current_assets") is None and cur.get("total_assets") and ca_ratio):
            cur["current_assets"] = cur["total_assets"] * ca_ratio
        if (cur.get("current_liabilities") is None and cur.get("total_assets") and cur.get("equity") and cl_ratio):
            cur["current_liabilities"] = (cur["total_assets"] - cur["equity"]) * cl_ratio
        if cur.get("gross_profit_margin") is None and gpm_med:
            cur["gross_profit_margin"] = gpm_med * 0.95
        for k, v in cur.items():
            fin[f"current_{k}"] = v
        return fin

# ------------------------------------------------------------
# ユーティリティ
# ------------------------------------------------------------
def build_prices_endpoint(stock_code: str, lookback_days: int = LOOKBACK_DAYS) -> str:
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return f"prices/daily_quotes?code={stock_code}&from={start}"

def enhance_stock_list_with_sectors(df: pd.DataFrame) -> pd.DataFrame:
    if "Code" not in df.columns:
        return df
    if "Sector33Name" not in df.columns:
        df["Sector33Name"] = df["Code"].astype(str).map(DynamicSectorAverages.get_sector_static).fillna("その他")
    if "MarketCode" not in df.columns:
        df["MarketCode"] = ""
    if "CompanyName" not in df.columns:
        df["CompanyName"] = ""
    return df[["Code","CompanyName","Sector33Name","MarketCode"]]

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

def calculate_adx_and_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[float,float,float]:
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
    plus_dm  = (high.diff().where(lambda x: x > 0, 0.0))
    minus_dm = (-low.diff().where(lambda x: x < 0, 0.0))
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    def clamp(x, lo, hi): 
        return float(max(lo, min(hi, x))) if np.isfinite(x) else float(lo)
    return clamp(adx.iloc[-1],5,80), clamp(plus_di.iloc[-1],5,95), clamp(minus_di.iloc[-1],5,95)

def calculate_moving_averages(prices: pd.Series, periods=[25,75,200]) -> Dict[str, float]:
    out = {}
    for p in periods:
        if len(prices) >= p:
            ma = prices.rolling(window=p).mean().iloc[-1]
            out[f"ma_{p}"] = float(ma) if np.isfinite(ma) else float(prices.iloc[-1])
        elif len(prices):
            out[f"ma_{p}"] = float(prices.iloc[-1])
        else:
            out[f"ma_{p}"] = None
    return out

# 置き換え: calculate_volatility 全体
def calculate_volatility(prices: pd.Series, period: int = 20) -> Tuple[Optional[float], Optional[float]]:
    if len(prices) < max(5, period):
        return None, None
    try:
        returns = prices.pct_change(fill_method=None).dropna()
    except TypeError:
        # 古いpandas互換
        returns = prices.pct_change().dropna()
    cur = returns.tail(period).std() * np.sqrt(252) if len(returns) >= period else returns.std() * np.sqrt(252)
    avg = returns.std() * np.sqrt(252)
    return float(cur), float(avg)


# ------------------------------------------------------------
# Piotroski（実データのみ）
# ------------------------------------------------------------
def calculate_piotroski_real(fin: dict) -> dict:
    def nz(x, default=0.0):
        if x in (None, "", "NA") or (isinstance(x, float) and (np.isnan(x) or not np.isfinite(x))):
            return default
        return float(x)
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
    try:
        if per is None or eps_growth_rate_pct is None:
            return None
        if per <= 0 or eps_growth_rate_pct <= 0:
            return None
        return float(per) / float(eps_growth_rate_pct)
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
# テーマタグ・ファンダメンタル補助
# ------------------------------------------------------------


@dataclass
class ThemeInfo:
    code: str
    name: str
    theme_tags: list[str]
    business_model: str


def load_theme_tags(path: str | Path = THEME_CONFIG_PATH) -> dict[str, ThemeInfo]:
    """
    YAML定義を読み込み、証券コード→ThemeInfoの辞書を返す。
    """
    target = Path(path)
    if not target.exists():
        logger.warning("theme_tags.yaml が見つかりません: %s", target)
        return {}
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.error("theme_tags.yaml の読み込みに失敗: %s", exc)
        return {}
    result: dict[str, ThemeInfo] = {}
    for raw_code, meta in payload.items():
        code = str(raw_code).zfill(4)
        name = str(meta.get("name") or "")
        tags = meta.get("theme_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        theme_tags = sorted({str(tag).strip().upper() for tag in tags if tag})
        business_model = str(meta.get("business_model") or "OTHER").strip().upper()
        result[code] = ThemeInfo(
            code=code,
            name=name,
            theme_tags=list(theme_tags),
            business_model=business_model if business_model else "OTHER",
        )
    return result


def _pick_numeric_field(record: dict, keys: Sequence[str]) -> Optional[float]:
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


def build_financial_history_from_statements(stmts: List[dict], max_years: int = 5) -> List[dict]:
    """
    財務諸表リストから最大 max_years 件の整形済み辞書を生成する（新しい年度順）。
    """
    if not stmts:
        return []
    sorted_stmts = sorted(stmts, key=_fiscal_year_from_statement, reverse=True)
    history: list[dict] = []
    for stmt in sorted_stmts:
        rec = {
            "fiscal_year": _fiscal_year_from_statement(stmt),
            "revenue": _pick_numeric_field(stmt, ["NetSales", "Revenue", "OperatingRevenue"]),
            "operating_income": _pick_numeric_field(stmt, ["OperatingIncome", "OperatingIncomeLoss", "OperatingProfit"]),
            "net_income": _pick_numeric_field(stmt, ["NetIncomeLoss", "Profit", "ProfitAttributableToOwnersOfParent", "NetIncome"]),
            "operating_cash_flow": _pick_numeric_field(
                stmt, ["NetCashProvidedByUsedInOperatingActivities", "CashFlowsFromOperatingActivities"]
            ),
            "total_assets": _pick_numeric_field(stmt, ["TotalAssets"]),
            "equity": _pick_numeric_field(stmt, ["EquityAttributableToOwnersOfParent", "Equity", "NetAssets"]),
            "current_assets": _pick_numeric_field(stmt, ["CurrentAssets"]),
            "current_liabilities": _pick_numeric_field(stmt, ["CurrentLiabilities"]),
            "gross_profit_margin": None,
            "shares_outstanding": _pick_numeric_field(
                stmt,
                [
                    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
                    "NumberOfIssuedAndOutstandingShares",
                ],
            ),
        }
        gross_profit = _pick_numeric_field(stmt, ["GrossProfit"])
        if rec["revenue"] and gross_profit and rec["revenue"] != 0:
            rec["gross_profit_margin"] = gross_profit / rec["revenue"]
        interest_bearing = _pick_numeric_field(
            stmt,
            [
                "InterestBearingDebt",
                "InterestBearingLiabilities",
                "ShortTermBorrowings",
                "LongTermBorrowings",
                "Loans",
            ],
        )
        if interest_bearing is None:
            total_liabilities = _pick_numeric_field(stmt, ["TotalLiabilities"])
            if total_liabilities is not None:
                interest_bearing = total_liabilities
            elif rec["total_assets"] is not None and rec["equity"] is not None:
                interest_bearing = max(rec["total_assets"] - rec["equity"], 0.0)
        rec["interest_bearing_debt"] = interest_bearing
        if rec["revenue"] and rec["revenue"] > 0 and rec["operating_income"] is not None:
            rec["operating_margin"] = rec["operating_income"] / rec["revenue"]
        else:
            rec["operating_margin"] = None
        history.append(rec)
        if len(history) >= max_years:
            break
    return history


def compute_sales_cagr(history: Sequence[dict], years: int = 3) -> Optional[float]:
    """
    history（最新順）から売上CAGRを算出する。
    """
    revenues = [rec.get("revenue") for rec in history if rec.get("revenue")]
    if len(revenues) <= years:
        return None
    latest = revenues[0]
    past = revenues[years]
    if not past or past <= 0 or not latest or latest <= 0:
        return None
    try:
        periods = years
        return (latest / past) ** (1 / periods) - 1
    except (ZeroDivisionError, OverflowError):
        return None


def compute_policy_stress_score(history: Sequence[dict]) -> int:
    """
    財政・政策ストレス環境で耐えられるかを0-4点で評価する。
    """
    records = list(history)
    if not records:
        return 0
    score = 0
    recent = records[:4]
    sales_cagr = compute_sales_cagr(records, years=3)
    if sales_cagr is not None and sales_cagr >= 0.08:
        score += 1

    margins = [rec.get("operating_margin") for rec in recent[:3] if rec.get("operating_margin") is not None]
    if len(margins) == 3 and all(m > 0 for m in margins) and margins[0] >= margins[-1]:
        score += 1

    ocf = [rec.get("operating_cash_flow") for rec in recent[:3] if rec.get("operating_cash_flow") is not None]
    if len(ocf) == 3 and all(val > 0 for val in ocf):
        score += 1

    latest = records[0]
    equity = latest.get("equity")
    debt = latest.get("interest_bearing_debt")
    if equity and equity > 0 and debt is not None:
        if debt / equity < 0.5:
            score += 1
    elif equity and equity > 0 and latest.get("total_assets") and latest["total_assets"] - equity <= 0:
        score += 1
    return score

# ------------------------------------------------------------
# 安全性・投機性
# ------------------------------------------------------------
def calculate_safety_score_v3(
    margin_ratio: float = None,
    short_selling_change_rate: float = None,
    yoy_eps_growth: float = None,
    dividend_status: str = None,
    avg_volume: int = None,
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

    # 信用・空売り
    margin_score = w['margin_ratio'] * (0.6 if margin_ratio is None else 1.0 if margin_ratio<=3 else 0.8 if margin_ratio<=5 else 0.6 if margin_ratio<=10 else 0.3 if margin_ratio<=20 else 0)
    short_score  = w['short_selling'] * (0.6 if short_selling_change_rate is None else 1.0 if short_selling_change_rate<=5 else 0.8 if short_selling_change_rate<=15 else 0.5 if short_selling_change_rate<=30 else 0.2 if short_selling_change_rate<=50 else 0)
    safety_score += margin_score + short_score
    details['信用安全性'] = f"{'不明' if margin_ratio is None else f'{margin_ratio:.1f}倍'} ({margin_score:.1f})"
    details['空売り安全性'] = f"{'不明' if short_selling_change_rate is None else f'{short_selling_change_rate:.1f}%'} ({short_score:.1f})"

    # 業績・配当
    eps_score = w['earnings_stability'] * (0.5 if yoy_eps_growth is None else 1.0 if yoy_eps_growth>=20 else 0.8 if yoy_eps_growth>=10 else 0.7 if yoy_eps_growth>=0 else 0.4 if yoy_eps_growth>=-10 else 0.2 if yoy_eps_growth>=-20 else 0)
    div_score = w['dividend_stability'] * (0.5 if not dividend_status else 1.0 if dividend_status=='増配' else 0.8 if dividend_status=='維持' else 0.3 if dividend_status=='未定' else 0.1 if dividend_status=='減配' else 0)
    safety_score += eps_score + div_score
    details['業績安定性'] = f"{'不明' if yoy_eps_growth is None else f'EPS成長率{yoy_eps_growth:.1f}%'} ({eps_score:.1f})"
    details['配当安定性'] = f"{dividend_status or '不明'} ({div_score:.1f})"

    # 流動性
    volume_score = w['liquidity'] * (0.5 if avg_volume is None else 1.0 if avg_volume>=500000 else 0.8 if avg_volume>=200000 else 0.6 if avg_volume>=100000 else 0.3 if avg_volume>=50000 else 0)
    safety_score += volume_score
    details['流動性'] = f"{'不明' if avg_volume is None else f'{avg_volume:,}株'} ({volume_score:.1f})"

    # モメンタム・ボラ
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

    # テクニカル
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
        elif margin_ratio >= 20: score += 15; flags.append(f"[警告] 信用倍率高: {margin_ratio:.1f}倍")
        elif margin_ratio >= 10: score += 8; risks.append(f"信用倍率やや高: {margin_ratio:.1f}倍")
    if short_selling_change_rate is not None:
        if short_selling_change_rate >= 100: score += 20; flags.append(f"🚨 空売り残急増: +{short_selling_change_rate:.1f}%")
        elif short_selling_change_rate >= 50: score += 12; flags.append(f"[警告] 空売り残増加: +{short_selling_change_rate:.1f}%")
        elif short_selling_change_rate >= 25: score += 6; risks.append(f"空売り残やや増加: +{short_selling_change_rate:.1f}%")
    if stagnant_days_after_spike is not None:
        if stagnant_days_after_spike >= 5: score += 15; flags.append(f"[下落] 急騰後の横ばい: {stagnant_days_after_spike}日")
        elif stagnant_days_after_spike >= 3: score += 8; risks.append(f"横ばい傾向: {stagnant_days_after_spike}日")
    if current_volatility is not None and average_volatility not in (None, 0):
        vr = current_volatility / average_volatility
        if vr >= 3.0: score += 20; flags.append(f"🚨 ボラティリティ異常: {vr:.1f}倍")
        elif vr >= 2.0: score += 12; flags.append(f"[警告] ボラティリティ高: {vr:.1f}倍")
        elif vr >= 1.5: score += 6; risks.append(f"ボラティリティやや高: {vr:.1f}倍")
    if below_ma25 and below_ma75: score += 8; flags.append("[警告] 25・75日線の両方割れ")
    elif below_ma25 or below_ma75: score += 4; risks.append("移動平均線の一部割れ")
    if avg_volume is not None and avg_volume < 30000: score += 8; flags.append(f"[警告] 流動性低: {avg_volume:,}株/日")
    if dividend_status in {"未定","減配"}: score += 6; risks.append(f"配当{dividend_status}")
    if yoy_eps_growth is not None and yoy_eps_growth < -30: score += 8; flags.append(f"[警告] EPS急減: {yoy_eps_growth:.1f}%")

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
        sector = sector_hint or DynamicSectorAverages.get_sector_static(code)
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
                    if k == c.lower(): return v
            return None

        c_close = _col(price_df, "Close","ClosePrice","EndPrice","AdjustmentClose","AdjClose")
        c_high  = _col(price_df, "High","HighPrice")
        c_low   = _col(price_df, "Low","LowPrice")
        c_vol   = _col(price_df, "Volume","TradingVolume")
        c_date  = _col(price_df, "Date","TradingDate")
        if c_date: price_df = price_df.sort_values(c_date)

        close = price_df[c_close].astype(float) if c_close in price_df.columns else pd.Series([], dtype=float)
        high  = price_df[c_high].astype(float)  if c_high  in price_df.columns else close
        low   = price_df[c_low].astype(float)   if c_low   in price_df.columns else close
        vol_s = price_df[c_vol].astype(float)   if c_vol   in price_df.columns else None

        current_price = float(close.iloc[-1]) if len(close) else None
        mas = calculate_moving_averages(close) if len(close) else {}
        rsi = float(calculate_rsi(close)) if len(close) else None
        adx, plus_di, minus_di = calculate_adx_and_di(high, low, close) if len(close) else (None, None, None)
        cur_vol, avg_vol = calculate_volatility(close) if len(close) else (None, None)
        below_ma25 = bool(current_price is not None and mas.get("ma_25") not in (None,) and current_price < mas["ma_25"])
        below_ma75 = bool(current_price is not None and mas.get("ma_75") not in (None,) and current_price < mas["ma_75"])
        avg_volume = int(vol_s.tail(30).mean()) if isinstance(vol_s, pd.Series) and len(vol_s) else None

        # 財務
        if offline:
            stmts = fc.load_statements(code)
        else:
            stmts = fdm.fetch_statements(code)

        financial_history = build_financial_history_from_statements(stmts, max_years=5)
        cur_fin = financial_history[0].copy() if financial_history else {}
        prv_fin = financial_history[1].copy() if len(financial_history) > 1 else {}

        fin = {"current": cur_fin, "previous": prv_fin, "current_price": current_price, "sector": sector}
        fin = fdm._fill_missing_fields(fin)
        if financial_history:
            financial_history[0].update(fin["current"])
        if len(financial_history) > 1:
            financial_history[1].update(fin["previous"])

        # 指標
        piot = calculate_piotroski_real(fin)
        val = calculate_valuation_metrics_ps_peg(
            current_price=current_price,
            net_income_current=fin["current"].get("net_income"),
            net_income_previous=fin["previous"].get("net_income"),
            revenue_current=fin["current"].get("revenue"),
            shares_outstanding=fin["current"].get("shares_outstanding"),
        )
        safety = calculate_safety_score_v3(
            yoy_eps_growth=val.get("eps_growth_rate"),
            avg_volume=avg_volume,
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

        shares_outstanding = fin["current"].get("shares_outstanding")
        market_cap = None
        if current_price is not None and shares_outstanding not in (None, 0):
            market_cap = current_price * shares_outstanding

        return {
            "stock_code": code, "company_name": name, "sector_name": sector,
            "current_price": current_price, "mas": mas, "rsi": rsi, "adx": adx,
            "plus_di": plus_di, "minus_di": minus_di,
            "volatility": cur_vol, "avg_volatility": avg_vol,
            "below_ma25": below_ma25, "below_ma75": below_ma75,
            "piotroski": piot,
            "ps_ratio": val.get("ps_ratio"), "peg_ratio": val.get("peg_ratio"), "per": val.get("per"),
            "revenue_per_share": val.get("revenue_per_share"),
            "safety": safety, "speculation": spec, "success": True,
            "avg_volume_30d": avg_volume,
            "financial_history": financial_history,
            "market_cap": market_cap,
            "shares_outstanding": shares_outstanding,
        }
    except Exception as e:
        return {"stock_code": code, "company_name": name, "sector_name": sector_hint or "その他", "error": f"{e}", "success": False}

def cache_status(session: requests.Session):
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    fc = FrozenCache()
    total = len(df)
    cached = sum(1 for c in df["Code"].astype(str) if fc.has_all(str(c)))
    print(f"📦 キャッシュ {cached}/{total} 銘柄  ({cached/total*100:.1f}%)")



# ------------------------------------------------------------
# 収集フェーズ
# ------------------------------------------------------------


def collect_batch(session: requests.Session, max_codes: int) -> dict:
    fdm = FinancialDataManager(session)
    df = fdm.get_stock_list_v2(force_refresh=False)
    df = df[df.apply(lambda r: check_company_name_validity(str(r.get("CompanyName","")))[0], axis=1)].reset_index(drop=True)
    fc = FrozenCache()
    pending = [str(c) for c in df["Code"].astype(str) if not fc.has_all(str(c))]
    picked  = pending[:max_codes]
    ok = 0; fail = 0
    start = time.time()
    for i, code in enumerate(picked, 1):
        ok_flag = collect_one_code(session, code)
        if ok_flag: ok += 1
        else: fail += 1
        if i % 20 == 0 or i == len(picked):
            elapsed = time.time() - start
            print(f"  ⏱ {i}/{len(picked)} 収集中 (OK={ok} FAIL={fail}) 経過{elapsed:.0f}s", flush=True)
    return {"tried": len(picked), "ok": ok, "fail": fail}


# ------------------------------------------------------------
# レポート出力
# ------------------------------------------------------------
def write_reports(flat: pd.DataFrame, outdir: Path, topn: int = 10) -> None:
    ok = flat[flat["ok"] == True].copy()
    if ok.empty:
        return
    for c in ["safety","piot","spec_score","per","peg","ps","rsi","adx"]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    rec = ok.sort_values(by=["safety","piot","spec_score"], ascending=[False,False,True]).head(topn)
    rec.to_csv(outdir / "top_recommended.csv", index=False, encoding="utf-8-sig")
    ok.sort_values(by=["safety","piot"], ascending=[False,False]).head(topn).to_csv(outdir / "top_safety.csv", index=False, encoding="utf-8-sig")
    ok.sort_values(by=["spec_score"], ascending=False).head(topn).to_csv(outdir / "top_speculative.csv", index=False, encoding="utf-8-sig")
    ok.sort_values(by=["piot","safety"], ascending=[False,False]).head(topn).to_csv(outdir / "top_piotroski.csv", index=False, encoding="utf-8-sig")

def write_markdown_report(flat: pd.DataFrame, outdir: Path, topn: int = 10) -> None:
    ok = flat[flat["ok"] == True].copy()
    if ok.empty: return
    ok["safety"] = pd.to_numeric(ok["safety"], errors="coerce")
    ok["piot"]   = pd.to_numeric(ok["piot"], errors="coerce")
    ok["spec_score"] = pd.to_numeric(ok["spec_score"], errors="coerce")
    rec = ok.sort_values(by=["safety","piot","spec_score"], ascending=[False,False,True]).head(topn)
    lines = ["# おすすめトップテン", ""]
    for _, r in rec.iterrows():
        lines.append(f"- **{r['code']} {r['name']}** | 安全 {r['safety']} | Pio {r['piot']} | 仕手 {r['spec_score']} | PER {r['per']} | PEG {r['peg']} | PS {r['ps']}")
    (outdir / "report_top10.md").write_text("\n".join(lines), encoding="utf-8")

# ==== 投資助言レポート生成 ====

def _grade_from_score(s: float) -> str:
    if s >= 85: return "A+"
    if s >= 75: return "A"
    if s >= 65: return "B+"
    if s >= 55: return "B"
    return "C"

def _val_score_from_ps_vs_sector(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x): return 6.0
    if x <= 0.5: return 12.5
    if x <= 1.0: return 10.0
    if x <= 1.5: return 8.0
    if x <= 2.0: return 5.0
    if x <= 3.0: return 2.0
    return 0.0

def _val_score_from_peg(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x): return 6.0
    if x <= 0.5: return 12.5
    if x <= 1.0: return 10.0
    if x <= 1.5: return 8.0
    if x <= 2.0: return 5.0
    if x <= 3.0: return 2.0
    return 0.0

def _tech_score(rsi: Optional[float], adx: Optional[float]) -> float:
    # RSI 15点 + ADX 10点 = 25点満点
    r = 0.0
    if rsi is not None and np.isfinite(rsi):
        if 45 <= rsi <= 60: r += 15.0
        elif (40 <= rsi < 45) or (60 < rsi <= 70): r += 10.0
        elif (30 <= rsi < 40) or (70 < rsi <= 80): r += 5.0
        else: r += 0.0
    else:
        r += 7.5
    if adx is not None and np.isfinite(adx):
        if 20 <= adx <= 40: r += 10.0
        elif 15 <= adx < 20 or 40 < adx <= 50: r += 6.0
        elif 10 <= adx < 15 or 50 < adx <= 60: r += 3.0
        else: r += 0.0
    else:
        r += 5.0
    return r

def _build_ranked(flat: pd.DataFrame) -> pd.DataFrame:
    df = flat.copy()
    # 数値化
    for c in ["ps","peg","per","rsi","adx","piot","safety","spec_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # セクター別PS中央値
    sec_med = df.groupby("sector")["ps"].median()
    def _ps_vs_sector(row):
        ps = row.get("ps")
        med = sec_med.get(row.get("sector"), np.nan)
        if pd.isna(ps) or pd.isna(med) or med <= 0: return np.nan
        return float(ps) / float(med)
    df["ps_vs_sector"] = df.apply(_ps_vs_sector, axis=1)

    # コンポーネント
    df["valuation_ps"]  = df["ps_vs_sector"].apply(_val_score_from_ps_vs_sector)
    df["valuation_peg"] = df["peg"].apply(_val_score_from_peg)
    df["valuation_score"] = df["valuation_ps"] + df["valuation_peg"]                      # 0-25
    df["safety_score_scaled"] = df["safety"].fillna(12.0) * (20.0/25.0)                  # 0-20
    df["financial_score"] = df["piot"].fillna(4.5) * (22.5/9.0)                          # 0-22.5
    df["technical_score"] = [_tech_score(rsi, adx) for rsi, adx in zip(df["rsi"], df["adx"])]  # 0-25
    df["spec_penalty"] = df["spec_score"].fillna(0.0).clip(lower=0, upper=100) * (10.0/100.0)  # 0-10

    # 総合
    df["total_score"] = (df["valuation_score"] + df["safety_score_scaled"] +
                         df["financial_score"] + df["technical_score"] - df["spec_penalty"])
    df["total_score"] = df["total_score"].clip(lower=0, upper=100)
    df["grade"] = df["total_score"].apply(_grade_from_score)

    # 表示補助
    df["pio_disp"] = df["piot"].fillna(0).astype(int).astype(str) + "/9"
    return df

def write_investment_advice_report(flat: pd.DataFrame, outdir: Path,
                                   topn: int = 15, details_n: int = 30) -> None:
    ok = flat[flat["ok"] == True].copy()
    if ok.empty: return
    ranked = _build_ranked(ok)

    # 概況
    now = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")
    n = len(ranked)
    avg_score = ranked["total_score"].mean()
    grade_counts = ranked["grade"].value_counts().reindex(["A+","A","B+","B","C"]).fillna(0).astype(int)

    # 分布
    ps_avg, ps_med = ranked["ps_vs_sector"].mean(skipna=True), ranked["ps_vs_sector"].median(skipna=True)
    ps_min, ps_max = ranked["ps_vs_sector"].min(skipna=True), ranked["ps_vs_sector"].max(skipna=True)
    peg_avg, peg_med = ranked["peg"].mean(skipna=True), ranked["peg"].median(skipna=True)
    peg_min, peg_max = ranked["peg"].min(skipna=True), ranked["peg"].max(skipna=True)

    # Topテーブル
    top = ranked.sort_values("total_score", ascending=False).head(topn)
    lines = []
    lines.append("# 🏆 PS・PEGレシオ対応 投資銘柄スクリーニング レポート")
    lines.append("")
    lines.append(f"**📅 生成日時:** {now}")
    lines.append(f"**[分析] 分析対象:** {n}銘柄")
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
    lines.append(f"## 🏆 投資推奨 Top{topn}銘柄")
    lines.append("")
    lines.append("| 順位 | 銘柄コード | 銘柄名 | グレード | スコア | セクター | PS比 | PEG | ピオトロスキー |")
    lines.append("|------|------------|--------|----------|--------|----------|------|-----|---------------|")
    for i, r in enumerate(top.itertuples(index=False), 1):
        lines.append(f"| {i} | {r.code} | {r.name} | {r.grade} | {r.total_score:.1f} | {r.sector} | "
                     f"{(0 if pd.isna(r.ps_vs_sector) else r.ps_vs_sector):.2f} | "
                     f"{(0 if pd.isna(r.peg) else r.peg):.2f} | {r.pio_disp} |")
    lines.append("")
    lines.append(f"## [分析] 詳細分析（上位{details_n}銘柄）")
    lines.append("")
    detail_df = ranked.sort_values("total_score", ascending=False).head(details_n)
    for i, r in enumerate(detail_df.itertuples(index=False), 1):
        lines.append(f"### {i}. [{r.code}] {r.name} ({r.sector})")
        lines.append(f"**総合スコア:** {r.total_score:.1f}点 | **グレード:** {r.grade}")
        lines.append("**スコア内訳:**")
        lines.append(f"- バリュエーション: {r.valuation_score:.1f}点")
        lines.append(f"- 安全性: {r.safety_score_scaled:.1f}点")
        lines.append(f"- 財務健全性: {r.financial_score:.1f}点")
        lines.append(f"- テクニカル: {r.technical_score:.1f}点")
        lines.append(f"- 仕手株ペナルティ: {r.spec_penalty:.1f}点")
        lines.append("")

    outdir.mkdir(exist_ok=True)
    (outdir / "ranked_with_scores.csv").write_text(ranked.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8")
    (outdir / "report_investment_advice.md").write_text("\n".join(lines), encoding="utf-8")

def build_offline_analysis_tasks(session: requests.Session) -> list[tuple[str, str, str, str | None]]:
    """
    凍結キャッシュが揃っている銘柄だけを抽出し、(code, name, market, sector_hint) のタスク配列を返す。
    これを使って analyze_single_stock_complete_v3 に“name”を渡す。
    """
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
    fc = FrozenCache()

    # キャッシュが両方あるコードだけ残す
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


def screen_aging_dx_alpha(session: requests.Session,
                          sector_averages: Optional[dict] = None,
                          *,
                          topn: int = 20,
                          theme_config_path: Path | str = THEME_CONFIG_PATH,
                          output_dir: Path = Path("output")) -> pd.DataFrame:
    """
    高齢化×DXテー マ向けの専用スクリーニングを実行し、結果DataFrameを返す。
    """
    theme_map = load_theme_tags(theme_config_path)
    if not theme_map:
        print("[警告] テーマタグ定義が存在しません。config/theme_tags.yaml を確認してください。")
        return pd.DataFrame()

    if sector_averages is None:
        sector_averages = DynamicSectorAverages(session).get_sector_averages()

    tasks = [task for task in build_offline_analysis_tasks(session) if task[0] in theme_map]
    if not tasks:
        print("[警告] aging_dx_alpha に該当するテーマ銘柄がキャッシュに存在しません。")
        return pd.DataFrame()

    def _normalize(value: Optional[float], upper: float) -> float:
        if value is None:
            return 0.0
        return max(0.0, min(1.0, value / upper)) if upper > 0 else 0.0

    results: list[dict] = []
    max_workers = max(4, min(16, (os.cpu_count() or 4) * 2))
    outdir = Path(output_dir); outdir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                analyze_single_stock_complete_v3,
                session, sector_averages, code, name, market, sector,
                offline=True
            ) for (code, name, market, sector) in tasks
        ]
        for fut in as_completed(futures):
            res = fut.result()
            if not res.get("success"):
                continue
            code = res.get("stock_code")
            theme = theme_map.get(str(code).zfill(4))
            if not theme:
                continue
            tags = set(theme.theme_tags)
            if not tags.intersection(AGING_DX_THEME_FILTER):
                continue
            business_model = theme.business_model
            if business_model == "POLICY_DEPENDENT":
                continue

            piot_score = (res.get("piotroski") or {}).get("score")
            if piot_score is None or piot_score < AGING_DX_MIN_PIOTROSKI:
                continue

            history = res.get("financial_history") or []
            if len(history) < 4:
                continue
            policy_score = compute_policy_stress_score(history)
            if policy_score < AGING_DX_MIN_POLICY_STRESS:
                continue
            sales_cagr = compute_sales_cagr(history, years=3)
            if sales_cagr is None or sales_cagr <= 0:
                continue

            ps_ratio = res.get("ps_ratio")
            if ps_ratio is None or not np.isfinite(ps_ratio) or (AGING_DX_MAX_PS_RATIO and ps_ratio > AGING_DX_MAX_PS_RATIO):
                continue
            market_cap = res.get("market_cap")
            if market_cap is None or not np.isfinite(market_cap) or market_cap < AGING_DX_MIN_MARKET_CAP or market_cap > AGING_DX_MAX_MARKET_CAP:
                continue
            current_price = res.get("current_price")
            avg_volume = res.get("avg_volume_30d")
            if current_price is None or not np.isfinite(current_price):
                continue
            if avg_volume in (None, 0) or not np.isfinite(avg_volume):
                continue
            avg_trading_value = current_price * avg_volume
            if avg_trading_value < AGING_DX_MIN_DAILY_TRADING_VALUE:
                continue

            per = res.get("per")
            peg = res.get("peg_ratio")

            norm_f = max(0.0, min(1.0, (piot_score or 0) / 9))
            norm_growth = _normalize(sales_cagr, upper=0.25)
            policy_component = policy_score  # 0-4
            bm_weight = BUSINESS_MODEL_WEIGHTS.get(business_model, 0.0)
            total_score = (
                AGING_DX_SCORE_WEIGHTS["f_score"] * norm_f +
                AGING_DX_SCORE_WEIGHTS["growth"] * norm_growth +
                AGING_DX_SCORE_WEIGHTS["policy"] * policy_component +
                AGING_DX_SCORE_WEIGHTS["moat"] * bm_weight
            )

            results.append({
                "code": code,
                "name": res.get("company_name") or theme.name,
                "theme_tags": ",".join(theme.theme_tags),
                "business_model": business_model,
                "policy_stress_score": policy_score,
                "sales_CAGR_3y": sales_cagr,
                "F_score": piot_score,
                "ps_ratio": ps_ratio,
                "per": per,
                "peg": peg,
                "market_cap": market_cap,
                "avg_trading_value": avg_trading_value,
                "total_score": total_score,
                "current_price": current_price,
                "avg_volume_30d": avg_volume,
            })

    if not results:
        print("[警告] aging_dx_alpha 条件を満たす銘柄がありませんでした。")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("total_score", ascending=False).head(topn)
    outfile = outdir / f"{AGING_DX_PROFILE_NAME}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ordered = df[AGING_DX_OUTPUT_COLUMNS].copy()
    ordered.to_csv(outfile, index=False, encoding="utf-8-sig")
    print(f"[OK] aging_dx_alpha 出力: {outfile}")
    return ordered


def lookup_company_name(session: requests.Session, code: str) -> str:
    """
    単銘柄分析用。コード→CompanyName を株主名簿から解決。
    見つからなければ空文字を返す。
    """
    fdm = FinancialDataManager(session)
    df_list = fdm.get_stock_list_v2(force_refresh=False)
    df_list = df_list.copy()
    df_list["Code"] = df_list["Code"].astype(str)
    hit = df_list[df_list["Code"] == str(code)]
    if not hit.empty:
        return str(hit.iloc[0].get("CompanyName") or "")
    return ""


# ------------------------------------------------------------
# インタラクティブUI / CLI
# ------------------------------------------------------------
def _flatten_result(d: dict) -> dict:
    pio = d.get("piotroski") or {}; saf = d.get("safety") or {}; spc = d.get("speculation") or {}
    return {
        "code": d.get("stock_code"), "name": d.get("company_name"), "sector": d.get("sector_name"),
        "price": d.get("current_price"), "ps": d.get("ps_ratio"), "peg": d.get("peg_ratio"), "per": d.get("per"),
        "rsi": d.get("rsi"), "adx": d.get("adx"),
        "piot": pio.get("score"), "piot_eval": pio.get("evaluation"),
        "safety": saf.get("total_score"), "safety_level": saf.get("safety_level"),
        "spec_score": spc.get("score"), "spec_level": spc.get("level"),
        "ok": d.get("success"), "error": d.get("error"),
    }


def run_interactive():
    session = get_authenticated_session_jquants()
    sector_avgs = DynamicSectorAverages(session).get_sector_averages()
    outdir = Path("output"); outdir.mkdir(exist_ok=True)

    while True:
        print("=== メニュー ===")
        print("1) 収集（価格+財務を凍結保存）")
        print("2) オフライン一括分析（トップ10出力）")
        print("3) 単銘柄分析（キャッシュ使用）")
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
                print("キャッシュ不足。先に収集を実行してください。"); continue
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
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outfile = outdir / f"screening_offline_{ts}.csv"
            flat.to_csv(outfile, index=False, encoding="utf-8-sig")
            write_reports(flat, outdir, topn=10)
            write_markdown_report(flat, outdir, topn=10)
            try:
                write_investment_advice_report(flat, outdir, topn=15)
            except Exception:
                pass
            print(f"[OK] 出力: {outfile}")
        elif choice == "3":
            code = input("銘柄コード4桁: ").strip()
            name = lookup_company_name(session, code)
            res = analyze_single_stock_complete_v3(session, sector_avgs, code, name=name, offline=True)
            df = pd.DataFrame([_flatten_result(res)])
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = outdir / f"single_{code}_{ts}.csv"
            df.to_csv(fp, index=False, encoding="utf-8-sig")
            print(f"[OK] 出力: {fp}")
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
    parser.add_argument("--profile",
                        choices=["default", AGING_DX_PROFILE_NAME],
                        default="default",
                        help="スクリーニングプロファイル（analyzeフェーズで使用）")
    # 追加フラグ
    parser.add_argument("--reset-pending", action="store_true")
    parser.add_argument("--refresh-days", type=int)
    parser.add_argument("--force-full", action="store_true")
    args = parser.parse_args()

    if args.phase == "interactive":
        run_interactive()
        return

    session = get_authenticated_session_jquants()
    sector_avgs = DynamicSectorAverages(session).get_sector_averages()
    outdir = Path("output"); outdir.mkdir(exist_ok=True)

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
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = outdir / f"single_{args.code}_{ts}.csv"
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        print(f"[OK] 単銘柄出力: {fp}")
        return

    if args.phase == "analyze":
        if args.profile == AGING_DX_PROFILE_NAME:
            screen_aging_dx_alpha(session,
                                  sector_averages=sector_avgs,
                                  topn=args.top,
                                  theme_config_path=THEME_CONFIG_PATH,
                                  output_dir=outdir)
            return
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
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = outdir / f"screening_offline_{ts}.csv"
        flat.to_csv(outfile, index=False, encoding="utf-8-sig")
        write_reports(flat, outdir, topn=max(10, args.top))
        write_markdown_report(flat, outdir, topn=max(10, args.top))
        try:
            write_investment_advice_report(flat, outdir, topn=max(10, args.top))
        except Exception:
            pass
        print(f"[OK] オフライン分析出力: {outfile}")



if __name__ == "__main__":
    # 引数未指定で△ボタン実行→メニュー表示
    main()
