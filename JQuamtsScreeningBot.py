# -*- coding: utf-8 -*-
"""J-Quants Standard equity screener — JQ-STD-CORRECTED-1.0-rc1.

Read README.md and IMPLEMENTATION_NOTES.md before operating.
No brokerage connection or order submission exists in this program.
Analysis is offline. Collection uses only the allowlisted J-Quants V2 endpoints.
Unknown financial/corporate-action inputs are NOT silently imputed.
For J-Quants fins/summary, JPY is an explicit configurable operational default; strict mode can disable it.
The original 7,967-line script is not imported and is not overwritten.
CLI1 restores the legacy Japanese menu (1-7); 8=audit, 9=date update.
The Standard screening rules and raw cache schema are unchanged.
Python >= 3.10; dependencies: numpy, pandas, requests.
"""
from __future__ import annotations

import argparse
import calendar as calmod
import configparser
import copy
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import requests

LOGIC_VERSION = 'JQ-STD-CORRECTED-1.0-rc1'
CODE_VERSION = '1.0.2-rc1-cli2'
CACHE_SCHEMA = 'jq-standard-raw-v1'
JST = dt.timezone(dt.timedelta(hours=9))
BASE_URL = 'https://api.jquants.com/v2'
LOG = logging.getLogger('jquants.standard')
SECTORS = dict(zip(
    ['0050','1050','2050','3050','3100','3150','3200','3250','3300','3350','3400',
     '3450','3500','3550','3600','3650','3700','3750','3800','4050','5050','5100',
     '5150','5200','5250','6050','6100','7050','7100','7150','7200','8050','9050'],
    ['水産・農林業','鉱業','建設業','食料品','繊維製品','パルプ・紙','化学','医薬品',
     '石油・石炭製品','ゴム製品','ガラス・土石製品','鉄鋼','非鉄金属','金属製品',
     '機械','電気機器','輸送用機器','精密機器','その他製品','電気・ガス業','陸運業',
     '海運業','空運業','倉庫・運輸関連業','情報・通信業','卸売業','小売業','銀行業',
     '証券・商品先物取引業','保険業','その他金融業','不動産業','サービス業']))
FINANCIAL_SECTORS = {'7050','7100','7150','7200'}
ALLOWED_ENDPOINTS = {
    'equities/master', 'equities/bars/daily', 'fins/summary',
    'indices/bars/daily/topix', 'markets/calendar', 'fins/earnings-date',
    'markets/margin-interest', 'markets/short-sale-report',
    'equities/earnings-calendar',
}
SOFT_BASES = {
    'ma200_reclaim_core':92., 'bottom_reversal_core':85., 'weak_reclaim_watch':78.,
    'watch_fundamental_core':75., 'extended_above_ma200':65., 'data_review_light':80.,
    'data_review':70., 'forward_downgrade_watch':62., 'cyclical_value_trap':45.,
    'excluded':30., 'satellite_valuation':72., 'satellite_ps_only':72.,
    'financial_quality_watch':55., 'model_scope_review':55., 'earnings_watch':75.,
    'risk_blocked':30.,
}
PRIORITIES = {
    'ma200_reclaim_core':10, 'bottom_reversal_core':20, 'watch_fundamental_core':40,
    'weak_reclaim_watch':45, 'earnings_watch':48, 'extended_above_ma200':50,
    'financial_quality_watch':55, 'forward_downgrade_watch':58, 'data_review':60,
    'cyclical_value_trap':70, 'satellite_valuation':80, 'satellite_ps_only':80,
    'model_scope_review':90, 'risk_blocked':990, 'excluded':999,
}
CORE_LANES = {'ma200_reclaim_core', 'bottom_reversal_core'}


class DataError(ValueError):
    """Unresolved data must block eligibility, never be silently repaired."""


class CollectionError(RuntimeError):
    pass


class AuthError(CollectionError):
    pass


def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def number(x: Any) -> Optional[float]:
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    try:
        v = float(str(x).replace(',', '').strip()) if isinstance(x, str) else float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError, OverflowError):
        return None


def boolean(x: Any) -> Optional[bool]:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return None
    text = str(x).strip().lower()
    if text in ('true','1','1.0','yes'):
        return True
    if text in ('false','0','0.0','no'):
        return False
    if text in ('','none','null','nan','na','<na>'):
        return None
    raise DataError(f'invalid_boolean:{text}')


def date(x: Any) -> Optional[dt.date]:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    if x is None:
        return None
    s = str(x).strip()
    for fmt in ('%Y-%m-%d','%Y%m%d','%Y/%m/%d'):
        try:
            return dt.datetime.strptime(s[:10] if fmt != '%Y%m%d' else s, fmt).date()
        except ValueError:
            pass
    return None


def timestamp(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    try:
        v = x if isinstance(x, dt.datetime) else dt.datetime.fromisoformat(str(x).replace('Z','+00:00'))
        return v.replace(tzinfo=JST) if v.tzinfo is None else v.astimezone(JST)
    except ValueError:
        return None


def published_at(row: dict, day_key: str = 'DiscDate', time_key: str = 'DiscTime') -> Optional[dt.datetime]:
    d = date(row.get(day_key))
    if d is None:
        return None
    # Date-only releases are conservatively available at the END of that day.
    s = str(row.get(time_key) or '23:59:59').strip()
    try:
        t = dt.time.fromisoformat(s)
    except ValueError:
        return None
    return dt.datetime.combine(d, t, JST)


def div(a: Any, b: Any, positive: bool = True) -> Optional[float]:
    a, b = number(a), number(b)
    if a is None or b is None or b == 0 or (positive and b < 0):
        return None
    return number(a / b)


def growth(a: Any, b: Any, scale: float = 1.0) -> Optional[float]:
    r = div(a, b)
    return None if r is None else (r-1.0)*scale


def clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def clean(x: Any) -> Any:
    """JSON finite/null contract; also used before CSV serialization."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, (dt.date, dt.datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    if x is pd.NA:
        return None
    return x


def encoded(obj: Any) -> bytes:
    return json.dumps(clean(obj), ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def digest(obj: Any) -> str:
    return hashlib.sha256(encoded(obj)).hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_bytes(path, encoded(obj))


def add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.year*12+d.month-1+n, 12)
    return dt.date(y, m+1, min(d.day, calmod.monthrange(y, m+1)[1]))


def normal_year(start: Any, end: Any) -> bool:
    s, e = date(start), date(end)
    return bool(s and e and add_months(s,12)-dt.timedelta(days=1) == e)


@dataclass
class Config:
    JQ_PLAN: str = 'standard'
    # /fins/summary の公式レスポンス仕様には通貨列がないため、
    # 日本株スクリーナーとして JPY を明示的な運用既定にする。
    # 厳格監査モードへ戻す場合は False。個別 override / 生 Currency が優先される。
    ASSUME_JQUANTS_SUMMARY_JPY: bool = True
    JQ_RPM: int = 60
    JQ_SUMMARY_RPM: int = 60
    JQ_DEFAULT_BUDGET: int = 800
    JQ_RPD: Optional[int] = None
    FINANCIAL_DETAILS_ENABLED: bool = False
    EDINET_GOVERNANCE_ENABLED: bool = False
    POSITION_SIZING_ENABLED: bool = False
    PRICE_TECH_WINDOW_CALENDAR_DAYS: int = 700
    ADJUSTMENT_HISTORY_YEARS: int = 6
    MIN_AVG_VOLUME_30D: int = 50000
    MIN_ADV_JPY_20D: int = 300000000
    MIN_MARKET_CAP_JPY: int = 50000000000
    MAX_PS_DEFENSIVE: float = 2.0
    MAX_PER_CORE: float = 60.0
    OP_INCOME_YEARS: int = 3
    OP_INCOME_DROP_FLOOR: float = 0.60
    SUMMARY_QUALITY_REQUIRED_ITEMS: int = 7
    RECLAIM_MIN_QUALITY_PASS: int = 5
    BOTTOM_MIN_QUALITY_PASS: int = 6
    RECLAIM_CORE_MIN_FUNDAMENTAL: float = 70.0
    MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY: float = 75.0
    WATCH_FUNDAMENTAL_EDGE_MIN: float = 60.0
    MA200_CROSS_LOOKBACK_DAYS: int = 20
    MIN_PRICE_SESSIONS: int = 220
    MA200_IDEAL_MAX_DISTANCE: float = 0.08
    MA200_EXTENDED_DISTANCE: float = 0.15
    MA200_BELOW_MIN_RATIO: float = 0.75
    BASING_MIN_REBOUND_FROM_LOW: float = 0.08
    BASING_LOOKBACK_LOW_DAYS: int = 120
    RECENT_LOW_LOOKBACK_DAYS: int = 60
    RECENT_LOW_NO_UPDATE_DAYS: int = 10
    MIN_SECTOR_VALID_SAMPLES: int = 5
    MAX_ANNUAL_DISCLOSURE_AGE_DAYS: int = 400
    MAX_ANNUAL_PERIOD_AGE_DAYS: int = 550
    MARKET_REGIME_RECLAIM_BOOST: float = 5.0
    EARNINGS_PRE_BLACKOUT_DAYS: int = 2
    EARNINGS_RECENT_WINDOW_DAYS: int = 30
    UPSIDE_MODE: str = 'diagnostic'
    PERSISTENCE_MODE: str = 'diagnostic'
    # Audit attestations: never filled automatically. See config.example.json.
    currency_rules: Optional[list] = None
    currency_overrides: Optional[dict] = None
    scope_overrides: Optional[dict] = None
    per_share_basis_overrides: Optional[dict] = None
    risk_overrides: Optional[list] = None
    scenarios: Optional[dict] = None

    def validate(self) -> None:
        if self.JQ_PLAN != 'standard':
            raise DataError('This build is Standard-only')
        if any((self.FINANCIAL_DETAILS_ENABLED, self.EDINET_GOVERNANCE_ENABLED,
                self.POSITION_SIZING_ENABLED)):
            raise DataError('details/EDINET/sizing must remain disabled in rc1')
        if self.UPSIDE_MODE != 'diagnostic' or self.PERSISTENCE_MODE != 'diagnostic':
            raise DataError('New indicators must remain diagnostic')
        if not (1 <= self.JQ_RPM <= 120 and 1 <= self.JQ_SUMMARY_RPM <= 60):
            raise DataError('Invalid Standard rate limit')
        if self.JQ_RPD is not None and self.JQ_RPD < 1:
            raise DataError('JQ_RPD must be positive or null')
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not math.isfinite(float(v)) or v < 0:
                    raise DataError('Invalid configuration: '+f.name)
        if (self.OP_INCOME_YEARS != 3 or self.SUMMARY_QUALITY_REQUIRED_ITEMS != 7
                or self.RECLAIM_MIN_QUALITY_PASS != 5 or self.BOTTOM_MIN_QUALITY_PASS != 6):
            raise DataError('rc1 quality contract requires 3 FY, 7 items, 5/6 passes')
        if self.MIN_PRICE_SESSIONS < 200+self.MA200_CROSS_LOOKBACK_DAYS:
            raise DataError('Insufficient MA200 lookback configuration')
        if not 0 < self.MA200_IDEAL_MAX_DISTANCE < self.MA200_EXTENDED_DISTANCE:
            raise DataError('Invalid MA200 bands')
        if not 0 < self.MA200_BELOW_MIN_RATIO < 1 or not 0 < self.OP_INCOME_DROP_FLOOR <= 1:
            raise DataError('Invalid floor ratio')
        if self.JQ_DEFAULT_BUDGET < 1 or self.MIN_SECTOR_VALID_SAMPLES < 5:
            raise DataError('Invalid budget/sector sample floor')
        for key in ('PRICE_TECH_WINDOW_CALENDAR_DAYS','ADJUSTMENT_HISTORY_YEARS',
                    'MA200_CROSS_LOOKBACK_DAYS','RECENT_LOW_LOOKBACK_DAYS',
                    'RECENT_LOW_NO_UPDATE_DAYS','BASING_LOOKBACK_LOW_DAYS'):
            if getattr(self,key)<1:raise DataError('Positive lookback required: '+key)
        if self.ADJUSTMENT_HISTORY_YEARS>10:raise DataError('Adjustment history exceeds Standard plan window')
        if self.BASING_LOOKBACK_LOW_DAYS>self.MIN_PRICE_SESSIONS or self.RECENT_LOW_LOOKBACK_DAYS+self.RECENT_LOW_NO_UPDATE_DAYS>self.MIN_PRICE_SESSIONS:
            raise DataError('Price history does not cover the configured low windows')
        for key in ('currency_rules','risk_overrides'):
            value=getattr(self,key)
            if value is not None and (not isinstance(value,list) or any(not isinstance(x,dict) for x in value)):
                raise DataError('Expected array of objects: '+key)
        for key in ('currency_overrides','scope_overrides','per_share_basis_overrides','scenarios'):
            value=getattr(self,key)
            if value is not None and (not isinstance(value,dict) or any(not isinstance(x,dict) for x in value.values())):
                raise DataError('Expected object keyed by code: '+key)
        for key in ('RECLAIM_CORE_MIN_FUNDAMENTAL','MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY','WATCH_FUNDAMENTAL_EDGE_MIN'):
            if not 0<=getattr(self,key)<=100:raise DataError('FE threshold out of range: '+key)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'Config':
        raw = json.loads(path.read_text('utf-8-sig')) if path else {}
        if not isinstance(raw,dict):raise DataError('Configuration must be a JSON object')
        known = {f.name for f in fields(cls)}
        if set(raw)-known:
            raise DataError('Unknown configuration fields: '+','.join(sorted(set(raw)-known)))
        defaults = cls()
        for key in known:
            if key.isupper() and key in os.environ and os.environ[key].strip():
                raw[key] = os.environ[key]
        for key,v in list(raw.items()):
            base = getattr(defaults,key)
            if isinstance(base,bool):
                raw[key] = boolean(v)
                if raw[key] is None:
                    raise DataError('Missing boolean: '+key)
            elif isinstance(base,int) or key == 'JQ_RPD':
                if v is not None:
                    nv = number(v)
                    if nv is None or not nv.is_integer():
                        raise DataError('Expected integer: '+key)
                    raw[key] = int(nv)
            elif isinstance(base,float):
                raw[key] = number(v)
                if raw[key] is None:
                    raise DataError('Expected finite number: '+key)
        obj = cls(**raw)
        obj.validate()
        return obj


# ---- Master identities and trading calendar ---------------------------------
def canonical_code(value: Any) -> str:
    # Accept only exact 4-character / ordinary-share 5-character identifiers.
    if not isinstance(value, str):
        raise DataError('invalid_code:not_string')
    s = value.strip().upper()
    if re.fullmatch(r'[0-9A-Z]{4}', s):
        return s
    if re.fullmatch(r'[0-9A-Z]{4}0', s):
        return s[:4]
    raise DataError('invalid_code:'+s)


def master_records(rows: list[dict]) -> list[dict]:
    result, seen = [], set()
    for raw in rows:
        api = str(raw.get('Code') or '').strip().upper()
        out = {'api_code':api, 'code':api, 'name':raw.get('CoName') or '',
               'S33':str(raw.get('S33') or ''), 'Mkt':str(raw.get('Mkt') or ''),
               'ProdCat':str(raw.get('ProdCat') or ''), 'instrument_type':'unknown',
               'master_date':raw.get('Date'), 'master_reason':''}
        try:
            code = canonical_code(api)
        except DataError:
            out['master_reason'] = 'not_ordinary_code_or_invalid'
            result.append(out)
            continue
        if code in seen:
            raise DataError('master_code_collision:'+code)
        seen.add(code)
        out['code'] = code
        # API five-character ordinary-share suffix is retained, not reconstructed
        # from an arbitrary user input. Non-zero suffixes remain out of scope.
        out['api_code'] = api
        prod, mkt, name = out['ProdCat'], out['Mkt'], out['name']
        nonstock = ('優先株', '優先出資', '投資法人', '受益証券', '上場投信', 'ETF', 'ETN', 'REIT')
        if any(x.lower() in name.lower() for x in nonstock):
            out['instrument_type'] = 'non_stock'
        elif prod and prod != '011':
            out['instrument_type'] = 'non_stock'
        elif mkt in {'0105','0111','0112','0113'} and name:
            out['instrument_type'] = 'stock'
        elif mkt == '0109':
            out['instrument_type'] = 'non_stock'
        out['sector'] = SECTORS.get(out['S33'], '業種不明')
        out['model_scope'] = 'financial' if out['S33'] in FINANCIAL_SECTORS else 'nonfinancial'
        result.append(out)
    return result


def resolve_code(value: str, master: list[dict]) -> str:
    c = canonical_code(value)
    if not any(m['code']==c and m['instrument_type']=='stock' for m in master):
        raise DataError('invalid_code:not_in_ordinary_master:'+value)
    return c


def trading_sessions(rows: list[dict]) -> pd.DatetimeIndex:
    values = []
    seen = {}
    for row in rows:
        d = date(row.get('Date'))
        hd = str(row.get('HolDiv',''))
        if not d or hd not in {'0','1','2','3'}:
            raise DataError('calendar_schema_invalid')
        if d in seen and seen[d] != hd:
            raise DataError('calendar_conflict')
        seen[d] = hd
        if hd in {'1','2'}:
            values.append(pd.Timestamp(d))
    return pd.DatetimeIndex(sorted(set(values)))


def complete_window(s: pd.Series, n: int) -> bool:
    return len(s) >= n and bool(np.isfinite(s.iloc[-n:].to_numpy(dtype=float)).all())


# ---- Prices: no adjustment of financial shares using rights-issue factors ----
def price_frame(rows: list[dict], api_code: str, sessions: pd.DatetimeIndex,
                asof: dt.date) -> pd.DataFrame:
    if not rows:
        raise DataError('price_missing')
    valid = []
    for row in rows:
        if str(row.get('Code') or api_code).strip().upper() != api_code:
            raise DataError('price_code_mismatch')
        d = date(row.get('Date'))
        if d is None:
            raise DataError('price_date_invalid')
        if d <= asof:
            valid.append(row)
    if not valid:
        raise DataError('price_asof_missing')
    df = pd.DataFrame(valid)
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].duplicated().any():
        raise DataError('duplicate_price_date')
    df = df.set_index('Date').sort_index()
    for c in ('O','H','L','C','Vo','Va','AdjFactor','MktCap'):
        df[c] = pd.to_numeric(df[c],errors='coerce') if c in df else np.nan
    if 'ExRT' not in df:
        df['ExRT'] = None
        df.attrs['exrt_column_present'] = False
    else:
        df.attrs['exrt_column_present'] = True
    exrt_present = df.attrs['exrt_column_present']
    idx = sessions[(sessions>=df.index.min()) & (sessions<=pd.Timestamp(asof))]
    df = df.reindex(idx)
    df.attrs['exrt_column_present'] = exrt_present
    if df.empty or df.index[-1].date() != asof or number(df['C'].iloc[-1]) is None:
        raise DataError('price_asof_missing')
    invalid = ((df[['O','H','L','C']]<=0).any(axis=1)
               | (df['H']<df[['O','L','C']].max(axis=1))
               | (df['L']>df[['O','H','C']].min(axis=1))
               | (df['Vo']<0) | (df['Va']<0) | (df['AdjFactor']<=0))
    if invalid.any():
        raise DataError('invalid_ohlcv_or_factor')
    # Factors at event day affect strictly earlier rows. A missing factor affects
    # all earlier rows; later complete price windows can still be evaluated.
    price_factor = np.full(len(df),np.nan)
    volume_factor = np.full(len(df),np.nan)
    cp, cv = 1.0, 1.0
    for i in range(len(df)-1,-1,-1):
        price_factor[i], volume_factor[i] = cp, cv
        f = number(df['AdjFactor'].iloc[i])
        typ = str(df['ExRT'].iloc[i]).strip()
        if f is None:
            cp, cv = float('nan'),float('nan')
        else:
            cp *= f
            if f != 1.0:
                if typ in {'1','2'}:
                    cv /= f
                elif typ == '3':
                    pass
                else:
                    cv = float('nan')
    df['price_factor_to_asof'] = price_factor
    df['volume_factor_to_asof'] = volume_factor
    for c in ('O','H','L','C'):
        df['A'+c] = df[c] * price_factor
    df['AVo'] = df['Vo'] * volume_factor
    return df


def share_factor(df: pd.DataFrame, basis_date: Any, asof: dt.date) -> tuple[Optional[float],str]:
    b = date(basis_date)
    if b is None or b > asof or df.empty:
        return None,'shares_basis_date_unknown'
    if df.index.min().date() > b:
        return None,'corporate_action_history_insufficient'
    z = df[(df.index>pd.Timestamp(b)) & (df.index<=pd.Timestamp(asof))]
    if z['AdjFactor'].isna().any():
        return None,'corporate_action_history_gap'
    factor = 1.0
    for _,r in z.iterrows():
        f, typ = number(r['AdjFactor']),str(r['ExRT']).strip()
        if typ == '3':
            return None,'rights_issue_shares_unresolved'
        if f is not None and f != 1.0:
            if typ not in {'1','2'}:
                return None,'corporate_action_basis_unknown'
            factor /= f
    return factor,'verified_split_factors'


def rsi_simple(s: pd.Series, n: int=14) -> Optional[float]:
    if not complete_window(s,n+1):
        return None
    d = s.iloc[-n-1:].diff().iloc[1:]
    g, l = float(d.clip(lower=0).mean()),float((-d.clip(upper=0)).mean())
    if l==0:
        return 100.0 if g>0 else 50.0
    return 100.-100./(1.+g/l)


def adx_ema(frame: pd.DataFrame, n: int=14) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index, columns=['adx','plus_di','minus_di'],dtype=float)
    # Use the trailing uninterrupted block. No time compression across holes.
    good = frame[['AH','AL','AC']].notna().all(axis=1)
    last_bad = np.where(~good.to_numpy())[0]
    block = frame.iloc[(last_bad[-1]+1 if len(last_bad) else 0):]
    if len(block) < n+5:
        return out
    h,l,c = block['AH'],block['AL'],block['AC']
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    up,down = h.diff(),-l.diff()
    pdm = up.where((up>0)&(up>down),0.)
    mdm = down.where((down>0)&(down>up),0.)
    ema = lambda x: x.ewm(alpha=2/(n+1),adjust=False).mean()
    atr,pe,me = ema(tr),ema(pdm),ema(mdm)
    p = (100*pe/atr).where(atr!=0,0.)
    m = (100*me/atr).where(atr!=0,0.)
    dx = (100*(p-m).abs()/(p+m)).where((p+m)!=0,0.)
    a = ema(dx)
    for name,s in [('adx',a),('plus_di',p),('minus_di',m)]:
        out.loc[block.index,name] = s.clip(0,100)
        out.loc[block.index[:n+4],name] = np.nan
    return out


def technicals(frame: pd.DataFrame, asof: dt.date, cfg: Config) -> dict:
    d = frame.loc[frame.index>=pd.Timestamp(asof-dt.timedelta(days=cfg.PRICE_TECH_WINDOW_CALENDAR_DAYS))]
    s = d['AC']
    result = {'price':number(frame['C'].iloc[-1]), 'current_price':number(frame['C'].iloc[-1]),
              'adjustment_basis':str(asof), 'adjustment_method':'raw_and_event_factors',
              'adjusted_close':number(s.iloc[-1]) if len(s) else None,
              'price_window_missing':int(s.tail(cfg.MIN_PRICE_SESSIONS).isna().sum())}
    for n in (25,75,200):
        result[f'ma_{n}'] = float(s.iloc[-n:].mean()) if complete_window(s,n) else None
    for n in (21,63,126,252):
        result[f'return_{n}d'] = growth(s.iloc[-1],s.iloc[-1-n]) if complete_window(s,n+1) else None
    for name,long,short in [('momentum_6m_1m',126,21),('momentum_6m_3m',126,63),
                            ('momentum_3m_1m',63,21),('return_12m_1m',252,21)]:
        result[name] = growth(s.iloc[-1-short],s.iloc[-1-long]) if complete_window(s,long+1) else None
    ma25s = s.rolling(25,min_periods=25).mean()
    result['slope25'] = growth(ma25s.iloc[-1],ma25s.iloc[-21]) if complete_window(s,45) else None
    result['rsi'] = rsi_simple(s)
    ad = adx_ema(d)
    for col in ad:
        result[col] = number(ad[col].iloc[-1]) if len(ad) else None
    result['adx_5d_ago'] = number(ad['adx'].iloc[-6]) if len(ad)>=6 else None
    r = s.pct_change(fill_method=None)
    v = float(r.iloc[-20:].std(ddof=1)) if complete_window(r,20) else None
    ref = number(r.std(ddof=1)) if r.notna().sum()>=20 else None
    result.update(vol_daily_20d=v,vol_annual_20d=None if v is None else v*math.sqrt(252),
                  vol_daily_reference=ref,vol_annual_reference=None if ref is None else ref*math.sqrt(252),
                  reference_return_count=int(r.notna().sum()),volatility=None if v is None else v*math.sqrt(252))
    result['max_drawdown'] = float((s/s.cummax()-1).min()) if len(s)>=2 and s.notna().all() else None
    for n in (20,60):
        result[f'adv_jpy_{n}d'] = float(d['Va'].iloc[-n:].mean()) if complete_window(d['Va'],n) else None
    result['avg_volume_30d'] = float(d['AVo'].iloc[-30:].mean()) if complete_window(d['AVo'],30) else None
    approx = d['C']*d['Vo']
    result['adv_jpy_20d_price_volume_proxy'] = float(approx.iloc[-20:].mean()) if complete_window(approx,20) else None
    result['turnover_persistence'] = (div(d['Va'].iloc[-20:].median(),d['Va'].iloc[-80:-20].median())
                                      if complete_window(d['Va'],80) else None)
    result['turnover_persistence_missing'] = max(0,80-len(d))+int(d['Va'].tail(80).isna().sum())
    result['positive_return_top3_share_63d'] = None
    if complete_window(s,64):
        lr = np.log(s.iloc[-64:]/s.iloc[-64:].shift()).iloc[1:]
        positives = lr[lr>0]
        result['positive_return_top3_share_63d'] = div(positives.nlargest(3).sum(),positives.sum())
    result['ma200_cross_count_60d'] = None
    if complete_window(s,260):
        ma = s.rolling(200).mean()
        above = s>=ma
        valid = ma.notna() & ma.shift().notna()
        result['ma200_cross_count_60d'] = int(((above!=above.shift()) & valid).iloc[-60:].sum())
    result.update(ma_state(s,result,cfg))
    return result


def ma_state(s: pd.Series, t: dict, cfg: Config) -> dict:
    o = dict(ma200_state='ma200_unknown', ma200_reason='insufficient_price_history',
             distance_from_ma200=None, crossed_above_ma200_recently=None,
             below_ma200_basing=None, below_ma200_downtrend=None,
             above_ma200_extended=None, recent_60d_low_update=None,
             downtrend_reasons=[],ma200_timing_score=None)
    if not complete_window(s,cfg.MIN_PRICE_SESSIONS) or t.get('ma_200') is None:
        return o
    p,ma = float(s.iloc[-1]),t['ma_200']
    dist = p/ma-1.
    mm = s.rolling(200,min_periods=200).mean()
    crossed = bool(((s.shift()<mm.shift())&(s>=mm)).iloc[-cfg.MA200_CROSS_LOOKBACK_DAYS:].any())
    prior_low = s.shift().rolling(cfg.RECENT_LOW_LOOKBACK_DAYS,min_periods=cfg.RECENT_LOW_LOOKBACK_DAYS).min()
    newlow = bool((s<=prior_low).iloc[-cfg.RECENT_LOW_NO_UPDATE_DAYS:].any())
    o.update(distance_from_ma200=dist,crossed_above_ma200_recently=crossed,
             recent_60d_low_update=newlow,below_ma200_basing=False,
             below_ma200_downtrend=False,above_ma200_extended=False)
    if p>=ma:
        if dist>cfg.MA200_EXTENDED_DISTANCE:
            o.update(ma200_state='above_ma200_extended',above_ma200_extended=True,ma200_timing_score=40.)
        elif crossed:
            o.update(ma200_state='ma200_reclaim',ma200_timing_score=95. if dist<=cfg.MA200_IDEAL_MAX_DISTANCE else 78.)
        else:
            o.update(ma200_state='above_ma200_near',ma200_timing_score=70.)
        o['ma200_reason'] = o['ma200_state']
        return o
    plus,minus,a,ap = (t.get(k) for k in ('plus_di','minus_di','adx','adx_5d_ago'))
    if any(x is None for x in (plus,minus,a,ap,t.get('slope25'),t.get('return_21d'))):
        o['ma200_reason']='missing_downtrend_inputs'
        return o
    ratio = (minus/plus if plus>0 else float('inf') if minus>0 else 0.)
    reasons = []
    if p<t['ma_25'] and t['slope25']<0: reasons.append('below_ma25_and_negative_slope')
    if newlow: reasons.append('recent_60d_low')
    if t['return_21d']<0: reasons.append('return_21d_negative')
    if ratio>1.3 and a>=20: reasons.append('minus_di_dominant')
    if dist<cfg.MA200_BELOW_MIN_RATIO-1: reasons.append('deep_below_ma200')
    lo120 = float(s.iloc[-cfg.BASING_LOOKBACK_LOW_DAYS:].min())
    lo60 = float(s.iloc[-cfg.RECENT_LOW_LOOKBACK_DAYS:].min())
    basing = (not reasons and dist>=cfg.MA200_BELOW_MIN_RATIO-1
              and (p>t['ma_25'] or t['slope25']>0) and t['return_21d']>0
              and not newlow and p/lo120-1>=cfg.BASING_MIN_REBOUND_FROM_LOW
              and (a<=35 or a<ap) and ratio<=1.2 and lo60>lo120*1.0001)
    o.update(below_ma200_basing=bool(basing),below_ma200_downtrend=not basing,
             ma200_state='below_ma200_basing' if basing else 'below_ma200_downtrend',
             ma200_reason='basing_pattern' if basing else ';'.join(reasons) or 'basing_unconfirmed',
             downtrend_reasons=reasons,ma200_timing_score=55. if basing else 15. if reasons else 22.)
    return o


def market_regime(rows: list[dict], sessions: pd.DatetimeIndex, asof: dt.date) -> dict:
    o = {'market_regime':'unknown','regime_source':'frozen_topix','return_topix_63d':None}
    df = pd.DataFrame(rows)
    if df.empty or not {'Date','C'}<=set(df):
        return o
    df['Date'] = pd.to_datetime(df['Date'],errors='coerce')
    if df['Date'].duplicated().any():
        return o
    s = pd.to_numeric(df.set_index('Date')['C'],errors='coerce').reindex(sessions[sessions<=pd.Timestamp(asof)])
    if not complete_window(s,200) or (s.tail(200)<=0).any():
        return o
    p,ma = float(s.iloc[-1]),float(s.iloc[-200:].mean())
    m25=s.rolling(25).mean()
    slope=float(m25.iloc[-1]/m25.iloc[-21]-1.)
    d=p/ma-1
    rg='risk_on' if p>ma and d>=.02 and slope>=-.005 else 'risk_off' if p<ma and (d<=-.03 or slope<-.01) else 'neutral'
    o.update(market_regime=rg,regime_d200=d,regime_slope25=slope,
             return_topix_63d=growth(s.iloc[-1],s.iloc[-64]) if complete_window(s,64) else None)
    return o

# ---- Financial statements ----------------------------------------------------
FIN_FIELDS = {
    'revenue':'Sales','operating_income':'OP','ordinary_income':'OdP','net_income':'NP',
    'operating_cash_flow':'CFO','investing_cash_flow':'CFI','financing_cash_flow':'CFF',
    'total_assets':'TA','net_assets':'Eq','shareholders_equity':'ShEq',
    'equity_ratio':'EqAR','cash_and_equivalents':'CashEq','eps_reported':'EPS',
    'diluted_eps':'DEPS','shares_outstanding':'ShOutFY','treasury_shares':'TrShFY',
    'average_shares':'AvgSh','dividend_per_share':'DivAnn','dividend_total':'DivTotalAnn',
    'payout_ratio_reported':'PayoutRatioAnn', 'roe_reported':'ROE',
}
FLAGS = ('RetroRst','ChgByASRev','ChgNoASRev','ChgAcEst','SigChgInC','MatChgSub')
CRITICAL = ('revenue','operating_income','net_income','operating_cash_flow',
            'total_assets','net_assets','shares_outstanding')


def document_scope(doc: Any) -> str:
    text = str(doc or '')
    if '_NonConsolidated_' in text:
        return 'nonconsolidated'
    if '_Consolidated_' in text:
        return 'consolidated'
    return 'unknown'


def currency_for(raw: dict, cfg: Config, code: str) -> tuple[Optional[str],str]:
    explicit = raw.get('Currency') or raw.get('AccountingCurrency')
    if explicit:
        return str(explicit).upper(),'explicit_source_currency'
    override = (cfg.currency_overrides or {}).get(code)
    if isinstance(override,dict) and override.get('source') and timestamp(override.get('verified_at')):
        return override.get('currency'),'audited_code_override:'+str(override['source'])
    for rule in cfg.currency_rules or []:
        if not rule.get('source') or not timestamp(rule.get('verified_at')):
            continue
        suffixes = rule.get('doc_type_suffixes') or []
        if suffixes and any(str(raw.get('DocType') or '').endswith(x) for x in suffixes):
            return rule.get('currency'),'audited_source_rule:'+str(rule['source'])

    # J-Quants /fins/summary の公開スキーマには Currency 列がない。
    # したがって「Currency 列がない＝全銘柄を severe review」にすると、
    # 引数なしの標準運用では eligible が構造的に 0 件になる。
    # 本ビルドでは日本株スクリーナーの運用仮定として JPY を明示採用する。
    # 生レスポンスの Currency / code override / audited rule は上で優先済み。
    if cfg.ASSUME_JQUANTS_SUMMARY_JPY:
        return 'JPY','jquants_fins_summary_operational_default'
    return None,'currency_not_verified'


def normalize_statement(raw: dict, code: str, cfg: Config) -> dict:
    doc = str(raw.get('DocType') or '')
    scope = document_scope(doc)
    override = (cfg.scope_overrides or {}).get(code)
    if scope=='unknown' and isinstance(override,dict) and override.get('source'):
        scope=override.get('scope','unknown')
    out = {'code':code,'doc_type':doc,'scope':scope,'disc_no':str(raw.get('DiscNo') or ''),
           'disclosed_at':published_at(raw),'period_type':str(raw.get('CurPerType') or '').upper(),
           'period_start':date(raw.get('CurPerSt')),'period_end':date(raw.get('CurPerEn')),
           'fy_start':date(raw.get('CurFYSt')),'fy_end':date(raw.get('CurFYEn')),
           'raw':copy.deepcopy(raw)}
    for name, key in FIN_FIELDS.items():
        # No mixing of consolidated and NC line items. Shared CF/share fields
        # pertain to the document's declared scope. NC forms may populate main keys.
        val = raw.get(key)
        if scope=='nonconsolidated' and 'NC'+key in raw and number(raw.get('NC'+key)) is not None:
            val=raw['NC'+key]
        out[name]=number(val)
    out['equity_ratio_reported']=out['equity_ratio']
    out['net_assets_ratio']=div(out['net_assets'],out['total_assets'])
    if out['equity_ratio'] is None:
        out['equity_ratio']=div(out['shareholders_equity'],out['total_assets'])
    out['currency'],out['currency_source']=currency_for(raw,cfg,code)
    out['comparability_flags']=[k for k in FLAGS if boolean(raw.get(k)) is True]
    out['field_sources']={k:out['disc_no'] for k in FIN_FIELDS if out[k] is not None}
    return out


def normalize_statements(rows: list[dict], api_code: str, code: str,
                         cutoff: dt.datetime, cfg: Config) -> tuple[list[dict],list[str]]:
    out, issues=[],[]
    for raw in rows:
        if str(raw.get('Code') or '').upper()!=api_code:
            issues.append('financial_code_mismatch')
            continue
        s=normalize_statement(raw,code,cfg)
        if s['disclosed_at'] is None:
            issues.append('disclosure_timestamp_missing')
        elif s['disclosed_at']<=cutoff:
            if (s['period_type']=='FY' and 'FinancialStatements' in s['doc_type']
                    and (not s['period_start'] or not s['period_end'] or s['scope']=='unknown')):
                issues.append('annual_identity_or_period_unresolved')
            out.append(s)
    return out,sorted(set(issues))


def annual_history(statements: list[dict]) -> list[dict]:
    grouped: dict[tuple,list[dict]]={}
    for s in statements:
        if (s['period_type']!='FY' or 'FinancialStatements' not in s['doc_type']
                or not s['period_start'] or not s['period_end'] or s['scope']=='unknown'):
            continue
        key=(s['scope'],s['period_start'],s['period_end'])
        grouped.setdefault(key,[]).append(s)
    records=[]
    for ss in grouped.values():
        ss=sorted(ss,key=lambda x:(x['disclosed_at'],x['disc_no']))
        merged=copy.deepcopy(ss[0])
        first=ss[0]['disclosed_at']
        for nxt in ss[1:]:
            # Actual-period partial corrections only; NOT a guidance fallback.
            for k in FIN_FIELDS:
                if nxt[k] is not None:
                    merged[k]=nxt[k]
                    merged['field_sources'][k]=nxt['disc_no']
            if nxt.get('equity_ratio_reported') is not None:
                merged['equity_ratio_reported']=nxt['equity_ratio_reported']
            for k in ('disc_no','disclosed_at','doc_type','raw','comparability_flags','currency','currency_source'):
                merged[k]=copy.deepcopy(nxt[k])
        merged['first_disclosed_at']=first
        merged['net_assets_ratio']=div(merged['net_assets'],merged['total_assets'])
        # Derive EqAR only from the correctly defined ShEq, never Eq.
        merged['equity_ratio']=(merged['equity_ratio_reported'] if merged.get('equity_ratio_reported') is not None
                                else div(merged['shareholders_equity'],merged['total_assets']))
        records.append(merged)
    records.sort(key=lambda x:(x['period_end'],x['scope']=='consolidated',x['disclosed_at']),reverse=True)
    # Prefer the consolidated form if two scopes exist for the same FY. A change
    # in scope across periods remains visible and fails comparability.
    used=set(); history=[]
    for rec in records:
        key=(rec['period_start'],rec['period_end'])
        if key in used:
            continue
        used.add(key);history.append(rec)
    return history[:5]


def consecutive(history: list[dict], n: int) -> bool:
    if len(history)<n:
        return False
    h=history[:n]
    if any(not normal_year(x.get('period_start'),x.get('period_end')) for x in h):
        return False
    if len({x.get('scope') for x in h})!=1 or h[0].get('scope') not in {'consolidated','nonconsolidated'}:
        return False
    # Accounting-standard/scope changes cannot be silently treated as comparable.
    if len({str(x.get('doc_type','')).split('_')[-1] for x in h})!=1:
        return False
    if any(any(k in x.get('comparability_flags',[]) for k in ('SigChgInC','MatChgSub')) for x in h):
        return False
    return all(date(a['period_start'])==date(b['period_end'])+dt.timedelta(days=1) for a,b in zip(h,h[1:]))


def operating_stability(history: list[dict], cfg: Config) -> dict:
    vals=[number(x.get('operating_income')) for x in history[:3]]
    o={'op_income_stable':None,'op_income_reason':'insufficient_annual_history',
       'years_checked':len(vals),'op_income_downside_score':0.}
    if len(vals)<3:
        return o
    if not consecutive(history,3):
        o['op_income_reason']='annual_not_comparable';return o
    if any(x is None for x in vals):
        o['op_income_reason']='operating_income_missing';return o
    stable=all(x>0 for x in vals) and vals[0]>=cfg.OP_INCOME_DROP_FLOOR*float(np.median(vals[1:]))
    o.update(op_income_stable=stable,op_income_reason='ok' if stable else 'deficit_or_drop_below_floor')
    if stable:
        span=3
        while span<len(history) and consecutive(history,span+1) and number(history[span].get('operating_income')) is not None:
            span+=1
        vs=[number(x['operating_income']) for x in history[:span]]
        if all(x>0 for x in vs):
            my=min(a/b-1 for a,b in zip(vs,vs[1:]))
            cv=float(np.std(vs)/np.mean(vs))
            o['op_income_downside_score']=clip(float(np.interp(my,[-.45,-.2,0,.2],[0,2,5.5,8]))-min(3,max(0,cv-.18)*8),0,8)
    return o


def adjust_financial_shares(history: list[dict], frame: pd.DataFrame, asof: dt.date,
                            cfg: Config, code: str) -> None:
    for s in history:
        f,why=share_factor(frame,s.get('period_end'),asof)
        s['share_adjustment_factor']=f
        s['share_basis_status']=why
        sh=number(s.get('shares_outstanding'))
        s['shares_adjusted']=sh*f if sh is not None and f is not None and sh>0 else None
        # DPS is NOT assumed to have the same basis as closing shares. Only use
        # explicit per-period attestation, or an event-free full fiscal interval.
        key=str(s.get('period_end'))
        ov=((cfg.per_share_basis_overrides or {}).get(code) or {}).get(key,{})
        if ov and ov.get('source') and number(ov.get('dps_factor_to_asof')) is not None and str(ov.get('asof'))==str(asof):
            dps_factor=number(ov['dps_factor_to_asof'])
        else:
            start=s.get('period_start')
            ff,_=share_factor(frame,start-dt.timedelta(days=1) if start else None,asof)
            # No corporate actions from the start of that financial year to T.
            dps_factor=1. if ff==1. and f==1. else None
        dp=number(s.get('dividend_per_share'))
        s['dps_adjusted']=dp*dps_factor if dp is not None and dps_factor is not None else None
        s['dps_basis_status']='verified' if dps_factor is not None else 'per_share_basis_unverified'


def summary_quality(history: list[dict]) -> dict:
    c=history[0] if history else {}
    p=history[1] if len(history)>1 else {}
    comp=consecutive(history,2)
    def greater(a,b):
        return None if a is None or b is None else bool(a>b)
    np0,cf=number(c.get('net_income')),number(c.get('operating_cash_flow'))
    q={
        'Q1':None if np0 is None else np0>0,
        'Q2':None if cf is None else cf>0,
        'Q3':greater(cf,np0),
        'Q4':greater(div(np0,c.get('total_assets')),div(p.get('net_income'),p.get('total_assets'))) if comp else None,
        'Q5':greater(div(cf,c.get('revenue')),div(p.get('operating_cash_flow'),p.get('revenue'))) if comp else None,
        'Q6':greater(number(p.get('shares_adjusted')),number(c.get('shares_adjusted'))) if comp else None,
        'Q7':None,
    }
    # Lower liabilities/assets = higher net_assets/assets.
    q['Q7']=greater(div(c.get('net_assets'),c.get('total_assets')),div(p.get('net_assets'),p.get('total_assets'))) if comp else None
    k=sum(v is True for v in q.values());n=sum(v is not None for v in q.values())
    return {**q,'summary_quality_score':k,'summary_quality_available':n,'summary_quality_coverage':n/7.,
            'quality_normalized_9':9.*k/n if n else None,'quality_model':'summary_quality_7',
            'quality_missing_items':[key for key,v in q.items() if v is None]}


def guidance(statements: list[dict], history: list[dict]) -> dict:
    o={'forecast_net_income':None,'forecast_eps':None,'forward_np_change':None,
       'forecast_revision_rate':None,'guidance_status':'missing','forward_guidance_warning':None,
       'guidance_target':None,'forward_guidance_disclosed':None,'forward_guidance_source':None}
    if not history:
        return o
    base=history[0];base_end=base['period_end']
    candidates=[]
    for s in statements:
        raw=s['raw'];doc=s['doc_type']
        if 'DividendForecastRevision' in doc:
            continue
        if s['scope'] not in {base['scope'],'unknown'}:
            continue
        if s['period_type']=='FY' and 'FinancialStatements' in doc:
            # A correction of an older FY cannot replace a current guidance.
            if s['period_end'] != base_end:
                continue
            ts,te=date(raw.get('NxtFYSt')),date(raw.get('NxtFYEn'))
            nk=('NxFNp','NxFNP');ek='NxFEPS';source='fy_next_guidance'
        else:
            ts,te=date(raw.get('CurFYSt')),date(raw.get('CurFYEn'))
            nk=('FNP',);ek='FEPS';source='quarter_or_revision_guidance'
            if te is not None and te<=base_end:
                continue
        prefix='NC' if s['scope']=='nonconsolidated' else ''
        # NC forecasts use separate field spellings, no selective scope mixing.
        if prefix:
            nk=('NxFNCNP',) if source=='fy_next_guidance' else ('FNCNP',)
            ek='NxFNCEPS' if source=='fy_next_guidance' else 'FNCEPS'
        has_forecast_keys=any(k in raw for k in (*nk,ek))
        if not has_forecast_keys and 'EarnForecastRevision' not in doc:
            continue
        ni=next((number(raw.get(k)) for k in nk if number(raw.get(k)) is not None),None)
        eps=number(raw.get(ek))
        candidates.append((s,ts,te,ni,eps,source))
    if not candidates:
        return o
    candidates.sort(key=lambda x:(x[0]['disclosed_at'],x[0]['disc_no']),reverse=True)
    s,ts,te,ni,eps,source=candidates[0]
    o.update(guidance_target=str(te) if te else None,forward_guidance_source=source,
             forward_guidance_disclosed=s['disclosed_at'].isoformat())
    if (not ts or not te or not normal_year(ts,te) or ts != base_end+dt.timedelta(days=1)):
        o.update(guidance_status='rejected_period',forward_guidance_warning='forward_period_mismatch_or_unknown')
        return o
    raw=s['raw']
    if boolean(raw.get('ForecastWithdrawn')) is True:
        o['guidance_status']='withdrawn';return o
    if ni is None:
        o.update(guidance_status='ambiguous' if 'EarnForecastRevision' in s['doc_type'] else 'missing',forecast_eps=eps)
        return o
    o.update(forecast_net_income=ni,forecast_eps=eps,guidance_status='valid',
             forward_np_change=growth(ni,base.get('net_income')))
    for old,ots,ote,oni,oe,src in candidates[1:]:
        if ots==ts and ote==te and oni is not None:
            o['forecast_revision_rate']=growth(ni,oni);break
    return o


def valuation(mcap: Any, history: list[dict], guide: dict, currency_ok: bool=True) -> dict:
    c=history[0] if history else {};p=history[1] if len(history)>1 else {}
    ni,fni=number(c.get('net_income')),number(guide.get('forecast_net_income'))
    m=number(mcap) if currency_ok else None
    actual=div(m,ni);forward=div(m,fni)
    cons=div(m,min(ni,fni)) if ni is not None and fni is not None else actual
    epc=div(ni,c.get('shares_adjusted'));epp=div(p.get('net_income'),p.get('shares_adjusted'))
    eg=growth(epc,epp,100.) if consecutive(history,2) else None
    return {'ps':div(m,c.get('revenue')),'per_actual':actual,'per_forward':forward,
            'per_conservative':cons,'per':cons,'per_basis':'market_cap_over_np',
            'eps_reported':c.get('eps_reported'),'eps_reported_basis':'not_verified_for_valuation',
            'eps_proxy_endshares_current':epc,'eps_proxy_endshares_previous':epp,
            'eps_proxy_endshares_yoy_pct':eg,
            'np_yoy_pct':growth(ni,p.get('net_income'),100.) if consecutive(history,2) else None,
            'forward_missing':fni is None,
            'reference_peg':div(cons,eg) if cons is not None and cons>0 else None}


def earnings_quality(history: list[dict], market_cap: Optional[float]=None) -> dict:
    c=history[0] if history else {}
    a=div((c['net_income']-c['operating_cash_flow']) if c.get('net_income') is not None and c.get('operating_cash_flow') is not None else None,c.get('total_assets'))
    bridge=div(c.get('net_income'),c.get('ordinary_income'),positive=False)
    warnings=[];diag=[]
    if a is not None and a>.12: warnings.append('high_accrual')
    if bridge is not None and (bridge<.45 or bridge>.90): warnings.append('np_ordinary_bridge_outlier')
    z=None;normalized_np=normalized_eps=normalized_per=gap=None
    if consecutive(history,3):
        margins=[div(x.get('operating_income'),x.get('revenue')) for x in history]
        if all(x is not None for x in margins):
            sd=float(np.std(margins[1:]))
            z=(margins[0]-float(np.median(margins[1:])))/(sd or .01)
            if z>2: diag.append('op_margin_spike_diagnostic')
            if c.get('revenue') is not None:
                normalized_np=c['revenue']*float(np.median(margins[1:]))*.70
                normalized_eps=div(normalized_np,c.get('shares_adjusted'))
                normalized_per=div(market_cap,normalized_np)
                if c.get('net_income') is not None and c['net_income']>0:
                    gap=max(0.,1-normalized_np/c['net_income'])
                    if gap>=.35:diag.append('normalized_eps_gap_diagnostic')
    status='watch' if warnings else 'ok' if a is not None or bridge is not None else 'unknown'
    return {'accrual_ratio':a,'np_to_ordinary':bridge,'op_margin_z':z,
            'earnings_quality_flag':status,'earnings_quality_warnings':warnings,
            'earnings_quality_diagnostics':diag,'normalized_eps':normalized_eps,'normalized_per':normalized_per,
            'earnings_quality_gap':gap,'normalized_np_operating_margin_proxy':normalized_np,
            'normalized_proxy_assumption':'past_median_operating_margin_x_0.70_diagnostic_only'}


def shareholder_quality(history: list[dict]) -> dict:
    c=history[0] if history else {};p=history[1] if len(history)>1 else {}
    cfo,dtot,ni=number(c.get('operating_cash_flow')),number(c.get('dividend_total')),number(c.get('net_income'))
    dp=number(c.get('dividend_per_share'))
    capacity=0. if cfo is not None and cfo<=0 else clip(1-dtot/cfo,0,1) if cfo is not None and cfo>0 and dtot is not None and dtot>=0 else None
    payout=div(dtot,ni) if dtot is not None and dtot>=0 else None
    loss_payout=bool(dtot is not None and dtot>0 and ni is not None and ni<=0)
    penalty=min(.35,max(0,payout-.8)*.6) if payout is not None else 0. if dtot==0 else None
    presence=15. if dp is not None and dp>0 else 0. if dp==0 else None
    credibility=consistency=None
    span=3
    if consecutive(history,3):
        while span<len(history) and consecutive(history,span+1):span+=1
        h=history[:span]
        ds=[x.get('dps_adjusted') for x in h]
        if all(x is not None and x>=0 for x in ds):
            cuts=sum(a<b*.95 for a,b in zip(ds,ds[1:]));incs=sum(a>b*1.02 for a,b in zip(ds,ds[1:]))
            credibility=0. if all(x==0 for x in ds) else clip(.45+.12*incs-.25*cuts,0,1)
        sh=[x.get('shares_adjusted') for x in h[:4]]
        if all(x is not None and x>0 for x in sh):
            cg=(sh[0]/sh[-1])**(1/(len(sh)-1))-1
            consistency=1. if cg<-.01 else .5 if cg<0 else 0.
    vals=[c.get('equity_ratio'),p.get('equity_ratio'),c.get('cash_and_equivalents'),p.get('cash_and_equivalents'),payout]
    strain=None
    if consecutive(history,2) and all(v is not None for v in vals):
        strain=float(vals[0]<vals[1]-.03 and vals[2]<vals[3] and vals[4]>.6)
    elif dtot==0:
        strain=0. # no payout; logically no payout-related strain
    parts={'capacity':capacity,'dividend_credibility':credibility,'share_decrease_consistency':consistency,
           'dividend_presence_points':presence,'balance_strain':strain,'payout_penalty':penalty}
    coverage=sum(v is not None for v in parts.values())/len(parts)
    score=None
    if coverage==1. and not loss_payout:
        score=clip(35*capacity+30*credibility+20*consistency+presence-20*strain-100*penalty,0,100)
    return {'shareholder_return_score':score,'shareholder_return_coverage':coverage,
            'shareholder_return_components':parts,'dividend_payout_ratio':payout,
            'payout_status':'loss_making_payout' if loss_payout else 'known' if payout is not None else 'unknown',
            'dividend_total_source':'reported_same_FY' if dtot is not None else 'missing'}


def peg_quality(peg: Any, eg: Any, forward_known: bool, quality_watch: bool) -> dict:
    peg,eg=number(peg),number(eg);warnings=[];trusted=False;grade='missing'
    if peg is not None and peg>0:
        if peg<.10: warnings.append('extremely_low_possible_oneoff')
        elif peg<.30: warnings.append('very_low_check_cyclical');trusted='caution'
        elif peg<=1.20: trusted=True;grade='ok'
        else: trusted=True;grade='expensive_or_moderate'
    if eg is not None and eg>80:
        warnings.append('eps_growth_too_high_oneoff_risk');trusted=False
    if not forward_known:
        warnings.append('forward_guidance_missing');trusted=False
    if quality_watch:
        warnings.append('earnings_quality_watch');trusted=False
    if peg is None:
        warnings.append('peg_missing')
    return {'peg_trusted':trusted,'peg_warning':warnings[0] if warnings else grade,
            'peg_warnings':warnings,'peg_quality_class':grade}


def earnings_context(rows: list[dict], api_code: str, cutoff: dt.datetime,
                     sessions: pd.DatetimeIndex, next_trade: Optional[dt.date], cfg: Config) -> dict:
    # Latest publication within each quarter/year-end-month-day stream. FYE is
    # NOT converted into a fiscal year. Past schedules never become future ones.
    streams={};ambiguous=False
    for r in rows:
        if str(r.get('Code','')).upper()!=api_code:
            raise DataError('earnings_code_mismatch')
        pub=published_at(r,'PubDate','PubTime')
        if pub is None or pub>cutoff:
            continue
        key=(r.get('FQName'),str(r.get('FYE') or ''))
        prev=streams.get(key)
        if prev and pub==prev[0] and r.get('SchDate')!=prev[1].get('SchDate'):
            ambiguous=True
        if prev is None or pub>=prev[0]: streams[key]=(pub,r)
    future=[];unknown_update=False
    for pub,r in streams.values():
        scheduled=date(r.get('SchDate'))
        if scheduled is None:
            unknown_update=True
        elif next_trade is not None and scheduled>=next_trade:
            future.append((scheduled,pub,r))
    o={'earnings_status':'unknown','earnings_blackout':False,'earnings_scheduled_date':None,
       'pretrade_check_required':True,'earnings_reason':'no_current_schedule'}
    if ambiguous:
        o['earnings_reason']='conflicting_latest_schedule';return o
    if not future or next_trade is None:
        if unknown_update:o['earnings_reason']='undetermined_latest_schedule'
        return o
    scheduled,pub,r=min(future,key=lambda x:x[0])
    if pd.Timestamp(scheduled) not in sessions or pd.Timestamp(next_trade) not in sessions:
        o['earnings_reason']='calendar_horizon_or_date_unverified';return o
    gap=int(sessions.get_loc(pd.Timestamp(scheduled))-sessions.get_loc(pd.Timestamp(next_trade)))
    o.update(earnings_status='known',earnings_scheduled_date=scheduled.isoformat(),
             earnings_blackout=0<=gap<=cfg.EARNINGS_PRE_BLACKOUT_DAYS,
             earnings_days_from_next_trade=gap,earnings_reason='reported_schedule',
             earnings_pub_date=pub.date().isoformat())
    return o


def risk_context(cfg: Config, code: str, cutoff: dt.datetime) -> dict:
    risks=[]
    for r in cfg.risk_overrides or []:
        if str(r.get('code'))!=code:
            continue
        confirmed,expiry=timestamp(r.get('confirmed_at')),timestamp(r.get('expires_at'))
        if not r.get('source') or not confirmed or not expiry:
            raise DataError('risk_override_requires_source_and_dates')
        if confirmed<=cutoff<=expiry:
            if r.get('severity') not in {'watch','severe'} or r.get('kind') not in {'accounting','governance'}:
                raise DataError('invalid_risk_override')
            risks.append(r)
    return {'confirmed_severe_risk':any(r['severity']=='severe' for r in risks),
            'accounting_risk':'watch' if any(r['kind']=='accounting' and r['severity']=='watch' for r in risks) else 'not_checked',
            'governance_status':'watch' if any(r['kind']=='governance' and r['severity']=='watch' for r in risks) else 'not_checked',
            'risk_evidence':risks}


def margin_diagnostic(bundle: dict, cutoff: dt.datetime) -> dict:
    o={'margin_ratio':None,'margin_ratio_state':'unknown','margin_reference_date':None,
       'margin_observed_at':bundle.get('collected_at'),'margin_status':bundle.get('status','not_collected')}
    observed=timestamp(bundle.get('collected_at'))
    # Use first observation as conservative availability when PubDate is absent.
    usable=[]
    for r in bundle.get('rows',[]):
        pub=published_at(r,'PubDate','PubTime') or observed
        if pub and pub<=cutoff and date(r.get('Date')) and date(r['Date'])<=cutoff.date():usable.append(r)
    if not usable:return o
    r=max(usable,key=lambda x:str(x['Date']))
    lng,srt=number(r.get('LongVol')),number(r.get('ShrtVol'))
    o.update(margin_reference_date=r['Date'],margin_long=lng,margin_short=srt)
    if lng is None or srt is None or lng<0 or srt<0:return o
    o['margin_ratio_state']='finite' if srt>0 else 'zero_short_balance' if lng>0 else 'both_zero'
    o['margin_ratio']=lng/srt if srt>0 else None
    return o


# ---- Fixed legacy reference component functions (S1; not new-entry gates) ----
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
    # 入力カバレッジを記録し、信用残・空売り・配当・業績・需給系のシグナルが全滅なら
    # data_review 側で "spec_inputs_insufficient" を付けられるよう observed_count を返す。
    fundamental_inputs_observed = sum(
        1 for x in (
            margin_ratio,
            short_selling_change_rate,
            yoy_eps_growth,
            dividend_status if dividend_status not in (None, "") else None,
            stagnant_days_after_spike,
        ) if x is not None
    )
    technical_inputs_observed = sum(
        1 for cond in (
            (avg_volume is not None),
            (current_volatility is not None and average_volatility not in (None, 0)),
            below_ma25 or below_ma75 or (mas is not None and len(mas) > 0),
        ) if cond
    )
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
    if fundamental_inputs_observed >= 2:
        model_scope = "fundamental_plus_technical"
    elif fundamental_inputs_observed == 1:
        model_scope = "partial_fundamental_technical"
    else:
        model_scope = "technical_only"
    inputs_insufficient = bool(fundamental_inputs_observed == 0 and technical_inputs_observed <= 1)
    return {
        "score": score,
        "level": level,
        "warning_flags": flags,
        "risk_factors": risks,
        "max_score": 100,
        "model_scope": model_scope,
        "scope_note": "需給・信用残などの外部データ未接続時は、価格・出来高・ボラ・移動平均に基づく簡易プロキシです。",
        "fundamental_inputs_observed": fundamental_inputs_observed,
        "technical_inputs_observed": technical_inputs_observed,
        "spec_inputs_insufficient": inputs_insufficient,
    }

def _val_score_from_ps_vs_sector(x: Optional[float]) -> float:
    if x is None or not np.isfinite(x): return 0.0
    if x <= 0.6: return 10.0
    if x <= 0.9: return 8.0
    if x <= 1.2: return 6.0
    if x <= 1.6: return 3.0
    if x <= 2.0: return 1.0
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


# ---- One-time scoring, gates and diagnostic upside ---------------------------
def fe_score(r: dict, cfg: Config) -> dict:
    ps,rel,per=r.get('ps'),r.get('ps_vs_sector'),r.get('per')
    eg,q9=r.get('eps_proxy_endshares_yoy_pct'),r.get('quality_normalized_9')
    k,n=r.get('summary_quality_score',0),r.get('summary_quality_available',0)
    pg=r.get('peg_trusted'); pc=r.get('peg_quality_class')
    def bucket(v,bounds,values):
        if number(v) is None:return 0.
        for b,p in zip(bounds,values):
            if v<=b:return p
        return values[-1]
    a={
        'ps':bucket(ps,[.35,.7,1.2,cfg.MAX_PS_DEFENSIVE,math.inf],[18,15,11,7,3]) if ps is not None and ps>0 else 0.,
        'sector_ps':bucket(rel,[.55,.8,1.,1.1,math.inf],[22,18,14,9,3]),
        'peg_quality':14. if pg is True and pc=='ok' else 8. if pg is True and pc=='expensive_or_moderate' else 5. if pg=='caution' else 0.,
        'summary_quality':22*k/n if n else 0.,
        'operating_stability':9. if r.get('op_income_stable') is True else 0.,
        'sales_cagr':0.,'eps_growth':0.,
    }
    cagr=r.get('sales_cagr')
    if cagr is not None:a['sales_cagr']=6. if cagr>=.15 else 4. if cagr>=.08 else 2.5 if cagr>=0 else 0.
    if eg is not None:a['eps_growth']=5. if 5<=eg<=45 else 2. if 45<eg<=80 else 0.
    p={}
    cheap=(ps is not None and ps<=.65) or (rel is not None and rel<=.72)
    if cheap and q9 is not None and q9<6 and not (pg is True and pc=='ok'):p['cheap_weak_quality']=14.
    if per is not None and per>90:p['very_high_per']=6. if per>120 else 3.
    missing=r.get('critical_missing_count',0)
    if missing:p['critical_missing']=min(18.,3.5*missing)
    if not r.get('annual_comparable'):p['annual_not_comparable']=10.
    if r.get('annual_stale'):p['annual_stale']=10.
    if q9 is not None and q9<5:p['quality_normalized_below_5']=15.
    if n==7 and k<=3:p['quality_passes_at_most_3']=12.
    drift=0.
    age=r.get('days_since_original_annual_release');oy=r.get('op_income_yoy_pct')
    if age is not None and 0<=age<=cfg.EARNINGS_RECENT_WINDOW_DAYS and oy is not None:
        drift=6. if oy>=15 else 3. if oy>=5 else -6. if oy<=-15 else -3. if oy<=-5 else 0.
    return {'fundamental_edge_score':clip(sum(a.values())-sum(p.values())+drift,0,100),
            'fe_components':a,'fe_penalties':p,'annual_release_adjustment':drift}


def cap_entry(raw: float, lane: str, severe: bool=False) -> tuple[float,float]:
    b=55. if lane=='data_review' and severe else SOFT_BASES[lane]
    return (raw if raw<=b else min(100.,b+(100.-b)*(1-math.exp(-(raw-b)/24))*.35),b)


def all_gates(r: dict, cfg: Config) -> dict:
    common=list(r.get('common_issues',[]));medium=list(r.get('financial_issues',[]));light=list(r.get('light_issues',[]))
    if r.get('market_regime') not in {'risk_on','neutral','risk_off'}:common.append('market_regime_unknown')
    if r.get('ma200_state')=='ma200_unknown':common.append('price_state_unknown')
    if r.get('liquidity_ok') is None:common.append('liquidity_unknown')
    if r.get('market_cap') is None:common.append('market_cap_unknown')
    if r.get('annual_stale'):medium.append('annual_stale')
    n,k=r.get('summary_quality_available',0),r.get('summary_quality_score',0)
    if r.get('model_scope')!='financial':
        if n<7:medium.append('summary_quality_items_missing')
        if r.get('op_income_stable') is None:medium.append('operating_stability_unknown')
    common=sorted(set(common));medium=sorted(set(medium));light=sorted(set(light))
    severe_fin=bool(set(medium)&{'currency_unknown_or_not_jpy','currency_not_comparable','invalid_total_assets'})
    level='severe' if common or r.get('annual_stale') or severe_fin else 'medium' if medium else 'light' if light else 'none'
    r['data_review_level']=level
    r['data_review_reasons']=common+medium+light
    fe=r['fundamental_edge_score'];st=r.get('ma200_state')
    pgw=r.get('peg_warnings',[]);eg=r.get('eps_proxy_endshares_yoy_pct');peg=r.get('reference_peg')
    opbad=r.get('op_income_stable') is False or (r.get('op_income_yoy_pct') is not None and r['op_income_yoy_pct']<0)
    trap=bool(((peg is not None and 0<peg<.30) or (eg is not None and eg>80)) and opbad)
    fchg=r.get('forward_np_change');fn=r.get('forecast_net_income')
    downgraded=(fchg is not None and fchg<=-.30) or (fn is not None and fn<=0)
    qualitybad=(r.get('op_income_stable') is False or r.get('Q1') is False or r.get('Q2') is False or (n==7 and k<5))
    fundamental_base=bool(r.get('instrument_type')=='stock' and r.get('liquidity_ok') is True
                          and r.get('market_cap_ok') is True and r.get('op_income_stable') is True
                          and r.get('critical_missing_count',0)==0 and n==7 and not common and not medium
                          and r.get('model_scope')!='financial')
    ps,per=r.get('ps'),r.get('per')
    core=bool(fundamental_base and ps is not None and 0<ps<=cfg.MAX_PS_DEFENSIVE
              and per is not None and 0<per<=cfg.MAX_PER_CORE)
    sat=bool(fundamental_base and not core and ps is not None and ps>0 and per is not None and per>0 and not qualitybad and not downgraded)
    # Under this strict data contract PS-only normally has no admissible cases;
    # the compatibility lane/file exists but never rescues unknown NP or losses.
    val_lane='core' if core else 'satellite_valuation' if sat else 'excluded'
    rq=bool(n==7 and k>=5 and r.get('Q1') is True and r.get('Q2') is True and r.get('op_income_stable') is True
            and not set(pgw)&{'extremely_low_possible_oneoff','eps_growth_too_high_oneoff_risk'})
    bq=bool(rq and k>=6 and r.get('return_21d') is not None and r['return_21d']>0 and r.get('recent_60d_low_update') is False)
    rmin=cfg.RECLAIM_CORE_MIN_FUNDAMENTAL+(cfg.MARKET_REGIME_RECLAIM_BOOST if r.get('market_regime')=='risk_off' else 0.)
    reasons=[]
    # Record all restrictions before selecting the first applicable main lane.
    if common:reasons+=common
    if medium:reasons+=medium
    if light:reasons+=light
    for cond,key in [(r.get('confirmed_severe_risk'),'confirmed_severe_risk'),(trap,'cyclical_value_trap'),
                     (downgraded,'forward_decline_or_loss'),(qualitybad,'financial_quality_failed'),
                     (r.get('below_ma200_downtrend'),'downtrend_or_basing_unconfirmed'),
                     (r.get('liquidity_ok') is False,'liquidity_failed'),(r.get('market_cap_ok') is False,'market_cap_failed'),
                     (r.get('earnings_blackout'),'earnings_blackout'),(st=='above_ma200_extended','overextended'),
                     (r.get('model_scope')=='financial','model_scope_financial')]:
        if cond:reasons.append(key)
    if r.get('instrument_type')!='stock':lane='excluded';reasons.append('not_ordinary_stock')
    elif r.get('confirmed_severe_risk'):lane='risk_blocked'
    elif common:lane='data_review'
    elif r.get('model_scope')=='financial':lane='model_scope_review'
    elif medium:lane='data_review'
    elif trap:lane='cyclical_value_trap'
    elif downgraded:lane='forward_downgrade_watch'
    elif qualitybad:lane='financial_quality_watch'
    elif r.get('below_ma200_downtrend'):lane='excluded'
    elif r.get('liquidity_ok') is False or r.get('market_cap_ok') is False:lane='excluded'
    elif sat:lane='satellite_valuation'
    elif not core:lane='excluded';reasons.append('valuation_not_eligible')
    elif light:lane='data_review_light'
    elif r.get('earnings_blackout'):lane='earnings_watch'
    elif st=='above_ma200_extended':lane='extended_above_ma200'
    elif st=='ma200_reclaim' and rq and fe>=rmin:lane='ma200_reclaim_core'
    elif st=='ma200_reclaim' and rq and cfg.WATCH_FUNDAMENTAL_EDGE_MIN<=fe<rmin:lane='weak_reclaim_watch'
    elif st=='below_ma200_basing' and bq and fe>=cfg.MIN_FUNDAMENTAL_EDGE_FOR_BOTTOM_BUY and r.get('market_regime')!='risk_off':lane='bottom_reversal_core'
    elif fe>=cfg.WATCH_FUNDAMENTAL_EDGE_MIN:lane='watch_fundamental_core';reasons.append('entry_conditions_not_met')
    else:lane='excluded';reasons.append('fundamental_edge_below_threshold')
    eligible=bool(core and lane in CORE_LANES and level=='none' and not r.get('earnings_blackout') and not r.get('confirmed_severe_risk') and not common)
    return {'valuation_lane':val_lane,'core_candidate':core,'satellite_candidate':sat,
            'candidate_lane':lane,'eligible_new_entry':eligible,'gate_reasons':sorted(set(reasons)),
            'data_review_level':level,'data_review_reasons':r['data_review_reasons'],
            'reclaim_quality':rq,'bottom_quality':bq,'reclaim_fe_threshold':rmin}


def recommendation(r: dict) -> dict:
    b={'ma200_reclaim':8.,'below_ma200_basing':4.,'above_ma200_near':3.,
       'above_ma200_extended':-6.,'below_ma200_downtrend':-14.,'ma200_unknown':-3.}
    d=r.get('distance_from_ma200');st=r.get('ma200_state')
    penalty=min(14.,max(0,d-.08)*65) if d is not None and st in {'ma200_reclaim','above_ma200_near','above_ma200_extended'} else 0.
    raw=clip(r['fundamental_edge_score']+b.get(st,-3.)-penalty,0,100)
    entry,base=cap_entry(raw,r['candidate_lane'],r['data_review_level']=='severe')
    adj={};fc=r.get('forward_np_change')
    if fc is None:adj['forward_missing']=-6.
    elif fc<=-.20:adj['forward_decline']=-12.
    elif fc<=-.10:adj['forward_decline']=-6.
    elif fc<0:adj['forward_decline']=-2.
    if str(r.get('forward_guidance_warning') or '').startswith('forward_period'):adj['forward_period_mismatch']=-5.
    if r.get('earnings_quality_flag')=='watch':adj['earnings_quality']=-8.
    if r.get('accounting_risk')=='watch':adj['accounting_risk']=-10.
    if r.get('governance_status')=='watch':adj['governance_risk']=-8.
    if r.get('shareholder_return_score') is not None and r['shareholder_return_score']>=70 and r.get('shareholder_return_coverage')==1.:adj['shareholder_return']=3.
    score=.6*r['fundamental_edge_score']+.4*entry+sum(adj.values())
    lane=r['candidate_lane']
    priority=30 if lane=='data_review_light' and st=='ma200_reclaim' else 65 if lane=='data_review_light' else PRIORITIES[lane]
    return {'entry_score_raw':raw,'entry_score':entry,'entry_soft_cap_base':base,
            'recommendation_score':score,'rec_priority':priority,
            'rec_secondary':r['fundamental_edge_score'] if lane=='watch_fundamental_core' else score,
            'recommendation_adjustments':adj}


def persistence(r: dict, history: list[dict]) -> dict:
    c=history[0] if history else {};p=history[1] if len(history)>1 else {}
    np0,rev,op,cfo=(c.get(k) for k in ('net_income','revenue','operating_income','operating_cash_flow'))
    o={'net_profit_margin':div(np0,rev),'operating_margin':div(op,rev),'cfo_margin':div(cfo,rev),
       'normalized_np_3y':None,'per_normalized_np_3y':None,'positive_cfo_years_3y':None,
       'cumulative_cfo_to_np_3y':None,'op_margin_change_yoy':None,
       'np_growth_actual':None if r.get('np_yoy_pct') is None else r['np_yoy_pct']/100.,
       'return_rel_topix_63d':None,'persistence_mode':'diagnostic','upside_mode':'diagnostic'}
    if consecutive(history,3):
        ns=[x.get('net_income') for x in history[:3]];cs=[x.get('operating_cash_flow') for x in history[:3]]
        if all(x is not None for x in ns):
            o['normalized_np_3y']=float(np.median(ns));o['per_normalized_np_3y']=div(r.get('market_cap'),o['normalized_np_3y'])
        if all(x is not None for x in cs):o['positive_cfo_years_3y']=sum(x>0 for x in cs)
        if all(x is not None for x in ns+cs):o['cumulative_cfo_to_np_3y']=div(sum(cs),sum(ns))
    pm=div(p.get('operating_income'),p.get('revenue'))
    if consecutive(history,2) and pm is not None and o['operating_margin'] is not None:o['op_margin_change_yoy']=o['operating_margin']-pm
    if r.get('return_63d') is not None and r.get('return_topix_63d') is not None:
        o['return_rel_topix_63d']=(1+r['return_63d'])/(1+r['return_topix_63d'])-1
    return o


def scenario_values(r: dict, cfg: Config, cutoff: dt.datetime) -> dict:
    o={'scenario_status':'not_configured','scenario_confidence':None,'suggested_take_profit':None,
       'recommended_shares':None,'portfolio_capital_jpy':None,'risk_budget_ratio':None}
    for side in ('bear','base','bull'):
        for key in ('scenario_np','scenario_per','scenario_price','upside'):o[key+'_'+side]=None
    sc=(cfg.scenarios or {}).get(r['code'])
    if not sc:return o
    created=timestamp(sc.get('created_at'))
    if not sc.get('source') or not created or created>cutoff or not sc.get('horizon') or not sc.get('assumptions'):
        o['scenario_status']='invalid_metadata';return o
    if boolean(sc.get('shares_constant')) is not True:
        o['scenario_status']='share_change_model_required';return o
    if any(not isinstance(sc.get(s),dict) or number(sc[s].get('np')) is None or number(sc[s].get('per')) is None for s in ('bear','base','bull')):
        o['scenario_status']='all_three_scenarios_required';return o
    if r.get('market_cap') is None or r.get('price') is None:
        o['scenario_status']='valuation_missing';return o
    for side in ('bear','base','bull'):
        ni,pe=number(sc[side]['np']),number(sc[side]['per'])
        if ni<=0 or pe<=0:
            o['scenario_status']='nonpositive_np_or_per_not_supported';return o
    for side in ('bear','base','bull'):
        ni,pe=number(sc[side]['np']),number(sc[side]['per'])
        price=r['price']*ni*pe/r['market_cap']
        o.update({f'scenario_np_{side}':ni,f'scenario_per_{side}':pe,f'scenario_price_{side}':price,
                  f'upside_{side}':price/r['price']-1})
    o.update(scenario_status='conditional_not_forecast',scenario_confidence=sc.get('confidence'),
             scenario_horizon=sc['horizon'],scenario_source=sc['source'],scenario_created_at=sc['created_at'],
             scenario_assumptions=sc['assumptions'])
    return o


def legacy_reference_scores(records: list[dict], cfg: Config) -> None:
    """Mutates records once, after the full-universe percentiles have been fixed."""
    for r in records:
        if r.get('instrument_type')!='stock':
            r.update(legacy_total=None,total_score=None,grade_raw='NR',grade='NR');continue
        n,k=r.get('summary_quality_available',0),r.get('summary_quality_score',0)
        q9=r.get('quality_normalized_9')
        pct=r.get('quality_pct_in_sector');gp=r.get('sales_cagr_pct_in_sector')
        fscore=(pct*12 if pct is not None else _quality_score_from_piotroski(q9)*12/14)
        fscore+=r.get('op_income_downside_score',0.)
        fscore+=gp*10 if gp is not None else _growth_score_from_sales_cagr(r.get('sales_cagr'))
        val=_val_score_from_ps_vs_sector(r.get('ps_vs_sector'))*.8
        safe_crit=calculate_safety_criteria_v1(r.get('ps'),r.get('cash_and_equivalents'),r.get('market_cap'),
                                           r.get('operating_cash_flow'),r.get('equity_ratio'),r.get('sales_cagr'),r.get('max_drawdown'))
        resilience=safe_crit['total_score']*.15+_resilience_score_from_drawdown(r.get('max_drawdown'))*10/8
        tech=clip(_momentum_score_from_returns(*(r.get(k) for k in ('return_21d','return_63d','return_126d',
                'momentum_6m_1m','momentum_6m_3m','momentum_3m_1m')),False),0,22)*.65
        # Only price/volume/volatility/verified performance inputs are connected.
        # Margin/short reports remain diagnostics until availability adapters pass
        # real-data audit; their absence is not declared safe.
        below25=(r.get('adjusted_close') is not None and r.get('ma_25') is not None and r['adjusted_close']<r['ma_25'])
        below75=(r.get('adjusted_close') is not None and r.get('ma_75') is not None and r['adjusted_close']<r['ma_75'])
        sa=calculate_safety_score_v3(yoy_eps_growth=r.get('eps_proxy_endshares_yoy_pct'),
                dividend_status=r.get('dividend_status'),avg_volume=r.get('avg_volume_30d'),
                avg_trading_value=r.get('adv_jpy_20d'),current_volatility=r.get('vol_daily_20d'),
                average_volatility=r.get('vol_daily_reference'),below_ma25=below25,below_ma75=below75)
        technical_known=all(number(r.get(k)) is not None for k in ('adjusted_close','ma_25','ma_75'))
        if not technical_known:
            # The legacy helper assumes both booleans are observed; remove that
            # contribution rather than falsely awarding above-MA points.
            sa['total_score']=max(0.,sa['total_score']-(1.5 if below25 or below75 else 3.))
            sa['coverage_ratio']=max(0.,sa['coverage_ratio']-1/8.)
        sp=detect_speculative_manipulation_v2(yoy_eps_growth=r.get('eps_proxy_endshares_yoy_pct'),
                dividend_status=r.get('dividend_status'),avg_volume=r.get('avg_volume_30d'),
                current_volatility=r.get('vol_daily_20d'),average_volatility=r.get('vol_daily_reference'),
                below_ma25=below25,below_ma75=below75,mas={k:r.get(k) for k in ('ma_25','ma_75','ma_200') if r.get(k) is not None})
        adv=r.get('adv_jpy_20d') or 0.
        safety=clip(sa['total_score']*.4+(2 if adv>=1e9 else 1 if adv>=5e8 else 0),0,12)
        neutral=[r.get(k) for k in ('quality_pct_in_sector','return_252d_pct_in_sector','return_12m_1m_pct_in_sector','per_pct_in_sector','ps_pct_in_sector')]
        neutral=[v for v in neutral if v is not None]
        bonus=float(np.mean(neutral))*4 if neutral else 0.
        pen=sp['score']*.08
        perpen=4. if r.get('per') is not None and r['per']>cfg.MAX_PER_CORE else 0.
        dp=min(r.get('critical_missing_count',0),4)*1.5
        age=r.get('annual_disclosure_age_days')
        if age is not None:dp+=6 if age>540 else 3 if age>365 else 0
        dp=min(12.,dp)
        total=clip(val+clip(fscore,0,30)+clip(resilience,0,25)+tech+safety+bonus-pen-perpen-dp,0,100)
        grade='A+' if total>=70 else 'A' if total>=62 else 'B+' if total>=54 else 'B' if total>=46 else 'C'
        rawgrade=grade;caps=[]
        if r.get('data_review_level') in {'medium','severe'} or r.get('model_scope')=='financial':grade='NR'
        else:
            fc=r.get('forward_np_change')
            if fc is not None and fc<=-.5:caps.append('C')
            elif fc is not None and fc<=-.3:caps.append('B')
            if r.get('earnings_quality_flag')=='watch' or r.get('accounting_risk')=='watch' or r.get('governance_status')=='watch':caps.append('B')
            if r.get('confirmed_severe_risk'):caps.append('C')
            order=['C','B','B+','A','A+']
            for cap in caps:
                if order.index(cap)<order.index(grade):grade=cap
        r.update(legacy_total=total,total_score=total,grade_raw=rawgrade,grade=grade,
                 financial_score=clip(fscore,0,30),valuation_score=val,resilience_score=clip(resilience,0,25),
                 technical_score=tech,safety_score_scaled=safety,sector_neutral_bonus=bonus,spec_penalty=pen,
                 per_penalty=perpen,data_penalty=dp,spec_score=sp['score'],spec_model_scope=sp['model_scope'],
                 spec_inputs_coverage=(sp['fundamental_inputs_observed']+sp['technical_inputs_observed'])/8.,
                 spec_status='reference_only_not_manipulation_detection',safety_coverage=sa['coverage_ratio'],
                 legacy_scope='available_inputs_reference_only',grade_caps=caps)


def rank_key(r: dict) -> tuple:
    def neg(k):return -(r.get(k) if r.get(k) is not None else -1e100)
    return (r.get('rec_priority',999),neg('rec_secondary'),neg('fundamental_edge_score'),
            neg('entry_score'),neg('legacy_total'),str(r.get('code','')))


def fix_benchmarks(records: list[dict], cfg: Config, universe_id: str) -> dict:
    stats={}
    for sec in SECTORS:
        rows=[r for r in records if r.get('S33')==sec and r.get('benchmark_input_valid')]
        ps=[r['ps'] for r in rows if r.get('ps') is not None and r['ps']>0]
        stats[sec]={'ps_median':float(np.median(ps)) if len(ps)>=cfg.MIN_SECTOR_VALID_SAMPLES else None,'sample_count':len(ps)}
    for r in records:
        stat=stats.get(r.get('S33'),{'ps_median':None,'sample_count':0})
        r.update(sector_ps_benchmark=stat['ps_median'],sector_sample_count=stat['sample_count'],
                 sector_benchmark_source='frozen_full_universe',percentile_universe_id=universe_id,
                 ps_vs_sector=div(r.get('ps'),stat['ps_median']),
                 rerating_space_ps=growth(stat['ps_median'],r.get('ps')))
        if stat['ps_median'] is None and r.get('instrument_type')=='stock' and r.get('model_scope')!='financial':
            r.setdefault('light_issues',[]).append('sector_benchmark_insufficient')
    # Each indicator gets its own non-missing observation count.
    for source,target,high in [
        ('summary_quality_coverage_rate','quality_pct_in_sector',True),
        ('sales_cagr','sales_cagr_pct_in_sector',True),('return_252d','return_252d_pct_in_sector',True),
        ('return_12m_1m','return_12m_1m_pct_in_sector',True),('per','per_pct_in_sector',False),('ps','ps_pct_in_sector',False)]:
        for r in records:r[target]=None
        for sec in SECTORS:
            eligible=[r for r in records if r.get('S33')==sec and r.get('benchmark_input_valid')
                      and r.get(source) is not None and (source!='summary_quality_coverage_rate' or r.get('summary_quality_available')==7)]
            if len(eligible)<cfg.MIN_SECTOR_VALID_SAMPLES:continue
            series=pd.Series([r[source] for r in eligible],dtype=float)
            pct=series.rank(method='average',pct=True,ascending=high)
            for row,score in zip(eligible,pct):row[target]=float(score)
    return stats

# ---- Immutable, checksummed cache and bounded API client ---------------------
class ProcessLock:
    def __init__(self, root: Path, name: str):
        self.path=root/(name+'.lock');self.fd=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as e:
            raise CollectionError(f'Another process or stale lock exists: {self.path}. Check the PID before removing it.') from e
        os.write(self.fd,encoded({'pid':os.getpid(),'created_at':now_jst().isoformat()}))
        return self
    def __exit__(self,*exc):
        if self.fd is not None:os.close(self.fd)
        self.path.unlink(missing_ok=True)


class Store:
    def __init__(self, root: Path):
        self.root=Path(root)
    def ensure(self):
        self.root.mkdir(parents=True,exist_ok=True)
        schema=self.root/'schema.json'
        if schema.exists():
            if json.loads(schema.read_text('utf-8')).get('schema')!=CACHE_SCHEMA:
                raise DataError('cache_schema_mismatch')
        else:atomic_json(schema,{'schema':CACHE_SCHEMA})
    def put(self, obj: Any) -> str:
        self.ensure();payload=encoded(obj);h=hashlib.sha256(payload).hexdigest()
        path=self.root/'objects'/(h+'.json.gz')
        if not path.exists():atomic_bytes(path,gzip.compress(payload,mtime=0))
        return h
    def get(self, ref: str) -> Any:
        if not isinstance(ref,str) or not re.fullmatch(r'[0-9a-f]{64}',ref):
            raise DataError('invalid_cache_reference')
        try:payload=gzip.decompress((self.root/'objects'/(ref+'.json.gz')).read_bytes())
        except (OSError,EOFError) as e:raise DataError('missing_or_corrupt_cache:'+ref[:12]) from e
        if hashlib.sha256(payload).hexdigest()!=ref:raise DataError('cache_hash_mismatch')
        return json.loads(payload)
    def snapshot(self, ident: Optional[str]=None) -> dict:
        p=self.root/'latest_snapshot.json'
        if ident is None:
            if not p.exists():return {}
            ident=json.loads(p.read_text('utf-8'))['snapshot_id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',ident):raise DataError('invalid_snapshot_id')
        path=self.root/'snapshots'/(ident+'.json')
        if not path.exists():raise DataError('snapshot_not_found:'+ident)
        out=json.loads(path.read_text('utf-8'))
        if out.get('schema')!=CACHE_SCHEMA:raise DataError('snapshot_schema_mismatch')
        return out
    def save_snapshot(self, snap: dict, promote: bool) -> None:
        self.ensure();name=snap['snapshot_id']
        path=self.root/'snapshots'/(name+'.json')
        if path.exists():raise DataError('immutable_snapshot_already_exists')
        atomic_json(path,snap)
        if promote:
            if snap.get('status')=='incomplete':raise DataError('cannot_promote_incomplete')
            atomic_json(self.root/'latest_snapshot.json',{'snapshot_id':name})


class RateLimiter:
    def __init__(self, cfg: Config, quota_path: Optional[Path]=None,
                 clock: Callable[[],float]=time.monotonic, sleep: Callable[[float],None]=time.sleep):
        self.cfg=cfg;self.clock=clock;self.sleep=sleep;self.lock=threading.Lock()
        self.general=deque();self.summary=deque();self.last=-1e10
        self.quota_path=quota_path;self.quota_day='';self.quota_count=0
        if quota_path and quota_path.exists():
            q=json.loads(quota_path.read_text('utf-8'))
            self.quota_day=q.get('date','');self.quota_count=int(q.get('count',0))
    def reserve(self, endpoint: str):
        with self.lock:
            day=now_jst().date().isoformat()
            if day!=self.quota_day:self.quota_day=day;self.quota_count=0
            if self.cfg.JQ_RPD is not None and self.quota_count>=self.cfg.JQ_RPD:
                raise CollectionError('daily_self_quota_reached')
            while True:
                now=self.clock()
                for q in (self.general,self.summary):
                    while q and now-q[0]>=60.1:q.popleft()
                waits=[0.,self.last+60./self.cfg.JQ_RPM-now]
                if len(self.general)>=self.cfg.JQ_RPM:waits.append(self.general[0]+60.1-now)
                if endpoint=='fins/summary' and len(self.summary)>=self.cfg.JQ_SUMMARY_RPM:waits.append(self.summary[0]+60.1-now)
                wait=max(waits)
                if wait<=0:break
                if wait>=2:LOG.info('[待機] APIレート制御: %.1f秒（Ctrl+Cで中断）',wait)
                self.sleep(wait)
            now=self.clock();self.general.append(now);self.last=now
            if endpoint=='fins/summary':self.summary.append(now)
            self.quota_count+=1
            if self.quota_path:atomic_json(self.quota_path,{'date':day,'count':self.quota_count})


class Client:
    def __init__(self, api_key: str, cfg: Config, store: Store,
                 session: Optional[requests.Session]=None, limiter: Optional[RateLimiter]=None,
                 sleep: Callable[[float],None]=time.sleep):
        if not api_key:raise AuthError('API key missing')
        self.session=session if session is not None else requests.Session()
        self.session.headers.update({'x-api-key':api_key,'User-Agent':'JQuants-Standard-Screener/'+CODE_VERSION})
        self.store=store;self.cfg=cfg;self.sleep=sleep
        self.limiter=limiter if limiter else RateLimiter(cfg,store.root/'quota.json')
    def fetch(self, endpoint: str, params: Optional[dict]=None, max_pages: int=1000) -> dict:
        if endpoint not in ALLOWED_ENDPOINTS:raise CollectionError('endpoint_not_allowlisted:'+endpoint)
        params=dict(params or {})
        if 'cursor' in params:raise CollectionError('Standard does not use the Premium cursor')
        pages=[];rows=[];key=None;seen=set();started=now_jst().isoformat()
        try:
            for _ in range(max_pages):
                p=dict(params)
                if key:p['pagination_key']=key
                response=None
                for attempt in range(5):
                    self.limiter.reserve(endpoint)
                    try:
                        response=self.session.get(BASE_URL+'/'+endpoint,params=p,timeout=(10,60),allow_redirects=False)
                    except requests.RequestException:
                        if attempt==4:raise CollectionError('network_failure:'+endpoint)
                        LOG.warning('[再試行] %s 通信エラー: %s/5',endpoint,attempt+1)
                        self.sleep(min(30,2**(attempt+1)));continue
                    if response.status_code in {401,403}:
                        raise AuthError(f'HTTP {response.status_code}: {endpoint}')
                    if response.status_code==429 or response.status_code>=500:
                        if attempt==4:raise CollectionError(f'HTTP {response.status_code} retry_exhausted:{endpoint}')
                        retry=number(response.headers.get('Retry-After'))
                        if retry is None and response.headers.get('Retry-After'):
                            from email.utils import parsedate_to_datetime
                            try:retry=max(0.,(parsedate_to_datetime(response.headers['Retry-After'])-dt.datetime.now(dt.timezone.utc)).total_seconds())
                            except (ValueError,TypeError):retry=None
                        LOG.warning('[再試行] %s HTTP %s: %s/5（Ctrl+Cで中断）',endpoint,response.status_code,attempt+1)
                        self.sleep(max(2**(attempt+1),retry or 0.));continue
                    if response.status_code!=200:raise CollectionError(f'HTTP {response.status_code}:{endpoint}')
                    break
                try:body=response.json()
                except (ValueError,AttributeError) as e:raise CollectionError('invalid_json:'+endpoint) from e
                if not isinstance(body,dict) or not isinstance(body.get('data'),list) or any(not isinstance(r,dict) for r in body['data']):
                    raise CollectionError('unexpected_response_schema:'+endpoint)
                pages.append(body);rows.extend(body['data'])
                key=body.get('pagination_key')
                if not key:
                    return {'endpoint':endpoint,'params':params,'rows':rows,'raw_pages':pages,
                            'status':'complete','started_at':started,'collected_at':now_jst().isoformat()}
                if not isinstance(key,str) or key in seen:raise CollectionError('pagination_cycle:'+endpoint)
                seen.add(key)
            raise CollectionError('pagination_limit_reached:'+endpoint)
        except BaseException:
            # Raw pages already received are retained, but NEVER marked complete.
            self.store.put({'endpoint':endpoint,'params':params,'rows':rows,'raw_pages':pages,
                            'status':'incomplete','started_at':started,'collected_at':now_jst().isoformat()})
            raise
    def close(self):self.session.close()


def api_key_from_env_or_ini(path: Path) -> str:
    key=(os.getenv('JQUANTS_API_KEY') or os.getenv('JQ_API_KEY') or '').strip()
    if not key and path.exists():
        cfg=configparser.ConfigParser(interpolation=None)
        cfg.read(path,encoding='utf-8-sig');key=cfg['DEFAULT'].get('API_KEY','').strip()
    if not key or key.upper().startswith(('PASTE_','YOUR_','REPLACE_')):
        raise AuthError('JQUANTS_API_KEY / JQ_API_KEY または api.ini [DEFAULT] API_KEY を設定してください')
    return key


def bundle_rows(store: Store, ref: Optional[str]) -> list[dict]:
    if not ref:return []
    blob=store.get(ref)
    if blob.get('status')!='complete':raise DataError('incomplete_cache_object')
    return blob.get('rows',[])


def new_id(prefix: str='') -> str:
    return prefix+now_jst().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:6]


def collect_globals(client: Client, asof: Optional[dt.date]=None) -> tuple[dict,dt.date]:
    now=now_jst()
    start=add_months(now.date(),-12*client.cfg.ADJUSTMENT_HISTORY_YEARS)-dt.timedelta(days=10)
    LOG.info('[準備 1/3] 取引カレンダーを取得しています')
    calendar_blob=client.fetch('markets/calendar',{'from':start.isoformat(),'to':(now.date()+dt.timedelta(days=45)).isoformat()})
    if not calendar_blob['rows']:raise CollectionError('calendar_empty')
    sessions=trading_sessions(calendar_blob['rows'])
    LOG.info('[準備 2/3] TOPIX日足・分析基準日を確認しています')
    topix_blob=client.fetch('indices/bars/daily/topix',{'from':start.isoformat(),'to':now.date().isoformat()})
    # Conservative daily bar finalization rule: use today's bar only after 18 JST.
    limit=now.date() if now.hour>=18 else now.date()-dt.timedelta(days=1)
    actual_dates=[date(r.get('Date')) for r in topix_blob['rows'] if number(r.get('C')) is not None]
    valid=[d for d in actual_dates if d and d<=limit and pd.Timestamp(d) in sessions]
    if not valid:raise CollectionError('no_finalized_topix_date')
    t=asof or max(valid)
    if t>max(valid) or pd.Timestamp(t) not in sessions:raise CollectionError('asof_not_available')
    LOG.info('[準備 3/3] 銘柄マスタを取得しています（基準日 %s）',t)
    master_blob=client.fetch('equities/master',{'date':t.isoformat()})
    if not master_blob['rows']:raise CollectionError('master_empty')
    master_records(master_blob['rows']) # validation BEFORE promotion
    return {'calendar':client.store.put(calendar_blob),'topix':client.store.put(topix_blob),
            'master':client.store.put(master_blob)},t



def collect(client: Client, budget: int, force: bool=False, only_codes: Optional[list[str]]=None,
            refresh_days: Optional[int]=None, asof: Optional[dt.date]=None,
            reset_pending: bool=False) -> dict:
    """Collect one bounded batch; resume only from an explicit, parent-checked journal.

    A journal may reference an interrupted snapshot, but that snapshot is NEVER
    promoted for analysis. Only completed per-symbol bundles are reused. A new
    request with explicit codes or reset_pending starts a new target selection.
    """
    if budget<1:raise DataError('budget must be positive')
    store=client.store;previous=store.snapshot()
    resume_path=store.root/'collection_resume.json'
    resume=None;remaining=None;completed_codes=[];failed_codes=[]
    request={'force':bool(force),'refresh_days':refresh_days,
             'asof':asof.isoformat() if asof else None,'config_hash':digest(asdict(client.cfg)),
             'only_codes':sorted({canonical_code(c) for c in only_codes}) if only_codes else None}
    if reset_pending:
        resume_path.unlink(missing_ok=True)
        LOG.info('[再選択] 収集待ちリストを作り直します。既存キャッシュ・履歴は削除しません')
    elif resume_path.exists():
        try:
            journal=json.loads(resume_path.read_text('utf-8'))
            saved=journal.get('request',{})
            compatible=(journal.get('schema')=='jq-collection-resume-v1'
                and journal.get('latest_snapshot_id')==previous.get('snapshot_id')
                and saved.get('config_hash')==request['config_hash']
                and (only_codes is None or saved.get('only_codes')==request['only_codes'])
                and (not force or saved.get('force') is True)
                and (refresh_days is None or saved.get('refresh_days')==refresh_days)
                and (asof is None or saved.get('asof')==asof.isoformat()))
            targets=journal.get('remaining_codes')
            if compatible and isinstance(targets,list) and targets:
                resume=store.snapshot(journal['snapshot_id'])
                remaining=list(dict.fromkeys(canonical_code(c) for c in targets))
                request=saved.copy();force=bool(saved.get('force'))
                refresh_days=saved.get('refresh_days');asof=date(saved.get('asof'))
                LOG.info('[再開] 保存済みの収集待ちリストを使用します: 残り%s銘柄',len(remaining))
            else:
                LOG.info('[収集] 再開情報は現在の指定・基準スナップショットと異なるため再選択します')
        except (DataError,OSError,ValueError,TypeError,KeyError) as e:
            LOG.warning('[収集] 再開情報を利用できないため、確定済みキャッシュから再選択します: %s',e)
            resume=None;remaining=None
    symbols=copy.deepcopy(previous.get('symbols',{}))
    if resume is not None:
        # An incomplete symbol must not replace a previously valid symbol.
        for c,item in resume.get('symbols',{}).items():
            if item.get('complete') is True:symbols[c]=copy.deepcopy(item)
    snap={'schema':CACHE_SCHEMA,'snapshot_id':new_id('s_'),'status':'incomplete',
          'created_at':now_jst().isoformat(),'globals':{},'symbols':symbols,
          'errors':[],'logic_version':LOGIC_VERSION,'parent_snapshot_id':previous.get('snapshot_id'),
          'collection_request':request,'resumed_from_snapshot_id':resume.get('snapshot_id') if resume else None}
    completed=False;planned=None;attempted=[]
    try:
        snap['globals'],t=collect_globals(client,asof)
        snap['price_asof_date']=t.isoformat()
        master=master_records(bundle_rows(store,snap['globals']['master']))
        all_stocks=[m for m in master if m['instrument_type']=='stock']
        stocks=all_stocks
        if only_codes:
            allowed={resolve_code(c,master) for c in only_codes};stocks=[m for m in stocks if m['code'] in allowed]
        candidates=[]
        if remaining is not None:
            by_code={m['code']:m for m in all_stocks}
            candidates=[by_code[c] for c in remaining if c in by_code]
            dropped=len(remaining)-len(candidates)
            if dropped:LOG.warning('[再開] 現在の普通株マスタにない%s銘柄を対象から外しました',dropped)
        else:
            for m in stocks:
                old=snap['symbols'].get(m['code'])
                olddate=date((old or {}).get('checked_price_date'))
                stale=refresh_days is not None and (olddate is None or (t-olddate).days>=refresh_days)
                if force or stale or old is None or old.get('complete') is not True:candidates.append(m)
        planned=[m['code'] for m in candidates]
        count=min(len(candidates),budget)
        mode='強制再収集' if force else f'{refresh_days}日以上古いものを再収集' if refresh_days is not None else '未取得・未完了のみ'
        LOG.info('[収集開始] 対象=%s銘柄 / このバッチ=%s銘柄 / モード=%s',len(candidates),count,mode)
        if not count:LOG.info('[収集] 今回の対象はありません')
        started=time.monotonic()
        start=add_months(t,-12*client.cfg.ADJUSTMENT_HISTORY_YEARS)-dt.timedelta(days=10)
        for i,m in enumerate(candidates[:budget],1):
            code,api=m['code'],m['api_code'];item={'complete':False,'checked_price_date':str(t)}
            attempted.append(code)
            LOG.info('[収集 %s/%s] %s %s（価格・財務・決算予定・需給）',i,count,code,m.get('name',''))
            for kind,ep,p in [
                ('prices','equities/bars/daily',{'code':api,'from':str(start),'to':str(t)}),
                ('summary','fins/summary',{'code':api}),('earnings','fins/earnings-date',{'code':api})]:
                blob=client.fetch(ep,p)
                item[kind]=store.put(blob)
                if kind!='earnings' and not blob['rows']:
                    snap['errors'].append({'code':code,'reason':kind+'_empty_http200'})
            for kind,ep,p in [
                ('margin','markets/margin-interest',{'code':api,'from':str(t-dt.timedelta(days=90)),'to':str(t)}),
                ('short','markets/short-sale-report',{'code':api,'disc_date_from':str(t-dt.timedelta(days=90)),'disc_date_to':str(t)})]:
                if kind=='margin' and now_jst().date()>=dt.date(2026,9,28):
                    item[kind]=store.put({'status':'schema_review_required','rows':[],'collected_at':now_jst().isoformat(),
                        'reason':'2026-09-28 margin schema migration requires adapter audit'});continue
                try:item[kind]=store.put(client.fetch(ep,p))
                except AuthError as e:
                    if '401' in str(e):raise
                    item[kind]=store.put({'status':'permission_unavailable','rows':[],'collected_at':now_jst().isoformat()})
                except CollectionError as e:
                    item[kind]=store.put({'status':'collection_failed','rows':[],'reason':str(e),'collected_at':now_jst().isoformat()})
            item['complete']=bool(bundle_rows(store,item['prices']) and bundle_rows(store,item['summary']))
            item['collected_at']=now_jst().isoformat()
            item['financial_checked_through']=item['collected_at']
            item['earnings_checked_through']=item['collected_at']
            snap['symbols'][code]=item
            if item['complete']:completed_codes.append(code)
            else:failed_codes.append(code)
            if i==1 or i%20==0 or i==count:
                LOG.info('[進捗] %s/%s 完了 / 成功=%s 未完了=%s / 経過%.0f秒',
                         i,count,len(completed_codes),len(failed_codes),time.monotonic()-started)
        snap['pending_count']=sum(not snap['symbols'].get(m['code'],{}).get('complete',False) for m in all_stocks)
        snap['attempted_count']=len(attempted)
        snap['attempted_codes']=attempted
        snap['pending_codes']=[m['code'] for m in candidates[budget:]]
        snap['status']='partial_universe' if snap['pending_count'] or snap['errors'] else 'complete'
        snap['information_cutoff']=now_jst().isoformat()
        snap['created_at']=snap['information_cutoff'];completed=True
        return snap
    finally:
        snap['batch_success_count']=len(completed_codes)
        snap['batch_incomplete_count']=len(failed_codes)
        if not completed:
            snap['status']='incomplete';snap['failure_at']=now_jst().isoformat()
            snap['attempted_codes']=attempted;snap['attempted_count']=len(attempted)
        store.save_snapshot(snap,promote=completed)
        # The journal is a collection-only pointer, never an analysis pointer.
        if planned is not None:
            done=set(completed_codes)
            todo=[c for c in planned if c not in done]
            if todo:
                atomic_json(resume_path,{'schema':'jq-collection-resume-v1','snapshot_id':snap['snapshot_id'],
                    'latest_snapshot_id':snap['snapshot_id'] if completed else previous.get('snapshot_id'),
                    'request':request,'remaining_codes':todo,'saved_at':now_jst().isoformat()})
            else:resume_path.unlink(missing_ok=True)
        if not completed:
            LOG.warning('[保存] 未完了スナップショット %s / 今回完了=%s銘柄。分析用の最新状態には反映していません',
                        snap['snapshot_id'],len(completed_codes))


def merge_daily_rows(old: list[dict], new: list[dict], kind: str) -> list[dict]:
    if kind=='prices':key=lambda r:(str(r.get('Code')),str(r.get('Date')))
    elif kind=='summary':
        key=lambda r:(str(r.get('Code')),str(r.get('DiscNo')) if r.get('DiscNo') else digest(r))
    elif kind=='earnings':key=lambda r:(str(r.get('Code')),str(r.get('PubDate')),str(r.get('FQName')),str(r.get('FYE')))
    else:key=lambda r:digest(r)
    merged={key(r):r for r in old}
    merged.update({key(r):r for r in new})
    return [merged[k] for k in sorted(merged)]


def update_daily(client: Client, from_date: Optional[dt.date]=None) -> dict:
    """Refresh all previously initialized symbols via date requests, not API cursor.

    The overlap begins at the oldest per-symbol verification date. Calendar days
    are included for financial/earnings publication. Late corrections carrying
    older disclosure dates require periodic full symbol refresh (documented).
    """
    store=client.store;previous=store.snapshot()
    if not previous:raise CollectionError('First run collect; there is no initial history to update')
    snap=copy.deepcopy(previous);snap.update(snapshot_id=new_id('s_'),status='incomplete',errors=[],
                                           parent_snapshot_id=previous.get('snapshot_id'))
    completed=False
    try:
        snap['globals'],t=collect_globals(client)
        snap['price_asof_date']=str(t)
        master=master_records(bundle_rows(store,snap['globals']['master']))
        allowed={m['api_code']:m['code'] for m in master if m['instrument_type']=='stock'}
        symbols={c:i for c,i in snap.get('symbols',{}).items() if i.get('complete') and c in set(allowed.values())}
        if not symbols:raise CollectionError('No complete symbol histories')
        oldest=min(date(i.get('financial_checked_through')) or date(previous['price_asof_date']) for i in symbols.values())
        start=from_date or oldest
        if start>oldest:raise CollectionError('update-from would skip unverified financial dates')
        end=now_jst().date()
        data={c:{k:bundle_rows(store,item.get(k)) for k in ('prices','summary','earnings')} for c,item in symbols.items()}
        sessions=trading_sessions(bundle_rows(store,snap['globals']['calendar']))
        d=start;raw_update_refs=[]
        while d<=end:
            endpoints=[('summary','fins/summary'),('earnings','fins/earnings-date')]
            if d<=t and pd.Timestamp(d) in sessions:endpoints.insert(0,('prices','equities/bars/daily'))
            for kind,ep in endpoints:
                blob=client.fetch(ep,{'date':str(d)})
                raw_update_refs.append(store.put(blob))
                for raw in blob['rows']:
                    c=allowed.get(str(raw.get('Code') or ''))
                    if c in data:data[c][kind]=merge_daily_rows(data[c][kind],[raw],kind)
            LOG.info('日付更新 %s',d);d+=dt.timedelta(days=1)
        observed=now_jst().isoformat()
        for c,content in data.items():
            for kind,rows in content.items():
                snap['symbols'][c][kind]=store.put({'status':'complete','rows':rows,'collected_at':observed,
                    'provenance':'prior_full_history_plus_complete_date_updates','parent_ref':symbols[c].get(kind),
                    'update_raw_refs':raw_update_refs})
            snap['symbols'][c].update(checked_price_date=str(t),collected_at=observed,
                                     financial_checked_through=observed,earnings_checked_through=observed)
        snap['information_cutoff']=observed;snap['created_at']=observed
        n=sum(m['instrument_type']=='stock' for m in master)
        snap['status']='complete' if len(symbols)==n else 'partial_universe'
        snap['pending_count']=n-len(symbols);completed=True
        return snap
    finally:
        if not completed:snap['status']='incomplete';snap['failure_at']=now_jst().isoformat()
        store.save_snapshot(snap,promote=completed)


# ---- Analysis pipeline: no Client, Session or credential lookup --------------
def analyze_symbol(m: dict, item: dict, store: Store, sessions: pd.DatetimeIndex,
                   asof: dt.date, cutoff: dt.datetime, regime: dict, cfg: Config,
                   next_trade: Optional[dt.date], global_issues: list[str]) -> tuple[dict,list[dict]]:
    r={**m,**regime,'price_asof_date':str(asof),'information_cutoff':cutoff.isoformat(),
       'logic_version':LOGIC_VERSION,'code_version':CODE_VERSION,'financial_data_mode':'summary_only',
       'structural_limitations':['summary_only','gross_margin_not_in_model','current_ratio_not_in_model'],
       'common_issues':list(global_issues),'financial_issues':[],'light_issues':[],
       'collected_at':item.get('collected_at'),'pretrade_check_required':True,
       'benchmark_input_valid':False,'annual_comparable':False,'critical_missing_count':0,
       'summary_quality_score':0,'summary_quality_available':0,'liquidity_ok':None,
       'market_cap':None,'market_cap_ok':None,'op_income_stable':None,
       'ma200_state':'ma200_unknown','earnings_blackout':False,
       'data_review_level':'none','model_scope':m.get('model_scope','nonfinancial')}
    r.update(risk_context(cfg,m['code'],cutoff))
    if m['instrument_type']!='stock':return r,[]
    if not item.get('complete'):
        r['common_issues'].append('symbol_collection_incomplete');return r,[]
    try:
        frame=price_frame(bundle_rows(store,item.get('prices')),m['api_code'],sessions,asof)
        r.update(technicals(frame,asof,cfg))
    except (DataError,KeyError,ValueError) as e:
        r['common_issues'].append(str(e));return r,[]
    liq=[r.get('avg_volume_30d'),r.get('adv_jpy_20d')]
    r['liquidity_ok']=None if any(x is None for x in liq) else liq[0]>=cfg.MIN_AVG_VOLUME_30D and liq[1]>=cfg.MIN_ADV_JPY_20D
    try:
        raw_stmts=bundle_rows(store,item.get('summary'))
        statements,issues=normalize_statements(raw_stmts,m['api_code'],m['code'],cutoff,cfg)
        hist=annual_history(statements)
        adjust_financial_shares(hist,frame,asof,cfg,m['code'])
    except (DataError,KeyError,ValueError) as e:
        r['financial_issues'].append(str(e));return r,[]
    if r['model_scope']!='financial':r['financial_issues']+=issues
    current=hist[0] if hist else {}
    r['financial_asof']=str(current.get('period_end')) if current.get('period_end') else None
    r['financial_disclosed_at']=clean(current.get('disclosed_at'))
    r['financial_field_sources']=current.get('field_sources',{})
    r['comparability_flags']=current.get('comparability_flags',[])
    r['currency']=current.get('currency');r['currency_source']=current.get('currency_source')
    if r['model_scope']!='financial':
        if current.get('currency')!='JPY':r['financial_issues'].append('currency_unknown_or_not_jpy')
        if any(x.get('currency')!=current.get('currency') for x in hist[:3]):r['financial_issues'].append('currency_not_comparable')
    r['annual_comparable']=consecutive(hist,3)
    r['annual_disclosure_age_days']=(cutoff.date()-current['disclosed_at'].date()).days if current.get('disclosed_at') else None
    r['annual_period_age_days']=(asof-current['period_end']).days if current.get('period_end') else None
    r['annual_stale']=bool((r['annual_disclosure_age_days'] is not None and r['annual_disclosure_age_days']>cfg.MAX_ANNUAL_DISCLOSURE_AGE_DAYS)
                          or (r['annual_period_age_days'] is not None and r['annual_period_age_days']>cfg.MAX_ANNUAL_PERIOD_AGE_DAYS))
    r['days_since_original_annual_release']=(cutoff.date()-current['first_disclosed_at'].date()).days if current.get('first_disclosed_at') else None
    if r['model_scope']!='financial':
        missing=[k for k in CRITICAL if current.get(k) is None]
        r['critical_missing_fields']=missing;r['critical_missing_count']=len(missing)
        if missing:r['financial_issues'].append('critical_financial_fields_missing')
        if not r['annual_comparable']:r['financial_issues'].append('insufficient_or_noncomparable_annual_history')
        if current.get('total_assets') is not None and current['total_assets']<=0:r['financial_issues'].append('invalid_total_assets')
    r['financial_history_audit']=[{k:clean(v) for k,v in h.items() if k!='raw'} for h in hist]
    # A collection watermark older than the price day cannot certify fresh guidance.
    for name in ('financial','earnings'):
        checked=timestamp(item.get(name+'_checked_through'))
        if checked is None or checked.date()<asof:
            r['common_issues'].append(name+'_collection_stale')
    mcap=number(frame['MktCap'].iloc[-1])
    shares_rows=[s for s in statements if s.get('shares_outstanding') is not None and s.get('period_end') and s['period_end']<=asof]
    latest_sh=max(shares_rows,key=lambda s:(s['period_end'],s['disclosed_at'],s['scope']=='consolidated')) if shares_rows else {}
    r['shares_asof_date']=clean(latest_sh.get('period_end'))
    r['shares_age_days']=(asof-latest_sh['period_end']).days if latest_sh else None
    if mcap is not None and mcap>0:
        r['market_cap']=mcap*1_000_000.;r['market_cap_source']='jquants_daily'
    else:
        f,why=share_factor(frame,latest_sh.get('period_end'),asof)
        sh=number(latest_sh.get('shares_outstanding'))
        if f is not None and sh is not None and sh>0:
            r['market_cap']=r['price']*sh*f;r['market_cap_source']='disclosed_shares_proxy'
            r['light_issues'].append('disclosed_shares_market_cap_proxy')
        else:r['common_issues'].append('market_cap_basis_unresolved:'+why)
    r['market_cap_ok']=None if r['market_cap'] is None else r['market_cap']>=cfg.MIN_MARKET_CAP_JPY
    r.update(operating_stability(hist,cfg));r.update(summary_quality(hist))
    r['summary_quality_coverage_rate']=r['summary_quality_score']/7. if r['summary_quality_available']==7 else None
    if r['model_scope']!='financial' and any(h.get('share_adjustment_factor') is None for h in hist[:2]):
        r['financial_issues'].append('corporate_action_or_share_basis_unresolved')
    r.update(guidance(statements,hist))
    r.update(valuation(r['market_cap'],hist,r,current.get('currency')=='JPY'))
    r.update(earnings_quality(hist,r['market_cap']));r.update(shareholder_quality(hist))
    r.update(peg_quality(r.get('reference_peg'),r.get('eps_proxy_endshares_yoy_pct'),
                        r.get('forecast_net_income') is not None,r['earnings_quality_flag']=='watch'))
    r['sales_cagr']=None
    if consecutive(hist,4) and all(h.get('revenue') is not None and h['revenue']>0 for h in hist[:4]):
        r['sales_cagr']=(hist[0]['revenue']/hist[3]['revenue'])**(1/3)-1
    r['op_income_yoy_pct']=growth(current.get('operating_income'),hist[1].get('operating_income'),100) if consecutive(hist,2) else None
    for k in ('operating_cash_flow','cash_and_equivalents','equity_ratio','net_assets_ratio','net_income','revenue','operating_income'):
        r[k]=current.get(k)
    r['dividend_status']=None
    if consecutive(hist,2) and all(h.get('dps_adjusted') is not None for h in hist[:2]):
        a,b=hist[0]['dps_adjusted'],hist[1]['dps_adjusted']
        r['dividend_status']='増配' if a>b*1.02 else '減配' if a<b*.95 else '維持' if a>0 else '無配'
    try:
        if not item.get('earnings'):raise DataError('earnings_not_collected')
        r.update(earnings_context(bundle_rows(store,item['earnings']),m['api_code'],cutoff,sessions,next_trade,cfg))
    except (DataError,KeyError,ValueError) as e:r['common_issues'].append(str(e))
    r.update(margin_diagnostic(store.get(item['margin']) if item.get('margin') else {},cutoff))
    short=store.get(item['short']) if item.get('short') else {}
    visible_shorts=[x for x in short.get('rows',[]) if published_at(x) is not None and published_at(x)<=cutoff]
    r.update(short_report_status=short.get('status','not_collected'),short_report_event_count=len(visible_shorts),
             short_total_balance=None,short_selling_change_rate=None,
             short_report_last_date=max((x.get('DiscDate','') for x in visible_shorts),default=None))
    r['benchmark_input_valid']=bool(not r['common_issues'] and r['ps'] is not None and r['ps']>0
                                  and r['S33'] in SECTORS and current.get('currency')=='JPY'
                                  and current and normal_year(current.get('period_start'),current.get('period_end'))
                                  and not r['annual_stale'])
    r.update(persistence(r,hist))
    return r,hist


def score_snapshot(store: Store, snap: dict, cfg: Config, asof: Optional[dt.date]=None,
                   cutoff: Optional[dt.datetime]=None) -> tuple[list[dict],dict,dict]:
    cfg.validate()
    t=asof or date(snap.get('price_asof_date'))
    information=cutoff or timestamp(snap.get('information_cutoff'))
    if t is None or information is None:raise DataError('snapshot_missing_asof_or_cutoff')
    if t>information.date():raise DataError('price_after_information_cutoff')
    snapshot_cutoff=timestamp(snap.get('information_cutoff'))
    if snapshot_cutoff is not None and information>snapshot_cutoff:
        raise DataError('information_cutoff_after_snapshot_verification')
    global_issues=[];globals=snap.get('globals',{})
    # This daily batch build deliberately uses a conservative 18:00 JST
    # publication boundary, NOT an assertion about the exchange closing time.
    if information.date()==t and information.time()<dt.time(18):
        global_issues.append('daily_publication_window_not_confirmed')
    if snap.get('status')=='incomplete':global_issues.append('snapshot_incomplete')
    calrows=bundle_rows(store,globals.get('calendar'))
    sessions=trading_sessions(calrows) if calrows else pd.DatetimeIndex([])
    if len(sessions)==0 or pd.Timestamp(t) not in sessions:global_issues.append('calendar_missing_or_asof_invalid')
    future=[s.date() for s in sessions if dt.datetime.combine(s.date(),dt.time(9),JST)>information]
    next_trade=min(future) if future else None
    if next_trade is None:global_issues.append('next_trading_session_unknown')
    rawmaster=bundle_rows(store,globals.get('master'))
    if not rawmaster:global_issues.append('master_missing')
    master=master_records(rawmaster)
    if any(date(m.get('master_date'))!=t for m in master):global_issues.append('master_asof_mismatch')
    topix=bundle_rows(store,globals.get('topix'))
    regime=market_regime(topix,sessions,t)
    if regime['market_regime']=='unknown':global_issues.append('topix_required')
    records=[]
    for idx,m in enumerate(master,1):
        if m['instrument_type']!='stock':continue
        item=snap.get('symbols',{}).get(m['code'],{})
        try:r,hist=analyze_symbol(m,item,store,sessions,t,information,regime,cfg,next_trade,global_issues)
        except DataError as e:
            r={**m,**regime,'common_issues':[str(e)],'financial_issues':[],'light_issues':[],
               'benchmark_input_valid':False,'annual_comparable':False,'critical_missing_count':0,
               'liquidity_ok':None,'market_cap':None,'market_cap_ok':None,'op_income_stable':None,
               'ma200_state':'ma200_unknown','summary_quality_score':0,'summary_quality_available':0,
               'price_asof_date':str(t),'information_cutoff':information.isoformat()}
        records.append(r)
        if idx%250==0:LOG.info('オフライン指標計算 %s/%s',idx,len(master))
    universe_id=digest({'snapshot':snap.get('snapshot_id'),'asof':str(t),'cutoff':information.isoformat(),'config':asdict(cfg)})[:20]
    benchmarks=fix_benchmarks(records,cfg,universe_id)
    for r in records:
        r.update(fe_score(r,cfg));r.update(all_gates(r,cfg));r.update(recommendation(r))
        r.update(scenario_values(r,cfg,information))
        stop=None;method=None
        if r['candidate_lane'] in CORE_LANES and r.get('price') is not None:
            if r['candidate_lane']=='ma200_reclaim_core' and r.get('ma_200') is not None and r.get('adjusted_close'):
                stop=r['ma_200']*r['price']/r['adjusted_close']*.95;method='ma200_x_0.95_reference'
            elif r.get('vol_daily_20d') is not None:
                stop=r['price']*(1-clip(2.5*r['vol_daily_20d'],.05,.18));method='daily_vol_reference'
        r.update(stop_reference_price=stop,stop_distance_ratio=1-stop/r['price'] if stop is not None else None,
                 stop_method=method,legacy_tp_reference=(r['price']*(1.25 if r['candidate_lane']=='bottom_reversal_core' else 1.12 if r['candidate_lane']=='extended_above_ma200' else 1.20)) if r.get('price') else None,
                 reward_to_risk_reference=None,eligible_is_order_permission=False)
        if stop is not None and r['price']>stop and r.get('scenario_price_base') is not None:
            r['reward_to_risk_reference']=(r['scenario_price_base']-r['price'])/(r['price']-stop)
    legacy_reference_scores(records,cfg)
    records.sort(key=rank_key)
    complete_count=sum(snap.get('symbols',{}).get(r['code'],{}).get('complete',False) for r in records)
    status='diagnostic_only' if global_issues else 'partial_universe' if complete_count<len(records) or any(r.get('common_issues') for r in records) else 'complete'
    meta={'price_asof_date':str(t),'information_cutoff':information.isoformat(),'snapshot_id':snap.get('snapshot_id'),
          'logic_version':LOGIC_VERSION,'code_version':CODE_VERSION,'schema':CACHE_SCHEMA,
          'next_trade_date':str(next_trade) if next_trade else None,'run_status':status,
          'global_issues':global_issues,'ordinary_master_count':len(records),'complete_symbol_count':int(complete_count),
          'eligible_count':sum(r['eligible_new_entry'] for r in records),'percentile_universe_id':universe_id,
          'pit_quality':'archived_input_snapshot_not_guaranteed_historical_vintage',
          'non_stock_master_rows':[m for m in master if m['instrument_type']!='stock'],
          'config':asdict(cfg),'config_hash':digest(asdict(cfg)),'input_hash':digest(snap)}
    return clean(records),clean(benchmarks),clean(meta)

# ---- Reports: filter only, never re-score ------------------------------------
MIN_COLUMNS = [
    'run_id','logic_version','code_version','code','api_code','name','S33','sector','instrument_type',
    'price_asof_date','information_cutoff','collected_at','financial_asof','shares_asof_date',
    'financial_data_mode','currency','currency_source','structural_limitations','data_review_level',
    'data_review_reasons','price','market_cap','market_cap_source','ps','per_actual','per_forward',
    'per_conservative','per_basis','eps_proxy_endshares_yoy_pct','np_yoy_pct','forward_np_change',
    'summary_quality_score','summary_quality_available','summary_quality_coverage',
    'Q1','Q2','Q3','Q4','Q5','Q6','Q7','quality_normalized_9','ma_25','ma_75','ma_200',
    'distance_from_ma200','rsi','adx','plus_di','minus_di','return_21d','return_63d','return_126d','return_252d',
    'vol_daily_20d','vol_annual_20d','vol_daily_reference','vol_annual_reference','ma200_state',
    'crossed_above_ma200_recently','above_ma200_extended','below_ma200_basing','below_ma200_downtrend',
    'valuation_lane','core_candidate','candidate_lane','eligible_new_entry',
    'fundamental_edge_score','entry_score_raw','entry_score','entry_soft_cap_base','recommendation_score',
    'rec_priority','rec_secondary','legacy_total','total_score','grade','grade_raw','gate_reasons',
    'percentile_universe_id','fe_components','fe_penalties','recommendation_adjustments',
    'guidance_status','guidance_target','earnings_status','earnings_blackout','pretrade_check_required',
    'rerating_space_ps','positive_cfo_years_3y','cumulative_cfo_to_np_3y','return_rel_topix_63d',
    'ma200_cross_count_60d','turnover_persistence','positive_return_top3_share_63d',
    'scenario_price_bear','scenario_price_base','scenario_price_bull','scenario_confidence',
    'stop_reference_price','stop_distance_ratio','stop_method','legacy_tp_reference',
    'suggested_take_profit','recommended_shares',
]


def csv_bytes(records: list[dict], columns: list[str]) -> bytes:
    f=io.StringIO(newline='');writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n')
    writer.writeheader()
    for r in records:
        row={}
        for k in columns:
            v=clean(r.get(k))
            if isinstance(v,(list,dict)):v=encoded(v).decode('utf-8')
            row[k]='' if v is None else v
        writer.writerow(row)
    return f.getvalue().encode('utf-8-sig')


def md_escape(x: Any) -> str:
    return str(x if x is not None else 'N/A').replace('|','\\|').replace('\n',' ')


def fmt(x: Any) -> str:
    v=number(x);return f'{v:.2f}' if v is not None else 'N/A'


def markdown_report(records: list[dict], meta: dict, top: int) -> str:
    buy=[r for r in records if r.get('eligible_new_entry') is True][:top]
    watch=[r for r in records if r.get('candidate_lane') in {'watch_fundamental_core','weak_reclaim_watch','financial_quality_watch','extended_above_ma200','earnings_watch','forward_downgrade_watch'}][:top]
    review=[r for r in records if r.get('candidate_lane') in {'data_review','data_review_light','model_scope_review'}][:top]
    lines=['# Standard版スクリーニング結果','',f"Run: `{meta['run_id']}` / {LOGIC_VERSION}",
           f"価格基準日: {meta['price_asof_date']} / 情報締切: {meta['information_cutoff']}",
           f"状態: **{meta['run_status']}** / 普通株マスタ {meta['ordinary_master_count']} / 初回収集済 {meta['complete_symbol_count']}",
           '', '**定量候補であり注文許可ではありません。ガバナンス等の個別確認は必要です。**',
           'FEとentryは共通成分を持ちます。PEG品質はFE/レーンに影響します。',
           '上昇余地の業種倍率差・持続性列は診断用で、期待リターンではありません。',
           'summary_onlyは正常モードです。通貨・株数基準不明は別途レビューします。','']
    if meta.get('global_issues'):lines+=['グローバル不足: '+', '.join(meta['global_issues']),'']
    for title,part in [('新規定量候補',buy),('監視候補',watch),('データ・適用範囲レビュー',review)]:
        lines+=['## '+title,'']
        if not part:lines+=['候補なし。',''];continue
        lines+=['|コード|銘柄|レーン|FE|entry複合点|推奨順位点|品質 k/7（取得n/7）|PS|PER|業種倍率差|理由|',
                '|---|---|---|---:|---:|---:|---|---:|---:|---:|---|']
        for r in part:
            vals=[r.get('code'),r.get('name'),r.get('candidate_lane'),fmt(r.get('fundamental_edge_score')),
                  fmt(r.get('entry_score')),fmt(r.get('recommendation_score')),
                  f"{r.get('summary_quality_score',0)}/7（{r.get('summary_quality_available',0)}/7）",
                  fmt(r.get('ps')),fmt(r.get('per')),fmt(r.get('rerating_space_ps')),
                  ', '.join(r.get('gate_reasons',[]))]
            lines.append('|'+ '|'.join(md_escape(v) for v in vals)+'|')
        lines.append('')
    return '\n'.join(lines)+'\n'


def write_run(records: list[dict], benchmarks: dict, meta: dict, output: Path,
              top: int=10, run_id: Optional[str]=None) -> Path:
    run_id=run_id or new_id('r_')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id):raise DataError('invalid_run_id')
    rows=copy.deepcopy(records);meta=copy.deepcopy(meta);meta['run_id']=run_id
    meta['generated_at']=now_jst().isoformat()
    source=Path(__file__)
    meta['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    for r in rows:r['run_id']=run_id;r.setdefault('logic_version',LOGIC_VERSION);r.setdefault('code_version',CODE_VERSION)
    meta['result_hash_excluding_run_id']=digest([{k:v for k,v in r.items() if k!='run_id'} for r in rows])
    history=output/'history';history.mkdir(parents=True,exist_ok=True)
    final=history/run_id
    if final.exists():raise DataError('immutable_run_already_exists')
    stage=history/('.incomplete_'+run_id);stage.mkdir()
    cols=MIN_COLUMNS+sorted({k for r in rows for k in r}-set(MIN_COLUMNS))
    groups={
        'screening_offline.csv':rows,
        f'screening_offline_{run_id}.csv':rows,
        'core_candidates.csv':[r for r in rows if r.get('core_candidate') is True],
        'top_recommended_core.csv':[r for r in rows if r.get('eligible_new_entry') is True][:top],
        'watch_candidates.csv':[r for r in rows if r.get('candidate_lane') in {'watch_fundamental_core','weak_reclaim_watch','extended_above_ma200','earnings_watch','forward_downgrade_watch','financial_quality_watch'}],
        'data_review_candidates.csv':[r for r in rows if r.get('candidate_lane') in {'data_review','data_review_light','model_scope_review'}],
        'excluded.csv':[r for r in rows if r.get('candidate_lane') in {'excluded','risk_blocked','cyclical_value_trap'}],
        'satellite_valuation_candidates.csv':[r for r in rows if r.get('candidate_lane')=='satellite_valuation'],
        'satellite_ps_only_candidates.csv':[r for r in rows if r.get('candidate_lane')=='satellite_ps_only'],
    }
    for lane in SOFT_BASES:
        groups.setdefault(lane+'_candidates.csv',[r for r in rows if r.get('candidate_lane')==lane])
    groups['non_stock_excluded.csv']=meta.get('non_stock_master_rows',[])
    try:
        for name,data in groups.items():atomic_bytes(stage/name,csv_bytes(data,cols))
        atomic_json(stage/'screening_results.json',rows)
        atomic_json(stage/'sector_benchmarks.json',benchmarks)
        counts={k:dict(Counter('unknown' if r.get(k) is None else str(r.get(k)).lower() for r in rows)) for k in
                ('liquidity_ok','market_cap_ok','op_income_stable','core_candidate','eligible_new_entry')}
        summary={'run_id':run_id,'eligible_count':sum(r.get('eligible_new_entry') is True for r in rows),
                 'lane_counts':dict(Counter(r.get('candidate_lane') for r in rows)),
                 'gate_counts':counts,'reason_counts':dict(Counter(x for r in rows for x in r.get('gate_reasons',[])))}
        atomic_json(stage/'filter_summary.json',summary)
        report=markdown_report(rows,meta,top).encode('utf-8')
        atomic_bytes(stage/'report_core_top10.md',report)
        atomic_bytes(stage/'report_investment_advice_core.md',report)
        meta['outputs']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.iterdir())}
        atomic_json(stage/'run_manifest.json',meta)
        os.replace(stage,final)
        # Complete history is immutable; publish display and then its pointer.
        latest_parent=output/'reports';latest_parent.mkdir(parents=True,exist_ok=True)
        temp_latest=latest_parent/('.latest_'+run_id)
        shutil.copytree(final,temp_latest)
        latest=latest_parent/'latest';previous=latest_parent/('.previous_'+run_id)
        if latest.exists():os.replace(latest,previous)
        try:
            os.replace(temp_latest,latest)
            atomic_json(output/'latest_run.json',{'run_id':run_id,'history_path':str(final.resolve())})
        except BaseException:
            if latest.exists() and not temp_latest.exists():os.replace(latest,temp_latest)
            if previous.exists() and not latest.exists():os.replace(previous,latest)
            raise
        if previous.exists():
            try:shutil.rmtree(previous) # display copy only; never history
            except OSError:LOG.warning('Old display copy retained: %s',previous)
        return final
    except BaseException:
        if stage.exists():atomic_json(stage/'run_manifest.json',{**meta,'run_status':'incomplete'})
        raise


def analyze_store(store: Store, cfg: Config, output: Path, top: int=10,
                  snapshot_id: Optional[str]=None, asof: Optional[dt.date]=None,
                  cutoff: Optional[dt.datetime]=None) -> Path:
    snap=store.snapshot(snapshot_id)
    if not snap:raise DataError('凍結スナップショットがありません。先に collect を実行してください')
    LOG.info('[分析開始] スナップショット=%s / 収集済み=%s銘柄（API通信なし）',snap.get('snapshot_id'),sum(i.get('complete') is True for i in snap.get('symbols',{}).values()))
    records,stats,meta=score_snapshot(store,snap,cfg,asof,cutoff)
    with ProcessLock(output,'analysis'):
        path=write_run(records,stats,meta,output,top)
    LOG.info('分析完了: eligible=%s status=%s -> %s',meta['eligible_count'],meta['run_status'],path)
    return path


def single_from_run(code: str, output: Path, run_id: Optional[str]=None) -> dict:
    c=canonical_code(code)
    if run_id is None:
        pointer=output/'latest_run.json'
        if not pointer.exists():raise DataError('先に analyze を実行してください。単独で業種順位は再計算しません')
        run_id=json.loads(pointer.read_text('utf-8'))['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id):raise DataError('invalid_run_id')
    root=output/'history'/run_id
    manifest=json.loads((root/'run_manifest.json').read_text('utf-8'))
    data=(root/'screening_results.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest['outputs']['screening_results.json']:
        raise DataError('result_hash_mismatch')
    hit=[r for r in json.loads(data) if r.get('code')==c]
    if not hit:raise DataError('code_not_in_selected_run:'+c)
    dst=output/'single'/run_id
    atomic_json(dst/(c+'.json'),hit[0])
    cols=MIN_COLUMNS+sorted(set(hit[0])-set(MIN_COLUMNS))
    atomic_bytes(dst/(c+'.csv'),csv_bytes(hit,cols))
    return hit[0]


def audit_snapshot(store: Store, output: Path, snapshot_id: Optional[str]=None) -> Path:
    snap=store.snapshot(snapshot_id)
    if not snap:raise DataError('監査用スナップショットがありません')
    counter={};symbol_audit=[]
    def accumulate(kind,rows):
        for r in rows:
            for k,v in r.items():
                key=(kind,k);b=counter.setdefault(key,{'dataset':kind,'field':k,'present_count':0,'nonnull_count':0,'example':None})
                b['present_count']+=1
                if v is not None and str(v).strip()!='':
                    b['nonnull_count']+=1
                    if b['example'] is None:b['example']=str(v)[:150]
    accumulate('master',bundle_rows(store,snap.get('globals',{}).get('master')))
    for code,item in snap.get('symbols',{}).items():
        latest={}
        for kind in ('prices','summary','earnings','margin','short'):
            if not item.get(kind):continue
            blob=store.get(item[kind]);rows=blob.get('rows',[]);accumulate(kind,rows)
            latest[kind+'_rows']=len(rows);latest[kind+'_status']=blob.get('status')
            if kind=='prices' and rows:
                p=max(rows,key=lambda r:str(r.get('Date','')))
                latest.update(MktCap=p.get('MktCap'),ExRT_column_present='ExRT' in p,AdjFactor=p.get('AdjFactor'),Va=p.get('Va'))
            if kind=='summary' and rows:
                fy=[r for r in rows if r.get('CurPerType')=='FY' and 'FinancialStatements' in str(r.get('DocType',''))]
                if fy:
                    r=max(fy,key=lambda x:(str(x.get('CurPerEn','')),str(x.get('DiscDate',''))))
                    latest.update(latest_FY=r.get('CurPerEn'),DocType=r.get('DocType'),Sales=r.get('Sales'),NP=r.get('NP'),CFO=r.get('CFO'),
                                  currency_field=r.get('Currency'),ShOutFY=r.get('ShOutFY'),DivAnn=r.get('DivAnn'),DivTotalAnn=r.get('DivTotalAnn'))
        symbol_audit.append({'code':code,**latest})
    root=output/'audits'/new_id('audit_');root.mkdir(parents=True)
    rows=list(counter.values())
    atomic_bytes(root/'raw_fields.csv',csv_bytes(rows,['dataset','field','present_count','nonnull_count','example']))
    cols=['code']+sorted({k for r in symbol_audit for k in r}-{'code'})
    atomic_bytes(root/'symbols.csv',csv_bytes(symbol_audit,cols))
    atomic_json(root/'snapshot_manifest.json',snap)
    atomic_bytes(root/'AUDIT_REQUIRED.md',(
        '# 実レスポンス監査\n\nMktCapは百万円、Vaは円。財務通貨は値の桁だけで推定しません。\n'
        '決算短信と生summaryのSales/NP等を照合し、通貨・単位が確認できた範囲だけconfigのcurrency_rules/currency_overridesへ記録してください。\n'
        '本コマンドは監査を自動合格にしません。ExRT列、AdjFactor、FY期間、ShOutFY、Eq/ShEq、DivAnn/DivFYを確認してください。\n').encode('utf-8'))
    return root



class MenuCancelled(Exception):
    """Return from a sub-prompt to the menu without starting an operation."""


def _interactive_input(prompt: str) -> str:
    # Propagate Ctrl+C / Ctrl+Z(EOF) to main(), including at every sub-prompt.
    return input(prompt).strip()


def _input_integer(prompt: str, default: int, minimum: int=1) -> int:
    while True:
        text=_interactive_input(prompt)
        if text.lower()=='q':raise MenuCancelled()
        if not text:return default
        try:value=int(text)
        except ValueError:value=minimum-1
        if value>=minimum:return value
        print(f'{minimum}以上の整数を入力してください。Enterで既定値、qでメニューに戻ります。',flush=True)


def wait_until_next_day() -> None:
    """Only collect_all waits on an explicitly configured self daily quota."""
    now=now_jst()
    target=dt.datetime.combine(now.date()+dt.timedelta(days=1),dt.time(0,0,10),JST)
    LOG.warning('[日次上限] JQ_RPDの自己設定上限に達しました。%sまで待機します（Ctrl+Cで中断・再開情報は保存済み）',target.isoformat())
    seconds=max(0.,(target-now).total_seconds())
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        time.sleep(min(1.,max(0.,until-time.monotonic())))


def update_sector_statistics(store: Store, cfg: Config, output: Path,
                             snapshot_id: Optional[str]=None, asof: Optional[dt.date]=None,
                             cutoff: Optional[dt.datetime]=None) -> Path:
    """Offline canonical full-universe calculation, not a subset re-ranking.

    This command exports an audited sector table; it does not change the latest
    stock-analysis run. analyze always fixes benchmarks from its own same input.
    """
    snap=store.snapshot(snapshot_id)
    if not snap:raise DataError('キャッシュ不足。先にメニュー1または5で収集してください')
    LOG.info('[セクター平均] 凍結キャッシュ全体から計算しています（API通信なし、中央値を使用）')
    records,stats,meta=score_snapshot(store,snap,cfg,asof,cutoff)
    table=[]
    for sec in SECTORS:
        info=dict(stats.get(sec,{}))
        per=[r['per'] for r in records if r.get('S33')==sec and r.get('benchmark_input_valid')
             and r.get('per') is not None and r['per']>0]
        info.update(per_median=float(np.median(per)) if len(per)>=cfg.MIN_SECTOR_VALID_SAMPLES else None,
                    per_sample_count=len(per))
        table.append({'S33':sec,'sector':SECTORS[sec],**info})
    with ProcessLock(output,'sector-update'):
        root=output/'audits'/new_id('sector_');root.mkdir(parents=True)
        atomic_bytes(root/'sector_averages.csv',csv_bytes(table,['S33','sector','ps_median','sample_count','per_median','per_sample_count']))
        payload={'snapshot_id':snap['snapshot_id'],'price_asof_date':meta['price_asof_date'],
                 'config_hash':meta['config_hash'],'percentile_universe_id':meta['percentile_universe_id'],
                 'run_status':meta['run_status'],'global_issues':meta['global_issues'],'sectors':table}
        atomic_json(root/'sector_averages.json',payload)
        atomic_json(output/'sector_averages.json',payload)
    for row in table:
        if row['sample_count'] or row['per_sample_count']:
            ps_text='N/A' if row['ps_median'] is None else f"{row['ps_median']:.2f}"
            per_text='N/A' if row['per_median'] is None else f"{row['per_median']:.2f}"
            print(f"  {row['sector']}: PS={ps_text} PER={per_text} PS有効数={row['sample_count']} PER有効数={row['per_sample_count']}",flush=True)
    LOG.info('[セクター平均] 保存完了。銘柄の採点済み結果は変更していません')
    return root


def run_interactive(args: argparse.Namespace, cfg: Config) -> int:
    default_budget=cfg.JQ_DEFAULT_BUDGET if args.budget is None else args.budget
    print(f'キャッシュ: {args.cache_dir.resolve()}',flush=True)
    print(f'出力先: {args.output_dir.resolve()}',flush=True)
    print('Ctrl+C / Ctrl+Z で終了。入力途中の q はメニューに戻ります。',flush=True)
    while True:
        print('\n=== メニュー ===',flush=True)
        print('1) 収集（価格+財務を凍結保存）')
        print('2) オフライン一括分析（core/satellite/excluded 出力）')
        print('3) 単銘柄分析（キャッシュ使用）')
        print('4) セクター平均を更新（キャッシュから計算）')
        print('5) 全銘柄ゆっくり収集（自動待機・再開可）')
        print('6) 鮮度で取り直し収集（例: 7日より古いものだけ）')
        print('7) 全銘柄“強制”再収集（pending初期化＋当日再取得）')
        print('8) キャッシュ・生フィールド監査（通信なし）')
        print('9) 日付単位で差分更新（収集済み銘柄）')
        print('q) 終了',flush=True)
        choice=_interactive_input('選択: ').lower()
        if choice=='q':
            print('[終了] 終了しました',flush=True);return 0
        mapping={'1':'collect','2':'analyze','3':'single','4':'sector-update',
                 '5':'collect_all','6':'collect_all','7':'collect_all','8':'cache-audit','9':'update'}
        if choice not in mapping:
            print('無効な選択です。1〜9 または q を入力してください。',flush=True);continue
        nxt=copy.copy(args);nxt.phase=mapping[choice];nxt._menu_child=True
        try:
            if choice in {'1','5','6','7'}:
                # Each menu operation has the legacy meaning, not flags left over
                # from a different operation selected in this same process.
                nxt.force_refresh=False;nxt.force_full=False;nxt.force=False
                nxt.reset_pending=False;nxt.refresh_days=None
                nxt.code=None;nxt.codes=None
                if choice=='6':
                    nxt.refresh_days=_input_integer('何日より古ければ取り直すか（日数。例: 7）: ',7,0)
                    nxt.reset_pending=True
                if choice=='7':nxt.force_full=True;nxt.reset_pending=True
                prompt=(f'本日収集する銘柄数（推奨既定{default_budget}、Enter で既定）: ' if choice=='1'
                        else f'1バッチあたりの最大収集銘柄数（既定{default_budget}、Enter で既定）: ')
                nxt.budget=_input_integer(prompt,default_budget)
                if choice in {'5','6','7'}:
                    print('V2では銘柄数は1バッチの上限です。JQ_RPD未設定なら同じ日も続行します。',flush=True)
            elif choice=='3':
                nxt.code=_interactive_input('銘柄コード（4文字・英字可）: ')
                if nxt.code.lower()=='q':raise MenuCancelled()
                canonical_code(nxt.code)
                if not nxt.run_id and not (nxt.output_dir/'latest_run.json').exists():
                    print('採点済み結果がありません。先にメニュー2で一括分析してください。',flush=True);continue
            if choice in {'2','4','8'} and not Store(nxt.cache_dir).snapshot(nxt.snapshot_id):
                print('キャッシュ不足。先にメニュー1または5で収集を実行してください。',flush=True);continue
            if choice=='9' and not Store(nxt.cache_dir).snapshot():
                print('差分更新の元データがありません。先にメニュー1または5で収集してください。',flush=True);continue
            result=dispatch(nxt,cfg)
            if result==130:raise KeyboardInterrupt()
            if result==1:print('[確認] 処理中に問題がありました。直前のログを確認してください。',flush=True)
        except MenuCancelled:
            print('メニューに戻りました。',flush=True)
        except (CollectionError,DataError,OSError,ValueError,KeyError) as e:
            LOG.error('%s',e)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='J-Quants V2 Standard スクリーナー（旧版互換メニュー・発注なし）')
    p.add_argument('--phase',choices=['collect','collect_all','update','analyze','single','interactive','fields-audit','cache-audit','sector-update'],default='interactive',help='引数なしで日本語の対話メニュー')
    default_config = Path(__file__).resolve().parent / 'config.json'
    p.add_argument('--config',type=Path,default=(default_config if default_config.is_file() else None),
                   help='JSON設定。未指定時はスクリプトと同じ場所の config.json を自動読込（なければ既定値＋環境変数）')
    p.add_argument('--cache-dir',type=Path,default=Path('.jquants_cache_v2_standard'),help='Standard版の凍結キャッシュ')
    p.add_argument('--output-dir',type=Path,default=Path('output_standard'),help='分析結果の保存先')
    p.add_argument('--api-ini',type=Path,default=Path(__file__).resolve().parent/'api.ini',help='V2 APIキーのiniファイル')
    p.add_argument('--budget',type=int,help='1バッチあたりの最大収集銘柄数（未指定は設定値）')
    p.add_argument('--top',type=int,default=10,help='上位出力件数（既定10）')
    p.add_argument('--code',help='銘柄コード（英字可）');p.add_argument('--codes',help='収集する普通株コードをカンマ区切りで指定')
    p.add_argument('--snapshot-id');p.add_argument('--run-id',help='single: 採点済みrun_idを指定')
    p.add_argument('--asof',help='共通株価基準日 YYYY-MM-DD（対応する凍結マスタが必要）')
    p.add_argument('--information-cutoff',help='情報締切日時 JST、例: 2026-09-11T08:00:00+09:00')
    p.add_argument('--update-from',help='日付更新の開始日（最古の未確認期間を飛ばさないこと）')
    p.add_argument('--force-refresh',action='store_true',help='収集対象を再取得')
    p.add_argument('--force-full',action='store_true',help='全普通株を強制再収集')
    p.add_argument('--force',action='store_true',help='--force-full と同じ')
    p.add_argument('--reset-pending',action='store_true',help='再開リストだけを初期化。既存の生データ・履歴は残す')
    p.add_argument('--refresh-days',type=int,help='この日数以上古い価格確認日の銘柄を再取得')
    p.add_argument('--max-batches',type=int,help='collect_allの最大完了バッチ数')
    p.add_argument('--once',action='store_true',help='1バッチだけで停止（--max-batches 1）')
    p.add_argument('--version',action='version',version=CODE_VERSION+' / '+LOGIC_VERSION)
    return p



def dispatch(args: argparse.Namespace, cfg: Config) -> int:
    store=Store(args.cache_dir);phase=args.phase
    budget=cfg.JQ_DEFAULT_BUDGET if args.budget is None else args.budget
    if budget<1 or args.top<1 or (args.max_batches is not None and args.max_batches<1):raise DataError('budget/top/max-batches must be positive')
    if args.refresh_days is not None and args.refresh_days<0:raise DataError('refresh-days must be nonnegative')
    asof=date(args.asof) if args.asof else None
    cutoff=timestamp(args.information_cutoff) if args.information_cutoff else None
    if args.asof and asof is None:raise DataError('invalid --asof')
    if args.information_cutoff and cutoff is None:raise DataError('invalid --information-cutoff')
    if not getattr(args,'_menu_child',False):
        LOG.info('plan=standard logic=%s code=%s cache=%s config_hash=%s',LOGIC_VERSION,CODE_VERSION,CACHE_SCHEMA,digest(asdict(cfg))[:12])
    if phase=='interactive':return run_interactive(args,cfg)
    if phase=='analyze':
        path=analyze_store(store,cfg,args.output_dir,args.top,args.snapshot_id,asof,cutoff)
        print('[出力] '+str(path),flush=True)
        meta=json.loads((path/'run_manifest.json').read_text('utf-8'))
        summary=json.loads((path/'filter_summary.json').read_text('utf-8'))
        gates=summary.get('gate_counts',{})
        core_count=int((gates.get('core_candidate') or {}).get('true',0) or 0)
        print('[集計] 収集済み={}/{} / core={} / eligible={}'.format(
            meta.get('complete_symbol_count',0),meta.get('ordinary_master_count',0),
            core_count,summary.get('eligible_count',0)),flush=True)
        reasons=summary.get('reason_counts') or {}
        if reasons:
            top_reasons=sorted(reasons.items(),key=lambda kv:(-int(kv[1]),str(kv[0])))[:8]
            print('[主要ゲート] '+', '.join(f'{k}={v}' for k,v in top_reasons),flush=True)
        if meta['run_status']=='diagnostic_only':
            LOG.warning('[診断のみ] 入力条件が不足しています。run_manifest.jsonのglobal_issuesを確認してください')
        elif meta['run_status']=='partial_universe':
            LOG.info('[部分母集団] 未収集または個別入力不足を含みます。メニュー5で収集を継続できます')
        return 2 if meta['run_status']=='diagnostic_only' else 0
    if phase=='single':
        if not args.code:raise DataError('--code is required')
        r=single_from_run(args.code,args.output_dir,args.run_id)
        print(json.dumps({k:r.get(k) for k in ('run_id','code','name','candidate_lane','eligible_new_entry','fundamental_edge_score','recommendation_score','gate_reasons')},ensure_ascii=False,indent=2),flush=True)
        return 0
    if phase=='cache-audit':
        print('[監査出力] '+str(audit_snapshot(store,args.output_dir,args.snapshot_id)),flush=True);return 0
    if phase=='sector-update':
        print('[セクター出力] '+str(update_sector_statistics(store,cfg,args.output_dir,args.snapshot_id,asof,cutoff)),flush=True);return 0
    # Credential lookup is reachable ONLY in explicit collection phases.
    LOG.info('[収集] APIキー・保存先を確認しています。Ctrl+Cで中断できます')
    key=api_key_from_env_or_ini(args.api_ini)
    store.ensure()
    with ProcessLock(store.root,'collector'):
        client=Client(key,cfg,store)
        try:
            if phase=='update':
                uf=date(args.update_from) if args.update_from else None
                if args.update_from and uf is None:raise DataError('invalid --update-from')
                LOG.info('[日付更新] 保存済み銘柄を日付単位で更新します')
                snap=update_daily(client,uf);print('snapshot_id='+snap['snapshot_id'],flush=True);return 0
            codes=args.codes.split(',') if args.codes else [args.code] if args.code else None
            force=args.force_refresh or args.force_full or args.force
            if phase=='collect_all':
                first=True;batches=0;remaining=None;reset=args.reset_pending;refresh_days=args.refresh_days
                limit=1 if args.once else args.max_batches
                while first or remaining:
                    try:
                        snap=collect(client,budget,force,remaining if remaining is not None else codes,refresh_days,asof,reset_pending=reset)
                    except CollectionError as e:
                        reset=False
                        if cfg.JQ_RPD is not None and 'daily_self_quota_reached' in str(e):
                            wait_until_next_day();continue
                        raise
                    reset=False;batches+=1;first=False
                    remaining=snap.get('pending_codes',[])
                    # A normal menu 1/5 may resume a previously forced job.
                    request=snap.get('collection_request',{})
                    force=bool(request.get('force',force))
                    refresh_days=request.get('refresh_days',refresh_days)
                    asof=date(request.get('asof')) if request.get('asof') else asof
                    LOG.info('[バッチ %s 完了] tried=%s 成功=%s 未完了=%s 次バッチ対象=%s snapshot=%s',
                        batches,snap.get('attempted_count',0),snap.get('batch_success_count',0),
                        snap.get('batch_incomplete_count',0),len(remaining),snap['snapshot_id'])
                    if snap.get('errors'):
                        LOG.warning('[停止] データ未取得があります。無限再試行を避けて停止します。再開リストは保存済みです');return 1
                    if limit is not None and batches>=limit:
                        LOG.info('[停止] 指定バッチ数に到達しました。続きはメニュー5で再開できます');break
                    if not remaining:break
                    LOG.info('[継続] 次の収集バッチへ進みます（Ctrl+Cで中断）')
                LOG.info('[収集終了] 未完了の新規収集銘柄数=%s',snap.get('pending_count',0))
                return 0
            snap=collect(client,budget,force,codes,args.refresh_days,asof,reset_pending=args.reset_pending)
            LOG.info('[収集終了] tried=%s 成功=%s 未完了=%s snapshot=%s status=%s',snap.get('attempted_count',0),
                     snap.get('batch_success_count',0),snap.get('batch_incomplete_count',0),snap['snapshot_id'],snap['status'])
            if phase=='fields-audit':print('[監査出力] '+str(audit_snapshot(store,args.output_dir,snap['snapshot_id'])),flush=True)
            return 1 if snap.get('errors') else 0
        finally:client.close()



def main(argv: Optional[list[str]]=None) -> int:
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    # Preserve the console encoding; only replace unrepresentable diagnostics.
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            try:stream.reconfigure(errors='backslashreplace')
            except (ValueError,OSError,AttributeError):pass
    args=build_parser().parse_args(argv)
    try:return dispatch(args,Config.load(args.config))
    except (KeyboardInterrupt,EOFError):
        LOG.warning('[中断] キーボード終了を受け付けました。未完了の分析結果は公開しません')
        LOG.info('[終了] 終了しました。収集済みの完了分は、同じ保存先のメニュー1/5で再開時に再利用します')
        return 130
    except (DataError,CollectionError,OSError,ValueError,KeyError) as e:
        LOG.error('%s',e);return 1


if __name__=='__main__':
    raise SystemExit(main())
