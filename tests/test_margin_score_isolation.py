"""Daily margin diagnostics must not change any existing decision or ranking."""
import copy
import datetime as dt
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import JQuamtsScreeningBot as bot


@pytest.fixture(scope="module")
def baseline():
    path = Path(__file__).resolve().parents[1] / "archive" / "jquants-1.0.3-rc1-cli3-all-20260930" / "JQuamtsScreeningBot.py"
    name = "_margin_isolation_pre_daily_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves annotations through the module registry.
    spec.loader.exec_module(module)
    return module


def valid_record(code="1001", **changes):
    return {"code": code, "instrument_type": "stock", "S33": "5250",
            "model_scope": "nonfinancial", "common_issues": [], "financial_issues": [], "light_issues": [],
            "market_regime": "risk_on", "ma200_state": "ma200_reclaim", "distance_from_ma200": 0.02,
            "liquidity_ok": True, "market_cap_ok": True, "market_cap": 100_000_000_000,
            "ps": 0.6, "ps_vs_sector": 0.7, "per": 12.0, "annual_comparable": True,
            "annual_stale": False, "critical_missing_count": 0, "summary_quality_score": 7,
            "summary_quality_available": 7, "quality_normalized_9": 9.0, "Q1": True, "Q2": True,
            "op_income_stable": True, "op_income_yoy_pct": 15.0, "op_income_downside_score": 5.0,
            "eps_proxy_endshares_yoy_pct": 15.0, "peg_trusted": True, "peg_quality_class": "ok",
            "reference_peg": 0.8, "peg_warnings": [], "sales_cagr": 0.1,
            "days_since_original_annual_release": 100, "annual_disclosure_age_days": 100,
            "forecast_net_income": 9_000_000_000, "forward_np_change": 0.08,
            "earnings_quality_flag": "ok", "accounting_risk": "unknown", "governance_status": "unknown",
            "shareholder_return_score": 75.0, "shareholder_return_coverage": 1.0,
            "earnings_blackout": False, "confirmed_severe_risk": False,
            "below_ma200_downtrend": False, "recent_60d_low_update": False,
            "return_21d": 0.04, "return_63d": 0.08, "return_126d": 0.15,
            "return_252d": 0.2, "momentum_6m_1m": 0.1, "momentum_6m_3m": 0.07,
            "momentum_3m_1m": 0.04, "max_drawdown": -0.15,
            "cash_and_equivalents": 20_000_000_000, "operating_cash_flow": 12_000_000_000,
            "equity_ratio": 0.65, "dividend_status": "増配", "price": 1020.0,
            "adjusted_close": 1020.0, "ma_25": 1005.0, "ma_75": 995.0, "ma_200": 1000.0,
            "avg_volume_30d": 1_000_000, "adv_jpy_20d": 1_020_000_000,
            "vol_daily_20d": 0.012, "vol_daily_reference": 0.014,
            "benchmark_input_valid": True, **changes}


DIAGNOSTICS = [
    pytest.param({"margin_status": "complete", "margin_ratio": 0.1,
                  "margin_short_volume": 1_000_000, "short_selling_change_rate": -99}, id="low-ratio"),
    pytest.param({"margin_status": "complete", "margin_ratio": 200,
                  "margin_short_volume": 1, "short_selling_change_rate": 1000,
                  "margin_short_change_5d_pct": 200, "margin_short_change_20d_pct": 1000}, id="extreme-values"),
    pytest.param({"margin_status": "zero_short_balance", "margin_ratio": None,
                  "margin_short_volume": 0, "margin_long_volume": 1_000_000}, id="zero-short"),
    pytest.param({"margin_status": "both_zero", "margin_ratio": None,
                  "margin_short_volume": 0, "margin_long_volume": 0}, id="both-zero"),
    pytest.param({"margin_status": "no_rows", "margin_ratio": None}, id="no-rows"),
    pytest.param({"margin_status": "collection_failed", "margin_ratio": None,
                  "margin_issues": ["collection_failed"]}, id="failure"),
    pytest.param({"margin_status": "schema_review_required", "margin_ratio": None}, id="schema"),
    pytest.param({"margin_status": "not_applicable", "margin_issue_type": "3",
                  "margin_ratio": None}, id="other-issue-type"),
    pytest.param({"margin_status": "not_yet_available", "margin_ratio": None,
                  "margin_date": "2026-09-25", "margin_after_price_asof": True}, id="future-information"),
]


def score_stages(module, record):
    row = copy.deepcopy(record)
    cfg = module.Config()
    stages = {}
    for name in ("fe_score", "all_gates", "recommendation"):
        result = getattr(module, name)(row, cfg) if name != "recommendation" else module.recommendation(row)
        stages[name] = copy.deepcopy(result)
        row.update(result)
    keys_before = set(row)
    module.legacy_reference_scores([row], cfg)
    stages["legacy_reference_scores"] = {key: row[key] for key in row.keys() - keys_before}
    stages["rank_key"] = module.rank_key(row)
    return stages, row


@pytest.mark.parametrize("diagnostics", DIAGNOSTICS)
def test_margin_fields_cannot_change_scoring_gates_or_rank_key(baseline, diagnostics):
    original = valid_record()
    expected, reference = score_stages(baseline, original)
    current_clean, _ = score_stages(bot, original)
    current_diagnostic, result = score_stages(bot, {**original, **diagnostics})
    assert reference["eligible_new_entry"] is True  # Avoid a vacuous all-excluded comparison.
    assert reference["candidate_lane"] == "ma200_reclaim_core"
    assert expected == current_clean == current_diagnostic
    assert result["data_review_level"] == "none"


def test_margin_fields_cannot_change_legacy_tie_break_ranking(baseline):
    # Equal recommendation and fundamental scores leave legacy_total as a real tie breaker.
    rows = [valid_record("1001", max_drawdown=-0.6), valid_record("1002", max_drawdown=-0.1),
            valid_record("1003", earnings_blackout=True)]
    old = [score_stages(baseline, row)[1] for row in rows]
    new = [score_stages(bot, {**row, "margin_ratio": 1000 if i == 1 else 0.1,
                             "short_selling_change_rate": 1000 if i == 1 else -99})[1]
           for i, row in enumerate(rows)]
    assert old[0]["recommendation_score"] == old[1]["recommendation_score"]
    assert old[0]["legacy_total"] != old[1]["legacy_total"]
    assert [r["code"] for r in sorted(old, key=baseline.rank_key)] == [
        r["code"] for r in sorted(new, key=bot.rank_key)]
    assert old[2]["eligible_new_entry"] is False


def make_snapshot(tmp_path):
    """Synthetic, fully local financial and price fixtures; no historical market claims."""
    store = bot.Store(tmp_path / "snapshot-cache")
    asof = dt.date(2026, 9, 11)
    cutoff = "2026-09-30T19:00:00+09:00"
    sessions = pd.bdate_range("2021-03-31", "2026-10-30")
    past = sessions[sessions <= pd.Timestamp(asof)]
    put = lambda rows: store.put({"status": "complete", "rows": rows, "collected_at": cutoff})
    master = [{"Date": str(asof), "Code": f"{code}0", "CoName": f"Synthetic {code}",
               "Mkt": "0111", "ProdCat": "011", "S33": "5250"}
              for code in ("1001", "1002", "1003", "1004", "1005")]
    snapshot = {"schema": bot.CACHE_SCHEMA, "snapshot_id": "synthetic_score_isolation",
                "status": "complete", "price_asof_date": str(asof), "information_cutoff": cutoff,
                "globals": {"calendar": put([{"Date": str(d.date()), "HolDiv": "1"} for d in sessions]),
                            "master": put(master),
                            "topix": put([{"Date": str(d.date()), "C": 2000 + i} for i, d in enumerate(past)])},
                "symbols": {}}
    for index, m in enumerate(master):
        api = m["Code"]
        closes = np.full(len(past), 1000.0 + index * 10)
        closes[-30:-1] = np.linspace(930, 980, 29) + index * 10
        closes[-1] = 1020 + index * 10
        prices = [{"Date": str(day.date()), "Code": api, "O": c, "H": c + 10, "L": c - 10,
                   "C": c, "Vo": 1_000_000, "Va": c * 1_000_000,
                   "AdjFactor": 1, "ExRT": "0", "MktCap": 100_000 + index * 5000}
                  for day, c in zip(past, closes)]
        statements = []
        for year in range(2022, 2027):
            n = year - 2022
            statements.append({"Code": api, "DiscDate": f"{year}-05-15", "DiscTime": "15:00:00",
                "DiscNo": f"{year}-{api}", "DocType": "FYFinancialStatements_Consolidated_JP",
                "CurPerType": "FY", "CurPerSt": f"{year-1}-04-01", "CurPerEn": f"{year}-03-31",
                "CurFYSt": f"{year-1}-04-01", "CurFYEn": f"{year}-03-31",
                "NxtFYSt": f"{year}-04-01", "NxtFYEn": f"{year+1}-03-31",
                "Sales": 140e9 + n * 15e9, "OP": 10e9 + n * 1e9,
                "NP": 6e9 + n * 0.8e9, "OdP": 10e9 + n * 1e9, "CFO": 12e9 + n * 1e9,
                "TA": 120e9 + n * 5e9, "Eq": 70e9 + n * 5e9, "ShEq": 70e9 + n * 5e9,
                "CashEq": 20e9 + n * 2e9, "ShOutFY": 100e6 - n * 2e6,
                "DivAnn": 30 + n * 2, "DivTotalAnn": 3e9, "NxFNp": 7e9 + n * 0.8e9})
        snapshot["symbols"][api[:4]] = {"complete": True, "collected_at": cutoff,
            "financial_checked_through": cutoff, "earnings_checked_through": cutoff,
            "prices": put(prices), "summary": put(statements), "earnings": put([]), "short": put([]),
            "margin": put([])}
    return store, snapshot


SCORE_FIELDS = ("code", "recommendation_score", "eligible_new_entry", "candidate_lane", "valuation_lane",
                "fundamental_edge_score", "entry_score", "legacy_total", "total_score", "grade",
                "safety_score_scaled", "spec_score", "spec_penalty", "rec_priority", "rec_secondary",
                "gate_reasons", "data_review_level", "common_issues", "financial_issues", "light_issues")


def test_score_snapshot_remains_identical_to_archive_with_margin_bundles(tmp_path, baseline):
    store, snapshot = make_snapshot(tmp_path)
    old, _, old_meta = baseline.score_snapshot(store, snapshot, baseline.Config())
    original, _, _ = bot.score_snapshot(store, snapshot, bot.Config())
    assert old_meta["run_status"] == "complete"
    assert any(r["eligible_new_entry"] for r in old)
    changed = copy.deepcopy(snapshot)
    statuses = ("complete", "collection_failed", "schema_review_required", "complete", "complete")
    for index, (code, item) in enumerate(changed["symbols"].items()):
        status = statuses[index]
        short = 0 if index == 3 else 10
        row = {"Code": code + "0", "Date": "2026-09-25", "PubDate": "2026-09-28", "IssType": "3" if index == 4 else "2",
               "LongVol": 100_000, "ShrtVol": short, "LongNegVol": 50_000, "LongStdVol": 50_000,
               "ShrtNegVol": short, "ShrtStdVol": 0,
               "LongVal": 100_000_000, "ShrtVal": short * 1000, "LongNegVal": 50_000_000,
               "LongStdVal": 50_000_000, "ShrtNegVal": short * 1000, "ShrtStdVal": 0}
        item["margin"] = store.put({"status": status, "rows": [row] if status == "complete" else [],
                                    "collected_at": "2026-09-28T17:00:00+09:00"})
    updated, _, new_meta = bot.score_snapshot(store, changed, bot.Config())
    project = lambda records: [{k: row.get(k) for k in SCORE_FIELDS} for row in records]
    assert project(old) == project(original) == project(updated)
    assert old_meta["eligible_count"] == new_meta["eligible_count"]
    assert old_meta["run_status"] == new_meta["run_status"]
