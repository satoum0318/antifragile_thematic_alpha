"""Financial and recommendation regressions using the maintained helpers."""
import datetime as dt

import pandas as pd
import pytest

import JQuamtsScreeningBot as bot


def annual_raw(year, shares=200_000_000, dps=60.0):
    return {"Code": "99990", "DiscDate": f"{year}-05-15", "DiscTime": "15:00:00",
            "DiscNo": str(year), "DocType": "FYFinancialStatements_Consolidated_JP",
            "CurPerType": "FY", "CurPerSt": f"{year-1}-04-01", "CurPerEn": f"{year}-03-31",
            "CurFYSt": f"{year-1}-04-01", "CurFYEn": f"{year}-03-31",
            "ShOutFY": shares, "DivAnn": dps, "DivTotalAnn": 13_000_000_000,
            "NP": 31_000_000_000, "CFO": 33_000_000_000, "ShEq": 5_000_000_000,
            "Eq": 5_000_000_000, "TA": 10_000_000_000, "CashEq": 1_000_000_000}


def statements_and_history(rows):
    statements = [bot.normalize_statement(row, "9999", bot.Config()) for row in rows]
    return statements, bot.annual_history(statements)


def action_frame():
    return pd.DataFrame({"AdjFactor": 1.0, "ExRT": "0"},
                        index=pd.bdate_range("2021-03-31", "2026-06-30"))


def test_verified_split_adjusts_shares_but_dps_requires_its_own_basis():
    rows = [annual_raw(2026, 205_624_838, 68), annual_raw(2025, 208_624_838, 63),
            annual_raw(2024, 213_624_838, 120), annual_raw(2023, 109_812_419, 102),
            annual_raw(2022, 116_812_419, 96)]
    _, history = statements_and_history(rows)
    frame = action_frame()
    frame.loc[pd.Timestamp("2024-03-29"), ["AdjFactor", "ExRT"]] = [0.5, "1"]
    asof = dt.date(2026, 6, 30)

    assert bot.share_factor(frame, dt.date(2023, 3, 31), asof) == (2.0, "verified_split_factors")
    bot.adjust_financial_shares(history, frame, asof, bot.Config(), "9999")
    assert [r["shares_adjusted"] for r in history] == [205_624_838, 208_624_838, 213_624_838,
                                                       219_624_838, 233_624_838]
    assert history[2]["dps_adjusted"] is None
    assert bot.shareholder_quality(history)["shareholder_return_score"] is None

    overrides = {str(r["period_end"]): {"source": "synthetic verified dividend basis",
                  "asof": str(asof), "dps_factor_to_asof": 0.5} for r in history[2:]}
    cfg = bot.Config(per_share_basis_overrides={"9999": overrides})
    bot.adjust_financial_shares(history, frame, asof, cfg, "9999")
    assert [r["dps_adjusted"] for r in history] == [68, 63, 60, 51, 48]
    quality = bot.shareholder_quality(history)
    assert quality["shareholder_return_components"]["share_decrease_consistency"] == 1.0
    assert quality["shareholder_return_components"]["dividend_credibility"] > 0.85
    assert quality["shareholder_return_score"] > 70


def test_event_free_share_and_dividend_bases_are_unchanged():
    rows = [annual_raw(2026, 6_884_244_856, 7.3), annual_raw(2025, 7_154_182_647, 7),
            annual_raw(2024, 7_637_068_986, 5.56), annual_raw(2023, 7_633_501_686, 5.56)]
    _, history = statements_and_history(rows)
    bot.adjust_financial_shares(history, action_frame(), dt.date(2026, 6, 30), bot.Config(), "9999")
    assert [r["share_adjustment_factor"] for r in history] == [1.0] * 4
    assert [r["shares_adjusted"] for r in history] == [r["ShOutFY"] for r in rows]
    assert [r["dps_adjusted"] for r in history] == [r["DivAnn"] for r in rows]
    assert bot.shareholder_quality(history)["shareholder_return_components"]["share_decrease_consistency"] == 1.0


def test_forward_guidance_requires_explicit_matching_next_fiscal_period():
    row = {**annual_raw(2026), "NP": 1000, "NxFNp": 1200,
           "NxtFYSt": "2026-04-01", "NxtFYEn": "2027-03-31"}
    statements, history = statements_and_history([row])
    good = bot.guidance(statements, history)
    assert good["forecast_net_income"] == 1200
    assert good["forward_np_change"] == pytest.approx(0.2)
    assert good["guidance_status"] == "valid"
    assert good["guidance_target"] == "2027-03-31"
    assert good["forward_guidance_warning"] is None

    for bad_period in ({"NxtFYSt": None, "NxtFYEn": None},
                       {"NxtFYSt": "2025-04-01", "NxtFYEn": "2026-03-31"}):
        statements, history = statements_and_history([{**row, **bad_period}])
        rejected = bot.guidance(statements, history)
        assert rejected["forecast_net_income"] is None
        assert rejected["guidance_status"] == "rejected_period"
        assert rejected["forward_guidance_warning"] == "forward_period_mismatch_or_unknown"


def test_forward_guidance_does_not_fall_back_to_stale_or_withdrawn_forecasts():
    stale_quarter = {**annual_raw(2026), "DiscDate": "2026-02-02", "CurPerType": "3Q",
                     "DocType": "3QFinancialStatements_Consolidated_JP", "CurPerEn": "2025-12-31",
                     "FNP": 900}
    statements, history = statements_and_history([stale_quarter, {**annual_raw(2026), "NP": 1000}])
    assert bot.guidance(statements, history)["forecast_net_income"] is None

    valid = {**annual_raw(2026), "NxFNp": 1200, "NxtFYSt": "2026-04-01", "NxtFYEn": "2027-03-31"}
    revision = {**valid, "DiscDate": "2026-06-01", "DiscNo": "latest",
                "DocType": "EarnForecastRevision_Consolidated_JP", "CurPerType": "1Q",
                "CurFYSt": "2026-04-01", "CurFYEn": "2027-03-31", "FNP": None}
    statements, history = statements_and_history([valid, revision])
    result = bot.guidance(statements, history)
    assert result["guidance_status"] == "ambiguous"
    assert result["forecast_net_income"] is None
    revision["ForecastWithdrawn"] = True
    statements, history = statements_and_history([valid, revision])
    assert bot.guidance(statements, history)["guidance_status"] == "withdrawn"


def recommendation_input(**changes):
    return {"fundamental_edge_score": 60.0, "ma200_state": "ma200_reclaim",
            "distance_from_ma200": 0.02, "candidate_lane": "ma200_reclaim_core",
            "data_review_level": "none", "forward_np_change": 0.05,
            "earnings_quality_flag": "ok", "accounting_risk": "clean",
            "governance_status": "unknown", "shareholder_return_score": 50,
            "shareholder_return_coverage": 1.0, **changes}


def test_recommendation_prefers_quality_over_favorable_entry_alone():
    quality = bot.recommendation(recommendation_input(fundamental_edge_score=87,
                                  ma200_state="above_ma200_extended", candidate_lane="extended_above_ma200"))
    entry = bot.recommendation(recommendation_input(fundamental_edge_score=58))
    assert quality["entry_score_raw"] - 87 < entry["entry_score_raw"] - 58
    assert quality["recommendation_score"] > entry["recommendation_score"]


def test_recommendation_penalizes_accounting_watch_and_missing_forward():
    clean = bot.recommendation(recommendation_input())
    watch = bot.recommendation(recommendation_input(forward_np_change=None,
                                  accounting_risk="watch", shareholder_return_score=90))
    assert watch["recommendation_adjustments"]["forward_missing"] == -6
    assert watch["recommendation_adjustments"]["accounting_risk"] == -10
    assert watch["recommendation_adjustments"]["shareholder_return"] == 3
    assert clean["recommendation_score"] - watch["recommendation_score"] == pytest.approx(13)
