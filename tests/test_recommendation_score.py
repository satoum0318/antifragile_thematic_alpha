import datetime

import pandas as pd

import JQuamtsScreeningBot as bot


def _annual(year, shares, dps, net_income=31_000_000_000, cfo=33_000_000_000):
    return {
        "fiscal_year": year,
        "period_end_date": f"{year}-03-31",
        "shares_outstanding": shares,
        "dividend_per_share": dps,
        "forecast_dividend_per_share": None,
        "net_income": net_income,
        "operating_cash_flow": cfo,
        "equity": 5_000_000_000,
        "total_assets": 10_000_000_000,
        "cash_and_equivalents": 1_000_000_000,
    }


def test_split_adjustment_recovers_buyback_and_dividend_growth():
    # 1:2 split at the FY2024 year end: raw share count doubles, raw DPS halves afterwards
    history = [
        _annual(2026, 205_624_838, 68.0),
        _annual(2025, 208_624_838, 63.0),
        _annual(2024, 213_624_838, 120.0),
        _annual(2023, 109_812_419, 102.0),
        _annual(2022, 116_812_419, 96.0),
    ]
    srq = bot.compute_shareholder_return_quality(history)
    assert srq["buyback_consistency"] == 1.0
    assert srq["dividend_policy_credibility"] > 0.85
    assert srq["shareholder_return_score"] > 70.0

    adjusted = bot.split_adjusted_series(history, "dividend_per_share", per_share=True)
    assert all(
        cur > prev for cur, prev in zip(adjusted, adjusted[1:])
    ), f"adjusted DPS should be monotonically rising, got {adjusted}"


def test_split_adjustment_is_noop_without_split():
    history = [
        _annual(2026, 6_884_244_856, 7.3, net_income=193_000_000_000, cfo=662_000_000_000),
        _annual(2025, 7_154_182_647, 7.0, net_income=153_000_000_000, cfo=519_000_000_000),
        _annual(2024, 7_637_068_986, 5.56, net_income=113_000_000_000, cfo=316_000_000_000),
        _annual(2023, 7_633_501_686, 5.56, net_income=178_000_000_000, cfo=93_000_000_000),
    ]
    factors = bot.build_split_adjustment_factors(history)
    assert factors == [1.0] * len(history)
    assert bot.compute_shareholder_return_quality(history)["buyback_consistency"] == 1.0


def test_forward_guidance_flags_period_mismatch():
    stmts = [
        {
            "DisclosedDate": "2026-05-15",
            "CurrentPeriodType": "FY",
            "CurrentPeriodEndDate": "2026-03-31",
            "CurrentFiscalYearEndDate": "2026-03-31",
            "NetIncomeLoss": "1000",
            "NextForecastNetIncome": "1200",
        }
    ]
    good = bot.resolve_forward_guidance(stmts, as_of_date=datetime.date(2026, 7, 27))
    assert good["forecast_net_income"] == 1200.0
    assert good["forward_guidance_horizon_months"] == 12
    assert good["forward_guidance_warning"] is None
    assert good["forward_guidance_target_fy_end"] == "2027-03-31"


def test_forward_guidance_rejects_stale_quarter_forecast():
    stmts = [
        {
            "DisclosedDate": "2026-02-02",
            "CurrentPeriodType": "3Q",
            "CurrentPeriodEndDate": "2025-12-31",
            "CurrentFiscalYearEndDate": "2026-03-31",
            "ForecastNetIncome": "900",
        },
        {
            "DisclosedDate": "2026-05-15",
            "CurrentPeriodType": "FY",
            "CurrentPeriodEndDate": "2026-03-31",
            "CurrentFiscalYearEndDate": "2026-03-31",
            "NetIncomeLoss": "1000",
        },
    ]
    out = bot.resolve_forward_guidance(stmts, as_of_date=datetime.date(2026, 7, 27))
    assert out["forecast_net_income"] is None


def test_recommendation_score_prefers_quality_over_entry_alone():
    df = pd.DataFrame(
        [
            {
                "code": "A",
                "fundamental_edge_score": 87.0,
                "entry_score": 83.0,
                "forward_np_change": 0.04,
                "earnings_quality_flag": "ok",
                "accounting_flag": "clean",
                "governance_flag": "unknown",
                "shareholder_return_score": 55.0,
            },
            {
                "code": "B",
                "fundamental_edge_score": 58.0,
                "entry_score": 95.0,
                "forward_np_change": 0.04,
                "earnings_quality_flag": "ok",
                "accounting_flag": "clean",
                "governance_flag": "unknown",
                "shareholder_return_score": 55.0,
            },
            {
                "code": "C",
                "fundamental_edge_score": 70.0,
                "entry_score": 78.0,
                "forward_np_change": -0.09,
                "earnings_quality_flag": "watch",
                "accounting_flag": "clean",
                "governance_flag": "unknown",
                "shareholder_return_score": 76.0,
            },
        ]
    )
    score = bot.compute_recommendation_score(df)
    # A: quality+entry healthy > B: entry-only dominance > C: mild decline + eq watch
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]


def test_recommendation_score_penalizes_accounting_watch_and_missing_forward():
    df = pd.DataFrame(
        [
            {
                "fundamental_edge_score": 60.0,
                "entry_score": 66.0,
                "forward_np_change": 0.05,
                "earnings_quality_flag": "ok",
                "accounting_flag": "clean",
                "governance_flag": "unknown",
                "shareholder_return_score": 50.0,
            },
            {
                "fundamental_edge_score": 60.0,
                "entry_score": 66.0,
                "forward_np_change": None,
                "earnings_quality_flag": "ok",
                "accounting_flag": "watch",
                "governance_flag": "unknown",
                "shareholder_return_score": 90.0,
            },
        ]
    )
    score = bot.compute_recommendation_score(df)
    assert score.iloc[0] > score.iloc[1]
