import math

from real_data_screening import compute_policy_stress_score, compute_sales_cagr


def _build_history():
    return [
        {
            "revenue": 120.0,
            "operating_income": 12.0,
            "operating_cash_flow": 18.0,
            "operating_margin": 0.10,
            "equity": 80.0,
            "interest_bearing_debt": 30.0,
        },
        {
            "revenue": 110.0,
            "operating_income": 10.0,
            "operating_cash_flow": 16.0,
            "operating_margin": 0.09,
            "equity": 75.0,
            "interest_bearing_debt": 35.0,
        },
        {
            "revenue": 100.0,
            "operating_income": 8.0,
            "operating_cash_flow": 14.0,
            "operating_margin": 0.08,
            "equity": 70.0,
            "interest_bearing_debt": 32.0,
        },
        {
            "revenue": 92.0,
            "operating_income": 7.0,
            "operating_cash_flow": 12.0,
            "operating_margin": 0.076,
            "equity": 68.0,
            "interest_bearing_debt": 30.0,
        },
    ]


def test_compute_sales_cagr_three_years():
    history = _build_history()
    cagr = compute_sales_cagr(history, years=3)
    assert cagr is not None
    assert math.isclose(cagr, (120.0 / 92.0) ** (1 / 3) - 1, rel_tol=1e-6)


def test_policy_stress_score_hits_upper_threshold():
    history = _build_history()
    score = compute_policy_stress_score(history)
    assert score >= 3

