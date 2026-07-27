import pandas as pd

import JQuamtsScreeningBot as bot


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
