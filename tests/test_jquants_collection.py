import pandas as pd

import JQuamtsScreeningBot as bot


def test_collect_one_code_treats_empty_price_http200_as_permanent(monkeypatch):
    def fake_fetch_prices(*args, **kwargs):
        return pd.DataFrame(), {
            "http": 200,
            "rows": 0,
            "transient": False,
            "empty_http200": True,
        }

    monkeypatch.setattr(bot, "fetch_prices_v2_with_meta", fake_fetch_prices)

    result = bot.collect_one_code_result(object(), "9999")

    assert result["status"] == "permanent_missing_financials"
    assert result["reason"] == "price_empty_http200"


def test_collect_one_code_stock_empty_price_http200_skips_financials(monkeypatch):
    def fake_fetch_prices(*args, **kwargs):
        return pd.DataFrame(), {
            "http": 200,
            "rows": 0,
            "transient": False,
            "empty_http200": False,
        }

    def fail_fetch_statements(*args, **kwargs):
        raise AssertionError("financial fetch should not run when price is empty HTTP 200")

    monkeypatch.setattr(bot, "fetch_prices_v2_with_meta", fake_fetch_prices)
    monkeypatch.setattr(bot.FinancialDataManager, "fetch_statements", fail_fetch_statements)

    result = bot.collect_one_code_result(object(), "9999")

    assert result["status"] == "permanent_missing_financials"
    assert result["reason"] == "price_empty_http200"


def test_collect_one_code_keeps_transient_price_errors_pending(monkeypatch):
    def fake_fetch_prices(*args, **kwargs):
        return pd.DataFrame(), {
            "http": 503,
            "rows": 0,
            "transient": True,
            "empty_http200": False,
        }

    monkeypatch.setattr(bot, "fetch_prices_v2_with_meta", fake_fetch_prices)

    result = bot.collect_one_code_result(object(), "9999")

    assert result["status"] == "transient_error"
    assert result["reason"] == "price_transient_http_503"
