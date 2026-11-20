"""Regressions for the swing-screening review using synthetic local evidence."""
import copy
import datetime as dt

import pandas as pd
import pytest

import JQuamtsScreeningBot as bot
import jq_margin
from test_universe_integration import current_snapshot


@pytest.mark.parametrize("period_type,forecast_key,forecast_eps_key", [
    ("FY", "NxFNp", "NxFEPS"),
    ("FY", "NxFNP", "NxFEPS"),
    ("2Q", "FNP", "FEPS"),
])
def test_standalone_statements_use_the_main_forecast_fields(period_type, forecast_key, forecast_eps_key):
    base = {"scope": "nonconsolidated", "period_end": dt.date(2026, 3, 31), "net_income": 304_000_000}
    raw = {"Code": "79020", "DiscNo": "1", "DiscDate": "2026-11-12", "DiscTime": "15:30:00",
           "DocType": period_type + "FinancialStatements_NonConsolidated_JP", "CurPerType": period_type,
           "CurPerEn": "2026-03-31" if period_type == "FY" else "2026-09-30",
           "CurFYSt": "2026-04-01", "CurFYEn": "2027-03-31",
           "NxtFYSt": "2026-04-01", "NxtFYEn": "2027-03-31",
           forecast_key: "260000000", forecast_eps_key: "26", "FNCNP": "", "NxFNCNP": ""}
    result = bot.guidance([bot.normalize_statement(raw, "7902", bot.Config())], [base])
    assert result["guidance_status"] == "valid"
    assert result["forecast_net_income"] == 260_000_000
    assert result["forecast_eps"] == 26.
    assert result["forward_np_change"] == pytest.approx(260 / 304 - 1)


def test_scope_free_revision_uses_standalone_forecast_against_standalone_actuals():
    base = {"scope": "nonconsolidated", "period_end": dt.date(2026, 3, 31), "net_income": 304_000_000}
    raw = {"Code": "79020", "DiscNo": "2", "DiscDate": "2026-11-12", "DiscTime": "15:30:00",
           "DocType": "EarnForecastRevision", "CurPerType": "2Q", "CurPerEn": "2026-09-30",
           "CurFYSt": "2026-04-01", "CurFYEn": "2027-03-31",
           "FNP": "480000000", "FNCNP": "260000000", "FEPS": "48", "FNCEPS": "26"}
    statement = bot.normalize_statement(raw, "7902", bot.Config())
    assert statement["scope"] == "unknown"
    result = bot.guidance([statement], [base])
    assert result["guidance_status"] == "valid"
    assert result["forecast_net_income"] == 260_000_000
    assert result["forecast_eps"] == 26.
    assert result["forward_np_change"] == pytest.approx(-.14473684210526316)


@pytest.mark.parametrize("observed,expected_date,blackout", [
    ("2026-11-16T18:00:00+09:00", "2026-11-17", True),
    ("2026-11-16T20:00:00+09:00", "2026-12-02", False),
    (None, "2026-12-02", False),
])
def test_same_day_earnings_changes_require_observation_before_the_cutoff(observed, expected_date, blackout):
    old = {"Code": "79020", "PubDate": "2026-11-10", "FQName": "2Q", "FYE": "0331", "SchDate": "2026-12-02"}
    changed = {**old, "PubDate": "2026-11-16", "SchDate": "2026-11-17"}
    if observed is not None:
        changed = bot.observed_earnings_rows({"rows": [changed], "collected_at": observed})[0]
    result = bot.earnings_context([old, changed], "79020", bot.timestamp("2026-11-16T19:00:00+09:00"),
                                 pd.bdate_range("2026-11-01", "2026-12-10"), dt.date(2026, 11, 17), bot.Config())
    assert result["earnings_scheduled_date"] == expected_date
    assert result["earnings_blackout"] is blackout
    assert result["earnings_status"] == "known"


def test_earnings_merge_keeps_first_observation_only_for_the_same_content():
    raw = {"Code": "79020", "PubDate": "2026-11-16", "FQName": "2Q", "FYE": "0331", "SchDate": "2026-11-17"}
    old = bot.observed_earnings_rows({"rows": [raw], "collected_at": "2026-11-16T10:10:00+09:00"})
    later = bot.observed_earnings_rows({"rows": [raw], "collected_at": "2026-11-16T18:00:00+09:00"})
    before = copy.deepcopy((old, later))
    merged = bot.merge_daily_rows(old, later, "earnings")
    assert merged[0]["_first_observed_at"] == "2026-11-16T10:10:00+09:00"
    assert (old, later) == before
    changed = bot.observed_earnings_rows({"rows": [{**raw, "SchDate": "2026-11-18"}],
                                         "collected_at": "2026-11-16T18:00:00+09:00"})
    corrected = bot.merge_daily_rows(merged, changed, "earnings")
    assert corrected[0]["SchDate"] == "2026-11-18"
    assert corrected[0]["_first_observed_at"] == "2026-11-16T18:00:00+09:00"


def test_valuation_market_cap_supports_the_pipeline_without_daily_market_cap(tmp_path):
    store, snap = current_snapshot(tmp_path)
    rows = []
    for code, item in snap["symbols"].items():
        prices = store.get(item["prices"])
        for price in prices["rows"]:
            price.pop("MktCap", None)
        item["prices"] = store.put(prices)
        rows.append({"Code": code + "0", "Date": snap["price_asof_date"], "MktCap": 100_000})
    snap["globals"]["valuation"] = store.put({"status": "complete", "rows": rows,
                                                "collected_at": snap["information_cutoff"]})
    records, _, _ = bot.score_snapshot(store, snap, bot.Config())
    assert all(row["market_cap"] == 100_000_000_000 for row in records)
    assert all(row["market_cap_source"] == "jquants_valuation_treasury_excluded" for row in records)
    assert all(row["registration_a_eligible"] is True for row in records)
    assert all(row["registration_b_eligible"] is True for row in records)


def test_explicit_credit_reference_selects_exact_application_session(tmp_path):
    store, snap = current_snapshot(tmp_path)
    code, item = next(iter(snap["symbols"].items()))
    calendar = store.get(snap["globals"]["calendar"])["rows"]
    kwargs = {"expected_code": code + "0", "price_asof_date": bot.date(snap["price_asof_date"]),
              "calendar_rows": calendar, "price_rows": store.get(item["prices"])["rows"],
              "reference_date": dt.date(2026, 10, 23)}
    bundle = store.get(item["margin"])
    result = jq_margin.margin_diagnostic(bundle, bot.timestamp(snap["information_cutoff"]), **kwargs)
    assert result["margin_reference_date"] == "2026-10-23"
    assert result["margin_diagnostic_status"] == "available"
    bundle["rows"] = [row for row in bundle["rows"] if row["Date"] != "2026-10-23"]
    missing = jq_margin.margin_diagnostic(bundle, bot.timestamp(snap["information_cutoff"]), **kwargs)
    assert missing["margin_diagnostic_status"] != "available"
    assert missing["margin_reference_date"] != "2026-10-26"


def test_publication_rechecks_after_copying_the_display_report(tmp_path, monkeypatch):
    store, snap = current_snapshot(tmp_path)
    rows, stats, meta = bot.score_snapshot(store, snap, bot.Config())
    output = tmp_path / 'reports'
    bot.write_run(rows, stats, meta, output, run_id='previous')
    pointer = (output / 'latest_run.json').read_bytes()
    report = (output / 'reports' / 'latest' / 'run_manifest.json').read_bytes()
    copied = []
    original_copy = bot.shutil.copytree
    def copy(source, target, *args, **kwargs):
        result = original_copy(source, target, *args, **kwargs)
        copied.append(True)
        return result
    monkeypatch.setattr(bot.shutil, 'copytree', copy)
    def check():
        if copied:
            raise bot.DataError('entry_session_changed_during_analysis')
    with pytest.raises(bot.DataError, match='entry_session_changed'):
        bot.write_run(rows, stats, meta, output, run_id='opening_crossed', publication_check=check)
    assert (output / 'latest_run.json').read_bytes() == pointer
    assert (output / 'reports' / 'latest' / 'run_manifest.json').read_bytes() == report
