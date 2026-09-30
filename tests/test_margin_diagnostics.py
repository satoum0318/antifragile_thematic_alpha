"""Offline contract tests: fixtures are synthetic, never live API responses."""
import copy
import datetime as dt

import pytest

from jq_margin import ADAPTER_VERSION, DAILY_START, VALUE_FIELDS, adapt_margin_bundle, margin_diagnostic

JST = dt.timezone(dt.timedelta(hours=9))
HOLIDAYS = {dt.date(2026, 9, 21), dt.date(2026, 9, 22), dt.date(2026, 9, 23), dt.date(2026, 10, 12)}


def calendar(begin="2026-09-01", end="2026-10-30"):
    day, last = dt.date.fromisoformat(begin), dt.date.fromisoformat(end)
    rows = []
    while day <= last:
        rows.append({"Date": day.isoformat(), "HolDiv": "1" if day.weekday() < 5 and day not in HOLIDAYS else "0"})
        day += dt.timedelta(days=1)
    return rows


CAL = calendar()
SESSIONS = [dt.date.fromisoformat(r["Date"]) for r in CAL if r["HolDiv"] == "1"]


def row(day="2026-09-25", long=200, short=100, issue="2", **extra):
    date = dt.date.fromisoformat(day)
    result = {"Date": day, "Code": "12340", "IssType": issue,
              "LongVol": long, "ShrtVol": short, "LongNegVol": long, "ShrtNegVol": short,
              "LongStdVol": 0, "ShrtStdVol": 0}
    if date >= DAILY_START:
        result["PubDate"] = SESSIONS[SESSIONS.index(date) + 1].isoformat()
        result.update({key: 1000 if key in {"LongVal", "ShrtVal", "LongNegVal", "ShrtNegVal"} else 0 for key in VALUE_FIELDS})
    result.update(extra)
    return result


def bundle(rows=None, at="2026-10-27T18:00:00+09:00", status="complete"):
    return {"status": status, "collected_at": at, "rows": [row()] if rows is None else rows,
            "endpoint": "markets/margin-interest"}


def prices():
    return [{"Date": day.isoformat(), "Code": "12340", "AdjFactor": 1, "ExRT": None, "Vo": 1000} for day in SESSIONS]


def diagnose(blob=None, cutoff="2026-10-27T18:00:00+09:00", asof="2026-10-26", **kwargs):
    return margin_diagnostic(blob or bundle(), dt.datetime.fromisoformat(cutoff), expected_code="12340",
                             price_asof_date=dt.date.fromisoformat(asof), calendar_rows=kwargs.pop("calendar_rows", CAL),
                             price_rows=kwargs.pop("price_rows", prices()), **kwargs)


def daily_rows(end="2026-10-26"):
    return [row(day.isoformat(), long=200 + i * 2, short=100 + i)
            for i, day in enumerate(d for d in SESSIONS if DAILY_START <= d <= dt.date.fromisoformat(end))]


def test_adapter_preserves_raw_unknown_fields_and_legacy_nulls_without_mutation():
    original = bundle([row("2026-09-18", NewServerField={"foo": 42})])
    before = copy.deepcopy(original)
    result = adapt_margin_bundle(original, "12340")
    assert original == before
    assert result["rows"] == original["rows"]
    assert result["status"] == "complete"
    assert result["adapter_version"] == ADAPTER_VERSION
    version = result["row_versions"][0]
    assert version["frequency"] == "weekly"
    assert version["validation_state"] == "valid"
    assert all(version["normalized_row"][key] is None for key in (*VALUE_FIELDS, "PubDate"))
    assert version["row"]["NewServerField"] == {"foo": 42}


def test_content_versions_keep_first_observation_and_do_not_backdate_corrections():
    first = adapt_margin_bundle(bundle(at="2026-09-28T16:10:00+09:00"), "12340")
    same = adapt_margin_bundle(bundle(at="2026-09-29T16:10:00+09:00"), "12340", first)
    assert same["row_versions"][0]["first_observed_at"] == "2026-09-28T16:10:00+09:00"
    corrected = adapt_margin_bundle(bundle([row(short=120)], at="2026-09-30T16:10:00+09:00"), "12340", same)
    assert len(corrected["row_version_history"]) == 2
    assert corrected["row_versions"][0]["first_observed_at"] == "2026-09-30T16:10:00+09:00"
    assert diagnose(corrected, cutoff="2026-09-29T17:00:00+09:00", asof="2026-09-29")["margin_short"] == 100
    assert diagnose(corrected, cutoff="2026-09-30T17:00:00+09:00", asof="2026-09-30")["margin_short"] == 120
    reverted = adapt_margin_bundle(bundle(at="2026-10-01T16:10:00+09:00"), "12340", corrected)
    assert diagnose(reverted, cutoff="2026-10-01T17:00:00+09:00", asof="2026-10-01")["margin_short"] == 100
    assert diagnose(reverted, cutoff="2026-09-30T17:00:00+09:00", asof="2026-09-30")["margin_short"] == 120


def test_duplicate_and_conflicting_rows_are_distinct():
    same = bundle([row(), row()])
    assert len(adapt_margin_bundle(same)["row_versions"]) == 1
    assert len(adapt_margin_bundle(same)["rows"]) == 2
    assert diagnose(same)["margin_ratio"] == 2
    conflict = diagnose(bundle([row(), row(short=101)]))
    assert conflict["margin_diagnostic_status"] == "conflicting_rows"
    assert conflict["margin_short"] is None


@pytest.mark.parametrize("status,rows,wanted", [
    ("not_collected", [], "not_collected"), ("failed", [], "failed"),
    ("permission_denied", [], "permission_denied"), ("incomplete", [row()], "incomplete"),
    ("schema_review_required", [], "schema_review_required"), ("complete", [], "no_rows"),
])
def test_failure_and_empty_never_restore_previous_success_or_claim_zero(status, rows, wanted):
    previous = adapt_margin_bundle(bundle())
    current = adapt_margin_bundle(bundle(rows, status=status), previous_bundle=previous)
    result = diagnose(current)
    assert current["row_version_history"]
    assert result["margin_diagnostic_status"] == wanted
    assert result["margin_status"] == status
    assert result["margin_short"] is None
    assert result["margin_ratio"] is None


@pytest.mark.parametrize("long,short,state,ratio", [(100, 0, "zero_short_balance", None),
    (0, 0, "both_zero", None), (0, 100, "finite", 0)])
def test_real_zero_balances_are_not_fabricated_finite_low_ratios(long, short, state, ratio):
    result = diagnose(bundle([row(long=long, short=short)]))
    assert result["margin_ratio_state"] == state
    assert result["margin_ratio"] == ratio
    assert result["margin_short"] == short
    assert result["margin_score_connected"] is False


def test_latest_invalid_or_out_of_scope_row_is_not_replaced_by_older_good_row():
    old = row()
    for latest, status in [(row("2026-09-28", short=-1), "invalid"),
                            (row("2026-09-28", issue="3"), "not_applicable")]:
        result = diagnose(bundle([old, latest]))
        assert result["margin_reference_date"] == "2026-09-28"
        assert result["margin_diagnostic_status"] == status
        assert result["margin_short"] is None


def test_daily_missing_required_field_is_unsupported_and_null_is_invalid():
    missing = row()
    missing.pop("PubDate")
    assert diagnose(bundle([missing]))["margin_diagnostic_status"] == "unsupported_schema"
    assert diagnose(bundle([row(LongVal=None)]))["margin_diagnostic_status"] == "invalid"
    assert diagnose(bundle([row(Code="56780")]))["margin_diagnostic_status"] == "invalid"
    assert diagnose(bundle([row(issue="9")]))["margin_target_state"] == "issue_type_unknown"


def test_publication_date_does_not_grant_midnight_knowledge_but_same_day_observation_does():
    blob = bundle(at="2026-09-28T16:10:00+09:00")
    unavailable = diagnose(blob, cutoff="2026-09-28T16:09:59+09:00", asof="2026-09-28")
    assert unavailable["margin_short"] is None
    available = diagnose(blob, cutoff="2026-09-28T16:10:00+09:00", asof="2026-09-28")
    assert available["margin_short"] == 100
    assert available["margin_pub_date"] == "2026-09-28"
    assert available["margin_available_at"] == "2026-09-28T16:10:00+09:00"
    assert available["margin_availability_basis"] == "archived_observation"


def test_naive_or_missing_observation_cannot_supply_historical_data():
    for at in (None, "2026-09-28T16:00:00"):
        result = diagnose(bundle(at=at))
        assert result["margin_short"] is None
        assert result["margin_diagnostic_status"] == "observation_time_unknown"
    with pytest.raises(ValueError, match="timezone"):
        margin_diagnostic(bundle(), dt.datetime(2026, 9, 28))


def test_invalid_publication_order_is_not_accepted_even_after_observation():
    assert diagnose(bundle([row(PubDate="2026-09-25")]))["margin_short"] is None
    result = diagnose(bundle([row(PubDate="2026-09-29")], at="2026-09-28T16:00:00+09:00"))
    assert result["margin_diagnostic_status"] == "observed_before_publication_date"


def test_five_and_twenty_sessions_require_six_and_twenty_one_daily_observations():
    for end, n, expected in [("2026-10-02", 5, 5.0), ("2026-10-26", 20, 20.0)]:
        result = diagnose(bundle(daily_rows(end)))
        assert result[f"margin_short_change_{n}bd_pct"] == pytest.approx(expected)
        assert result[f"margin_long_change_{n}bd_pct"] == pytest.approx(expected)
        rows = daily_rows(end)
        rows.pop(2)
        assert diagnose(bundle(rows))[f"margin_short_change_{n}bd_pct"] is None
    assert diagnose(bundle(daily_rows("2026-10-01")))["margin_short_change_5bd_pct"] is None
    assert diagnose(bundle(daily_rows("2026-10-23")))["margin_short_change_20bd_pct"] is None


def test_zero_baseline_and_issue_change_disable_change_without_forward_fill():
    rows = daily_rows("2026-10-02")
    rows[0] = row(short=0)
    result = diagnose(bundle(rows))
    assert result["margin_short_change_5bd_pct"] is None
    assert result["margin_metric_reasons"]["margin_short_change_5bd_pct"] == "zero_baseline"
    rows = daily_rows("2026-10-02")
    rows[2]["IssType"] = "1"
    assert diagnose(bundle(rows))["margin_metric_reasons"]["margin_short_change_5bd_pct"] == "issue_type_changed"
    rows = daily_rows("2026-10-02")
    rows[-1]["ShrtVol"] = rows[-1]["ShrtNegVol"] = 0
    assert diagnose(bundle(rows))["margin_short_change_5bd_pct"] == -100


def test_weekly_bridge_records_actual_two_session_interval_not_one_daily_step():
    result = diagnose(bundle([row("2026-09-18", short=50), row()]))
    assert result["margin_bridge_business_days"] == 2
    assert result["margin_bridge_calendar_days"] == 7
    assert result["margin_bridge_short_change_pct"] == 100
    assert result["margin_short_change_1bd_pct"] is None
    earlier = diagnose(bundle([row("2026-09-11", short=50), row()]))
    assert earlier["margin_bridge_business_days"] == 7
    assert earlier["margin_bridge_start_date"] == "2026-09-11"


@pytest.mark.parametrize("change,reason", [({"AdjFactor": 0.5, "ExRT": "1"}, "corporate_action_in_window"),
    ({"ExRT": "3"}, "corporate_action_in_window"), ({"AdjFactor": 0.5}, "corporate_action_basis_unknown"),
    ({"ExRT": "X"}, "corporate_action_basis_unknown"), ({"AdjFactor": None}, "corporate_action_unverified")])
def test_corporate_actions_and_unknown_factors_invalidate_share_changes(change, reason):
    raw_prices = prices()
    next(r for r in raw_prices if r["Date"] == "2026-09-30").update(change)
    result = diagnose(bundle(daily_rows("2026-10-02")), price_rows=raw_prices)
    assert result["margin_short_change_5bd_pct"] is None
    assert result["margin_metric_reasons"]["margin_short_change_5bd_pct"] == reason
    assert result["margin_ratio"] == 2


def test_old_price_asof_prevents_using_future_cached_prices_and_volume():
    result = diagnose(bundle(daily_rows()), asof="2026-09-11")
    assert result["margin_after_price_asof"] is True
    assert result["margin_short"] == 120
    assert result["margin_short_change_20bd_pct"] is None
    assert result["margin_metric_reasons"]["margin_short_change_20bd_pct"] == "corporate_action_unverified"
    assert result["margin_short_to_volume_20bd"] is None


def test_turnover_uses_same_date_twenty_raw_volume_sessions_and_no_adjustment():
    result = diagnose(bundle(daily_rows()))
    assert result["margin_short_to_volume_20bd"] == pytest.approx(0.12)
    zero_prices = prices()
    for item in zero_prices:
        item["Vo"] = 0
    result = diagnose(bundle(daily_rows()), price_rows=zero_prices)
    assert result["margin_short_to_volume_20bd"] is None
    assert result["margin_metric_reasons"]["margin_short_to_volume_20bd"] == "zero_volume_denominator"


def test_calendar_hole_cannot_be_treated_as_holiday():
    missing = [r for r in CAL if r["Date"] != "2026-09-29"]
    result = diagnose(bundle(daily_rows("2026-10-02")), calendar_rows=missing)
    assert result["margin_short_change_5bd_pct"] is None
    assert result["margin_metric_reasons"]["margin_short_change_5bd_pct"] == "calendar_unverified"
    assert diagnose(calendar_rows=[])["margin_short"] is None


def test_weekly_amounts_remain_null_and_have_no_inferred_currency():
    result = diagnose(bundle([row("2026-09-18")]))
    assert result["margin_short_value"] is None
    assert result["margin_value_ratio"] is None
    assert result["margin_metric_reasons"]["margin_value_ratio"] == "not_provided_weekly"
    assert result["margin_amount_unit"] == "unverified"


def test_inputs_are_immutable_and_diagnostics_reproducible():
    source = adapt_margin_bundle(bundle(daily_rows()))
    before = copy.deepcopy(source)
    first, second = diagnose(source), diagnose(source)
    assert first == second
    assert source == before
    assert "short_selling_change_rate" not in first
    assert "eligible_new_entry" not in first
    assert "recommendation_score" not in first


def test_target_scope_survives_missing_values_and_does_not_revive_old_rows():
    for issue, target in (("3", "not_applicable"), (None, "issue_type_unknown")):
        latest = row("2026-09-28", issue=issue, LongVal=None, ShrtVol=None)
        result = diagnose(bundle([row(), latest]))
        assert result["margin_target_state"] == target
        assert result["margin_reference_date"] == "2026-09-28"
        assert result["margin_short"] is None


def test_inconsistent_general_short_is_not_rendered_as_valid_percentage():
    result = diagnose(bundle([row(ShrtNegVol=101)]))
    assert result["margin_short"] == 100
    assert result["margin_general_short_pct"] is None
    assert result["margin_metric_reasons"]["margin_general_short_pct"] == "general_short_exceeds_total"


def test_malformed_rows_container_is_quarantined_without_zero_values():
    broken = bundle()
    broken["rows"] = {"not": "a list"}
    result = diagnose(broken)
    assert result["margin_diagnostic_status"] == "unsupported_schema"
    assert result["margin_short"] is None
    assert adapt_margin_bundle(broken)["rows"] == {"not": "a list"}


def test_latest_unparseable_date_does_not_resurrect_old_valid_row():
    result = diagnose(bundle([row(), row(Date="unknown")]))
    assert result["margin_short"] is None
    assert result["margin_diagnostic_reasons"] == ["invalid_date"]


def test_snapshot_replay_does_not_move_first_observation_to_new_collection_time():
    old = adapt_margin_bundle(bundle(at="2026-09-28T16:10:00+09:00"))
    fresh = adapt_margin_bundle(bundle(at="2026-09-30T16:10:00+09:00"), previous_bundle=old)
    result = diagnose(fresh, cutoff="2026-09-28T17:00:00+09:00", asof="2026-09-28")
    assert result["margin_short"] == 100
    assert result["margin_first_observed_at"] == "2026-09-28T16:10:00+09:00"
