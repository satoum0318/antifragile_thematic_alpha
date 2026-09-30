"""Synthetic, offline contracts for independent manual-registration candidates."""
import copy
import math

import pytest

from jq_universe import UNIVERSE_LOGIC_VERSION, UniverseConfig, evaluate, from_mapping, rank_key


def candidate(**changes):
    row = {
        "code": "1234", "instrument_type": "stock", "model_scope": "nonfinancial",
        "common_issues": [], "financial_issues": [], "light_issues": [],
        "confirmed_severe_risk": False, "liquidity_ok": True, "market_cap_ok": True,
        "annual_comparable": True, "annual_stale": False, "critical_missing_count": 0,
        "summary_quality_available": 7, "summary_quality_score": 6,
        "Q1": True, "Q2": True, "op_income_stable": True,
        "market_regime": "risk_on", "earnings_blackout": False, "earnings_status": "known",
        "accounting_risk": "not_checked", "governance_status": "not_checked", "earnings_quality_flag": "ok",
        "price_asof_date": "2026-11-02", "information_cutoff": "2026-11-02T19:00:00+09:00",
        "registration_price_age_bd": 0, "ma200_state": "above_ma200_near",
        "below_ma200_downtrend": False, "above_ma200_extended": False,
        "recent_60d_low_update": False,
        "financial_asof": "2026-03-31", "financial_disclosed_at": "2026-05-15T15:00:00+09:00",
        "sales_cagr": .15, "op_income_yoy_pct": 25., "forward_np_change": .25,
        "guidance_status": "valid", "guidance_target": "2027-03-31",
        "forward_guidance_disclosed": "2026-05-15T15:00:00+09:00", "forecast_net_income": 100_000_000,
        "ps": 1., "ps_vs_sector": .8, "per": 20., "return_21d": .04, "return_63d": .20,
        "return_rel_topix_63d": .10, "turnover_persistence": 1.25,
        "margin_diagnostic_status": "available", "margin_validation_state": "valid",
        "margin_target_state": "applicable", "margin_frequency": "daily",
        "margin_information_cutoff": "2026-11-02T19:00:00+09:00",
        "margin_available_at": "2026-11-02T18:00:00+09:00",
        "margin_price_asof_date": "2026-11-02", "margin_reference_date": "2026-10-30",
        "margin_pub_date": "2026-11-02", "margin_after_price_asof": False,
        "margin_application_age_bd": 1, "margin_short": 500_000, "margin_long": 1_000_000,
        "margin_ratio": 2., "margin_ratio_state": "finite",
        "margin_short_to_volume_20bd": 2., "margin_long_to_volume_20bd": 4.,
        "margin_short_change_20bd_pct": 20., "margin_short_change_5bd_pct": 5.,
        "margin_long_change_20bd_pct": 3., "margin_long_change_5bd_pct": 2.,
        "eligible_new_entry": False, "recommendation_score": 73., "candidate_lane": "watch_fundamental_core",
    }
    row["margin_metric_states"] = {key: "available" for key in (
        "margin_application_age_bd", "margin_short", "margin_long", "margin_short_to_volume_20bd",
        "margin_long_to_volume_20bd", "margin_short_change_20bd_pct", "margin_short_change_5bd_pct",
        "margin_long_change_20bd_pct", "margin_long_change_5bd_pct")}
    row["margin_metric_reasons"] = {key: None for key in row["margin_metric_states"]}
    row.update(changes)
    return row


def test_uptrend_continuation_is_eligible_without_changing_legacy_or_input():
    row = candidate()
    original = copy.deepcopy(row)
    result = evaluate(row)
    assert row == original
    assert all(key.startswith("registration_") for key in result)
    assert not set(result) & {"eligible_new_entry", "recommendation_score", "candidate_lane"}
    assert result["registration_logic_version"] == UNIVERSE_LOGIC_VERSION
    assert result["registration_a_eligible"] is True
    assert result["registration_b_eligible"] is True
    assert result["registration_overlap"] is True
    assert result["registration_is_order_permission"] is False
    assert result["registration_price_state"] == "uptrend_continuation"
    assert result["registration_a_score"] != result["registration_b_score"]


def test_high_valuation_growth_candidate_is_separate_b_and_not_a_rescue():
    result = evaluate(candidate(ps=6., per=80.))
    assert result["registration_a_status"] == "excluded"
    assert result["registration_b_status"] == "eligible"
    poor = evaluate(candidate(ps=6., per=80., Q2=False, summary_quality_score=3))
    assert poor["registration_b_status"] == "excluded"
    assert "Q2_failed" in poor["registration_b_reasons"]


@pytest.mark.parametrize("changes", [
    {"sales_cagr": 0., "op_income_yoy_pct": 0., "forward_np_change": 0.},
    {"sales_cagr": -.05}, {"op_income_yoy_pct": -5.}, {"forward_np_change": -.10},
])
def test_cheap_stagnation_or_contradicted_growth_is_not_a_candidate(changes):
    result = evaluate(candidate(ps=.2, per=5., **changes))
    assert result["registration_a_status"] == "excluded"
    assert result["registration_growth_a_state"] == "contradicted"


def test_growth_thresholds_and_units_are_distinct_from_percentage_points():
    result = evaluate(candidate(sales_cagr=.03, op_income_yoy_pct=5., forward_np_change=.05))
    assert result["registration_a_status"] == "eligible"
    assert result["registration_b_status"] == "excluded"
    mixed = evaluate(candidate(sales_cagr=.02))
    assert mixed["registration_growth_a_state"] == "mixed"
    assert mixed["registration_a_status"] == "excluded"


@pytest.mark.parametrize("changes", [
    {"sales_cagr": None}, {"op_income_yoy_pct": None}, {"forward_np_change": None},
    {"guidance_status": "withdrawn"}, {"guidance_target": "2026-03-31"},
    {"forward_guidance_disclosed": "2026-11-03T12:00:00+09:00"},
    {"forward_guidance_disclosed": "2026-05-15T15:00:00"},
])
def test_missing_unpublished_or_period_invalid_guidance_is_hold_and_no_reweight(changes):
    result = evaluate(candidate(**changes))
    assert result["registration_a_status"] == "hold"
    assert result["registration_a_score"] is None
    assert result["registration_a_components"]["growth"] is None
    assert result["registration_a_score_coverage"] < 1.


@pytest.mark.parametrize("changes,reason", [
    ({"confirmed_severe_risk": True}, "confirmed_severe_risk"),
    ({"liquidity_ok": False}, "liquidity_ok_failed"),
    ({"market_cap_ok": False}, "market_cap_ok_failed"),
    ({"instrument_type": "non_stock"}, "not_ordinary_stock"),
    ({"model_scope": "financial"}, "outside_nonfinancial_model"),
    ({"op_income_stable": False}, "op_income_stable_failed"),
])
def test_high_score_cannot_override_financial_or_scope_restrictions(changes, reason):
    result = evaluate(candidate(sales_cagr=.6, op_income_yoy_pct=100., forward_np_change=1., **changes))
    for route in ("a", "b"):
        assert result[f"registration_{route}_status"] == "excluded"
        assert reason in result[f"registration_{route}_reasons"]


@pytest.mark.parametrize("changes,reason", [
    ({"common_issues": ["snapshot_incomplete"]}, "common_issues:snapshot_incomplete"),
    ({"financial_issues": ["share_basis_unknown"]}, "financial_issues:share_basis_unknown"),
    ({"light_issues": ["disclosed_shares_market_cap_proxy"]}, "light_issues:disclosed_shares_market_cap_proxy"),
    ({"summary_quality_available": 6}, "summary_quality_incomplete_or_invalid"),
    ({"registration_price_age_bd": 4}, "price_stale"),
    ({"registration_price_age_bd": None}, "price_age_unknown"),
    ({"earnings_blackout": True}, "earnings_blackout"),
    ({"ma200_state": "above_ma200_extended"}, "price_condition:overheated_wait"),
    ({"ma200_state": "below_ma200_downtrend"}, "price_condition:downtrend_wait"),
])
def test_data_and_price_wait_states_remain_reviewable_not_top_candidates(changes, reason):
    result = evaluate(candidate(**changes))
    assert result["registration_a_status"] == "hold"
    assert result["registration_b_status"] == "hold"
    assert reason in result["registration_a_reasons"]


def test_basing_is_a_registration_waiting_setup_without_order_permission():
    result = evaluate(candidate(ma200_state="below_ma200_basing"))
    assert result["registration_a_status"] == "eligible"
    assert result["registration_price_state"] == "basing_wait"
    assert result["registration_is_order_permission"] is False


@pytest.mark.parametrize("changes,reason", [
    ({"summary_quality_score": 5}, "basing_quality_requires_six_of_seven"),
    ({"return_21d": 0.}, "basing_positive_return_unconfirmed"),
    ({"return_21d": None}, "basing_positive_return_unconfirmed"),
    ({"recent_60d_low_update": True}, "basing_no_recent_low_update_unconfirmed"),
    ({"recent_60d_low_update": None}, "basing_no_recent_low_update_unconfirmed"),
    ({"market_regime": "risk_off"}, "basing_market_risk_off"),
])
def test_basing_keeps_existing_additional_quality_and_market_checks(changes, reason):
    result = evaluate(candidate(ma200_state="below_ma200_basing", **changes))
    for route in ("a", "b"):
        assert result[f"registration_{route}_status"] == "hold"
        assert reason in result[f"registration_{route}_reasons"]


@pytest.mark.parametrize("warning", ["extremely_low_possible_oneoff", "eps_growth_too_high_oneoff_risk"])
def test_existing_oneoff_growth_warning_is_not_removed_by_high_growth_score(warning):
    result = evaluate(candidate(peg_warnings=[warning]))
    for route in ("a", "b"):
        assert result[f"registration_{route}_status"] == "hold"
        assert "growth_oneoff_risk_review_required" in result[f"registration_{route}_reasons"]


def test_a_sector_relative_valuation_changes_score_without_changing_b():
    cheap = evaluate(candidate(ps_vs_sector=.5))
    costly = evaluate(candidate(ps_vs_sector=1.5))
    missing = evaluate(candidate(ps_vs_sector=None))
    assert cheap["registration_a_score"] > costly["registration_a_score"]
    assert cheap["registration_b_score"] == costly["registration_b_score"]
    assert missing["registration_a_status"] == "hold"
    assert missing["registration_a_components"]["valuation"] is None
    assert missing["registration_a_score"] is None
    assert missing["registration_b_status"] == "eligible"


@pytest.mark.parametrize("status", ["no_rows", "fetch_failed", "unsupported_schema", "not_applicable"])
def test_credit_missing_has_no_a_score_benefit_and_holds_b(status):
    original = evaluate(candidate())
    result = evaluate(candidate(margin_diagnostic_status=status))
    assert result["registration_a_status"] == "eligible"
    assert result["registration_a_score"] == original["registration_a_score"]
    assert result["registration_supply_state"] == "unassessed"
    assert "supply_unassessed" in result["registration_a_warnings"]
    assert result["registration_b_status"] == "hold"
    assert result["registration_b_score"] is None


@pytest.mark.parametrize("changes,reason", [
    ({"margin_reference_date": "2026-11-03", "margin_pub_date": "2026-11-04"}, "margin_after_price_asof"),
    ({"margin_available_at": "2026-11-02T20:00:00+09:00"}, "margin_not_observed_by_cutoff"),
    ({"margin_information_cutoff": "2026-11-03T19:00:00+09:00"}, "margin_cutoff_mismatch_or_unknown"),
    ({"margin_price_asof_date": "2026-10-30"}, "margin_price_basis_unknown_or_mismatch"),
    ({"margin_application_age_bd": 4}, "margin_stale"),
    ({"margin_pub_date": None}, "margin_publication_unverified"),
])
def test_credit_point_in_time_failures_cannot_feed_b(changes, reason):
    result = evaluate(candidate(**changes))
    assert result["registration_b_status"] == "hold"
    assert result["registration_b_components"]["supply"] is None
    assert reason in result["registration_supply_reasons"]


def test_twenty_session_decline_cannot_be_rescued_by_five_session_rise():
    result = evaluate(candidate(margin_short_change_20bd_pct=-1., margin_short_change_5bd_pct=100.))
    assert result["registration_b_status"] == "excluded"
    assert result["registration_supply_evidence"]["short_window_bd"] == 20
    assert "short_balance_not_accumulating" in result["registration_b_reasons"]


def test_five_session_fallback_requires_explicit_available_state():
    row = candidate()
    row["margin_metric_states"]["margin_short_change_20bd_pct"] = "unavailable"
    row["margin_metric_reasons"]["margin_short_change_20bd_pct"] = "daily_history_insufficient"
    result = evaluate(row)
    assert result["registration_b_status"] == "eligible"
    assert result["registration_supply_evidence"]["short_window_bd"] == 5
    row["margin_metric_states"]["margin_short_change_5bd_pct"] = "unavailable"
    row["margin_metric_reasons"]["margin_short_change_5bd_pct"] = "corporate_action_in_window"
    row["margin_short_change_1bd_pct"] = 1000.
    row["margin_metric_states"]["margin_short_change_1bd_pct"] = "available"
    result = evaluate(row)
    assert result["registration_b_status"] == "hold"
    assert result["registration_supply_evidence"]["short_change_pct"] is None


def test_available_but_nonfinite_twenty_day_value_does_not_fall_back():
    result = evaluate(candidate(margin_short_change_20bd_pct=math.nan, margin_short_change_5bd_pct=100.))
    assert result["registration_b_status"] == "hold"
    assert result["registration_supply_evidence"]["short_window_bd"] == 20


def test_zero_short_balance_is_observation_not_a_finite_low_ratio_or_b_entry():
    result = evaluate(candidate(margin_short=0., margin_short_to_volume_20bd=0.,
                                margin_ratio=None, margin_ratio_state="zero_short_balance"))
    assert result["registration_a_status"] == "eligible"
    assert result["registration_b_status"] == "excluded"
    assert "zero_short_balance_not_low_ratio" in result["registration_supply_observations"]
    assert result["registration_evidence"]["margin_ratio"] is None


def test_long_pressure_requires_joint_balance_growth_and_price_evidence():
    normal = evaluate(candidate())
    pressure = candidate(margin_long_to_volume_20bd=11., margin_long_change_20bd_pct=35., return_21d=-.06)
    result = evaluate(pressure)
    assert result["registration_supply_state"] == "caution"
    assert "extreme_long_balance_growth_with_weak_price" in result["registration_supply_warnings"]
    assert result["registration_a_status"] == "eligible"
    assert result["registration_a_score"] == normal["registration_a_score"]
    pressure["return_21d"] = .05
    assert "extreme_long_balance_growth_with_weak_price" not in evaluate(pressure)["registration_supply_warnings"]
    pressure["return_21d"] = -.06
    pressure["margin_metric_states"]["margin_long_change_20bd_pct"] = "unavailable"
    pressure["margin_metric_states"]["margin_long_change_5bd_pct"] = "unavailable"
    assert "extreme_long_balance_growth_with_weak_price" not in evaluate(pressure)["registration_supply_warnings"]


@pytest.mark.parametrize("changes", [
    {"return_63d": 0.}, {"return_rel_topix_63d": -.01}, {"turnover_persistence": .99},
    {"return_63d": None}, {"return_rel_topix_63d": None}, {"turnover_persistence": None},
])
def test_b_needs_multiday_price_and_volume_evidence_besides_short_balance(changes):
    result = evaluate(candidate(**changes))
    assert result["registration_b_status"] == "hold"
    assert result["registration_a_status"] == "eligible"


@pytest.mark.parametrize("overrides", [
    {"UNKNOWN": 1}, {"A_MAX_PS": True}, {"A_MAX_PS": "2"}, {"A_MAX_PS": float("inf")},
    {"A_MIN_SALES_CAGR": 0}, {"MAX_PRICE_AGE_BD": 3.5}, {"MIN_QUALITY_PASSES": 4},
    {"B_MAX_PER": 40}, {"A_WEIGHT_GROWTH": 36}, {"LONG_EXTREME_CHANGE_PCT": 5},
    {"WEAK_PRICE_RETURN_21D": .05},
])
def test_invalid_or_ambiguous_rc_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        from_mapping(overrides)


def test_thresholds_and_weights_are_explicit_configurable_and_do_not_mutate_defaults():
    rules = UniverseConfig.from_mapping({"A_MIN_SALES_CAGR": .04,
                                        "A_WEIGHT_GROWTH": 40., "A_WEIGHT_VALUATION": 20.})
    assert UniverseConfig().A_MIN_SALES_CAGR == .03
    result = evaluate(candidate(sales_cagr=.035), rules)
    assert result["registration_a_status"] == "excluded"
    assert result["registration_growth_a_evidence"]["thresholds"]["sales_cagr"] == .04


def test_route_rank_is_deterministic_and_does_not_fill_with_held_candidates():
    rows = []
    for code, kwargs in [("2222", {}), ("1111", {}), ("0001", {"earnings_blackout": True})]:
        row = candidate(code=code, **kwargs)
        row.update(evaluate(row))
        rows.append(row)
    assert [row["code"] for row in sorted(rows, key=lambda row: rank_key(row, "a"))] == ["1111", "2222", "0001"]
    assert len([row for row in rows if row["registration_a_eligible"]][:10]) == 2
    with pytest.raises(ValueError):
        rank_key(rows[0], "combined")
