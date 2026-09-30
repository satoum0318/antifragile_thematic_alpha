"""Independent A/B registration-candidate rules; never an order permission.

All thresholds and weights are explicit release-candidate defaults, not claims
of demonstrated investment performance. Inputs must be point-in-time audited by
the caller. This module does no I/O and never changes legacy screening fields.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, Mapping

UNIVERSE_LOGIC_VERSION = "JQ-UNIVERSE-AB-1.0-rc1"
JST = dt.timezone(dt.timedelta(hours=9))


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (str, Real)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _timestamp(value: Any) -> dt.datetime | None:
    try:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.astimezone(JST) if result.tzinfo is not None and result.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def _unit(value: float) -> float:
    return max(0., min(1., value))


@dataclass(frozen=True)
class UniverseConfig:
    A_MIN_SALES_CAGR: float = .03
    A_MIN_OP_GROWTH_PCT: float = 5.
    A_MIN_FORWARD_GROWTH: float = .05
    A_MAX_PS: float = 2.
    A_MAX_PER: float = 60.
    B_MIN_SALES_CAGR: float = .10
    B_MIN_OP_GROWTH_PCT: float = 15.
    B_MIN_FORWARD_GROWTH: float = .15
    B_MAX_PS: float = 10.
    B_MAX_PER: float = 100.
    MIN_QUALITY_PASSES: int = 5
    MAX_PRICE_AGE_BD: int = 3
    MAX_MARGIN_AGE_BD: int = 3
    B_MIN_SHORT_TO_VOLUME_20BD: float = 1.
    B_MIN_SHORT_CHANGE_PCT: float = 0.
    B_MIN_RETURN_63D: float = 0.
    B_MIN_RELATIVE_RETURN_63D: float = 0.
    B_MIN_TURNOVER_PERSISTENCE: float = 1.
    LONG_PRESSURE_TO_VOLUME: float = 10.
    LONG_PRESSURE_CHANGE_PCT: float = 10.
    LONG_EXTREME_CHANGE_PCT: float = 30.
    WEAK_PRICE_RETURN_21D: float = -.05
    A_WEIGHT_QUALITY: float = 35.
    A_WEIGHT_GROWTH: float = 35.
    A_WEIGHT_VALUATION: float = 25.
    A_WEIGHT_TECHNICAL: float = 5.
    B_WEIGHT_QUALITY: float = 25.
    B_WEIGHT_GROWTH: float = 40.
    B_WEIGHT_VALUATION: float = 10.
    B_WEIGHT_TECHNICAL: float = 20.
    B_WEIGHT_SUPPLY: float = 5.

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None = None) -> "UniverseConfig":
        if overrides is not None and not isinstance(overrides, Mapping):
            raise ValueError("REGISTRATION_RULES must be an object or null")
        values = dict(overrides or {})
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown registration rule: " + ",".join(sorted(map(str, unknown))))
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("Registration rule must be a finite number: " + key)
            if key != "WEAK_PRICE_RETURN_21D" and value < 0:
                raise ValueError("Registration rule must be nonnegative: " + key)
        for key in ("MIN_QUALITY_PASSES", "MAX_PRICE_AGE_BD", "MAX_MARGIN_AGE_BD"):
            if not isinstance(values[key], int):
                raise ValueError("Registration rule must be an integer: " + key)
        if not 5 <= self.MIN_QUALITY_PASSES <= 7:
            raise ValueError("Registration quality requires at least 5 of 7 items")
        for route in ("A", "B"):
            for name in ("MIN_SALES_CAGR", "MIN_OP_GROWTH_PCT", "MIN_FORWARD_GROWTH", "MAX_PS", "MAX_PER"):
                if values[route + "_" + name] <= 0:
                    raise ValueError("Positive growth and valuation thresholds required")
            weights = [value for key, value in values.items() if key.startswith(route + "_WEIGHT_")]
            if not math.isclose(sum(weights), 100., rel_tol=0., abs_tol=1e-9):
                raise ValueError(route + " registration weights must sum to 100")
        for name in ("MIN_SALES_CAGR", "MIN_OP_GROWTH_PCT", "MIN_FORWARD_GROWTH", "MAX_PS", "MAX_PER"):
            if values["B_" + name] < values["A_" + name]:
                raise ValueError("B thresholds must be at least the corresponding A thresholds")
        if self.B_MIN_SHORT_TO_VOLUME_20BD <= 0 or self.LONG_PRESSURE_TO_VOLUME <= 0:
            raise ValueError("Positive balance-to-volume thresholds required")
        if self.LONG_EXTREME_CHANGE_PCT < self.LONG_PRESSURE_CHANGE_PCT:
            raise ValueError("Extreme long-balance threshold must not be below caution threshold")
        if not -1 < self.WEAK_PRICE_RETURN_21D < 0:
            raise ValueError("Weak-price threshold must be a negative fractional return above -1")


def from_mapping(overrides: Mapping[str, Any] | None = None) -> UniverseConfig:
    return UniverseConfig.from_mapping(overrides)


def _metric(record: dict, name: str) -> float | None:
    states = record.get("margin_metric_states")
    if not isinstance(states, dict) or states.get(name) != "available":
        return None
    return _number(record.get(name))


def _change(record: dict, side: str) -> tuple[int | None, float | None]:
    """An available 20-session observation always wins, including a decline."""
    states = record.get("margin_metric_states") or {}
    for window in (20, 5):
        key = f"margin_{side}_change_{window}bd_pct"
        if isinstance(states, dict) and states.get(key) == "available":
            return window, _number(record.get(key))
    return None, None


def _growth(record: dict, rules: UniverseConfig, route: str) -> dict:
    cutoff = _timestamp(record.get("information_cutoff"))
    disclosed = _timestamp(record.get("forward_guidance_disclosed"))
    target, financial = _date(record.get("guidance_target")), _date(record.get("financial_asof"))
    forecast = _number(record.get("forecast_net_income"))
    valid = (record.get("guidance_status") == "valid" and cutoff is not None
             and disclosed is not None and disclosed <= cutoff and target is not None
             and financial is not None and target > financial and forecast is not None)
    values = {"sales_cagr": _number(record.get("sales_cagr")),
              "op_income_yoy_pct": _number(record.get("op_income_yoy_pct")),
              "forward_np_change": _number(record.get("forward_np_change")) if valid else None}
    limits = {"sales_cagr": getattr(rules, route + "_MIN_SALES_CAGR"),
              "op_income_yoy_pct": getattr(rules, route + "_MIN_OP_GROWTH_PCT"),
              "forward_np_change": getattr(rules, route + "_MIN_FORWARD_GROWTH")}
    reasons = []
    for key, value in values.items():
        if value is None:
            reasons.append("growth_unavailable:" + key)
        elif value < limits[key]:
            reasons.append("growth_below_threshold:" + key)
    if not valid:
        reasons.append("guidance_unverified_or_unavailable")
    if valid and forecast <= 0:
        reasons.append("forecast_loss_or_zero")
    contradiction = any(value is not None and value <= 0 for value in values.values()) or (valid and forecast <= 0)
    state = ("contradicted" if contradiction else "insufficient" if any(value is None for value in values.values())
             else "confirmed" if not reasons else "mixed")
    score = None if any(value is None for value in values.values()) else sum(
        _unit(value / (3 * limits[key])) for key, value in values.items()) / 3
    return {"state": state, "reasons": reasons, "values": values, "thresholds": limits, "unit_score": score}


def _supply(record: dict, rules: UniverseConfig) -> dict:
    problems = []
    cutoff = _timestamp(record.get("information_cutoff"))
    credit_cutoff = _timestamp(record.get("margin_information_cutoff"))
    available = _timestamp(record.get("margin_available_at"))
    price_day = _date(record.get("price_asof_date"))
    credit_price_day = _date(record.get("margin_price_asof_date"))
    application, publication = _date(record.get("margin_reference_date")), _date(record.get("margin_pub_date"))
    if record.get("margin_diagnostic_status") != "available":
        problems.append("margin_unavailable:" + str(record.get("margin_diagnostic_status", "unknown")))
    if record.get("margin_validation_state") != "valid" or record.get("margin_target_state") != "applicable":
        problems.append("margin_contract_unverified")
    if cutoff is None or credit_cutoff != cutoff:
        problems.append("margin_cutoff_mismatch_or_unknown")
    if available is None or cutoff is None or available > cutoff:
        problems.append("margin_not_observed_by_cutoff")
    if price_day is None or credit_price_day != price_day or application is None:
        problems.append("margin_price_basis_unknown_or_mismatch")
    elif application > price_day or record.get("margin_after_price_asof") is True:
        problems.append("margin_after_price_asof")
    if record.get("margin_frequency") == "daily":
        if publication is None or cutoff is None or publication > cutoff.date():
            problems.append("margin_publication_unverified")
        elif application is None or publication <= application:
            problems.append("margin_publication_order_invalid")
        if publication is not None and available is not None and available.date() < publication:
            problems.append("margin_observed_before_publication")
    age = _metric(record, "margin_application_age_bd")
    if age is None or age < 0:
        problems.append("margin_age_unknown")
    elif age > rules.MAX_MARGIN_AGE_BD:
        problems.append("margin_stale")
    price_age = _number(record.get("registration_price_age_bd"))
    if price_age is None or price_age < 0 or price_age > rules.MAX_PRICE_AGE_BD:
        problems.append("price_not_fresh_for_supply_comparison")
    short_window, short_change = _change(record, "short")
    long_window, long_change = _change(record, "long")
    short_level = _metric(record, "margin_short_to_volume_20bd")
    long_level = _metric(record, "margin_long_to_volume_20bd")
    short = _metric(record, "margin_short")
    warnings, observations = [], []
    change21 = _number(record.get("return_21d"))
    if not problems:
        if (long_level is not None and long_level >= rules.LONG_PRESSURE_TO_VOLUME
                and long_change is not None and change21 is not None and change21 <= rules.WEAK_PRICE_RETURN_21D):
            if long_change >= rules.LONG_EXTREME_CHANGE_PCT:
                warnings.append("extreme_long_balance_growth_with_weak_price")
            elif long_change >= rules.LONG_PRESSURE_CHANGE_PCT:
                warnings.append("long_balance_growth_with_weak_price")
        if short_change is not None and short_change > 0 and change21 is not None:
            if change21 <= rules.WEAK_PRICE_RETURN_21D:
                warnings.append("short_balance_growth_with_weak_price")
            elif change21 >= 0:
                observations.append("short_balance_growth_with_nonnegative_price_return")
        if record.get("margin_ratio_state") == "zero_short_balance":
            observations.append("zero_short_balance_not_low_ratio")
    return {"state": "unassessed" if problems else "caution" if warnings else "observed",
            "reasons": sorted(set(problems)), "warnings": warnings, "observations": observations,
            "short_window_bd": short_window, "short_change_pct": short_change,
            "long_window_bd": long_window, "long_change_pct": long_change,
            "short_balance": short, "short_to_volume_20bd": short_level,
            "long_to_volume_20bd": long_level,
            "comparison_valid": not problems}


def evaluate(record: dict, rules: UniverseConfig | Mapping[str, Any] | None = None) -> dict:
    """Return new registration_* fields; do not mutate record or legacy scores."""
    rules = rules if isinstance(rules, UniverseConfig) else UniverseConfig.from_mapping(rules)
    rules.validate()
    hold, excluded, warnings = [], [], list(record.get("light_issues") or [])
    for key in ("common_issues", "financial_issues", "light_issues"):
        if record.get(key):
            hold.extend(key + ":" + str(issue) for issue in record[key])
    if record.get("instrument_type") != "stock":
        excluded.append("not_ordinary_stock")
    if record.get("model_scope") != "nonfinancial":
        excluded.append("outside_nonfinancial_model")
    if record.get("confirmed_severe_risk") is True:
        excluded.append("confirmed_severe_risk")
    for key in ("liquidity_ok", "market_cap_ok", "annual_comparable", "Q1", "Q2", "op_income_stable"):
        if record.get(key) is False:
            excluded.append(key + "_failed")
        elif record.get(key) is not True:
            hold.append(key + "_unknown")
    if record.get("annual_stale") is not False:
        hold.append("annual_stale_or_unknown")
    if _number(record.get("critical_missing_count")) != 0:
        hold.append("critical_financial_fields_missing_or_unknown")
    count, passes = _number(record.get("summary_quality_available")), _number(record.get("summary_quality_score"))
    if count != 7 or passes is None or not passes.is_integer() or not 0 <= passes <= 7:
        hold.append("summary_quality_incomplete_or_invalid")
    elif passes < rules.MIN_QUALITY_PASSES:
        excluded.append("financial_quality_below_threshold")
    if record.get("market_regime") not in {"risk_on", "neutral", "risk_off"}:
        hold.append("market_regime_unknown")
    elif record.get("market_regime") == "risk_off":
        warnings.append("market_risk_off")
    if record.get("earnings_blackout") is True:
        hold.append("earnings_blackout")
    if record.get("earnings_status") != "known":
        warnings.append("earnings_schedule_not_confirmed_pretrade_check_required")
    for key in ("accounting_risk", "governance_status"):
        if record.get(key) == "watch":
            hold.append(key + "_watch")
        else:
            warnings.append(key + "_not_comprehensively_verified")
    if record.get("earnings_quality_flag") == "watch":
        hold.append("earnings_quality_watch")
    oneoff_warnings = {"extremely_low_possible_oneoff", "eps_growth_too_high_oneoff_risk"}
    if oneoff_warnings.intersection(record.get("peg_warnings") or []):
        hold.append("growth_oneoff_risk_review_required")
    age = _number(record.get("registration_price_age_bd"))
    if age is None or age < 0:
        hold.append("price_age_unknown")
    elif age > rules.MAX_PRICE_AGE_BD:
        hold.append("price_stale")
    price_states = {"ma200_reclaim": "trend_transition", "above_ma200_near": "uptrend_continuation",
                    "above_ma200_extended": "overheated_wait", "below_ma200_basing": "basing_wait",
                    "below_ma200_downtrend": "downtrend_wait", "ma200_unknown": "unknown"}
    price_state = price_states.get(record.get("ma200_state"), "unknown")
    if record.get("below_ma200_downtrend") is True:
        price_state = "downtrend_wait"
    if record.get("above_ma200_extended") is True:
        price_state = "overheated_wait"
    if price_state in {"downtrend_wait", "overheated_wait", "unknown"}:
        hold.append("price_condition:" + price_state)
    if price_state == "basing_wait":
        if count != 7 or passes is None or passes < 6:
            hold.append("basing_quality_requires_six_of_seven")
        return21 = _number(record.get("return_21d"))
        if return21 is None or return21 <= 0:
            hold.append("basing_positive_return_unconfirmed")
        if record.get("recent_60d_low_update") is not False:
            hold.append("basing_no_recent_low_update_unconfirmed")
        if record.get("market_regime") == "risk_off":
            hold.append("basing_market_risk_off")
    supply = _supply(record, rules)
    out = {"registration_logic_version": UNIVERSE_LOGIC_VERSION,
           "registration_is_order_permission": False, "registration_price_state": price_state,
           "registration_price_warnings": sorted(set(reason for reason in hold if
                reason.startswith("price_") or reason == "earnings_blackout")),
           "registration_supply_state": supply["state"], "registration_supply_reasons": supply["reasons"],
           "registration_supply_warnings": supply["warnings"],
           "registration_supply_observations": supply["observations"],
           "registration_supply_evidence": supply,
           "registration_horizon_primary": "6-12_months", "registration_horizon_secondary": "1-3_years"}
    quality = (None if count != 7 or passes is None or not passes.is_integer() or not 0 <= passes <= 7
               or record.get("op_income_stable") not in (True, False) or record.get("Q2") not in (True, False)
               else .6 * passes / 7 + .2 * int(record["op_income_stable"]) + .2 * int(record["Q2"]))
    ps, per = _number(record.get("ps")), _number(record.get("per"))
    sector_ps = _number(record.get("ps_vs_sector"))
    for route in ("a", "b"):
        prefix, upper = "registration_" + route + "_", route.upper()
        route_hold, route_excluded = list(hold), list(excluded)
        growth = _growth(record, rules, upper)
        if growth["state"] == "insufficient":
            route_hold.extend(growth["reasons"])
        elif growth["state"] != "confirmed":
            route_excluded.extend(growth["reasons"])
        max_ps, max_per = getattr(rules, upper + "_MAX_PS"), getattr(rules, upper + "_MAX_PER")
        for name, value, maximum in (("ps", ps, max_ps), ("per", per, max_per)):
            if value is None:
                route_hold.append(name + "_unknown")
            elif value <= 0 or value > maximum:
                route_excluded.append(name + "_outside_route_range")
        valuation = None if ps is None or per is None or ps <= 0 or per <= 0 else .5 * _unit(1 - ps / max_ps) + .5 * _unit(1 - per / max_per)
        if route == "a":
            if sector_ps is None or sector_ps <= 0:
                route_hold.append("sector_ps_comparison_unavailable")
                valuation = None
            elif valuation is not None:
                valuation = .4 * _unit(1 - ps / max_ps) + .3 * _unit(1 - sector_ps / 2) + .3 * _unit(1 - per / max_per)
        technical = {"uptrend_continuation": 1., "trend_transition": .9, "basing_wait": .65,
                     "overheated_wait": .2, "downtrend_wait": 0.}.get(price_state)
        if route == "b":
            if supply["reasons"]:
                route_hold.extend(supply["reasons"])
            if record.get("margin_frequency") != "daily":
                route_hold.append("daily_margin_required")
            for key in ("short_balance", "short_to_volume_20bd", "short_change_pct"):
                if supply[key] is None:
                    route_hold.append("short_evidence_missing:" + key)
            if not supply["reasons"]:
                if supply["short_balance"] is not None and supply["short_balance"] <= 0:
                    route_excluded.append("short_balance_not_positive")
                if supply["short_to_volume_20bd"] is not None and supply["short_to_volume_20bd"] < rules.B_MIN_SHORT_TO_VOLUME_20BD:
                    route_excluded.append("short_balance_to_volume_below_threshold")
                if supply["short_change_pct"] is not None and supply["short_change_pct"] <= rules.B_MIN_SHORT_CHANGE_PCT:
                    route_excluded.append("short_balance_not_accumulating")
            ret, relative, turnover = (_number(record.get(key)) for key in
                                       ("return_63d", "return_rel_topix_63d", "turnover_persistence"))
            for name, value, valid in (("return_63d", ret, ret is not None and ret > rules.B_MIN_RETURN_63D),
                                       ("return_rel_topix_63d", relative, relative is not None and relative >= rules.B_MIN_RELATIVE_RETURN_63D),
                                       ("turnover_persistence", turnover, turnover is not None and turnover >= rules.B_MIN_TURNOVER_PERSISTENCE)):
                if value is None:
                    route_hold.append("momentum_unknown:" + name)
                elif not valid:
                    route_hold.append("momentum_wait:" + name)
            technical = None if any(value is None for value in (ret, relative, turnover)) else (
                .4 * _unit(ret / .30) + .3 * _unit(relative / .15) + .3 * _unit(turnover / 2))
        components = {"quality": None if quality is None else quality * getattr(rules, upper + "_WEIGHT_QUALITY"),
                      "growth": None if growth["unit_score"] is None else growth["unit_score"] * getattr(rules, upper + "_WEIGHT_GROWTH"),
                      "valuation": None if valuation is None else valuation * getattr(rules, upper + "_WEIGHT_VALUATION"),
                      "technical": None if technical is None else technical * getattr(rules, upper + "_WEIGHT_TECHNICAL")}
        if route == "b":
            components["supply"] = (None if supply["reasons"] or supply["short_to_volume_20bd"] is None or supply["short_change_pct"] is None
                                    else rules.B_WEIGHT_SUPPLY * (.5 * _unit(supply["short_to_volume_20bd"] / 3)
                                                                 + .5 * _unit(supply["short_change_pct"] / 50)))
        score = None if any(value is None for value in components.values()) else sum(components.values())
        if score is None:
            route_hold.append("score_components_unavailable")
        status = "excluded" if route_excluded else "hold" if route_hold else "eligible"
        out.update({prefix + "eligible": status == "eligible", prefix + "status": status,
                    prefix + "score": score, prefix + "components": components,
                    prefix + "reasons": sorted(set(route_excluded + route_hold)),
                    prefix + "warnings": sorted(set(warnings + supply["warnings"] +
                                                      (["supply_unassessed"] if supply["reasons"] else []))),
                    "registration_growth_" + route + "_state": growth["state"],
                    "registration_growth_" + route + "_evidence": growth,
                    prefix + "score_coverage": sum(value is not None for value in components.values()) / len(components)})
    out["registration_overlap"] = out["registration_a_eligible"] and out["registration_b_eligible"]
    out["registration_evidence"] = {key: record.get(key) for key in (
        "price_asof_date", "information_cutoff", "registration_price_age_bd", "financial_asof", "financial_disclosed_at",
        "sales_cagr", "op_income_yoy_pct", "guidance_status", "guidance_target", "forward_guidance_disclosed",
        "forecast_net_income", "forward_np_change", "forecast_revision_rate", "positive_cfo_years_3y",
        "cumulative_cfo_to_np_3y", "ps", "per", "ps_vs_sector", "summary_quality_score", "summary_quality_available",
        "return_21d", "return_63d", "return_rel_topix_63d", "turnover_persistence", "margin_long", "margin_short",
        "margin_ratio", "margin_ratio_state", "margin_reference_date", "margin_pub_date", "margin_available_at",
        "margin_frequency", "margin_application_age_bd", "margin_metric_reasons")}
    return out


def rank_key(record: dict, route: str = "a") -> tuple:
    """Deterministic within-route rank; never compare the two route scores."""
    if route not in {"a", "b"}:
        raise ValueError("Registration route must be 'a' or 'b'")
    prefix = "registration_" + route + "_"
    status = record.get(prefix + "status")
    score = _number(record.get(prefix + "score"))
    return ({"eligible": 0, "hold": 1, "excluded": 2}.get(status, 3),
            -(score if score is not None else -1.), str(record.get("code", "")))
