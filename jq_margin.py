"""Offline, versioned diagnostics for J-Quants credit balances.

This module intentionally has no network, persistence or scoring dependencies.
Amounts remain API values: their currency unit has not been verified.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from typing import Any

JST = dt.timezone(dt.timedelta(hours=9))
DAILY_START = dt.date(2026, 9, 25)
ADAPTER_VERSION = "margin-v1"
VOLUME_FIELDS = ("LongVol", "ShrtVol", "LongNegVol", "ShrtNegVol", "LongStdVol", "ShrtStdVol")
VALUE_FIELDS = ("LongVal", "ShrtVal", "LongNegVal", "ShrtNegVal", "LongStdVal", "ShrtStdVal")
FIELD_NAMES = {
    "LongVol": "margin_long", "ShrtVol": "margin_short",
    "LongNegVol": "margin_long_neg", "ShrtNegVol": "margin_short_neg",
    "LongStdVol": "margin_long_std", "ShrtStdVol": "margin_short_std",
    "LongVal": "margin_long_value", "ShrtVal": "margin_short_value",
    "LongNegVal": "margin_long_neg_value", "ShrtNegVal": "margin_short_neg_value",
    "LongStdVal": "margin_long_std_value", "ShrtStdVal": "margin_short_std_value",
}


def _date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _timestamp(value: Any) -> dt.datetime | None:
    try:
        result = value if isinstance(value, dt.datetime) else dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            return None
        return result.astimezone(JST)
    except (ValueError, TypeError, OverflowError):
        return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _code(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    if len(value) == 4 and value.isalnum():
        value += "0"
    return value if len(value) == 5 and value.isalnum() else None


def _hash(row: Any) -> str:
    # Hash raw values, including unknown fields, to avoid borrowing observation
    # evidence from a correction or silently changed server representation.
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _validate(row: Any, expected_code: str | None) -> dict:
    normal = copy.deepcopy(row) if isinstance(row, dict) else {}
    reasons = []
    day = _date(normal.get("Date"))
    frequency = "daily" if day and day >= DAILY_START else "weekly" if day else "unknown"
    required = {"Date", "Code", "IssType", *VOLUME_FIELDS}
    if frequency == "daily":
        required.update(("PubDate", *VALUE_FIELDS))
    missing = sorted(required - normal.keys())
    if missing:
        reasons.append("missing_required_fields:" + ",".join(missing))
    if day is None:
        reasons.append("invalid_date")
    code = _code(normal.get("Code"))
    if code is None:
        reasons.append("invalid_code")
    elif expected_code is not None and code != _code(expected_code):
        reasons.append("code_mismatch")
    issue = str(normal.get("IssType", ""))
    if issue not in {"1", "2", "3"}:
        reasons.append("issue_type_unknown")
    for key in VOLUME_FIELDS:
        value = _number(normal.get(key))
        if value is None or value < 0 or not value.is_integer():
            reasons.append("invalid_value:" + key)
    if frequency == "daily":
        pub = _date(normal.get("PubDate"))
        if pub is None:
            reasons.append("invalid_publication_date")
        elif day is not None and pub <= day:
            reasons.append("publication_not_after_application")
        for key in VALUE_FIELDS:
            value = _number(normal.get(key))
            if value is None or value < 0:
                reasons.append("invalid_value:" + key)
    elif frequency == "weekly":
        for key in ("PubDate", *VALUE_FIELDS):
            if normal.get(key) is not None:
                reasons.append("weekly_nonnull_daily_field:" + key)
            normal.setdefault(key, None)
    state = "unsupported_schema" if missing else "invalid" if reasons else "valid"
    return {"normalized_row": normal, "frequency": frequency,
            "validation_state": state, "reasons": reasons}


def adapt_margin_bundle(bundle: dict, expected_code: str | None = None,
                        previous_bundle: dict | None = None) -> dict:
    """Preserve the response and attach per-content observation evidence.

    Historical versions are audit evidence only: they never replace a failed or
    empty current response. An old raw bundle's saved collected_at is acceptable
    evidence; it is not claimed to be the true exchange publication time.
    """
    result = copy.deepcopy(bundle)
    history: dict[str, dict] = {}

    def remember(version: dict) -> None:
        if not isinstance(version, dict) or "row" not in version:
            return
        raw = copy.deepcopy(version["row"])
        digest = _hash(raw)
        old = history.get(digest)
        times = [version.get("first_observed_at"), *version.get("observed_at_times", [])]
        if old:
            times += old.get("observed_at_times", [])
        times = sorted({t.isoformat() for value in times if (t := _timestamp(value)) is not None})
        history[digest] = {"row": raw, "row_hash": digest,
                           "first_observed_at": times[0] if times else None,
                           "observed_at_times": times,
                           **_validate(raw, expected_code)}

    for source in (previous_bundle, bundle):
        if not isinstance(source, dict):
            continue
        for version in source.get("row_version_history", []) + source.get("row_versions", []):
            remember(version)
        if source.get("status") == "complete" and isinstance(source.get("rows"), list):
            for raw in source["rows"]:
                remember({"row": raw, "first_observed_at": source.get("collected_at")})

    rows = bundle.get("rows", [])
    versions = []
    seen = set()
    if isinstance(rows, list):
        for row in rows:
            digest = _hash(row)
            if digest in seen:
                continue
            seen.add(digest)
            known = history.get(digest)
            versions.append(copy.deepcopy(known) if known else {
                "row": copy.deepcopy(row), "row_hash": digest, "first_observed_at": None, "observed_at_times": [],
                **_validate(row, expected_code)})
    groups = defaultdict(list)
    for version in versions:
        row = version["normalized_row"]
        groups[(_code(row.get("Code")), row.get("Date"))].append(version)
    for group in groups.values():
        if len(group) > 1:
            for version in group:
                version["validation_state"] = "conflicting_rows"
                version["reasons"] = sorted(set(version["reasons"] + ["conflicting_rows"]))
                # Store the conflict evidence so re-adapting or replaying the
                # bundle cannot turn one member of this response into truth.
                history[version["row_hash"]] = copy.deepcopy(version)
    states = {v["validation_state"] for v in versions}
    if not isinstance(rows, list):
        status = "unsupported_schema"
    elif bundle.get("status") != "complete":
        status = bundle.get("status", "not_collected")
    elif not rows:
        status = "no_rows"
    elif "conflicting_rows" in states:
        status = "conflicting_rows"
    elif "unsupported_schema" in states:
        status = "unsupported_schema"
    elif "invalid" in states:
        status = "invalid"
    else:
        status = "valid"
    result.update(adapter_version=ADAPTER_VERSION, adapter_status=status,
                  adapter_reasons=sorted({r for v in versions for r in v["reasons"]}),
                  row_versions=versions,
                  row_version_history=sorted(history.values(), key=lambda v: (str(v["first_observed_at"]), v["row_hash"])))
    return result


def _calendar(rows: list[dict] | None) -> tuple[dict[dt.date, str], str | None]:
    days = {}
    for row in rows or []:
        day, kind = _date(row.get("Date")), str(row.get("HolDiv", ""))
        if day is None or kind not in {"0", "1", "2", "3"}:
            return {}, "calendar_invalid"
        if day in days and days[day] != kind:
            return {}, "calendar_conflict"
        days[day] = kind
    return days, None if days else "calendar_unverified"


def _sessions(days: dict[dt.date, str], begin: dt.date, end: dt.date) -> list[dt.date] | None:
    if begin > end:
        return None
    result = []
    day = begin
    while day <= end:
        if day not in days:
            return None
        if days[day] in {"1", "2"}:
            result.append(day)
        day += dt.timedelta(days=1)
    return result


def _availability(version: dict, cutoff: dt.datetime, days: dict[dt.date, str]) -> str | None:
    row = version["normalized_row"]
    observed = _timestamp(version.get("first_observed_at"))
    if observed is None:
        return "observation_time_unknown"
    if observed > cutoff:
        return "not_observed_by_cutoff"
    day = _date(row.get("Date"))
    if day is None:
        return "invalid_date"
    if day > cutoff.date():
        return "application_after_cutoff"
    if days.get(day) not in {"1", "2"}:
        return "application_calendar_unverified" if day not in days else "application_not_session"
    if version["frequency"] == "daily":
        pub = _date(row.get("PubDate"))
        if pub is None:
            return "invalid_publication_date"
        if pub > cutoff.date():
            return "not_published_by_cutoff"
        if pub <= day:
            return "publication_not_after_application"
        if pub > observed.date():
            return "observed_before_publication_date"
    return None


def _price_index(rows: list[dict] | None, code: str | None,
                 price_asof: dt.date | None) -> dict[dt.date, dict | None]:
    prices = {}
    for row in rows or []:
        day = _date(row.get("Date"))
        if day is None or price_asof is None or day > price_asof:
            continue
        if code is not None and _code(row.get("Code")) != code:
            continue
        if day in prices and prices[day] != row:
            prices[day] = None
        else:
            prices[day] = row
    return prices


def _corporate_action_reason(sessions: list[dt.date], prices: dict, *,
                             include_start: bool = False) -> str | None:
    # The baseline price must exist as evidence that the cache covers this
    # window, although only events in (baseline, end] change comparability.
    if not sessions or any(prices.get(day) is None for day in sessions):
        return "corporate_action_unverified"
    for day in sessions if include_start else sessions[1:]:
        row = prices[day]
        factor = _number(row.get("AdjFactor"))
        if "ExRT" not in row or factor is None or factor <= 0:
            return "corporate_action_unverified"
        kind = str(row.get("ExRT") if row.get("ExRT") is not None else "").strip()
        if kind in {"1", "2", "3"}:
            return "corporate_action_in_window"
        if factor != 1 or kind not in {"", "0"}:
            return "corporate_action_basis_unknown"
    return None


def margin_diagnostic(bundle: dict, cutoff: dt.datetime, *, expected_code: str | None = None,
                      price_asof_date: dt.date | None = None,
                      calendar_rows: list[dict] | None = None,
                      price_rows: list[dict] | None = None) -> dict:
    """Compute diagnostics from archived rows, never scoring or fetching data."""
    cutoff = _timestamp(cutoff)
    if cutoff is None:
        raise ValueError("margin_cutoff_requires_timezone")
    price_asof = _date(price_asof_date)
    if price_asof is not None and price_asof > cutoff.date():
        raise ValueError("price_after_information_cutoff")
    adapted = adapt_margin_bundle(bundle, expected_code=expected_code)
    states, reasons = {}, {}
    out = {"margin_ratio": None, "margin_ratio_state": "unknown",
           "margin_reference_date": None, "margin_pub_date": None,
           "margin_observed_at": bundle.get("collected_at"),
           "margin_first_observed_at": None, "margin_available_at": None,
           "margin_status": bundle.get("status", "not_collected"),
           "margin_adapter_version": ADAPTER_VERSION,
           "margin_validation_state": adapted["adapter_status"],
           "margin_diagnostic_status": "unknown", "margin_target_state": "unknown",
           "margin_frequency": None, "margin_issue_type": None,
           "margin_after_price_asof": None,
           "margin_price_asof_date": price_asof.isoformat() if price_asof else None,
           "margin_information_cutoff": cutoff.isoformat(),
           "margin_availability_basis": None, "margin_row_hash": None,
           "margin_pit_quality": "archived_observation_required",
           "margin_amount_unit": "unverified",
           "margin_amount_note": "API原値・単位未確認",
           "margin_metric_states": states, "margin_metric_reasons": reasons,
           "margin_diagnostic_reasons": [], "margin_current_row_count": len(adapted["row_versions"]),
           "margin_score_connected": False,
           "margin_bridge_start_date": None, "margin_bridge_end_date": None,
           "margin_bridge_business_days": None, "margin_bridge_calendar_days": None}
    metrics = [*FIELD_NAMES.values(), "margin_ratio", "margin_general_short_pct", "margin_value_ratio",
               "margin_short_to_volume_20bd", "margin_long_to_volume_20bd",
               "margin_application_age_bd", "margin_publication_age_days",
               "margin_bridge_long_change_pct", "margin_bridge_short_change_pct"]
    metrics += [f"margin_{side}_change_{n}bd_pct" for side in ("long", "short") for n in (1, 5, 20)]
    metrics += [f"margin_{side}_change_{n}bd_vol" for side in ("long", "short") for n in (1, 5, 20)]
    for key in metrics:
        out[key] = None
        states[key], reasons[key] = "unavailable", "no_usable_row"

    def set_metric(key, value, reason=None):
        if isinstance(value, float) and not math.isfinite(value):
            value, reason = None, "nonfinite_calculation"
        out[key] = value
        states[key] = "available" if value is not None else "unavailable"
        reasons[key] = reason

    def stop(reason, detail=None):
        out["margin_diagnostic_status"] = reason
        out["margin_diagnostic_reasons"] = detail or [reason]
        for key in metrics:
            if out[key] is None:
                reasons[key] = reason
        return out

    if bundle.get("status") != "complete":
        return stop(bundle.get("status", "not_collected"))
    if not isinstance(bundle.get("rows", []), list):
        return stop("unsupported_schema")
    if not adapted["row_versions"]:
        return stop("no_rows")
    days, calendar_problem = _calendar(calendar_rows)
    # No fallback to an old valid Date when a malformed current Date cannot be
    # ordered safely.
    if any(_date(v["normalized_row"].get("Date")) is None for v in adapted["row_versions"]):
        return stop("invalid", ["invalid_date"])

    current_dates = {_date(v["normalized_row"].get("Date")) for v in adapted["row_versions"]}
    versions_by_date = defaultdict(list)
    for version in adapted["row_version_history"] + adapted["row_versions"]:
        day = _date(version["normalized_row"].get("Date"))
        if day not in current_dates:
            continue
        if expected_code is not None and _code(version["normalized_row"].get("Code")) != _code(expected_code):
            # Keep current wrong-code responses visible as invalid, rather than
            # letting a matching historical row conceal a server contract error.
            if version in adapted["row_versions"] and not any(v["row_hash"] == version["row_hash"] for v in versions_by_date[day]):
                versions_by_date[day].append(version)
            continue
        if not any(v["row_hash"] == version["row_hash"] for v in versions_by_date[day]):
            versions_by_date[day].append(version)

    selected = {}
    for day, versions in versions_by_date.items():
        observed = []
        for item in versions:
            seen = [_timestamp(t) for t in item.get("observed_at_times", [item.get("first_observed_at")])]
            seen = [t for t in seen if t is not None and t <= cutoff]
            if seen:
                observed.append((max(seen), item))
        if observed:
            newest = max(t for t, _ in observed)
            latest = [v for t, v in observed if t == newest]
            chosen = copy.deepcopy(latest[0])
            if len(latest) > 1:
                chosen["validation_state"] = "conflicting_rows"
                chosen["reasons"] = sorted(set(chosen["reasons"] + ["conflicting_rows"]))
            selected[day] = chosen
    candidates = {d: v for d, v in selected.items() if d <= cutoff.date()}
    # A newer observed-but-invalid/unpublished row is reported, not silently
    # replaced with an earlier date. Future unobserved versions are excluded.
    if not candidates:
        problem = "observation_time_unknown" if not any(v.get("first_observed_at") for v in adapted["row_versions"]) else "not_available_by_cutoff"
        return stop(problem)
    day = max(candidates)
    version = candidates[day]
    row = version["normalized_row"]
    observed = _timestamp(version["first_observed_at"])
    pub = _date(row.get("PubDate"))
    out.update(margin_reference_date=day.isoformat(), margin_pub_date=pub.isoformat() if pub else None,
               margin_frequency=version["frequency"], margin_issue_type=row.get("IssType"),
               margin_first_observed_at=observed.isoformat(), margin_available_at=observed.isoformat(),
               margin_availability_basis="archived_observation", margin_row_hash=version["row_hash"],
               margin_pit_quality="archived_content_version_observed",
               margin_validation_state=version["validation_state"],
               margin_after_price_asof=(day > price_asof) if price_asof else None)
    issue = str(row.get("IssType", ""))
    out["margin_target_state"] = "applicable" if issue in {"1", "2"} else "not_applicable" if issue == "3" else "issue_type_unknown"
    if version["validation_state"] != "valid":
        return stop(version["validation_state"], version["reasons"])
    unavailable = _availability(version, cutoff, days)
    if unavailable:
        return stop(unavailable)
    if issue not in {"1", "2"}:
        return stop(out["margin_target_state"])
    out["margin_diagnostic_status"] = "available"
    for key, name in FIELD_NAMES.items():
        set_metric(name, _number(row.get(key)), "not_provided_weekly" if version["frequency"] == "weekly" and key in VALUE_FIELDS else None)
    lng, srt = out["margin_long"], out["margin_short"]
    out["margin_ratio_state"] = "finite" if srt > 0 else "zero_short_balance" if lng > 0 else "both_zero"
    set_metric("margin_ratio", lng / srt if srt > 0 else None, None if srt > 0 else out["margin_ratio_state"])
    general_problem = "zero_denominator" if srt == 0 else "general_short_exceeds_total" if out["margin_short_neg"] > srt else None
    set_metric("margin_general_short_pct", 100 * out["margin_short_neg"] / srt if not general_problem else None,
               general_problem)
    lv, sv = out["margin_long_value"], out["margin_short_value"]
    set_metric("margin_value_ratio", lv / sv if lv is not None and sv is not None and sv > 0 else None,
               None if sv is not None and sv > 0 else "not_provided_weekly" if version["frequency"] == "weekly" else "zero_denominator")
    age_sessions = _sessions(days, day, cutoff.date())
    set_metric("margin_application_age_bd", len(age_sessions) - 1 if age_sessions is not None else None,
               None if age_sessions is not None else calendar_problem or "calendar_unverified")
    set_metric("margin_publication_age_days", (cutoff.date() - pub).days if pub else None,
               None if pub else "publication_date_unknown")
    code = _code(expected_code) if expected_code else _code(row.get("Code"))
    prices = _price_index(price_rows, code, price_asof)

    def window_reason(window, *, require_daily=True):
        if window is None:
            return "calendar_unverified"
        window_rows = [selected.get(d) for d in window]
        if any(v is None for v in window_rows):
            return "margin_history_insufficient"
        for item in window_rows:
            if item["validation_state"] != "valid":
                return item["validation_state"]
            r = item["normalized_row"]
            if _code(r.get("Code")) != code:
                return "code_mismatch"
            if str(r.get("IssType")) != issue:
                return "issue_type_changed"
            if require_daily and item["frequency"] != "daily":
                return "daily_history_insufficient"
            problem = _availability(item, cutoff, days)
            if problem:
                return problem
        return _corporate_action_reason(window, prices)

    known_sessions = sorted(d for d, kind in days.items() if kind in {"1", "2"} and d <= day)
    for n in (1, 5, 20):
        window = None
        problem = "daily_history_insufficient"
        if version["frequency"] == "daily" and len(known_sessions) >= n + 1:
            begin = known_sessions[-n - 1]
            window = _sessions(days, begin, day)
            problem = window_reason(window)
            if window is not None and len(window) != n + 1:
                problem = "business_day_gap"
        for side, field in (("long", "LongVol"), ("short", "ShrtVol")):
            key = f"margin_{side}_change_{n}bd_pct"
            baseline = _number(selected[window[0]]["normalized_row"].get(field)) if not problem else None
            reason = problem or ("zero_baseline" if baseline == 0 else None)
            set_metric(key, 100 * (_number(row[field]) / baseline - 1) if not reason else None, reason)
            # Share changes use exactly the same eligibility as percentage
            # changes, including the zero-baseline guard, for paired evidence.
            set_metric(f"margin_{side}_change_{n}bd_vol",
                       _number(row[field]) - baseline if not reason else None, reason)

    bridge_daily = selected.get(DAILY_START)
    weekly = [d for d, v in selected.items() if d < DAILY_START and v["frequency"] == "weekly"]
    bridge_problem = "bridge_history_insufficient"
    bridge_start = max(weekly) if weekly else None
    if bridge_daily is not None and bridge_start is not None:
        bridge_window = _sessions(days, bridge_start, DAILY_START)
        out.update(margin_bridge_start_date=bridge_start.isoformat(), margin_bridge_end_date=DAILY_START.isoformat(),
                   margin_bridge_business_days=len(bridge_window) - 1 if bridge_window else None,
                   margin_bridge_calendar_days=(DAILY_START - bridge_start).days)
        bridge_problem = None if bridge_window else "calendar_unverified"
        for item in (selected[bridge_start], bridge_daily):
            item_row = item["normalized_row"]
            if item["validation_state"] != "valid":
                bridge_problem = item["validation_state"]
            elif str(item_row.get("IssType")) != issue:
                bridge_problem = "issue_type_changed"
            elif _code(item_row.get("Code")) != code:
                bridge_problem = "code_mismatch"
            else:
                bridge_problem = bridge_problem or _availability(item, cutoff, days)
        bridge_problem = bridge_problem or _corporate_action_reason(bridge_window or [], prices)
    for side, field in (("long", "LongVol"), ("short", "ShrtVol")):
        baseline = _number(selected[bridge_start]["normalized_row"].get(field)) if not bridge_problem else None
        problem = bridge_problem or ("zero_baseline" if baseline == 0 else None)
        value = 100 * (_number(bridge_daily["normalized_row"][field]) / baseline - 1) if not problem else None
        set_metric(f"margin_bridge_{side}_change_pct", value, problem)

    turnover_problem = "volume_history_insufficient"
    turnover_value = None
    long_turnover_value = None
    if len(known_sessions) >= 20:
        window = _sessions(days, known_sessions[-20], day)
        turnover_problem = "calendar_unverified" if window is None else _corporate_action_reason(window, prices)
        if not turnover_problem:
            volumes = [_number(prices[d].get("Vo")) for d in window]
            if any(v is None or v < 0 for v in volumes):
                turnover_problem = "volume_history_insufficient"
            elif sum(volumes) == 0:
                turnover_problem = "zero_volume_denominator"
            else:
                turnover_value = srt / (sum(volumes) / len(volumes))
                long_turnover_value = lng / (sum(volumes) / len(volumes))
    set_metric("margin_short_to_volume_20bd", turnover_value, turnover_problem)
    set_metric("margin_long_to_volume_20bd", long_turnover_value, turnover_problem)
    return out
