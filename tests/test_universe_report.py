"""The report projects frozen decisions without accepting watch rows as picks."""
import copy

import pytest

from jq_universe_report import groups, markdown_report, summarize


def candidate(code, a="eligible", b="excluded", a_score=70, b_score=None):
    return {
        "code": code, "name": "検証企業", "registration_a_eligible": a == "eligible",
        "registration_b_eligible": b == "eligible", "registration_a_status": a,
        "registration_b_status": b, "registration_a_score": a_score,
        "registration_b_score": b_score, "registration_a_reasons": [],
        "registration_b_reasons": [], "registration_a_warnings": [], "registration_b_warnings": [],
        "registration_growth_a_state": "confirmed", "registration_growth_b_state": "confirmed",
        "registration_price_state": "uptrend_continuation", "registration_supply_state": "unassessed",
        "registration_supply_reasons": ["supply_unassessed"], "summary_quality_score": 6,
        "summary_quality_available": 7, "sales_cagr": .12, "op_income_yoy_pct": 15,
        "forward_np_change": .2, "ps": 1.2, "per": 15, "eligible_new_entry": False,
        "recommendation_score": 64, "candidate_lane": "watch_fundamental_core",
    }


def metadata():
    return {"run_id": "r_example", "run_status": "complete", "price_asof_date": "2026-10-26",
            "information_cutoff": "2026-10-27T18:00:00+09:00", "registration_logic_version": "ab-rc1",
            "legacy_logic_version": "legacy-1", "registration_margin_input_mode": "explicit_snapshot",
            "registration_margin_snapshot_id": "ms_example", "registration_margin_source_snapshot_id": "s_example",
            "registration_price_age_bd": 1}


def test_groups_are_immutable_and_ties_are_stable_without_mixing_routes_or_holds():
    rows = [candidate("1200", b="eligible", b_score=80), candidate("1100", b="hold", b_score=99),
            candidate("1300", a="hold", a_score=100), candidate("1400", a="excluded", a_score=120)]
    before = copy.deepcopy(rows)
    first, shuffled = groups(rows, 1), groups(list(reversed(rows)), 1)
    assert rows == before
    assert set(first) == {"top_registration_a.csv", "top_registration_b.csv", "registration_candidates_a.csv",
                          "registration_candidates_b.csv", "registration_review.csv"}
    assert first == shuffled
    assert [row["code"] for row in first["top_registration_a.csv"]] == ["1100"]
    assert [row["code"] for row in first["registration_candidates_a.csv"]] == ["1100", "1200"]
    assert [row["code"] for row in first["top_registration_b.csv"]] == ["1200"]
    assert [row["code"] for row in first["registration_review.csv"]] == ["1300"]
    assert first["top_registration_a.csv"][0]["recommendation_score"] == 64


def test_eligibility_requires_the_true_boolean_and_candidates_are_not_padded():
    rows = [candidate("1000", a="hold"), candidate("2000", a="excluded")]
    rows[0]["registration_a_eligible"] = "True"
    rows[1]["registration_a_eligible"] = 1
    assert groups(rows, 10)["top_registration_a.csv"] == []
    assert groups([], 10)["top_registration_b.csv"] == []
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            groups(rows, invalid)


def test_summary_keeps_unknown_states_and_counts_overlapping_candidates_once():
    both = candidate("1000", b="eligible", b_score=80)
    held = candidate("2000", a="hold", b="hold")
    held["registration_a_reasons"] = ["growth_below_threshold:sales_cagr"]
    held["registration_a_warnings"] = ["supply_unassessed"]
    excluded = candidate("3000", a="excluded")
    counts = summarize([both, held, excluded, {"code": "4000"}])
    assert counts["record_count"] == 4
    assert counts["overlap_count"] == 1
    assert counts["review_count"] == 1
    assert counts["a"]["eligible_count"] == counts["b"]["eligible_count"] == 1
    assert counts["a"]["status_counts"] == {"eligible": 1, "excluded": 1, "hold": 1, "unknown": 1}
    assert counts["a"]["reason_counts"] == {"growth_below_threshold:sales_cagr": 1}
    assert counts["a"]["warning_counts"] == {"supply_unassessed": 1}
    assert counts["supply_state_counts"] == {"unassessed": 3, "unknown": 1}


def test_report_uses_selected_margin_evidence_and_distinguishes_zero_from_missing():
    record = candidate("1234")
    record.update(margin_long=999999, margin_short=888888,
                  registration_margin_diagnostic={
                      "margin_long": 0, "margin_short": None, "margin_ratio_state": "unknown",
                      "margin_diagnostic_status": "available", "margin_frequency": "daily",
                      "margin_reference_date": "2026-10-26", "margin_pub_date": "2026-10-27",
                      "margin_available_at": "2026-10-27T17:00:00+09:00", "margin_application_age_bd": 1,
                      "margin_long_change_5bd_vol": 0, "margin_long_change_5bd_pct": 0,
                      "margin_metric_reasons": {"margin_short": "custom_missing_balance"}})
    source, meta = [record], metadata()
    before, original_meta = copy.deepcopy(source), copy.deepcopy(meta)
    report = markdown_report(source, meta, 3)
    assert source == before and meta == original_meta
    assert "上位3件" in report and "上位10件" not in report
    assert "|1234|0株|未評価（custom_missing_balance）|" in report
    assert "買 0株（0.0%）" in report
    assert "999,999" not in report and "888,888" not in report
    assert "2026-10-27T17:00:00+09:00" in report
    assert "ms_example" in report and "s_example" in report
    assert "旧非適格 / 点 64.00" in report
    assert "未評価・欠測は安全を意味しません" in report


def test_empty_selected_diagnostic_does_not_resurrect_fallback_values():
    record = candidate("1234")
    record.update(margin_long=999999, margin_short=888888, registration_margin_diagnostic={})
    report = markdown_report([record], metadata())
    assert "|1234|未評価|未評価|" in report
    assert "999,999" not in report and "888,888" not in report
    record.pop("registration_margin_diagnostic")
    fallback = markdown_report([record], metadata())
    assert "|1234|999,999株|888,888株|" in fallback


def test_no_eligible_rows_explain_holds_and_preserve_unknown_reasons():
    record = candidate("1234", a="hold", b="hold")
    record["registration_a_reasons"] = ["growth_below_threshold:sales_cagr"]
    record["registration_b_reasons"] = ["untranslated_evidence_gap"]
    report = markdown_report([record], metadata())
    assert report.count("候補なし。") == 2
    assert "適格 0件 / 保留 1件" in report
    assert "成長条件未達：売上成長率 1件" in report
    assert "untranslated_evidence_gap 1件" in report
    assert "registration_review.csv" in report


def test_report_escapes_names_and_marks_credit_after_price_as_unusable():
    record = candidate("1234")
    record["name"] = "銘柄|<script>\n別行"
    record["registration_margin_diagnostic"] = {"margin_after_price_asof": True}
    report = markdown_report([record], metadata())
    assert "銘柄\\|&lt;script&gt; 別行" in report
    assert "<script>" not in report
    assert "価格基準日より後の残高・判定使用不可" in report
