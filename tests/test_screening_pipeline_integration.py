"""End-to-end temporary-store screening, with only market transport replaced.

The weekday-only fixture calendar is synthetic; these tests make no claim about
actual exchange sessions or historic investment performance. No live API is used.
"""
import copy
import datetime as dt
import hashlib
import json
from types import SimpleNamespace

import pytest

import JQuamtsScreeningBot as bot
import jq_screening_workflow as workflow
from test_universe_integration import current_snapshot, margin_rows, project_legacy


@pytest.mark.parametrize("empty_new_listing", [False, True], ids=["complete", "financial-empty-partial"])
def test_real_screening_pipeline_keeps_clocks_inputs_scope_and_publication_consistent(
        tmp_path, monkeypatch, empty_new_listing):
    store, source = current_snapshot(tmp_path)
    normal_codes = sorted(source["symbols"])
    fixed_now = dt.datetime(2026, 10, 27, 19, 5, tzinfo=bot.JST)
    monkeypatch.setattr(bot, "now_jst", lambda: fixed_now)
    monkeypatch.setenv("JQUANTS_API_KEY", "offline-fixture-not-a-real-key")
    cfg = bot.Config()
    args = SimpleNamespace(cache_dir=store.root, output_dir=tmp_path / "output",
                           api_ini=tmp_path / "must-not-be-read.ini", top=10, budget=2, max_retries=2)
    originals = {code: {kind: copy.deepcopy(store.get(item[kind])["rows"])
                        for kind in ("prices", "summary", "earnings")}
                 for code, item in source["symbols"].items()}
    master = copy.deepcopy(store.get(source["globals"]["master"])["rows"])
    master.append({"Date": "2026-10-27", "Code": "50010", "CoName": "Synthetic PRO stock",
                   "Mkt": "0105", "ProdCat": "011", "S33": "5250"})
    if empty_new_listing:
        master.append({"Date": "2026-10-27", "Code": "30010", "CoName": "Synthetic newly listed stock",
                       "Mkt": "0111", "ProdCat": "011", "S33": "5250"})
        new_prices = [{**row, "Code": "30010"} for row in originals[normal_codes[0]]["prices"][-5:]]
        originals["3001"] = {"prices": new_prices, "summary": [], "earnings": []}
    source["globals"]["master"] = store.put({"status": "complete", "rows": master,
                                               "collected_at": source["information_cutoff"]})
    old_full_at = "2026-09-01T19:00:00+09:00"
    for code, item in source["symbols"].items():
        # Keep legacy credit absent. The real refresh must provide the distinct
        # registration evidence without changing these old diagnostic inputs.
        item["margin"] = store.put({"status": "not_collected", "rows": [],
                                     "collected_at": source["information_cutoff"]})
        item.update(last_full_price_at=old_full_at, last_full_financial_at=old_full_at,
                    checked_price_date=source["price_asof_date"])
    source["snapshot_id"] = "pipeline_original"
    store.save_snapshot(source, promote=True)
    original_path = store.root / "snapshots" / "pipeline_original.json"
    original_bytes = original_path.read_bytes()
    original_refs = copy.deepcopy(source["symbols"])
    global_context = copy.deepcopy(source["globals"])
    calls, clients = [], []
    manual_resume = store.root / "margin_collection_resume.json"
    bot.atomic_json(manual_resume, {"manual_job": "must_remain_untouched"})
    manual_bytes = manual_resume.read_bytes()

    class OfflineClient:
        def __init__(self, key, config, cache):
            assert key == "offline-fixture-not-a-real-key"
            self.cfg, self.store, self.closed = config, cache, False
            clients.append(self)

        def fetch(self, endpoint, params):
            calls.append((endpoint, copy.deepcopy(params)))
            code = str(params.get("code", ""))[:4]
            if endpoint == "markets/margin-interest":
                assert code in originals
                rows = [] if code == "3001" else margin_rows(code)
            elif endpoint in {"equities/bars/daily", "fins/summary", "fins/earnings-date"}:
                kind = {"equities/bars/daily": "prices", "fins/summary": "summary",
                        "fins/earnings-date": "earnings"}[endpoint]
                if code:
                    assert code in originals, "PRO market stock must never receive a code-level request"
                    rows = originals[code][kind]
                    if kind == "prices":
                        rows = [row for row in rows if params["from"] <= row["Date"] <= params["to"]]
                else:
                    assert set(params) == {"date"}
                    field = "Date" if kind == "prices" else "DiscDate" if kind == "summary" else "PubDate"
                    rows = [row for data in originals.values() for row in data[kind]
                            if row.get(field) == params["date"]]
            else:
                pytest.fail("Unexpected endpoint in offline integration: " + endpoint)
            return {"status": "complete", "endpoint": endpoint, "params": copy.deepcopy(params),
                    "rows": copy.deepcopy(rows), "collected_at": fixed_now.isoformat()}

        def close(self):
            self.closed = True

    monkeypatch.setattr(bot, "Client", OfflineClient)
    monkeypatch.setattr(bot, "collect_globals", lambda client: (copy.deepcopy(global_context), fixed_now.date()))
    assert workflow.run_screening(bot, args, cfg) == (2 if empty_new_listing else 0)

    assert all(client.closed for client in clients)
    assert original_path.read_bytes() == original_bytes
    assert store.snapshot("pipeline_original")["symbols"] == original_refs
    assert manual_resume.read_bytes() == manual_bytes
    assert not (store.root / "screening_workflow_resume.json").exists()
    assert not (store.root / "screening_margin_resume.json").exists()
    assert not any(params.get("code") == "50010" for _, params in calls)
    assert sum(ep == "fins/summary" and params.get("code") == "30010" for ep, params in calls) == int(empty_new_listing)
    assert any(ep == "fins/summary" and "date" in params for ep, params in calls)
    assert any(ep == "equities/bars/daily" and "date" in params for ep, params in calls)

    final = store.snapshot()
    status = json.loads((store.root / "screening_workflow_status.json").read_text("utf-8"))
    pointer = json.loads((args.output_dir / "latest_run.json").read_text("utf-8"))
    path = args.output_dir / "history" / pointer["run_id"]
    manifest = json.loads((path / "run_manifest.json").read_text("utf-8"))
    rows = json.loads((path / "screening_results.json").read_text("utf-8"))
    assert status["run_id"] == pointer["run_id"] == manifest["run_id"]
    assert status["source_snapshot_id"] == manifest["snapshot_id"] == final["snapshot_id"]
    assert status["status"] == ("partial" if empty_new_listing else "complete")
    assert manifest["run_status"] == ("partial_universe" if empty_new_listing else "complete")
    assert final["screening_scope"]["included_codes"] == sorted(originals)
    assert final["screening_scope"]["excluded"][0]["code"] == "5001"
    assert manifest["registration_price_age_bd"] == 0
    assert manifest["price_asof_date"] == status["price_asof_date"] == "2026-10-27"
    assert manifest["information_cutoff"] == final["information_cutoff"] == fixed_now.isoformat()
    assert status["counts"]["target_count"] == len(normal_codes) + int(empty_new_listing)
    assert status["counts"]["complete_count"] == len(normal_codes)
    assert status["counts"]["unavailable_count"] == int(empty_new_listing)
    assert status["counts"]["financial_unavailable_count"] == int(empty_new_listing)
    assert status["counts"]["credit_completed_count"] == len(originals)
    assert status["counts"]["credit_empty_count"] == int(empty_new_listing)
    assert status["counts"]["scope_excluded_count"] == 1
    assert status["oldest_last_full_financial_at"] == fixed_now.isoformat()
    assert status["last_full_financial_at_unknown_count"] == int(empty_new_listing)
    assert manifest["screening_workflow"]["counts"]["credit_completed_count"] == len(originals)

    margin_id = status["margin_snapshot_id"]
    assert margin_id == manifest["registration_margin_snapshot_id"]
    margin = bot.jq_margin_workflow.load_margin_snapshot(bot, store, margin_id)
    margin_source = store.snapshot(margin["source_snapshot_id"])
    assert margin_source["snapshot_id"] != final["snapshot_id"]
    assert margin_source["price_asof_date"] == final["price_asof_date"]
    assert bot.timestamp(margin_source["information_cutoff"]) <= bot.timestamp(margin["information_cutoff"])
    assert bot.timestamp(margin["information_cutoff"]) <= bot.timestamp(final["information_cutoff"])
    assert manifest["registration_margin_source_snapshot_id"] == margin_source["snapshot_id"]
    assert margin["selected_codes"] == sorted(originals)
    assert margin["window_to"] == "2026-10-27"
    assert not margin["pending_codes"]
    for code in normal_codes:
        item = final["symbols"][code]
        assert item["last_full_price_at"] == item["last_full_financial_at"] == fixed_now.isoformat()
        assert item["financial_checked_through"] == item["earnings_checked_through"] == fixed_now.isoformat()
        assert item["margin"] == original_refs[code]["margin"]
    normal_rows = [row for row in rows if row["code"] in normal_codes]
    assert len(normal_rows) == len(normal_codes)
    assert all(row["registration_a_eligible"] and row["registration_b_eligible"] for row in normal_rows)
    assert all(row["eligible_new_entry"] is False for row in normal_rows)
    assert all(row["margin_diagnostic_status"] == "not_collected" for row in normal_rows)
    assert all(row["registration_margin_diagnostic"]["margin_diagnostic_status"] == "available" for row in normal_rows)
    assert all(row["registration_margin_diagnostic"]["margin_metric_states"]["margin_short_change_20bd_pct"] == "available"
               for row in normal_rows)
    assert all(row["registration_margin_snapshot_id"] == margin_id for row in normal_rows)
    assert all(row["registration_margin_diagnostic"]["margin_reference_date"] == "2026-10-26" for row in normal_rows)
    assert all(row["registration_margin_diagnostic"]["margin_pub_date"] == "2026-10-27" for row in normal_rows)
    if empty_new_listing:
        absent = next(row for row in rows if row["code"] == "3001")
        assert absent["registration_a_eligible"] is False and absent["registration_b_eligible"] is False
        assert final["symbols"]["3001"]["screening_availability"] == "financial_unavailable"
        assert final["symbols"]["3001"]["screening_next_retry_date"] == "2026-10-28"
    normal_credit_rows, _, _ = bot.score_snapshot(store, final, cfg)
    assert project_legacy(rows) == project_legacy(normal_credit_rows)
    assert not any(row["registration_b_eligible"] for row in normal_credit_rows)
    for name, checksum in manifest["outputs"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == checksum
    assert (path / "report_registration_top10.md").is_file()
