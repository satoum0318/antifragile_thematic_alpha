import copy
import datetime as dt
import json

import pytest

import JQuamtsScreeningBot as bot
import jq_margin_workflow as workflow


def calendar_rows(start=dt.date(2026, 7, 1), end=dt.date(2026, 10, 31)):
    holidays = {dt.date(2026, 9, day) for day in (21, 22, 23)} | {dt.date(2026, 10, 12)}
    return [{"Date": str(start+dt.timedelta(days=n)),
             "HolDiv": "1" if (start+dt.timedelta(days=n)).weekday() < 5 and start+dt.timedelta(days=n) not in holidays else "0"}
            for n in range((end-start).days+1)]


def margin_row(code="86970", day="2026-09-28", pub="2026-09-29", short=415100):
    return {"Code": code, "Date": day, "PubDate": pub, "IssType": "2",
            "LongVol": 299700, "ShrtVol": short,
            "LongNegVol": 99700, "ShrtNegVol": short-200000,
            "LongStdVol": 200000, "ShrtStdVol": 200000,
            "LongVal": 1000.0, "ShrtVal": 1500.0,
            "LongNegVal": 300.0, "ShrtNegVal": 500.0,
            "LongStdVal": 700.0, "ShrtStdVal": 1000.0}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real network/credentials are forbidden in offline workflow tests")
    monkeypatch.setattr(bot.requests.Session, "request", forbidden)
    monkeypatch.setattr(bot, "api_key_from_env_or_ini", forbidden)
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 9, 30, 11, 25, tzinfo=bot.JST))


def make_source(tmp_path, codes=("86970",), calendar=None):
    store = bot.Store(tmp_path / "cache")
    master = [{"Date": "2026-09-11", "Code": code, "CoName": "Test "+code,
               "S33": "5250", "Mkt": "0111", "ProdCat": "011"} for code in codes]
    def put(rows):
        return store.put({"status": "complete", "rows": rows,
                          "collected_at": "2026-09-11T19:00:00+09:00"})
    source = {"schema": bot.CACHE_SCHEMA, "snapshot_id": "source_1", "status": "complete",
              "price_asof_date": "2026-09-11", "information_cutoff": "2026-09-11T19:00:00+09:00",
              "globals": {"master": put(master), "calendar": put(calendar if calendar is not None else calendar_rows())},
              "symbols": {code[:4]: {"complete": True, "prices": put([]), "summary": put([])} for code in codes}}
    store.save_snapshot(source, promote=True)
    return store, source


class FakeClient:
    def __init__(self, store, actions=None):
        self.store = store
        self.cfg = bot.Config()
        self.actions = actions or {}
        self.calls = []

    def fetch(self, endpoint, params):
        self.calls.append((endpoint, copy.deepcopy(params)))
        if endpoint == "markets/calendar":
            rows = calendar_rows(bot.date(params["from"]), bot.date(params["to"]))
        else:
            queue = self.actions.get(params["code"])
            value = queue.pop(0) if queue else [margin_row(params["code"])]
            if isinstance(value, BaseException):
                raise value
            rows = value
        return {"status": "complete", "endpoint": endpoint, "params": copy.deepcopy(params),
                "rows": rows, "raw_pages": [{"data": rows}],
                "started_at": bot.now_jst().isoformat(), "collected_at": bot.now_jst().isoformat()}


def test_complete_symbols_refresh_without_touching_source_or_price(tmp_path):
    store, source = make_source(tmp_path)
    source_path = store.root / "snapshots" / "source_1.json"
    before_source = source_path.read_bytes()
    before_pointer = (store.root / "latest_snapshot.json").read_bytes()
    client = FakeClient(store)

    snapshot = workflow.margin_refresh(bot, client)

    assert snapshot["status"] == "complete"
    assert snapshot["pending_codes"] == []
    assert client.calls == [("markets/margin-interest", {"code": "86970", "from": "2026-07-03", "to": "2026-09-30"})]
    assert snapshot["source_price_asof_date"] == "2026-09-11"
    assert source_path.read_bytes() == before_source
    assert (store.root / "latest_snapshot.json").read_bytes() == before_pointer
    assert store.snapshot() == source
    rows, meta = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_short"] == 415100
    assert rows[0]["margin_after_price_asof"] is True
    assert rows[0]["margin_short_change_1bd_pct"] is None
    assert meta["source_information_cutoff"] == "2026-09-11T19:00:00+09:00"
    assert meta["information_cutoff"] == "2026-09-30T11:25:00+09:00"
    assert meta["screening_recomputed"] is False
    assert "recommendation_score" not in rows[0]


def test_checkpoint_resume_keeps_window_across_midnight(tmp_path, monkeypatch):
    store, _ = make_source(tmp_path, ("86970", "72030"))
    client = FakeClient(store)
    first = workflow.margin_refresh(bot, client, budget=1)
    assert first["pending_codes"] == ["8697"]
    assert first["unattempted_count"] == 1
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 10, 1, 9, tzinfo=bot.JST))
    second = workflow.margin_refresh(bot, client, budget=1)
    assert second["status"] == "complete"
    assert second["window_to"] == "2026-09-30"
    assert second["source_snapshot_id"] == "source_1"
    assert len(client.calls) == 2
    assert all(params["to"] == "2026-09-30" for _, params in client.calls)
    assert second["symbols"]["7203"]["margin_ref"] == first["symbols"]["7203"]["margin_ref"]
    assert not (store.root / "margin_collection_resume.json").exists()


def test_interrupt_saves_completed_symbols_and_resumes_pending_only(tmp_path):
    store, _ = make_source(tmp_path, ("72030", "86970"))
    client = FakeClient(store, {"86970": [KeyboardInterrupt(), [margin_row()]]})
    with pytest.raises(KeyboardInterrupt):
        workflow.margin_refresh(bot, client, budget=2)
    interrupted = workflow.load_margin_snapshot(bot, store)
    assert interrupted["status"] == "incomplete"
    assert interrupted["symbols"]["7203"]["last_attempt_status"] == "complete"
    assert interrupted["symbols"]["8697"]["last_attempt_status"] == "collection_interrupted"
    assert interrupted["pending_codes"] == ["8697"]
    assert interrupted["retry_counts"] == {}
    finished = workflow.margin_refresh(bot, client, budget=2)
    assert finished["status"] == "complete"
    assert [params["code"] for _, params in client.calls] == ["72030", "86970", "86970"]


def test_failures_are_bounded_and_old_success_is_separate(tmp_path):
    store, _ = make_source(tmp_path)
    workflow.margin_refresh(bot, FakeClient(store), max_retries=2)
    client = FakeClient(store, {"86970": [bot.CollectionError("network_failure"), bot.CollectionError("network_failure")]})
    first = workflow.margin_refresh(bot, client, max_retries=2)
    assert first["pending_codes"] == ["8697"]
    rows, _ = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_short"] is None
    assert rows[0]["margin_status"] == "collection_failed"
    assert rows[0]["margin_previous_observation"]["margin_short"] == 415100
    terminal = workflow.margin_refresh(bot, client, max_retries=2)
    assert terminal["pending_codes"] == []
    assert terminal["terminal_failed_codes"] == ["8697"]
    assert terminal["status"] == "partial"
    assert not (store.root / "margin_collection_resume.json").exists()


def test_successful_empty_result_does_not_restore_old_balance(tmp_path):
    store, _ = make_source(tmp_path)
    workflow.margin_refresh(bot, FakeClient(store))
    snapshot = workflow.margin_refresh(bot, FakeClient(store, {"86970": [[]]}))
    assert snapshot["status"] == "complete"
    assert snapshot["symbols"]["8697"]["last_attempt_status"] == "complete"
    rows, _ = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_diagnostic_status"] == "no_rows"
    assert rows[0]["margin_short"] is None
    assert rows[0]["margin_previous_observation"]["margin_short"] == 415100
    blob = store.get(snapshot["symbols"]["8697"]["margin_ref"])
    assert blob["rows"] == []
    assert blob["row_version_history"]


def test_missing_calendar_is_fetched_only_for_diagnostics(tmp_path):
    store, source = make_source(tmp_path, calendar=calendar_rows(end=dt.date(2026, 9, 11)))
    original_ref = source["globals"]["calendar"]
    client = FakeClient(store)
    snapshot = workflow.margin_refresh(bot, client)
    assert client.calls[0] == ("markets/calendar", {"from": "2026-07-03", "to": "2026-09-30"})
    assert snapshot["calendar_status"] == "complete"
    assert snapshot["calendar_ref"] != original_ref
    assert store.snapshot()["globals"]["calendar"] == original_ref


def test_resume_rejects_changed_request_without_discarding_journal(tmp_path):
    store, _ = make_source(tmp_path, ("72030", "86970"))
    client = FakeClient(store)
    workflow.margin_refresh(bot, client, budget=1)
    journal = store.root / "margin_collection_resume.json"
    before = journal.read_bytes()
    with pytest.raises(bot.DataError, match="margin_resume_request_mismatch"):
        workflow.margin_refresh(bot, client, only_codes=["7203"])
    assert journal.read_bytes() == before
    assert len(client.calls) == 1


def test_offline_analysis_and_immutable_artifacts_leave_screening_outputs_alone(tmp_path):
    store, _ = make_source(tmp_path)
    workflow.margin_refresh(bot, FakeClient(store))
    rows, meta = workflow.margin_analyze(bot, store)
    again = workflow.margin_analyze(bot, store)
    assert again == (rows, meta)
    with pytest.raises(bot.DataError, match="margin_cutoff_after_snapshot_verification"):
        workflow.margin_analyze(bot, store, cutoff="2026-10-01T00:00:00+09:00")
    output = tmp_path / "output"
    output.mkdir()
    screening_pointer = output / "latest_run.json"
    screening_pointer.write_text('{"run_id":"original_screening_run"}', encoding="utf-8")
    before = screening_pointer.read_bytes()
    path = workflow.write_margin_run(bot, rows, meta, output)
    assert path.parent.name == "margin_history"
    assert {p.name for p in path.iterdir()} == {"margin_diagnostics.json", "margin_diagnostics.csv", "report_margin.md", "manifest.json"}
    assert json.loads((path / "margin_diagnostics.json").read_text("utf-8")) == bot.clean(rows)
    manifest = json.loads((path / "manifest.json").read_text("utf-8"))
    assert manifest["result_hash"] == bot.digest(rows)
    assert manifest["screening_recomputed"] is False
    assert screening_pointer.read_bytes() == before
    previous = {p.name: p.read_bytes() for p in path.iterdir()}
    another = workflow.write_margin_run(bot, rows, meta, output)
    assert another != path
    assert {p.name: p.read_bytes() for p in path.iterdir()} == previous


def test_real_client_pagination_contract_with_injected_fake_session(tmp_path):
    store, _ = make_source(tmp_path)
    first = margin_row(day="2026-09-25", pub="2026-09-28", short=417000)
    second = margin_row()
    class Response:
        status_code = 200
        headers = {}
        def __init__(self, body):
            self.body = body
        def json(self):
            return self.body
    class Session:
        def __init__(self):
            self.headers = {}
            self.calls = []
        def get(self, url, params, **kwargs):
            self.calls.append(copy.deepcopy(params))
            return Response({"data": [second]} if params.get("pagination_key") else {"data": [first], "pagination_key": "page2"})
    class Limiter:
        def reserve(self, endpoint):
            pass
    session = Session()
    client = bot.Client("offline-test-only", bot.Config(), store, session=session, limiter=Limiter(), sleep=lambda _: None)
    snapshot = workflow.margin_refresh(bot, client)
    raw = store.get(snapshot["symbols"]["8697"]["raw_ref"])
    assert raw["rows"] == [first, second]
    assert len(raw["raw_pages"]) == 2
    assert len(session.calls) == 2
    assert session.calls[1]["pagination_key"] == "page2"
    assert all("published_date" not in params for params in session.calls)


def test_invalid_daily_schema_is_retried_boundedly(tmp_path):
    store, _ = make_source(tmp_path)
    invalid = margin_row()
    invalid.pop("PubDate")
    client = FakeClient(store, {"86970": [[invalid], [invalid]]})
    first = workflow.margin_refresh(bot, client, max_retries=2)
    assert first["symbols"]["8697"]["transport_status"] == "complete"
    assert first["symbols"]["8697"]["last_attempt_status"] == "unsupported_schema"
    assert first["pending_codes"] == ["8697"]
    assert first["unattempted_count"] == 0
    second = workflow.margin_refresh(bot, client, max_retries=2)
    assert second["terminal_failed_codes"] == ["8697"]
    rows, _ = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_diagnostic_status"] == "unsupported_schema"
    assert rows[0]["margin_short"] is None


def test_quota_stop_preserves_retry_budget_and_resume(tmp_path):
    store, _ = make_source(tmp_path)
    client = FakeClient(store, {"86970": [bot.CollectionError("daily_self_quota_reached"), [margin_row()]]})
    with pytest.raises(bot.CollectionError, match="daily_self_quota_reached"):
        workflow.margin_refresh(bot, client, max_retries=1)
    first = workflow.load_margin_snapshot(bot, store)
    assert first["pending_codes"] == ["8697"]
    assert first["retry_counts"] == {}
    assert first["symbols"]["8697"]["last_attempt_status"] == "quota_exhausted"
    assert workflow.margin_refresh(bot, client, max_retries=1)["status"] == "complete"


def test_calendar_permission_failure_does_not_skip_margin(tmp_path):
    store, _ = make_source(tmp_path, calendar=[])
    class Client(FakeClient):
        def fetch(self, endpoint, params):
            if endpoint == "markets/calendar":
                self.calls.append((endpoint, copy.deepcopy(params)))
                raise bot.AuthError("HTTP 403: markets/calendar")
            return super().fetch(endpoint, params)
    client = Client(store)
    snapshot = workflow.margin_refresh(bot, client)
    assert snapshot["calendar_status"] == "permission_unavailable"
    assert snapshot["symbols"]["8697"]["last_attempt_status"] == "complete"
    assert [endpoint for endpoint, _ in client.calls] == ["markets/calendar", "markets/margin-interest"]


def test_corrupt_optional_history_and_price_cache_do_not_abort_collection(tmp_path):
    store, source = make_source(tmp_path)
    source["snapshot_id"] = "source_corrupt_optional"
    source["symbols"]["8697"].update(margin="f"*64, prices="e"*64)
    store.save_snapshot(source, promote=True)
    snapshot = workflow.margin_refresh(bot, FakeClient(store))
    assert snapshot["status"] == "complete"
    rows, _ = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_short"] == 415100
    assert rows[0]["margin_price_cache_status"] == "cache_unavailable"
    assert rows[0]["margin_short_change_5bd_pct"] is None


def test_observation_history_survives_alternating_symbol_subsets(tmp_path, monkeypatch):
    store, _ = make_source(tmp_path, ("72030", "86970"))
    first = workflow.margin_refresh(bot, FakeClient(store), only_codes=["7203"])
    first_bundle = store.get(first["symbols"]["7203"]["margin_ref"])
    original_observed = first_bundle["row_versions"][0]["first_observed_at"]
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 10, 1, 9, tzinfo=bot.JST))
    second = workflow.margin_refresh(bot, FakeClient(store), only_codes=["8697"])
    assert set(second["observation_index"]) == {"7203", "8697"}
    assert set(second["symbols"]) == {"8697"}
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 10, 2, 9, tzinfo=bot.JST))
    third = workflow.margin_refresh(bot, FakeClient(store), only_codes=["7203"])
    third_bundle = store.get(third["symbols"]["7203"]["margin_ref"])
    assert third_bundle["row_versions"][0]["first_observed_at"] == original_observed


def test_failed_partial_raw_reference_remains_auditable(tmp_path):
    store, _ = make_source(tmp_path)
    partial_ref = store.put({"status": "incomplete", "rows": [margin_row()], "raw_pages": [{"data": [margin_row()]}]})
    error = bot.CollectionError("page2_failed")
    error.raw_cache_ref = partial_ref
    snapshot = workflow.margin_refresh(bot, FakeClient(store, {"86970": [error]}), max_retries=1)
    assert snapshot["symbols"]["8697"]["raw_ref"] == partial_ref
    rows, _ = workflow.margin_analyze(bot, store)
    assert rows[0]["margin_short"] is None
    assert rows[0]["margin_status"] == "collection_failed"
