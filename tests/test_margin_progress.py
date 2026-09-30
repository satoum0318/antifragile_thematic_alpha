"""Heartbeat visibility during blocked I/O and cleanup without real requests."""
import datetime as dt
import logging
import threading
import time

import pytest

import JQuamtsScreeningBot as bot
import jq_margin_workflow as workflow
from test_margin_workflow import FakeClient, make_source, margin_row


class RecordedLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.entries = []

    def emit(self, record):
        with self.condition:
            self.entries.append((threading.current_thread(), record.getMessage()))
            self.condition.notify_all()

    def wait_for(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                match = next((entry for entry in self.entries if predicate(entry)), None)
                if match is not None:
                    return match
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError("Expected progress log did not arrive: " + repr(self.entries))
                self.condition.wait(remaining)


@pytest.fixture
def logs():
    recorder = RecordedLogs()
    logger = logging.Logger("margin-progress-test", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(recorder)
    return logger, recorder


@pytest.fixture(autouse=True)
def offline_clock_and_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Progress tests must not read credentials or use a network")

    monkeypatch.setattr(bot, "api_key_from_env_or_ini", forbidden)
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 9, 30, 11, 25, tzinfo=bot.JST))


@pytest.mark.parametrize("abort", [False, True], ids=["normal-exit", "exception-exit"])
def test_heartbeat_thread_stops_on_every_context_exit(logs, abort):
    logger, recorder = logs
    caller = threading.current_thread()
    observed = None

    try:
        with workflow.MarginProgress(logger, interval=0.02) as progress:
            progress.update(total=3, completed=1, pending=2, attempted=1,
                            failed=0, code="8697", phase="blocked-fetch")
            observed = recorder.wait_for(lambda entry: entry[0] is not caller)
            assert observed[0].daemon
            assert "8697" in observed[1]
            if abort:
                raise ValueError("intentional context exit")
    except ValueError as exc:
        assert abort and str(exc) == "intentional context exit"

    assert observed is not None
    assert not observed[0].is_alive()


@pytest.mark.parametrize(
    "outcome,expected_status,expected_snapshot_status",
    [
        ("success", "complete", "complete"),
        ("empty", "complete", "complete"),
        ("failure", "collection_failed", "partial"),
        ("interrupt", "collection_interrupted", "incomplete"),
    ],
)
def test_blocked_fetch_emits_heartbeat_and_keeps_screening_cache_unchanged(
        tmp_path, monkeypatch, logs, outcome, expected_status, expected_snapshot_status):
    store, _ = make_source(tmp_path)
    # The normal collector journal and errors must survive a margin-only run.
    (store.root / "collection_resume.json").write_bytes(b'{"normal":"pending"}')
    (store.root / "collection_failures.json").write_bytes(b'{"normal":"failure"}')
    before = {path: path.read_bytes() for path in store.root.rglob("*") if path.is_file()}
    entered, release = threading.Event(), threading.Event()
    actions = {
        "success": [margin_row()], "empty": [],
        "failure": bot.CollectionError("injected-fetch-failure"),
        "interrupt": KeyboardInterrupt(),
    }

    class BlockingClient(FakeClient):
        def fetch(self, endpoint, params):
            if endpoint == "markets/margin-interest":
                entered.set()
                if not release.wait(3):
                    raise AssertionError("Test did not release blocked fetch")
            return super().fetch(endpoint, params)

    client = BlockingClient(store, {"86970": [actions[outcome]]})
    logger, recorder = logs
    original_progress = workflow.MarginProgress
    monkeypatch.setattr(workflow, "MarginProgress", lambda log: original_progress(log, interval=0.02))
    monkeypatch.setattr(bot, "LOG", logger)
    result, errors = [], []
    heartbeat = None

    def refresh():
        try:
            result.append(workflow.margin_refresh(bot, client, max_retries=1))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=refresh, name="offline-margin-fetch-test")
    worker.start()
    try:
        assert entered.wait(3), "Workflow never reached the injected fetch"
        heartbeat = recorder.wait_for(
            lambda entry: entry[0].daemon and "8697" in entry[1])
        assert worker.is_alive(), "Heartbeat must appear before the request returns"
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive()

    assert heartbeat is not None and not heartbeat[0].is_alive()
    if outcome == "interrupt":
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
        assert result == []
    else:
        assert errors == []
        assert len(result) == 1
    snapshot = workflow.load_margin_snapshot(bot, store)
    assert snapshot["status"] == expected_snapshot_status
    assert snapshot["symbols"]["8697"]["last_attempt_status"] == expected_status
    assert all(path.read_bytes() == original for path, original in before.items())
    records, _ = workflow.margin_analyze(bot, store)
    if outcome == "success":
        assert records[0]["margin_long"] == 299700
        assert records[0]["margin_short"] == 415100
    else:
        assert records[0]["margin_long"] is None
        assert records[0]["margin_short"] is None
    if outcome == "empty":
        assert records[0]["margin_diagnostic_status"] == "no_rows"
