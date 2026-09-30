"""Collection regressions through the maintained Client/collect contract."""
from collections import Counter
import datetime as dt
from urllib.parse import urlsplit

import JQuamtsScreeningBot as bot


class FakeResponse:
    def __init__(self, status, rows):
        self.status_code = status
        self.headers = {}
        self.rows = rows

    def json(self):
        return {"data": self.rows}


class FakeSession:
    def __init__(self, price_status=200):
        self.headers = {}
        self.calls = []
        self.price_status = price_status

    def get(self, url, **kwargs):
        endpoint = urlsplit(url).path.removeprefix("/v2/")
        self.calls.append(endpoint)
        rows = {
            "markets/calendar": [{"Date": "2026-09-11", "HolDiv": "1"}],
            "indices/bars/daily/topix": [{"Date": "2026-09-11", "C": 2700}],
            "equities/master": [{"Date": "2026-09-11", "Code": "99990",
                                 "CoName": "Synthetic ordinary stock", "Mkt": "0111",
                                 "ProdCat": "011", "S33": "5250"}],
            "equities/bars/daily": [],
            "fins/summary": [{"Code": "99990", "DiscDate": "2026-05-15"}],
            "fins/earnings-date": [],
        }
        assert endpoint in rows, "Optional diagnostics must not run with missing prices"
        status = self.price_status if endpoint == "equities/bars/daily" else 200
        return FakeResponse(status, rows[endpoint])


class FakeLimiter:
    def __init__(self):
        self.calls = Counter()

    def reserve(self, endpoint):
        self.calls[endpoint] += 1


def make_client(tmp_path, monkeypatch, price_status=200, session=None):
    monkeypatch.setattr(bot, "now_jst", lambda: dt.datetime(2026, 9, 30, 19, tzinfo=bot.JST))
    session = session if session is not None else FakeSession(price_status)
    sleeps = []
    client = bot.Client("fake-test-key", bot.Config(), bot.Store(tmp_path / "cache"),
                        session=session, limiter=FakeLimiter(), sleep=sleeps.append)
    return client, session, sleeps


def test_empty_http200_prices_are_not_a_completed_symbol(tmp_path, monkeypatch):
    client, session, _ = make_client(tmp_path, monkeypatch)
    snapshot = bot.collect(client, budget=1, asof=dt.date(2026, 9, 11))

    item = snapshot["symbols"]["9999"]
    assert item["complete"] is False
    assert item["failure_reason"] == "prices_empty_http200"
    assert snapshot["completed_codes"] == []
    assert snapshot["retryable_failed_codes"] == ["9999"]
    assert snapshot["status"] == "partial_universe"
    assert client.store.get(item["prices"])["rows"] == []
    assert client.store.get(item["prices"])["status"] == "complete"
    assert "markets/margin-interest" not in session.calls
    assert client.store.get(item["margin"])["status"] == "skipped_required_data_incomplete"
    assert client.store.get(item["short"])["status"] == "skipped_required_data_incomplete"


def test_empty_prices_exhaust_bounded_retries_without_becoming_success(tmp_path, monkeypatch):
    client, session, _ = make_client(tmp_path, monkeypatch)
    first = bot.collect(client, budget=1, asof=dt.date(2026, 9, 11), max_retries=2)
    assert first["pending_codes"] == ["9999"]

    second = bot.collect(client, budget=1, asof=dt.date(2026, 9, 11), max_retries=2)
    assert second["retry_counts"]["9999"] == 2
    assert second["terminal_failed_codes"] == ["9999"]
    assert second["pending_codes"] == []
    assert second["symbols"]["9999"]["complete"] is False
    assert second["terminal_failure_details"]["9999"]["reason"] == "prices_empty_http200"
    assert "markets/short-sale-report" not in session.calls


def test_transient_price_failure_remains_pending_after_http_retries(tmp_path, monkeypatch):
    client, session, sleeps = make_client(tmp_path, monkeypatch, price_status=503)
    snapshot = bot.collect(client, budget=1, asof=dt.date(2026, 9, 11))

    assert client.limiter.calls["equities/bars/daily"] == 5
    assert len(sleeps) == 4
    assert "fins/summary" not in session.calls
    assert snapshot["symbols"]["9999"]["complete"] is False
    assert "HTTP 503 retry_exhausted" in snapshot["symbols"]["9999"]["failure_reason"]
    assert snapshot["pending_codes"] == ["9999"]
    assert snapshot["terminal_failed_codes"] == []


def test_normal_collect_after_migration_fetches_margin_but_keeps_frozen_price_date(tmp_path, monkeypatch):
    weekly_row = {"Code": "99990", "Date": "2026-09-11", "IssType": "2",
                  "LongVol": 1000, "ShrtVol": 100, "LongNegVol": 400,
                  "LongStdVol": 600, "ShrtNegVol": 30, "ShrtStdVol": 70}

    class SuccessfulSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.queries = []

        def get(self, url, **kwargs):
            endpoint = urlsplit(url).path.removeprefix("/v2/")
            self.queries.append((endpoint, dict(kwargs["params"])))
            rows = {
                # A newer available price date must not advance a normal continuation.
                "markets/calendar": [{"Date": "2026-09-11", "HolDiv": "1"},
                                     {"Date": "2026-09-30", "HolDiv": "1"}],
                "indices/bars/daily/topix": [{"Date": "2026-09-11", "C": 2700},
                                             {"Date": "2026-09-30", "C": 2800}],
                "equities/bars/daily": [{"Code": "99990", "Date": "2026-09-11", "C": 1000}],
                "markets/margin-interest": [weekly_row],
                "markets/short-sale-report": [],
            }
            if endpoint in rows:
                self.calls.append(endpoint)
                return FakeResponse(200, rows[endpoint])
            return super().get(url, **kwargs)

    session = SuccessfulSession()
    client, _, _ = make_client(tmp_path, monkeypatch, session=session)
    client.store.save_snapshot({"schema": bot.CACHE_SCHEMA, "snapshot_id": "frozen_source",
                               "status": "partial_universe", "price_asof_date": "2026-09-11",
                               "information_cutoff": "2026-09-11T19:00:00+09:00",
                               "globals": {}, "symbols": {}}, promote=True)

    first = bot.collect(client, budget=1)
    assert first["price_asof_date"] == "2026-09-11"
    assert first["collection_request"]["asof"] == "2026-09-11"
    assert first["completed_codes"] == ["9999"]
    assert first["status"] == "complete"
    requests = [params for endpoint, params in session.queries if endpoint == "markets/margin-interest"]
    assert requests == [{"code": "99990", "from": "2026-06-13", "to": "2026-09-11"}]
    bundle = client.store.get(first["symbols"]["9999"]["margin"])
    assert bundle["status"] == "complete"
    assert bundle["adapter_status"] == "valid"
    assert bundle["adapter_version"]
    assert bundle["rows"] == [weekly_row]
    assert bundle["row_versions"][0]["frequency"] == "weekly"
    assert bundle["row_versions"][0]["first_observed_at"] == "2026-09-30T19:00:00+09:00"

    session.calls.clear()
    session.queries.clear()
    second = bot.collect(client, budget=1)
    assert second["price_asof_date"] == "2026-09-11"
    assert second["attempted_codes"] == []
    assert second["symbols"]["9999"] == first["symbols"]["9999"]
    assert set(session.calls) == {"markets/calendar", "indices/bars/daily/topix", "equities/master"}
    assert "markets/margin-interest" not in session.calls
