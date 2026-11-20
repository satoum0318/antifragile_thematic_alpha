import datetime as dt
import json

import pandas as pd
import pytest

import backtest_ic as backtest
import JQuamtsScreeningBot as bot


def price_series(values, days=None):
    days = days or pd.bdate_range("2026-11-17", periods=len(values)).date
    return pd.DataFrame({"date": days, "close": values})


def test_horizon_counts_elapsed_sessions_and_requires_complete_prices():
    prices = price_series([100., 110., 121.])
    baseline = dt.date(2026, 11, 16)
    assert backtest._forward_return_business_days(prices, baseline, 1) == pytest.approx(.1)
    assert backtest._forward_return_business_days(prices, baseline, 2) == pytest.approx(.21)
    assert backtest._forward_return_business_days(prices, baseline, 3) is None
    prices.loc[1, "close"] = None
    assert backtest._forward_return_business_days(prices, baseline, 2) is None
    with pytest.raises(ValueError, match="positive"):
        backtest._forward_return_business_days(prices, baseline, 0)


def test_manifest_publication_after_open_defers_reference_close(tmp_path, monkeypatch):
    master = tmp_path / "screening_offline.csv"
    pd.DataFrame([{"code": "1234", "price_asof_date": "2026-11-16",
                   "information_cutoff": "2026-11-17T08:00:00+09:00",
                   "registration_a_score": 70, "registration_a_eligible": True}]).to_csv(master, index=False)
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "price_asof_date": "2026-11-16", "information_cutoff": "2026-11-17T08:00:00+09:00",
        "generated_at": "2026-11-17T09:30:00+09:00", "next_trade_date": "2026-11-17"}), "utf-8")
    monkeypatch.setattr(backtest, "_load_prices_for_code", lambda code: price_series([100., 110., 121.]))
    returns, _, groups = backtest.run_single_snapshot(master, None, [1], ["registration_a_score"], "registration")
    assert returns.iloc[0]["entry_reference_date"] == "2026-11-18"
    assert returns.iloc[0]["fwd_ret_1d"] == pytest.approx(.1)
    assert groups.iloc[0]["lane"] == "a"
    with pytest.raises(ValueError, match="precedes"):
        backtest.run_single_snapshot(master, dt.date(2026, 11, 15), [1], [], "registration")


def test_missing_date_is_rejected_and_immature_groups_are_empty(tmp_path, monkeypatch):
    master = tmp_path / "screening_offline.csv"
    pd.DataFrame([{"code": "1234", "candidate_lane": "core", "entry_score": 10}]).to_csv(master, index=False)
    with pytest.raises(ValueError, match="Screening date unavailable"):
        backtest.run_single_snapshot(master, None, [1], ["entry_score"], "candidate_lane")
    monkeypatch.setattr(backtest, "_load_prices_for_code", lambda code: price_series([100.]))
    returns, _, groups = backtest.run_single_snapshot(master, dt.date(2026, 11, 16), [21], ["entry_score"], "candidate_lane")
    assert pd.isna(returns.iloc[0]["fwd_ret_21d"])
    assert groups.empty
    assert backtest.pool_lane_stats([groups]).empty


def test_route_ic_and_groups_use_only_that_routes_eligible_candidates(tmp_path, monkeypatch):
    master = tmp_path / "screening_offline.csv"
    rows = [{"code": str(1000+i), "price_asof_date": "2026-11-16", "registration_a_score": i,
             "registration_a_eligible": i <= 5, "registration_b_score": 10-i,
             "registration_b_eligible": i >= 5} for i in range(1, 7)]
    pd.DataFrame(rows).to_csv(master, index=False)
    monkeypatch.setattr(backtest, "_load_prices_for_code",
                        lambda code: price_series([100., 100.+int(code)-1000]))
    returns, ic, groups = backtest.run_single_snapshot(master, None, [1], backtest.DEFAULT_SCORE_COLUMNS, "registration")
    assert len(returns) == 6
    a = ic[ic["score"] == "registration_a_score"].iloc[0]
    assert a["n"] == 5 and a["ic_spearman"] == 1.
    assert groups.set_index("lane")["n"].to_dict() == {"a": 5, "b": 2}


def standard_snapshot(tmp_path, missing_day=False):
    store = bot.Store(tmp_path / "cache")
    days = pd.bdate_range("2026-11-16", periods=3)
    calendar = [{"Date": str(day.date()), "HolDiv": "1"} for day in days]
    rows = [{"Date": str(day.date()), "Code": "12340", "O": close, "H": close, "L": close,
             "C": close, "Vo": 100, "Va": close*100, "AdjFactor": factor, "ExRT": "1" if factor != 1 else None}
            for day, close, factor in zip(days, (100., 50., 55.), (1., .5, 1.))]
    if missing_day:
        rows.pop(1)
    snapshot = {"schema": bot.CACHE_SCHEMA, "snapshot_id": "future", "price_asof_date": "2026-11-18",
                "globals": {"calendar": store.put({"status": "complete", "rows": calendar})},
                "symbols": {"1234": {"prices": store.put({"status": "complete", "rows": rows})}}}
    store.save_snapshot(snapshot, promote=True)
    return store, snapshot


def test_standard_price_source_uses_split_factors_and_preserves_missing_sessions(tmp_path):
    store, snapshot = standard_snapshot(tmp_path)
    prices = backtest._load_standard_prices("1234", store, snapshot)
    assert prices["close"].tolist() == [50., 50., 55.]
    assert backtest._forward_return_business_days(prices, dt.date(2026, 11, 16), 1) == pytest.approx(.1)
    missing_store, missing_snapshot = standard_snapshot(tmp_path / "gap", missing_day=True)
    gap = backtest._load_standard_prices("1234", missing_store, missing_snapshot)
    assert len(gap) == 3 and pd.isna(gap.iloc[1]["close"])
    assert backtest._forward_return_business_days(gap, dt.date(2026, 11, 16), 1) is None


def test_cli_pins_future_snapshot_and_writes_its_evidence(tmp_path, monkeypatch):
    store, snapshot = standard_snapshot(tmp_path)
    master = tmp_path / "screening_offline.csv"
    pd.DataFrame([{"code": "1234", "price_asof_date": "2026-11-16", "registration_a_eligible": True,
                   "registration_a_score": 80}]).to_csv(master, index=False)
    outdir = tmp_path / "backtest"
    assert backtest.main(["--master", str(master), "--cache-dir", str(store.root),
                          "--price-snapshot-id", snapshot["snapshot_id"], "--horizons", "1",
                          "--outdir", str(outdir)]) == 0
    metadata = json.loads((outdir / "run_meta.json").read_text("utf-8"))
    assert metadata["price_snapshot_id"] == "future"
    assert metadata["price_snapshot_input_hash"] == bot.digest(snapshot)
    returns = pd.read_csv(outdir / "per_stock_returns.csv")
    assert returns.iloc[0]["fwd_ret_1d"] == pytest.approx(.1)
