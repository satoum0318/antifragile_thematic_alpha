"""Regression checks for daily coverage and scoped, staged workflow inputs."""
import copy
import datetime as dt
from types import SimpleNamespace

import pytest

import JQuamtsScreeningBot as bot


def source_fixture(tmp_path, monkeypatch, *, financial_day='2026-09-30'):
    monkeypatch.setattr(bot, 'now_jst', lambda: dt.datetime(2026, 10, 1, 19, tzinfo=bot.JST))
    store = bot.Store(tmp_path / 'cache')
    calendar = []
    day = dt.date(2026, 8, 1)
    while day <= dt.date(2026, 10, 15):
        calendar.append({'Date': str(day), 'HolDiv': '1' if day.weekday() < 5 else '0'})
        day += dt.timedelta(days=1)
    master = [{'Code': '99990', 'CoName': 'Example', 'Mkt': '0111', 'S33': '5250',
               'ProdCat': '011', 'Date': '2026-10-01'},
              {'Code': '88880', 'CoName': 'Professional market', 'Mkt': '0105', 'S33': '5250',
               'ProdCat': '011', 'Date': '2026-10-01'}]
    def bundle(rows):
        return store.put({'status': 'complete', 'rows': rows})
    globals_ = {'calendar': bundle(calendar), 'master': bundle(master),
                'topix': bundle([{'Date': '2026-10-01', 'C': 3000}])}
    item = {'complete': True, 'checked_price_date': '2026-09-11',
            'financial_checked_through': financial_day, 'earnings_checked_through': financial_day,
            'last_full_price_at': '2026-09-11T19:00:00+09:00',
            'last_full_financial_at': '2026-09-11T19:00:00+09:00',
            'prices': bundle([{'Code': '99990', 'Date': '2026-09-11', 'C': 100}]),
            'summary': bundle([{'Code': '99990', 'DiscDate': '2026-08-01', 'DiscNo': 'old'}]),
            'earnings': bundle([])}
    snap = {'schema': bot.CACHE_SCHEMA, 'snapshot_id': 'original', 'status': 'complete',
            'globals': globals_, 'price_asof_date': '2026-09-11',
            'information_cutoff': '2026-09-30T19:00:00+09:00',
            'screening_scope': {'kind': 'retail_ordinary', 'included_codes': ['9999'],
                                'excluded': [{'code': '8888', 'reason': 'tokyo_pro_market'}]},
            'symbols': {'9999': item}}
    store.save_snapshot(snap, promote=True)
    monkeypatch.setattr(bot, 'collect_globals', lambda client: (globals_, dt.date(2026, 10, 1)))
    calls = []
    def fetch(endpoint, params):
        calls.append((endpoint, params.copy()))
        rows = []
        if endpoint == 'equities/bars/daily':
            rows = [{'Code': '99990', 'Date': params['date'], 'C': 101}]
        elif endpoint == 'fins/summary' and params['date'] == '2026-09-30':
            rows = [{'Code': '99990', 'DiscDate': params['date'], 'DiscNo': 'new', 'NP': '1000'}]
        return {'status': 'complete', 'rows': rows}
    return SimpleNamespace(store=store, fetch=fetch), snap, globals_, calls


def test_newer_financial_clock_cannot_skip_missing_price_dates(tmp_path, monkeypatch):
    client, source, globals_, calls = source_fixture(tmp_path, monkeypatch)
    updated = bot.update_daily(client, source_snapshot=source,
                               global_context=(globals_, dt.date(2026, 10, 1)), promote=False)
    price_days = [p['date'] for endpoint, p in calls if endpoint == 'equities/bars/daily']
    assert '2026-09-14' in price_days
    assert '2026-09-29' in price_days
    assert updated['daily_update_window']['from'] == '2026-09-05'
    assert updated['symbols']['9999']['last_full_financial_at'] == source['symbols']['9999']['last_full_financial_at']
    assert updated['symbols']['9999']['last_full_price_at'] == source['symbols']['9999']['last_full_price_at']
    summary = client.store.get(updated['symbols']['9999']['summary'])['rows']
    assert {r['DiscNo'] for r in summary} == {'old', 'new'}
    assert client.store.snapshot()['snapshot_id'] == 'original'
    assert client.store.snapshot(updated['snapshot_id'])['status'] == 'complete'


def test_earnings_clock_and_overlap_are_not_skipped(tmp_path, monkeypatch):
    client, source, globals_, calls = source_fixture(tmp_path, monkeypatch)
    source['symbols']['9999']['earnings_checked_through'] = '2026-09-01'
    updated = bot.update_daily(client, source_snapshot=source, global_context=(globals_, dt.date(2026, 10, 1)), promote=False)
    assert updated['daily_update_window']['from'] == '2026-08-26'
    assert ('fins/earnings-date', {'date': '2026-09-02'}) in calls


def test_explicit_start_must_not_skip_price_coverage(tmp_path, monkeypatch):
    client, source, _, _ = source_fixture(tmp_path, monkeypatch)
    with pytest.raises(bot.CollectionError, match='skip unverified'):
        bot.update_daily(client, dt.date(2026, 9, 30), source_snapshot=source)
    assert client.store.snapshot()['snapshot_id'] == 'original'


def test_latest_empty_price_is_unavailable_not_fresh(tmp_path, monkeypatch):
    client, source, _, _ = source_fixture(tmp_path, monkeypatch)
    original_fetch = client.fetch
    def fetch(endpoint, params):
        if endpoint == 'equities/bars/daily' and params['date'] == '2026-10-01':
            return {'status': 'complete', 'rows': []}
        return original_fetch(endpoint, params)
    client.fetch = fetch
    updated = bot.update_daily(client)
    assert updated['status'] == 'partial_universe'
    assert updated['pending_count'] == 1
    assert updated['symbols']['9999']['complete'] is False
    assert updated['symbols']['9999']['failure_reason'] == 'latest_price_row_missing'
    assert updated['symbols']['9999']['checked_price_date'] == '2026-09-11'


def test_fatal_update_does_not_publish_new_snapshot(tmp_path, monkeypatch):
    client, _, _, _ = source_fixture(tmp_path, monkeypatch)
    def fail(endpoint, params):
        raise bot.CollectionError('HTTP 503 retry_exhausted')
    client.fetch = fail
    with pytest.raises(bot.CollectionError):
        bot.update_daily(client)
    assert client.store.snapshot()['snapshot_id'] == 'original'
    snapshots = list((client.store.root / 'snapshots').glob('*.json'))
    assert len(snapshots) == 2


def test_scope_is_explicit_and_does_not_change_legacy_master(tmp_path, monkeypatch):
    client, source, _, _ = source_fixture(tmp_path, monkeypatch)
    rows = bot.bundle_rows(client.store, source['globals']['master'])
    assert bot._snapshot_stock_counts(client.store, source) == (1, 1)
    current = bot.screening_master_records(rows, source)
    assert current[1]['instrument_type'] == 'out_of_scope'
    assert current[1]['master_reason'] == 'tokyo_pro_market'
    old = copy.deepcopy(source)
    old.pop('screening_scope')
    assert bot.screening_master_records(rows, old)[1]['instrument_type'] == 'stock'
    assert bot._snapshot_stock_counts(client.store, old) == (2, 1)


@pytest.mark.parametrize('status,age', [('diagnostic_only', 0), ('complete', 1), ('complete', None)])
def test_current_screening_cannot_publish_stale_or_unverified_result(tmp_path, monkeypatch, status, age):
    client, source, _, _ = source_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(bot, 'score_snapshot', lambda *args: ([], {}, {
        'run_status': status, 'registration_price_age_bd': age, 'global_issues': []}))
    def forbidden(*args, **kwargs):
        pytest.fail('Unverified analysis must not publish or replace saved results')
    monkeypatch.setattr(bot, 'write_run', forbidden)
    with pytest.raises(bot.DataError, match='not_current_or_verified'):
        bot.analyze_store(client.store, bot.Config(), tmp_path/'output', require_current=True)


def test_market_day_change_during_report_writing_blocks_publication(tmp_path, monkeypatch):
    client, source, _, _ = source_fixture(tmp_path, monkeypatch)
    meta = {'run_status': 'complete', 'registration_price_age_bd': 0, 'global_issues': [],
            'price_asof_date': '2026-10-01'}
    monkeypatch.setattr(bot, 'score_snapshot', lambda *args: ([], {}, meta))
    def write(records, stats, metadata, output, top, publication_check):
        monkeypatch.setattr(bot, 'now_jst', lambda: dt.datetime(2026, 10, 2, 19, tzinfo=bot.JST))
        publication_check()
        pytest.fail('A date change must stop publication')
    monkeypatch.setattr(bot, 'write_run', write)
    with pytest.raises(bot.DataError, match='date_changed_during_analysis'):
        bot.analyze_store(client.store, bot.Config(), tmp_path/'output', require_current=True)
    assert not (tmp_path/'output'/'latest_run.json').exists()


def test_historical_failure_registry_does_not_mislabel_scoped_current_run(tmp_path, monkeypatch):
    client, _, _, _ = source_fixture(tmp_path, monkeypatch)
    bot.atomic_json(client.store.root/'collection_failures.json', {
        'schema': 'jq-collection-failures-v1', 'failures': {
            '8888': {'reason': 'summary_empty_http200'},
            '9999': {'reason': 'old_failure_since_recovered'}}})
    status = bot.collection_status(client.store)
    assert status['total'] == status['complete'] == 1
    assert status['terminal_failures'] == 0
    assert status['legacy_terminal_failures'] == 2
    assert status['scope_excluded'] == 1


@pytest.mark.parametrize('option,value', [('--asof','2026-09-11'),('--code','9999'),('--refresh-days','0')])
def test_screen_rejects_advanced_overrides_before_starting_network(tmp_path, monkeypatch, option, value):
    args = bot.build_parser().parse_args(['--phase','screen','--cache-dir',str(tmp_path/'cache'),option,value])
    monkeypatch.setattr(bot.jq_screening_workflow, 'run_screening', lambda *a: pytest.fail('must reject first'))
    with pytest.raises(bot.DataError):
        bot.dispatch(args, bot.Config())
