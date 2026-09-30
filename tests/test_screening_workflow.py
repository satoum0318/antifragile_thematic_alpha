"""Offline orchestration checks using temporary stores and injected collectors."""
import copy
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import JQuamtsScreeningBot as bot
import jq_screening_workflow as workflow


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = [dt.datetime(2026, 9, 30, 19, tzinfo=bot.JST)]
    state = SimpleNamespace(clock=clock, calls=[], events=[], clients=[], empty_summary=False,
                            fail_endpoint=None, interrupt_code=None, fail_update=False,
                            fail_analysis=False, margin_partial=False, new_split=False,
                            masters=[('10010', '0111'), ('20010', '0105')], globals_calls=0,
                            rollover_on=None, runs=[])
    args = SimpleNamespace(cache_dir=tmp_path/'cache', output_dir=tmp_path/'out',
                           api_ini=tmp_path/'api.ini', max_retries=3, budget=10, top=10)
    cfg = bot.Config()
    store = bot.Store(args.cache_dir)
    store.ensure()
    state.args, state.cfg, state.store = args, cfg, store
    monkeypatch.setattr(bot, 'now_jst', lambda: clock[0])
    monkeypatch.setattr(bot, 'api_key_from_env_or_ini', lambda _: 'test-not-a-real-key')

    def blob(endpoint, params, rows, when=None):
        return {'status': 'complete', 'rows': rows, 'endpoint': endpoint, 'params': params,
                'collected_at': (when or clock[0]).isoformat()}

    state.blob = blob

    class Client:
        def __init__(self, key, config, cache):
            self.cfg, self.store, self.closed = config, cache, False
            state.clients.append(self)

        def fetch(self, endpoint, params):
            state.calls.append((endpoint, copy.deepcopy(params)))
            code = params.get('code')
            if code == state.interrupt_code:
                state.interrupt_code = None
                raise KeyboardInterrupt()
            if endpoint == state.fail_endpoint:
                raise bot.CollectionError('simulated_temporary_failure')
            if endpoint == 'equities/bars/daily':
                rows = [{'Code': code, 'Date': params['to'], 'C': 1000, 'AdjFactor': 1, 'ExRT': '0'}]
            elif endpoint == 'fins/summary':
                rows = [] if state.empty_summary else [{'Code': code, 'DiscDate': '2026-05-15', 'NP': 1}]
            elif endpoint == 'fins/earnings-date':
                rows = []
            else:
                pytest.fail('Unexpected endpoint: '+endpoint)
            return blob(endpoint, params, rows)

        def close(self):
            self.closed = True

    monkeypatch.setattr(bot, 'Client', Client)

    def globals_(client):
        state.globals_calls += 1
        if state.rollover_on == state.globals_calls:
            clock[0] += dt.timedelta(days=1)
        day = clock[0].date()
        master = [{'Date': str(day), 'Code': code, 'CoName': 'Synthetic '+code,
                   'Mkt': market, 'ProdCat': '011', 'S33': '5250'} for code, market in state.masters]
        result = {'master': store.put(blob('equities/master', {'date': str(day)}, master)),
                  'calendar': store.put(blob('markets/calendar', {}, [{'Date': str(day), 'HolDiv': '1'}])),
                  'topix': store.put(blob('indices/bars/daily/topix', {}, [{'Date': str(day), 'C': 2000}]))}
        return result, day

    monkeypatch.setattr(bot, 'collect_globals', globals_)

    def margin_refresh(module, client, source_snapshot_id, only_codes, **kwargs):
        state.events.append(('margin', source_snapshot_id))
        source = client.store.snapshot(source_snapshot_id)
        assert source['screening_scope']['included_codes'] == only_codes
        assert all(code != '2001' for code in only_codes)
        # The workflow's proxy must map this into its own resume namespace.
        scratch = client.store.root/'margin_collection_resume.json'
        assert scratch.name == 'screening_margin_resume.json'
        bot.atomic_json(scratch, {'source_snapshot_id': source_snapshot_id})
        clock[0] += dt.timedelta(seconds=1)
        ident = bot.new_id('ms_test_')
        result = {'schema': bot.jq_margin_workflow.MARGIN_SNAPSHOT_SCHEMA,
                  'snapshot_id': ident, 'source_snapshot_id': source_snapshot_id,
                  'status': 'partial' if state.margin_partial else 'complete', 'information_cutoff': clock[0].isoformat(),
                  'selected_codes': only_codes, 'targets': {},
                  'symbols': {code: {'last_attempt_status': 'collection_failed' if state.margin_partial else 'complete'}
                              for code in only_codes}, 'pending_codes': [],
                  'terminal_failed_codes': list(only_codes) if state.margin_partial else []}
        bot.atomic_json(store.root/'margin_snapshots'/(ident+'.json'), result)
        scratch.unlink()
        return result

    monkeypatch.setattr(bot.jq_margin_workflow, 'margin_refresh', margin_refresh)

    def update(client, *, source_snapshot, global_context, promote, progress):
        state.events.append(('update', source_snapshot.get('snapshot_id')))
        assert promote is False
        if state.fail_update:
            raise bot.CollectionError('simulated_fatal_daily_update')
        clock[0] += dt.timedelta(seconds=1)
        result = copy.deepcopy(source_snapshot)
        result.update(snapshot_id=bot.new_id('s_update_'), globals=global_context[0],
                      price_asof_date=str(global_context[1]), information_cutoff=clock[0].isoformat())
        for item in result.get('symbols', {}).values():
            if item.get('complete'):
                item.update(checked_price_date=str(global_context[1]),
                            financial_checked_through=clock[0].isoformat(), earnings_checked_through=clock[0].isoformat())
                if state.new_split:
                    raw = copy.deepcopy(store.get(item['prices']))
                    raw['rows'].append({'Date': str(global_context[1]), 'Code': '10010', 'C': 500,
                                        'AdjFactor': .5, 'ExRT': '1'})
                    item['prices'] = store.put(raw)
        client.store.save_snapshot(result, promote=False)
        return result

    monkeypatch.setattr(bot, 'update_daily', update)

    def analyze(cache, config, output, top, snapshot_id, asof, cutoff, margin_id, *, require_current):
        state.events.append(('analyze', snapshot_id))
        assert require_current is True
        source = cache.snapshot(snapshot_id)
        margin = bot.jq_margin_workflow.load_margin_snapshot(bot, cache, margin_id)
        assert bot.timestamp(margin['information_cutoff']) <= bot.timestamp(source['information_cutoff'])
        published = cache.snapshot()
        assert published.get('snapshot_id') != snapshot_id
        status = json.loads((cache.root/'screening_workflow_status.json').read_text('utf-8'))
        assert status['status'] == 'running' and status['stage'] == 'analyze'
        if state.fail_analysis:
            raise bot.DataError('screening_inputs_not_current_or_verified: price_not_current')
        state.runs.append((copy.deepcopy(source), copy.deepcopy(margin)))
        path = output/'history'/('r_'+str(len(state.runs)))
        path.mkdir(parents=True)
        bot.atomic_json(path/'run_manifest.json', {'run_id': path.name, 'run_status': source['status']})
        bot.atomic_json(output/'latest_run.json', {'run_id': path.name})
        return path

    monkeypatch.setattr(bot, 'analyze_store', analyze)
    state.run = lambda: workflow.run_screening(bot, args, cfg)

    def seed(financial_age=5, daily_wrapped=False, complete=True):
        at = clock[0]-dt.timedelta(days=financial_age)
        item = {'complete': complete}
        price_at = clock[0]-dt.timedelta(days=5)
        item['prices'] = store.put(blob('equities/bars/daily', {'code': '10010', 'from': '2020-09-20', 'to': '2026-09-29'},
                                               [{'Code': '10010', 'Date': '2026-09-29', 'C': 1000, 'AdjFactor': 1, 'ExRT': '0'}], price_at))
        summary = store.put(blob('fins/summary', {'code': '10010'}, [{'Code': '10010', 'NP': 1}], at))
        if daily_wrapped:
            summary = store.put({'status': 'complete', 'rows': [{'Code': '10010', 'NP': 1}],
                                 'provenance': 'prior_full_history_plus_complete_date_updates',
                                 'parent_ref': summary, 'collected_at': clock[0].isoformat()})
        item.update(summary=summary, earnings=store.put(blob('fins/earnings-date', {'code': '10010'}, [])),
                    checked_price_date='2026-09-29', financial_checked_through=clock[0].isoformat(),
                    earnings_checked_through=clock[0].isoformat())
        globals0, day = globals_(None)
        snap = {'schema': bot.CACHE_SCHEMA, 'snapshot_id': 'original', 'globals': globals0,
                'symbols': {'1001': item}, 'price_asof_date': '2026-09-29',
                'information_cutoff': (clock[0]-dt.timedelta(minutes=1)).isoformat(), 'status': 'complete'}
        store.save_snapshot(snap, promote=True)
        return snap

    state.seed = seed
    return state


def test_initial_job_excludes_pro_market_uses_exact_margin_and_preserves_manual_resume(harness):
    h = harness
    manual = h.store.root/'margin_collection_resume.json'
    bot.atomic_json(manual, {'manual_job': 'do_not_change'})
    before = manual.read_bytes()
    assert h.run() == 0
    assert manual.read_bytes() == before
    assert [kind for kind, _ in h.events] == ['margin', 'update', 'analyze']
    assert {params['code'] for _, params in h.calls} == {'10010'}
    snap, margin = h.runs[0]
    assert snap['screening_scope']['included_codes'] == ['1001']
    assert snap['screening_scope']['excluded'][0]['reason'] == 'tokyo_pro_market'
    assert snap['symbols']['1001']['last_full_financial_at']
    assert h.store.snapshot()['snapshot_id'] == snap['snapshot_id']
    assert all(client.closed for client in h.clients)
    assert not (h.store.root/'screening_workflow_resume.json').exists()


def test_recent_daily_updates_do_not_reset_monthly_financial_full_clock(harness):
    h = harness
    original = h.seed(financial_age=45, daily_wrapped=True)
    original_bytes = (h.store.root/'snapshots'/'original.json').read_bytes()
    assert h.run() == 0
    endpoints = Counter(endpoint for endpoint, _ in h.calls)
    assert endpoints == {'fins/summary': 1, 'fins/earnings-date': 1}
    assert (h.store.root/'snapshots'/'original.json').read_bytes() == original_bytes
    current = h.store.snapshot()['symbols']['1001']
    assert current['last_full_price_at'] != current['last_full_financial_at']
    assert current['summary'] != original['symbols']['1001']['summary']


def test_empty_financial_response_is_not_retried_three_times_or_again_same_day(harness):
    h = harness
    h.empty_summary = True
    assert h.run() == 2
    assert sum(endpoint == 'fins/summary' for endpoint, _ in h.calls) == 1
    item = h.store.snapshot()['symbols']['1001']
    assert item['complete'] is False
    assert item['screening_availability'] == 'financial_unavailable'
    assert item['screening_next_retry_date'] == '2026-10-01'
    assert h.run() == 2
    assert sum(endpoint == 'fins/summary' for endpoint, _ in h.calls) == 1
    h.clock[0] += dt.timedelta(days=1)
    assert h.run() == 2
    assert sum(endpoint == 'fins/summary' for endpoint, _ in h.calls) == 2


def test_failed_periodic_refresh_does_not_present_previous_complete_data_as_fresh(harness):
    h = harness
    h.seed(financial_age=45)
    h.fail_endpoint = 'fins/summary'
    assert h.run() == 2
    assert sum(endpoint == 'fins/summary' for endpoint, _ in h.calls) == 3
    item = h.store.snapshot()['symbols']['1001']
    assert item['complete'] is False and item['screening_availability'] == 'collection_failed'
    assert h.store.snapshot('original')['symbols']['1001']['complete'] is True


def test_earnings_failure_is_retried_next_run_despite_fresh_summary_marker(harness):
    h = harness
    h.fail_endpoint = 'fins/earnings-date'
    assert h.run() == 2
    item = h.store.snapshot()['symbols']['1001']
    assert item['last_full_financial_at'] and item['complete'] is False
    h.fail_endpoint = None
    assert h.run() == 0
    assert h.store.snapshot()['symbols']['1001']['complete'] is True


def test_interruption_resumes_finished_stocks_without_repeating_their_full_requests(harness):
    h = harness
    h.masters += [('30010', '0111')]
    h.interrupt_code = '30010'
    assert h.run() == 130
    assert h.store.snapshot() == {}
    assert not h.runs
    assert (h.store.root/'screening_workflow_resume.json').exists()
    first_count = sum(params.get('code') == '10010' for _, params in h.calls)
    assert first_count == 3
    assert h.run() == 0
    assert sum(params.get('code') == '10010' for _, params in h.calls) == first_count
    assert all(client.closed for client in h.clients)


def test_fatal_daily_failure_retains_old_snapshot_and_report_and_resumes_exact_margin(harness):
    h = harness
    h.seed()
    bot.atomic_json(h.args.output_dir/'latest_run.json', {'run_id': 'old_report'})
    h.fail_update = True
    assert h.run() == 1
    assert h.store.snapshot()['snapshot_id'] == 'original'
    assert json.loads((h.args.output_dir/'latest_run.json').read_text('utf-8'))['run_id'] == 'old_report'
    journal = json.loads((h.store.root/'screening_workflow_resume.json').read_text('utf-8'))
    selected_id = journal['margin_snapshot_id']
    h.fail_update = False
    assert h.run() == 0
    assert len([event for event in h.events if event[0] == 'margin']) == 1
    assert h.runs[-1][1]['snapshot_id'] == selected_id


def test_day_rollover_refreshes_margin_window_and_does_not_publish_old_price_date(harness):
    h = harness
    h.rollover_on = 2
    assert h.run() == 0
    assert len([event for event in h.events if event[0] == 'margin']) == 2
    assert h.store.snapshot()['price_asof_date'] == '2026-10-01'
    assert len(h.runs) == 1


def test_history_depth_change_invalidates_full_history_assumptions_on_resume(harness):
    h = harness
    h.masters += [('30010', '0111')]
    h.interrupt_code = '30010'
    assert h.run() == 130
    h.cfg.ADJUSTMENT_HISTORY_YEARS = 7
    assert h.run() == 0
    prices = [params for endpoint, params in h.calls if endpoint == 'equities/bars/daily' and params['code'] == '10010']
    assert len(prices) == 2
    assert prices[0]['from'].startswith('2020') and prices[1]['from'].startswith('2019')


def test_history_depth_change_after_completed_job_refreshes_prices_without_financial_full(harness):
    h = harness
    assert h.run() == 0
    first = len(h.calls)
    h.cfg.ADJUSTMENT_HISTORY_YEARS = 7
    assert h.run() == 0
    assert [endpoint for endpoint, _ in h.calls[first:]] == ['equities/bars/daily']
    assert h.store.snapshot()['symbols']['1001']['last_full_price_history_years'] == 7


def test_credit_only_failure_is_partial_with_preserved_credit_counts(harness):
    h = harness
    h.margin_partial = True
    assert h.run() == 2
    status = json.loads((h.store.root/'screening_workflow_status.json').read_text('utf-8'))
    assert status['counts']['unavailable_count'] == 0
    assert status['counts']['credit_failed_count'] == 1
    assert status['counts']['credit_completed_count'] == 0
    assert status['counts']['credit_pending_count'] == 0
    assert h.runs[-1][0]['screening_workflow']['collection_status'] == 'partial'


def test_publication_guard_keeps_prior_report_and_normal_snapshot(harness):
    h = harness
    h.seed()
    bot.atomic_json(h.args.output_dir/'latest_run.json', {'run_id': 'old_report'})
    h.fail_analysis = True
    assert h.run() == 1
    assert h.store.snapshot()['snapshot_id'] == 'original'
    assert json.loads((h.args.output_dir/'latest_run.json').read_text('utf-8'))['run_id'] == 'old_report'
    assert not h.runs


def test_new_split_triggers_targeted_price_history_check_and_final_cutoff_verification(harness):
    h = harness
    h.seed()
    h.new_split = True
    assert h.run() == 0
    assert Counter(endpoint for endpoint, _ in h.calls) == {'equities/bars/daily': 1}
    assert [kind for kind, _ in h.events] == ['margin', 'update', 'update', 'analyze']
    assert h.store.snapshot()['symbols']['1001']['last_full_price_at'].startswith('2026-09-30')


def test_corrupt_resume_fails_explicitly_before_any_api_request(harness):
    h = harness
    (h.store.root/'screening_workflow_resume.json').write_text('[1,2]', encoding='utf-8')
    assert h.run() == 1
    assert not h.calls and not h.clients and not h.runs
    status = json.loads((h.store.root/'screening_workflow_status.json').read_text('utf-8'))
    assert status['status'] == 'failed'
    assert 'workflow_state_must_be_object' in status['reason']


def test_cleanup_failure_after_publication_reports_published_result_not_old_retention(harness, monkeypatch, caplog):
    h = harness
    unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.name == 'screening_workflow_resume.json':
            raise OSError('simulated_cleanup_denied')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', fail_cleanup)
    assert h.run() == 2
    status = json.loads((h.store.root/'screening_workflow_status.json').read_text('utf-8'))
    assert status['run_id'] == 'r_1' and status['stage'] == 'done'
    assert status['reason'].startswith('result_published_metadata_warning:')
    assert h.store.snapshot()['snapshot_id'] == h.runs[0][0]['snapshot_id']
    assert json.loads((h.args.output_dir/'latest_run.json').read_text('utf-8'))['run_id'] == 'r_1'
    assert '公開済み' in caplog.text
    assert '公開済みランキングは保持' not in caplog.text
