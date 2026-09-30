"""Phase boundaries: offline analysis, bounded refresh, and launcher wiring."""
import subprocess
import sys
from pathlib import Path

import pytest

import JQuamtsScreeningBot as bot


def arguments(tmp_path, *extra):
    return bot.build_parser().parse_args([
        '--cache-dir', str(tmp_path / 'cache'), '--output-dir', str(tmp_path / 'out'), *extra])


def forbidden(*args, **kwargs):
    raise AssertionError('This phase must not collect or score screening data')


def test_margin_analyze_routes_offline_without_credentials_or_rescoring(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, 'api_key_from_env_or_ini', forbidden)
    monkeypatch.setattr(bot, 'Client', forbidden)
    monkeypatch.setattr(bot, 'score_snapshot', forbidden)
    calls = []

    def analyze(module, store, ident, cutoff):
        calls.append((module is bot, store.root, ident, cutoff.isoformat()))
        return [{'code': '8697'}], {'source_snapshot_id': 'source'}

    def write(module, records, meta, output):
        assert records == [{'code': '8697'}]
        assert meta == {'source_snapshot_id': 'source'}
        return output / 'margin_history' / 'test'

    monkeypatch.setattr(bot.jq_margin_workflow, 'margin_analyze', analyze)
    monkeypatch.setattr(bot.jq_margin_workflow, 'write_margin_run', write)
    args = arguments(tmp_path, '--phase', 'margin_analyze', '--margin-snapshot-id', 'margin_1',
                     '--information-cutoff', '2026-09-30T18:00:00+09:00')
    assert bot.dispatch(args, bot.Config()) == 0
    assert calls == [(True, tmp_path / 'cache', 'margin_1', '2026-09-30T18:00:00+09:00')]
    assert not (tmp_path / 'out' / 'margin-analyze.lock').exists()


@pytest.mark.parametrize('extra', [('--asof', '2026-09-11'), ('--force-full',),
                                  ('--information-cutoff', '2026-09-30T18:00:00+09:00')])
def test_margin_refresh_rejects_screening_clock_overrides_before_auth(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(bot, 'api_key_from_env_or_ini', forbidden)
    with pytest.raises(bot.DataError, match='today-fixed 90 days'):
        bot.dispatch(arguments(tmp_path, '--phase', 'margin_refresh', *extra), bot.Config())


def test_margin_refresh_once_leaves_pending_and_closes_client(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, 'collect', forbidden)
    monkeypatch.setattr(bot, 'update_daily', forbidden)
    monkeypatch.setattr(bot, 'score_snapshot', forbidden)
    monkeypatch.setattr(bot, 'api_key_from_env_or_ini', lambda path: 'offline-test')
    closed = []

    class Client:
        def __init__(self, key, cfg, store):
            self.store = store

        def close(self):
            closed.append(True)

    monkeypatch.setattr(bot, 'Client', Client)
    calls = []

    def refresh(module, client, **kw):
        calls.append(kw)
        return {'snapshot_id': 'margin_1', 'pending_codes': ['7203'], 'terminal_failed_codes': []}

    monkeypatch.setattr(bot.jq_margin_workflow, 'margin_refresh', refresh)
    args = arguments(tmp_path, '--phase', 'margin_refresh', '--snapshot-id', 'source_1',
                     '--codes', '8697,7203', '--budget', '1', '--once', '--reset-pending')
    assert bot.dispatch(args, bot.Config()) == 2
    assert calls == [dict(source_snapshot_id='source_1', budget=1, only_codes=['8697', '7203'],
                          max_retries=3, reset_pending=True)]
    assert closed == [True]
    assert not (tmp_path / 'cache' / 'collector.lock').exists()


def test_corrupt_optional_margin_input_stays_a_diagnostic():
    cutoff = bot.timestamp('2026-09-30T18:00:00+09:00')
    for malformed in ([], {'status': 'complete', 'rows': [{}], 'row_versions': None}):
        result = bot.margin_diagnostic(malformed, cutoff)
        assert result['margin_ratio'] is None
        assert result['margin_diagnostic_status'] == 'schema_unsupported'
        assert 'common_issues' not in result


def test_runner_can_start_and_exposes_daily_margin_phases():
    root = Path(bot.__file__).parent
    run = subprocess.run([sys.executable, '-B', str(root / 'runner.py'), '--help'],
                         cwd=root, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert 'margin_refresh' in run.stdout
    assert 'margin_analyze' in run.stdout
    assert '--margin-snapshot-id' in run.stdout


@pytest.mark.parametrize('body', [{'data': {'unexpected': 'schema'}}, ['unexpected'], {'data': [42]}])
def test_unsupported_api_response_body_is_retained_for_review(tmp_path, body):
    class Response:
        status_code = 200
        headers = {}

        def json(self):
            return body

    class Session:
        headers = {}

        def get(self, *args, **kwargs):
            return Response()

        def close(self):
            pass

    class Limiter:
        def reserve(self, endpoint):
            pass

    store = bot.Store(tmp_path / 'cache')
    client = bot.Client('offline-test', bot.Config(), store, session=Session(), limiter=Limiter())
    with pytest.raises(bot.CollectionError, match='unexpected_response_schema') as caught:
        client.fetch('markets/margin-interest', {'code': '86970'})
    saved = store.get(caught.value.raw_cache_ref)
    assert saved['status'] == 'incomplete'
    assert saved['rows'] == []
    assert saved['raw_pages'] == [body]


def test_margin_refresh_calendar_failure_is_not_success_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, 'api_key_from_env_or_ini', lambda path: 'offline-test')

    class Client:
        def __init__(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(bot, 'Client', Client)
    monkeypatch.setattr(bot.jq_margin_workflow, 'margin_refresh', lambda *a, **kw: {
        'snapshot_id': 'calendar_failed', 'status': 'partial',
        'calendar_status': 'collection_failed', 'pending_codes': [], 'terminal_failed_codes': []})
    assert bot.dispatch(arguments(tmp_path, '--phase', 'margin_refresh'), bot.Config()) == 2
