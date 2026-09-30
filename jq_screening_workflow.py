"""Resumable latest-data screening; all dependencies are supplied by the CLI.

Importing this module performs no I/O. A screening job owns its checkpoint and
pins the margin snapshot explicitly, never relying on an ambient latest result.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import threading
import time
from dataclasses import asdict


SCHEMA = "jq-screening-workflow-v1"
FULL_REFRESH_DAYS = 30


class _MarginRoot:
    """Redirect only the margin resume file, keeping raw objects immutable."""
    def __init__(self, root):
        self.root = root

    def __truediv__(self, name):
        if name == 'margin_collection_resume.json':
            name = 'screening_margin_resume.json'
        return self.root/name


class _MarginStore:
    def __init__(self, store):
        self.store, self.root = store, _MarginRoot(store.root)

    def __getattr__(self, name):
        return getattr(self.store, name)


class _MarginClient:
    def __init__(self, client):
        self.client, self.store = client, _MarginStore(client.store)

    def __getattr__(self, name):
        return getattr(self.client, name)


class ScreeningProgress:
    def __init__(self, logger, interval=30.):
        self.logger, self.interval = logger, interval
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = {"phase": "開始", "completed": 0, "total": 0, "code": None}
        self.started = time.monotonic()

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def report(self):
        with self.lock:
            state = dict(self.state)
        self.logger.info('[最新スクリーニング] %s / 完了=%s/%s / 銘柄=%s / 経過=%.0f秒',
                         state['phase'], state['completed'], state['total'],
                         state.get('code') or '-', time.monotonic()-self.started)

    def _beat(self):
        while not self.stop.wait(self.interval):
            self.report()

    def __enter__(self):
        self.thread = threading.Thread(target=self._beat, name='screening-heartbeat', daemon=True)
        self.thread.start()
        self.report()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def _read(path):
    if not path.exists():
        return None
    value = json.loads(path.read_text('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('workflow_state_must_be_object:'+path.name)
    return value


def _bundle(bot, store, ref):
    return bot.jq_margin_workflow._read_optional_bundle(bot, store, ref)


def _full_at(bot, store, item, kind, api):
    marker = 'last_full_price_at' if kind == 'prices' else 'last_full_financial_at'
    if bot.timestamp(item.get(marker)) is not None:
        return item[marker]
    endpoint = 'equities/bars/daily' if kind == 'prices' else 'fins/summary'
    ref, seen = item.get(kind), set()
    while ref and ref not in seen and len(seen) < 128:
        seen.add(ref)
        blob = _bundle(bot, store, ref)
        params = blob.get('params') or {}
        if (blob.get('status') == 'complete' and blob.get('rows') and blob.get('endpoint') == endpoint
                and isinstance(params, dict) and str(params.get('code')) == api
                and bot.timestamp(blob.get('collected_at')) is not None):
            return blob['collected_at']
        ref = blob.get('parent_ref')
    return None


def _scope(bot, store, globals_):
    master = bot.master_records(bot.bundle_rows(store, globals_['master']))
    targets = [m for m in master if m['instrument_type'] == 'stock' and str(m.get('Mkt')) != '0105']
    excluded = [{**m, 'reason': 'tokyo_pro_market'} for m in master
                if m['instrument_type'] == 'stock' and str(m.get('Mkt')) == '0105']
    return targets, {'kind': 'retail_ordinary', 'included_codes': sorted(m['code'] for m in targets),
                     'excluded': excluded}


def _counts(snap):
    codes = snap.get('screening_scope', {}).get('included_codes', [])
    items = [snap.get('symbols', {}).get(c, {}) for c in codes]
    return {'target_count': len(codes), 'complete_count': sum(i.get('complete') is True for i in items),
            'unavailable_count': sum(i.get('complete') is not True for i in items),
            'financial_unavailable_count': sum(i.get('screening_availability') == 'financial_unavailable' for i in items),
            'collection_failed_count': sum(i.get('screening_availability') == 'collection_failed' for i in items),
            'scope_excluded_count': len(snap.get('screening_scope', {}).get('excluded', []))}


def _full_summary(snap):
    items = [snap.get('symbols', {}).get(c, {}) for c in snap.get('screening_scope', {}).get('included_codes', [])]
    result = {}
    for key in ('last_full_price_at', 'last_full_financial_at'):
        values = [i[key] for i in items if i.get(key)]
        result['oldest_' + key] = min(values) if values else None
        result[key + '_unknown_count'] = len(items)-len(values)
    return result


def _save_draft(bot, store, work):
    draft = copy.deepcopy(work)
    draft.update(snapshot_id=bot.new_id('s_screen_'), schema=bot.CACHE_SCHEMA,
                 created_at=bot.now_jst().isoformat(), screening_stage='staged_not_published')
    draft['information_cutoff'] = draft['created_at']
    count = _counts(draft)
    draft.update(status='complete' if count['unavailable_count'] == 0 else 'partial_universe',
                 pending_count=count['unavailable_count'])
    store.save_snapshot(draft, promote=False)
    return draft


def _due(bot, store, item, kind, api, today, history_years=None):
    at = _full_at(bot, store, item, kind, api)
    marker = 'last_full_price_at' if kind == 'prices' else 'last_full_financial_at'
    if at:
        item[marker] = at
    blob = _bundle(bot, store, item.get(kind))
    timestamp = bot.timestamp(at)
    if kind == 'prices' and item.get('last_full_price_history_years') not in (None, history_years):
        return True
    return (blob.get('status') != 'complete' or not blob.get('rows') or timestamp is None
            or (today-timestamp.date()).days >= FULL_REFRESH_DAYS)


def _corporate_action_due(bot, store, item):
    at = bot.timestamp(item.get('last_full_price_at'))
    if at is None:
        return False
    for row in _bundle(bot, store, item.get('prices')).get('rows', []):
        day = bot.date(row.get('Date'))
        factor = bot.number(row.get('AdjFactor'))
        if day and day > at.date() and (factor is None or factor != 1 or str(row.get('ExRT') or '') not in {'', '0'}):
            return True
    return False


def _fetch(bot, client, endpoint, params, attempts, progress):
    for attempt in range(1, attempts+1):
        progress.update(phase=f'履歴取得 {endpoint}（試行{attempt}/{attempts}・API待機を含む）')
        try:
            blob = client.fetch(endpoint, params)
            if blob.get('status') != 'complete':
                raise bot.CollectionError('incomplete_response:' + endpoint)
            return blob
        except bot.AuthError:
            raise
        except (bot.CollectionError, bot.DataError, OSError, ValueError) as exc:
            if 'daily_self_quota_reached' in str(exc):
                raise
            if attempt == attempts:
                raise
            bot.LOG.warning('[再試行] %s %s / %s', endpoint, attempt, exc)


def _refresh_full(bot, client, work, target, attempts, progress, *, price_only=False):
    store, today = client.store, bot.now_jst().date()
    code, api = target['code'], target['api_code']
    item = work.setdefault('symbols', {}).setdefault(code, {'complete': False})
    deferred = bot.date(item.get('screening_next_retry_date'))
    if deferred is not None and today < deferred and not price_only:
        return
    incomplete = item.get('complete') is not True
    price_due = price_only or incomplete or _due(bot, store, item, 'prices', api, today,
                                               client.cfg.ADJUSTMENT_HISTORY_YEARS)
    financial_due = not price_only and (incomplete or _due(bot, store, item, 'summary', api, today))
    if not price_due and not financial_due:
        return
    # No stale successful bundle can certify a failed refresh in this run.
    item['complete'] = False
    try:
        requests = []
        if price_due:
            t = bot.date(work['price_asof_date'])
            begin = bot.add_months(t, -12*client.cfg.ADJUSTMENT_HISTORY_YEARS)-dt.timedelta(days=10)
            requests.append(('prices', 'equities/bars/daily', {'code': api, 'from': str(begin), 'to': str(t)}))
        if financial_due:
            requests += [('summary', 'fins/summary', {'code': api}), ('earnings', 'fins/earnings-date', {'code': api})]
        for kind, endpoint, params in requests:
            blob = _fetch(bot, client, endpoint, params, attempts, progress)
            item[kind] = store.put(blob)
            item['screening_last_attempt_at'] = blob.get('collected_at') or bot.now_jst().isoformat()
            if kind in {'prices', 'summary'} and not blob.get('rows'):
                item.update(screening_availability='financial_unavailable' if kind == 'summary' else 'price_unavailable',
                            screening_unavailable_reason=kind+'_empty_http200',
                            screening_next_retry_date=str(today+dt.timedelta(days=1)))
                return
            if kind == 'prices':
                item['last_full_price_at'] = item['screening_last_attempt_at']
                item['last_full_price_history_years'] = client.cfg.ADJUSTMENT_HISTORY_YEARS
                item['checked_price_date'] = work['price_asof_date']
            elif kind == 'summary':
                item['last_full_financial_at'] = item['screening_last_attempt_at']
                item['financial_checked_through'] = item['screening_last_attempt_at']
            else:
                item['earnings_checked_through'] = item['screening_last_attempt_at']
        item['complete'] = bool(_bundle(bot, store, item.get('prices')).get('rows')
                                and _bundle(bot, store, item.get('summary')).get('rows'))
        item.update(collected_at=bot.now_jst().isoformat(), screening_availability='available')
        for key in ('screening_unavailable_reason', 'screening_next_retry_date'):
            item.pop(key, None)
    except bot.AuthError:
        raise
    except (bot.CollectionError, bot.DataError, OSError, ValueError) as exc:
        item.update(screening_availability='collection_failed', screening_unavailable_reason=str(exc),
                    screening_last_attempt_at=bot.now_jst().isoformat())
        if 'daily_self_quota_reached' in str(exc):
            raise


def run_screening(bot, args, cfg) -> int:
    """Collect current evidence and publish A/B results under one explicit job."""
    store = bot.Store(args.cache_dir)
    store.ensure()
    resume_path = store.root/'screening_workflow_resume.json'
    status_path = store.root/'screening_workflow_status.json'
    journal, client, published_path = None, None, None
    status = {'schema': SCHEMA, 'status': 'running', 'stage': 'prepare',
              'started_at': bot.now_jst().isoformat(), 'reason': None, 'source_snapshot_id': None,
              'margin_snapshot_id': None, 'run_id': None, 'counts': {}}

    def checkpoint(stage, work=None):
        status.update(stage=stage, updated_at=bot.now_jst().isoformat())
        if work is not None:
            counts = {**_counts(work), **{k: v for k, v in status.get('counts', {}).items() if k.startswith('credit_')}}
            status.update(counts=counts, price_asof_date=work.get('price_asof_date'),
                          information_cutoff=work.get('information_cutoff'), **_full_summary(work))
            journal['work'] = work
        if journal is not None:
            journal['stage'] = stage
            bot.atomic_json(resume_path, journal)
        bot.atomic_json(status_path, status)

    def publication_warning(reason):
        status.update(status='partial', stage='done', reason='result_published_metadata_warning:'+str(reason),
                      run_id=published_path.name, output_path=str(published_path))
        try:
            bot.atomic_json(status_path, status)
        except (OSError, ValueError) as exc:
            bot.LOG.warning('[状態保存警告] %s', exc)
        bot.LOG.warning('[公開済み・保存情報要確認] 結果は公開済みです: %s / %s', published_path, reason)
        return 2

    with bot.ProcessLock(store.root, 'collector'), ScreeningProgress(bot.LOG) as progress:
        try:
            journal = _read(resume_path)
            if journal and journal.get('schema') != SCHEMA:
                raise bot.DataError('screening_resume_schema_mismatch')
            if journal:
                if not isinstance(journal.get('work'), dict) or not journal.get('started_at'):
                    raise bot.DataError('screening_resume_invalid')
                status['started_at'] = journal['started_at']
                if journal.get('config_hash') != bot.digest(asdict(cfg)):
                    if journal.get('history_years') != cfg.ADJUSTMENT_HISTORY_YEARS:
                        for item in journal.get('work', {}).get('symbols', {}).values():
                            item.pop('last_full_price_at', None)
                            item['complete'] = False
                    journal['margin_snapshot_id'] = journal['margin_source_snapshot_id'] = None
                    journal['config_hash'] = bot.digest(asdict(cfg))
                    journal['history_years'] = cfg.ADJUSTMENT_HISTORY_YEARS
                    bot.LOG.info('[再開設定変更] 保存済み履歴を照合し、取得元と信用残の組み合わせを作り直します')
            else:
                journal = {'schema': SCHEMA, 'job_id': bot.new_id('screen_'),
                           'started_at': status['started_at'], 'config_hash': bot.digest(asdict(cfg)),
                           'history_years': cfg.ADJUSTMENT_HISTORY_YEARS,
                           'work': copy.deepcopy(store.snapshot()), 'margin_snapshot_id': None}
            status['job_id'] = journal['job_id']
            checkpoint('prepare')
            key = bot.api_key_from_env_or_ini(args.api_ini)
            client = bot.Client(key, cfg, store)
            work = journal['work'] or {'schema': bot.CACHE_SCHEMA, 'symbols': {}, 'globals': {}}
            # A date change during a long collection is reconciled once; another
            # rollover stops with the checkpoint instead of publishing stale data.
            for cycle in range(2):
                progress.update(phase='最新カレンダー・市場・銘柄一覧の確認')
                globals_, t = bot.collect_globals(client)
                targets, scope = _scope(bot, store, globals_)
                if not targets:
                    raise bot.CollectionError('screening_scope_has_no_ordinary_stocks')
                same_day = journal.get('collection_date') == str(bot.now_jst().date())
                same_scope = work.get('screening_scope') == scope
                same_price = work.get('price_asof_date') == str(t)
                if not (same_day and same_scope and same_price):
                    journal['margin_snapshot_id'] = None
                    journal['margin_source_snapshot_id'] = None
                work.update(globals=globals_, price_asof_date=str(t), screening_scope=scope)
                journal['collection_date'] = str(bot.now_jst().date())
                progress.update(phase='価格・財務の全履歴更新期限を確認', total=len(targets), completed=0)
                for index, target in enumerate(targets, 1):
                    progress.update(code=target['code'], completed=index-1)
                    _refresh_full(bot, client, work, target, max(1, args.max_retries), progress)
                    if index % 10 == 0 or index == len(targets):
                        checkpoint('full_history', work)
                    if index == 1 or index % 20 == 0 or index == len(targets):
                        progress.update(completed=index)
                        progress.report()
                checkpoint('full_history', work)
                source_id = journal.get('margin_source_snapshot_id')
                if not source_id:
                    draft = _save_draft(bot, store, work)
                    source_id = draft['snapshot_id']
                    journal['margin_source_snapshot_id'] = source_id
                status['source_snapshot_id'] = source_id
                checkpoint('margin', work)
                if not journal.get('margin_snapshot_id'):
                    # The dedicated workflow must never resume an unrelated
                    # manual margin job. Its exact source is checked before reuse.
                    pending = _read(store.root/'screening_margin_resume.json')
                    owns_pending = bool(pending and pending.get('source_snapshot_id') == source_id
                                        and pending.get('max_retries') == args.max_retries
                                        and pending.get('selected_codes') == scope['included_codes'])
                    reset = not owns_pending
                    while True:
                        progress.update(phase='信用買残・売残の更新', code=None)
                        margin = bot.jq_margin_workflow.margin_refresh(bot, _MarginClient(client), source_snapshot_id=source_id,
                            only_codes=scope['included_codes'], budget=args.budget or cfg.JQ_DEFAULT_BUDGET,
                            max_retries=args.max_retries, reset_pending=reset)
                        reset = False
                        status['margin_snapshot_id'] = margin['snapshot_id']
                        checkpoint('margin', work)
                        if not margin.get('pending_codes'):
                            journal['margin_snapshot_id'] = margin['snapshot_id']
                            break
                margin_id = journal['margin_snapshot_id']
                status['margin_snapshot_id'] = margin_id
                checkpoint('daily_update', work)
                fresh_globals, fresh_t = bot.collect_globals(client)
                _, fresh_scope = _scope(bot, store, fresh_globals)
                if fresh_t != t or fresh_scope != scope or journal['collection_date'] != str(bot.now_jst().date()):
                    journal['margin_snapshot_id'] = journal['margin_source_snapshot_id'] = None
                    if cycle == 0:
                        continue
                    raise bot.CollectionError('screening_date_changed_during_collection:resume_required')
                progress.update(phase='株価・財務開示・決算予定の差分更新', code=None)
                updated = bot.update_daily(client, source_snapshot=work, global_context=(fresh_globals, fresh_t),
                                           promote=False, progress=progress)
                # New or corrected split evidence warrants a full price check
                # for affected stocks, without re-fetching all financial history.
                ca_targets = [m for m in targets if updated.get('symbols', {}).get(m['code'], {}).get('complete')
                              and _corporate_action_due(bot, store, updated['symbols'][m['code']])]
                if ca_targets:
                    for target in ca_targets:
                        progress.update(phase='企業行動に伴う価格履歴の再確認', code=target['code'])
                        _refresh_full(bot, client, updated, target, args.max_retries, progress, price_only=True)
                    updated = _save_draft(bot, store, updated)
                    updated = bot.update_daily(client, source_snapshot=updated, global_context=(fresh_globals, fresh_t),
                                               promote=False, progress=progress)
                work = updated
                checkpoint('verify', work)
                latest_globals, latest_t = bot.collect_globals(client)
                _, latest_scope = _scope(bot, store, latest_globals)
                if latest_t != fresh_t or latest_scope != scope or journal['collection_date'] != str(bot.now_jst().date()):
                    journal['margin_snapshot_id'] = journal['margin_source_snapshot_id'] = None
                    if cycle == 0:
                        continue
                    raise bot.CollectionError('screening_date_changed_during_collection:resume_required')
                break
            margin = bot.jq_margin_workflow.load_margin_snapshot(bot, store, margin_id)
            count = _counts(work)
            margin_items = list(margin.get('symbols', {}).values())
            count.update(credit_completed_count=sum(i.get('last_attempt_status') == 'complete' for i in margin_items),
                         credit_failed_count=len(margin.get('terminal_failed_codes', [])),
                         credit_pending_count=len(margin.get('pending_codes', [])),
                         credit_empty_count=sum(i.get('adapter_status') == 'no_rows' for i in margin_items))
            partial = count['unavailable_count'] > 0 or margin.get('status') != 'complete'
            status.update(status='running', counts=count,
                          reason='some_inputs_unavailable' if partial else None,
                          price_asof_date=work['price_asof_date'], information_cutoff=work['information_cutoff'])
            final = copy.deepcopy(work)
            status['source_snapshot_id'] = bot.new_id('s_screen_')
            final.update(snapshot_id=status['source_snapshot_id'], parent_snapshot_id=work.get('snapshot_id'),
                         screening_workflow={**copy.deepcopy(status), 'collection_status': 'partial' if partial else 'complete'},
                         status='partial_universe' if partial else 'complete', screening_stage='verified_for_analysis')
            store.save_snapshot(final, promote=False)
            checkpoint('analyze', final)
            progress.update(phase='業種比較・A/B候補のオフライン分析', code=None)
            path = bot.analyze_store(store, cfg, args.output_dir, args.top, final['snapshot_id'],
                                     None, None, margin_id, require_current=True)
            published_path = path
            manifest = json.loads((path/'run_manifest.json').read_text('utf-8'))
            status.update(status='partial' if partial else 'complete', stage='done', run_id=manifest['run_id'], output_path=str(path),
                          **_full_summary(final))
            if manifest.get('run_status') != 'complete':
                status.update(status='partial', reason='some_inputs_unavailable')
            bot.atomic_json(store.root/'latest_snapshot.json', {'snapshot_id': final['snapshot_id']})
            bot.atomic_json(status_path, status)
            resume_path.unlink(missing_ok=True)
            bot.LOG.info('[更新・分析完了] 対象=%s 取得済=%s 未評価=%s / %s',
                         count['target_count'], count['complete_count'], count['unavailable_count'], path)
            print('[A・Bレポート] '+str(path/'report_registration_top10.md'), flush=True)
            return 2 if status['status'] == 'partial' else 0
        except (KeyboardInterrupt, SystemExit):
            if published_path is not None:
                return publication_warning('interrupted_after_publication')
            status.update(status='interrupted', reason='user_interrupted')
            checkpoint(status['stage'], locals().get('work'))
            bot.LOG.warning('[中断] 完了済みの履歴取得を保存しました。最新スクリーニングで再開できます')
            return 130
        except (bot.CollectionError, bot.DataError, OSError, ValueError, KeyError, TypeError) as exc:
            if published_path is not None:
                return publication_warning(exc)
            status.update(status='failed', reason=str(exc))
            checkpoint(status['stage'], locals().get('work'))
            bot.LOG.error('[更新停止] %s。直前の公開済みランキングは保持します', exc)
            return 1
        finally:
            if client is not None:
                client.close()
