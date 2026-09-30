"""Offline A/B integration, input clocks, replay, and legacy comparison."""
import copy
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import JQuamtsScreeningBot as bot
import jq_margin_workflow as workflow
from test_margin_score_isolation import make_snapshot, SCORE_FIELDS


def current_snapshot(tmp_path):
    store, snap = make_snapshot(tmp_path)
    day = dt.date(2026, 10, 27)
    cutoff = '2026-10-27T19:00:00+09:00'
    observed = '2026-10-27T18:00:00+09:00'
    snap.update(price_asof_date=str(day), information_cutoff=cutoff, snapshot_id='ab_current')
    put = lambda rows: store.put({'status':'complete', 'rows':rows, 'collected_at':observed})
    days = pd.date_range('2021-03-31', '2026-10-30')
    sessions = days[days.dayofweek < 5]
    past = sessions[sessions <= pd.Timestamp(day)]
    snap['globals']['calendar'] = put([{'Date':str(d.date()), 'HolDiv':'1' if d.dayofweek < 5 else '0'} for d in days])
    snap['globals']['topix'] = put([{'Date':str(d.date()), 'C':2000+i} for i,d in enumerate(past)])
    master = store.get(snap['globals']['master'])['rows']
    for row in master:
        row['Date'] = str(day)
    snap['globals']['master'] = put(master)
    for code,item in snap['symbols'].items():
        prices = np.full(len(past), 1000.)
        prices[-80:] = np.linspace(1001., 1120., 80)
        item['prices'] = put([{'Date':str(d.date()), 'Code':code+'0', 'O':p, 'H':p+10, 'L':p-10,
            'C':p, 'Vo':1_000_000, 'Va':p*1_000_000, 'AdjFactor':1, 'ExRT':'0', 'MktCap':100_000}
            for d,p in zip(past,prices)])
        statements = store.get(item['summary'])['rows']
        for n,row in enumerate(statements):
            row.update(Sales=100e9*1.25**n, OP=10e9*1.30**n, OdP=10e9*1.30**n, NP=6e9*1.25**n,
                       CFO=10e9*1.3**n, NxFNp=6e9*1.25**(n+1))
        item.update(summary=put(statements), collected_at=observed,
                    financial_checked_through=observed, earnings_checked_through=observed,
                    earnings=put([]), short=put([]))
        item['margin'] = put(margin_rows(code))
    return store,snap


def margin_rows(code):
    rows=[]
    for i,d in enumerate(pd.bdate_range('2026-09-25','2026-10-26')):
        short=2_000_000+i*50_000
        long=4_000_000
        rows.append({'Code':code+'0', 'Date':str(d.date()), 'PubDate':str((d+pd.offsets.BDay()).date()),
            'IssType':'2', 'LongVol':long, 'ShrtVol':short,
            'LongNegVol':long//2, 'LongStdVol':long//2,
            'ShrtNegVol':short//2, 'ShrtStdVol':short//2,
            'LongVal':long*1000, 'ShrtVal':short*1000,
            'LongNegVal':long*500, 'LongStdVal':long*500,
            'ShrtNegVal':short*500, 'ShrtStdVal':short*500})
    return rows


def save_margin_snapshot(store,snap,ident,code='1001',status='complete',rows=None,observed='2026-10-27T18:00:00+09:00'):
    bundle={'status':status,'rows':margin_rows(code) if rows is None else rows,'collected_at':observed}
    ref=store.put(bundle)
    payload={'schema':workflow.MARGIN_SNAPSHOT_SCHEMA,'snapshot_id':ident,
             'source_snapshot_id':snap['snapshot_id'], 'information_cutoff':observed,
             'status':status,'selected_codes':[code], 'targets':{code:{'api_code':code+'0'}},
             'symbols':{code:{'margin_ref':ref,'last_attempt_status':status}}}
    bot.atomic_json(store.root/'margin_snapshots'/(ident+'.json'),payload)
    bot.atomic_json(store.root/'latest_margin_snapshot.json',{'snapshot_id':ident})
    return payload


def project_legacy(rows):
    return [{k:r.get(k) for k in SCORE_FIELDS} for r in rows]


def test_continuing_trend_is_a_registration_candidate_and_legacy_stays_watch(tmp_path):
    store,snap=current_snapshot(tmp_path)
    rows,stats,meta=bot.score_snapshot(store,snap,bot.Config())
    assert meta['run_status']=='complete'
    assert all(r['ma200_state']=='above_ma200_near' for r in rows)
    assert all(r['eligible_new_entry'] is False for r in rows)
    assert all(r['registration_a_eligible'] is True for r in rows)
    assert all(r['registration_b_eligible'] is True for r in rows)
    assert meta['registration_price_age_bd']==0
    again,stats2,meta2=bot.score_snapshot(store,snap,bot.Config())
    assert (rows,stats,meta)==(again,stats2,meta2)


def test_explicit_margin_changes_only_registration_source_and_never_legacy(tmp_path):
    store,snap=current_snapshot(tmp_path)
    original=copy.deepcopy(snap)
    base,_,_=bot.score_snapshot(store,snap,bot.Config())
    save_margin_snapshot(store,snap,'failed_margin',status='collection_failed',rows=[])
    automatic,_,automatic_meta=bot.score_snapshot(store,snap,bot.Config())
    assert automatic==base  # A latest pointer alone is never an opt-in.
    assert automatic_meta['registration_margin_snapshot_id'] is None
    selected,_,meta=bot.score_snapshot(store,snap,bot.Config(),margin_snapshot_id='latest')
    assert project_legacy(selected)==project_legacy(base)
    assert meta['registration_margin_snapshot_id']=='failed_margin'
    assert meta['registration_margin_snapshot_hash']
    by_code={r['code']:r for r in selected}
    assert by_code['1001']['registration_margin_diagnostic']['margin_diagnostic_status']=='collection_failed'
    assert by_code['1002']['registration_margin_diagnostic']['margin_diagnostic_status']=='not_selected'
    assert all(r['registration_a_eligible'] for r in selected)
    assert not any(r['registration_b_eligible'] for r in selected)
    assert all(r['margin_diagnostic_status']=='available' for r in selected)
    assert snap==original


def test_later_observations_cannot_extend_screening_cutoff(tmp_path):
    store,snap=current_snapshot(tmp_path)
    save_margin_snapshot(store,snap,'later_observation',observed='2026-10-28T18:00:00+09:00')
    rows,_,meta=bot.score_snapshot(store,snap,bot.Config(),margin_snapshot_id='later_observation')
    assert meta['information_cutoff']==snap['information_cutoff']
    assert not any(r['registration_b_eligible'] for r in rows)
    first=next(r for r in rows if r['code']=='1001')
    assert first['registration_margin_diagnostic']['margin_diagnostic_status']=='not_available_by_cutoff'
    with pytest.raises(bot.DataError,match='information_cutoff_after_snapshot_verification'):
        bot.score_snapshot(store,snap,bot.Config(),cutoff=bot.timestamp('2026-10-28T19:00:00+09:00'),
                           margin_snapshot_id='later_observation')


def test_new_margin_with_old_price_is_not_a_supply_signal(tmp_path):
    store,snap=current_snapshot(tmp_path)
    code='1001'
    future=margin_rows(code)[-1]
    future.update(Date='2026-10-28',PubDate='2026-10-29')
    snap['information_cutoff']='2026-10-29T19:00:00+09:00'
    save_margin_snapshot(store,snap,'newer_credit',rows=[future],observed='2026-10-29T18:00:00+09:00')
    rows,_,_=bot.score_snapshot(store,snap,bot.Config(),margin_snapshot_id='newer_credit')
    first=next(r for r in rows if r['code']==code)
    assert first['registration_margin_diagnostic']['margin_after_price_asof'] is True
    assert first['registration_b_eligible'] is False
    assert first['registration_supply_state']=='unassessed'


def test_old_price_uses_frozen_clock_and_holds_new_candidates(tmp_path):
    store,snap=make_snapshot(tmp_path)
    rows,_,meta=bot.score_snapshot(store,snap,bot.Config())
    assert not any(r['registration_a_eligible'] or r['registration_b_eligible'] for r in rows)
    # The fixture omits closed days, so new freshness certification is unknown.
    assert meta['registration_price_age_bd'] is None
    full=[{'Date':str(d.date()),'HolDiv':'1' if d.dayofweek<5 else '0'}
          for d in pd.date_range('2021-03-31','2026-10-30')]
    snap['globals']['calendar']=store.put({'status':'complete','rows':full})
    rows,_,meta=bot.score_snapshot(store,snap,bot.Config())
    assert meta['registration_price_age_bd']>3
    assert not any(r['registration_a_eligible'] or r['registration_b_eligible'] for r in rows)


def test_outputs_have_separate_rankings_audits_and_single_uses_saved_results(tmp_path,monkeypatch):
    store,snap=current_snapshot(tmp_path)
    rows,stats,meta=bot.score_snapshot(store,snap,bot.Config())
    output=tmp_path/'out'
    destination=bot.write_run(rows,stats,meta,output,top=2,run_id='ab_result')
    for route in ('a','b'):
        table=pd.read_csv(destination/f'top_registration_{route}.csv')
        assert len(table)==2
        assert table[f'registration_{route}_eligible'].all()
        assert list(table[f'registration_{route}_rank'])==[1,2]
    assert (destination/'legacy_top_recommended_core.csv').read_bytes()==(destination/'top_recommended_core.csv').read_bytes()
    manifest=json.loads((destination/'run_manifest.json').read_text('utf-8'))
    assert manifest['legacy_logic_version']==bot.LOGIC_VERSION
    assert manifest['registration_logic_version']==bot.REGISTRATION_LOGIC_VERSION
    assert 'jq_universe.py' in manifest['source_sha256_by_file']
    for name,expected in manifest['outputs'].items():
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest()==expected
    monkeypatch.setattr(bot,'score_snapshot',lambda *a,**k:pytest.fail('single must not rescore'))
    single=bot.single_from_run('1001',output)
    original=next(r for r in rows if r['code']=='1001')
    assert single['registration_a_score']==original['registration_a_score']
    assert (destination/'report_registration_top10.md').is_file()
    with pytest.raises(bot.DataError,match='immutable_run_already_exists'):
        bot.write_run(rows,stats,meta,output,run_id='ab_result')
