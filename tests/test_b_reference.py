"""Manual B references preserve the strict decisions and expose unmet conditions."""
import copy

import pytest

from jq_universe import evaluate, from_mapping
from jq_universe_report import groups, markdown_report, summarize
from test_universe_engine import candidate
from test_universe_report import metadata


def reference_input(**changes):
    row = candidate(name='参考企業', margin_short_to_volume_20bd=.08,
                    ma200_state='below_ma200_downtrend', below_ma200_downtrend=True,
                    return_63d=-.1, return_rel_topix_63d=-.15, turnover_persistence=.8)
    for n in (5, 20):
        key = f'margin_short_change_{n}bd_pct'
        row[key] = None
        row['margin_metric_states'][key] = 'unavailable'
        row['margin_metric_reasons'][key] = 'margin_history_insufficient'
    row.update(changes)
    return row


def scored(**changes):
    row = reference_input(**changes)
    return {**row, **evaluate(row)}


def test_reference_keeps_strict_failure_and_scores_only_complete_fundamental_components():
    row = reference_input()
    before = copy.deepcopy(row)
    result = evaluate(row)
    assert row == before
    assert result['registration_b_status'] == 'excluded'
    assert result['registration_b_eligible'] is False
    assert result['registration_b_score'] is None
    assert result['registration_b_reference_candidate'] is True
    parts = result['registration_b_components']
    assert result['registration_b_reference_score'] == sum(parts[k] for k in ('quality', 'growth', 'valuation'))
    assert result['registration_b_reference_score_max'] == 75
    assert result['registration_is_order_permission'] is False
    assert set(result['registration_b_reference_cautions']) == {
        'short_balance_to_volume_below_threshold', 'short_evidence_missing:short_change_pct',
        'price_condition:downtrend_wait', 'momentum_wait:return_63d',
        'momentum_wait:return_rel_topix_63d', 'momentum_wait:turnover_persistence'}


@pytest.mark.parametrize('changes', [
    {'sales_cagr': .09}, {'sales_cagr': None}, {'op_income_yoy_pct': 14.},
    {'forward_np_change': .14}, {'guidance_status': 'unknown'},
    {'summary_quality_score': 4}, {'summary_quality_available': 6},
    {'Q1': False}, {'Q2': False}, {'op_income_stable': False},
    {'liquidity_ok': False}, {'market_cap_ok': False}, {'annual_comparable': False},
    {'annual_stale': True}, {'critical_missing_count': 1}, {'ps': 11.}, {'per': 101.},
    {'confirmed_severe_risk': True}, {'earnings_quality_flag': 'watch'},
    {'accounting_risk': 'watch'}, {'governance_status': 'watch'}, {'earnings_blackout': True},
    {'peg_warnings': ['eps_growth_too_high_oneoff_risk']},
    {'common_issues': ['symbol_collection_incomplete']},
    {'financial_issues': ['currency_unknown_or_not_jpy']},
    {'registration_price_age_bd': 4}, {'registration_price_age_bd': None},
    {'ma200_state': 'ma200_unknown', 'below_ma200_downtrend': False},
    {'return_63d': None}, {'turnover_persistence': None},
    {'margin_diagnostic_status': 'collection_failed'}, {'margin_frequency': 'weekly'},
    {'margin_application_age_bd': 4}, {'margin_available_at': '2026-11-03T18:00:00+09:00'},
    {'margin_after_price_asof': True}, {'margin_short': 0},
    {'margin_short_to_volume_20bd': None},
])
def test_financial_risk_missing_inputs_and_unverified_credit_cannot_become_references(changes):
    result = evaluate(reference_input(**changes))
    assert result['registration_b_reference_candidate'] is False
    assert result['registration_b_reference_score'] is None
    assert result['registration_b_reference_blockers']


@pytest.mark.parametrize('reason', ['corporate_action_in_window', 'calendar_unverified',
                                  'zero_baseline', 'not_observed_by_cutoff', None])
def test_only_history_shortage_can_waive_missing_credit_change(reason):
    row = reference_input()
    row['margin_metric_reasons']['margin_short_change_5bd_pct'] = reason
    result = evaluate(row)
    assert result['registration_b_reference_candidate'] is False
    assert 'short_evidence_missing:short_change_pct' in result['registration_b_reference_blockers']


def test_known_declining_credit_is_explicit_manual_caution():
    result = evaluate(candidate(margin_short_change_20bd_pct=-5.))
    assert result['registration_b_status'] == 'excluded'
    assert result['registration_b_reference_candidate'] is True
    assert result['registration_b_reference_cautions'] == ['short_balance_not_accumulating']


def test_strict_eligible_rows_do_not_repeat_in_references():
    result = evaluate(candidate())
    assert result['registration_b_eligible'] is True
    assert result['registration_b_reference_candidate'] is False


def test_reference_score_respects_configured_weights_without_redistribution():
    rules = from_mapping({'B_WEIGHT_GROWTH': 35., 'B_WEIGHT_TECHNICAL': 25.})
    result = evaluate(reference_input(), rules)
    assert result['registration_b_reference_candidate'] is True
    assert result['registration_b_reference_score_max'] == 70
    assert result['registration_b_reference_score'] == sum(result['registration_b_reference_components'].values())


def test_references_have_stable_separate_rankings_and_never_fill_b_eligible_csv():
    rows = [scored(code='2000'), scored(code='1000'), scored(code='3000')]
    before = copy.deepcopy(rows)
    first = groups(rows)
    assert groups(list(reversed(rows))) == first
    assert [r['code'] for r in first['top_registration_reference_b.csv']] == ['1000', '2000']
    assert len(first['registration_reference_b.csv']) == 3
    assert first['top_registration_b.csv'] == first['registration_candidates_b.csv'] == []
    assert summarize(rows)['b']['reference_count'] == 3
    assert len(groups(rows, 1)['top_registration_reference_b.csv']) == 1
    assert groups([], 10)['top_registration_reference_b.csv'] == []
    assert len(groups(rows[:1])['top_registration_reference_b.csv']) == 1
    assert rows == before


def test_reference_report_exposes_missing_history_weak_price_and_partial_score():
    row = scored()
    row['registration_margin_diagnostic'] = {k: v for k, v in row.items() if k.startswith('margin_')}
    row['margin_short'] = 999999999  # An unrelated regular source must not override the selected one.
    report = markdown_report([row], metadata())
    assert '人が確認するB参考候補' in report
    assert '通過 1件 / 表示 1件' in report
    assert '正式なB適格候補' in report and '適格 0件' in report
    assert '500,000株' in report and '999,999,999' not in report
    assert ' / 75' in report and '欠測分の再配点もしません' in report
    assert '下降・条件待ち' in report and '-10.0%' in report
    assert '信用履歴不足' in report and '信用売残の根拠不足' in report
    assert 'top_registration_reference_b.csv' in report


def test_reference_names_and_cautions_are_escaped():
    row = scored(name='参考|<script>\n別行')
    row['registration_b_reference_cautions'].append('raw|<script>')
    report = markdown_report([row], metadata())
    assert '参考\\|&lt;script&gt; 別行' in report
    assert 'raw\\|&lt;script&gt;' in report
    assert '<script>' not in report
