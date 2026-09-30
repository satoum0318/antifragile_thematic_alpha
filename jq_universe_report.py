"""Read-only projections of the A/B registration-candidate results."""
from __future__ import annotations

import html
import math
from collections import Counter
from typing import Any

import jq_universe


LABELS = {
    "eligible": "適格", "hold": "保留", "excluded": "対象外",
    "confirmed": "成長根拠を確認", "insufficient": "成長根拠不足",
    "mixed": "成長指標が混在", "contradicted": "成長条件と矛盾",
    "unassessed": "未評価", "caution": "注意", "observed": "需給指標を確認",
    "trend_transition": "上昇への転換局面", "uptrend_continuation": "上昇継続",
    "basing_wait": "底固め・条件待ち", "overheated_wait": "過熱・条件待ち",
    "downtrend_wait": "下降・条件待ち",
    "quality": "財務品質", "growth": "成長", "valuation": "評価額", "technical": "価格局面", "supply": "信用需給",
    "supply_unassessed": "信用需給は未評価", "price_condition": "価格の条件待ち",
    "momentum_wait": "モメンタムの条件待ち", "momentum_unknown": "モメンタム未評価",
    "short_evidence_missing": "信用売残の根拠不足", "growth_below_threshold": "成長条件未達",
    "score_components_unavailable": "点数に必要な情報が不足",
    "short_balance_not_positive": "信用売残が正の値ではない",
    "short_balance_to_volume_below_threshold": "出来高に対する信用売残が条件未達",
    "short_balance_not_accumulating": "信用売残の積み上がりを未確認",
    "sales_cagr": "売上成長率", "return_63d": "63営業日の株価変化",
    "return_rel_topix_63d": "63営業日のTOPIX相対変化", "turnover_persistence": "出来高の継続性",
    "short_change_pct": "信用売残の変化率",
    "available": "確認可能", "unavailable": "未評価", "unknown": "未評価",
    "not_collected": "未取得", "no_rows": "該当する信用残なし",
    "not_applicable": "信用残の対象外", "collection_failed": "取得失敗",
    "daily": "日次", "weekly": "週次", "finite": "倍率算出可能",
    "zero_short_balance": "売残ゼロ・倍率未定義", "both_zero": "買残・売残ともゼロ",
    "ma200_reclaim": "200日線を再奪還", "above_ma200_near": "200日線より上",
    "above_ma200_extended": "200日線から上方乖離", "below_ma200_basing": "200日線より下で底固め",
    "below_ma200_downtrend": "200日線より下で下降", "ma200_unknown": "価格局面未評価",
    "corporate_action_in_window": "対象期間に株数調整あり",
    "corporate_action_unverified": "対象期間の株数調整を未確認",
    "corporate_action_basis_unknown": "株数調整の基準不明",
    "daily_history_insufficient": "日次履歴不足", "margin_history_insufficient": "信用履歴不足",
    "volume_history_insufficient": "出来高履歴不足", "zero_volume_denominator": "出来高の分母がゼロ",
    "zero_baseline": "比較開始時の残高がゼロ", "calendar_unverified": "営業日未確認",
    "not_available_by_cutoff": "締切時点では未確認", "not_observed_by_cutoff": "締切時点では未取得",
    "not_published_by_cutoff": "締切時点では未公表", "observation_time_unknown": "取得時刻不明",
    "financial_quality_failed": "財務品質の条件未達", "forward_decline_or_loss": "会社予想の悪化・赤字",
    "confirmed_severe_risk": "確認済みの重大リスク", "earnings_blackout": "決算前後の保留期間",
    "liquidity_failed": "流動性の条件未達", "market_cap_failed": "企業規模の条件未達",
    "model_scope_financial": "金融業のため一般企業モデル対象外",
}


def _items(value: Any) -> list:
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _label(value: Any) -> str:
    if value is None:
        return "未評価"
    text = str(value)
    if text in LABELS:
        return LABELS[text]
    if ":" in text:
        prefix, detail = text.split(":", 1)
        if prefix in LABELS:
            return LABELS[prefix] + "：" + LABELS.get(detail, detail)
    return text


def _escape(value: Any) -> str:
    text = str(value if value is not None else "未評価").replace("\r", " ").replace("\n", " ")
    return html.escape(text, quote=False).replace("|", "\\|").replace("`", "\\`")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _number(value)
    return "未評価" if number is None else f"{number:,.{digits}f}{suffix}"


def _percent(value: Any, *, fraction: bool = False) -> str:
    number = _number(value)
    return _fmt(number * 100 if number is not None and fraction else number, 1, "%")


def _reasons(value: Any) -> str:
    reasons = [_label(item) for item in _items(value)]
    return "、".join(reasons) if reasons else "記録なし"


def _valid_top(top: int) -> None:
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ValueError("top must be a positive integer")


def _eligible(record: dict, lane: str) -> bool:
    return record.get(f"registration_{lane}_eligible") is True


def _review(record: dict) -> bool:
    return (not _eligible(record, "a") and not _eligible(record, "b")
            and any(record.get(f"registration_{lane}_status") == "hold" for lane in ("a", "b")))


def groups(records: list[dict], top: int = 10) -> dict[str, list[dict]]:
    """Filter and sort existing scores, without changing or scoring input rows."""
    _valid_top(top)
    result = {}
    for lane in ("a", "b"):
        candidates = sorted((row for row in records if _eligible(row, lane)),
                            key=lambda row: jq_universe.rank_key(row, lane))
        result[f"top_registration_{lane}.csv"] = candidates[:top]
        result[f"registration_candidates_{lane}.csv"] = candidates
    result["registration_review.csv"] = sorted((row for row in records if _review(row)),
                                                key=lambda row: str(row.get("code", "")))
    return result


def summarize(records: list[dict]) -> dict:
    """Return counts from stored decisions; unknown states remain unknown."""
    result = {"record_count": len(records), "review_count": sum(_review(row) for row in records),
              "overlap_count": sum(_eligible(row, "a") and _eligible(row, "b") for row in records)}
    for lane in ("a", "b"):
        prefix = f"registration_{lane}_"
        statuses = Counter(row.get(prefix + "status") or "unknown" for row in records)
        result[lane] = {
            "eligible_count": sum(_eligible(row, lane) for row in records),
            "hold_count": statuses["hold"], "excluded_count": statuses["excluded"],
            "status_counts": dict(sorted(statuses.items())),
            "reason_counts": dict(sorted(Counter(str(reason) for row in records
                                                  for reason in _items(row.get(prefix + "reasons"))).items())),
            "warning_counts": dict(sorted(Counter(str(reason) for row in records
                                                   for reason in _items(row.get(prefix + "warnings"))).items())),
        }
    for field in ("supply", "price"):
        result[field + "_state_counts"] = dict(sorted(Counter(
            row.get(f"registration_{field}_state") or "unknown" for row in records).items()))
    return result


def _margin(record: dict) -> dict:
    selected = record.get("registration_margin_diagnostic")
    # An explicitly selected empty diagnostic must not revive old margin values.
    return selected if isinstance(selected, dict) else record


def _metric(margin: dict, key: str, *, digits: int = 0, suffix: str = "") -> str:
    value = _fmt(margin.get(key), digits, suffix)
    if value != "未評価":
        return value
    reason = (margin.get("margin_metric_reasons") or {}).get(key)
    return f"未評価（{_label(reason)}）" if reason else value


def _change(margin: dict, n: int, side: str) -> str:
    absolute = _metric(margin, f"margin_{side}_change_{n}bd_vol", suffix="株")
    percentage = _metric(margin, f"margin_{side}_change_{n}bd_pct", digits=1, suffix="%")
    return f"{absolute}（{percentage}）"


def _growth(record: dict, lane: str) -> str:
    return (f"{_label(record.get(f'registration_growth_{lane}_state'))} / "
            f"売上CAGR {_percent(record.get('sales_cagr'), fraction=True)} / "
            f"営業利益前年比 {_percent(record.get('op_income_yoy_pct'))} / "
            f"予想利益変化 {_percent(record.get('forward_np_change'), fraction=True)}")


def _table(lines: list[str], headers: list[str], rows: list[list]) -> None:
    lines.append("|" + "|".join(_escape(value) for value in headers) + "|")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("|" + "|".join(_escape(value) for value in row) + "|" for row in rows)
    lines.append("")


def markdown_report(records: list[dict], meta: dict, top: int = 10) -> str:
    """Render independent A/B candidates and their observed evidence in Japanese."""
    selected, counts = groups(records, top), summarize(records)
    lines = ["# 売買Botへの登録検討候補", "",
             f"価格基準日: {_escape(meta.get('price_asof_date'))} / 情報締切: {_escape(meta.get('information_cutoff'))}",
             f"実行ID: {_escape(meta.get('run_id'))} / 入力状態: {_escape(meta.get('run_status'))}",
             f"登録候補仕様: {_escape(meta.get('registration_logic_version'))} / 旧比較仕様: {_escape(meta.get('legacy_logic_version'))}",
             f"信用入力: {_escape(meta.get('registration_margin_input_mode'))} / 信用snapshot: {_escape(meta.get('registration_margin_snapshot_id'))}",
             f"信用取得元snapshot: {_escape(meta.get('registration_margin_source_snapshot_id'))} / 価格経過営業日: {_escape(meta.get('registration_price_age_bd'))}",
             "", "銘柄選定と資金配分はご自身で管理し、売買タイミングは別の売買Botが判断します。候補入りは注文許可ではありません。",
             "主軸の評価期間は6〜12か月、補助は1〜3年です。最低保有期間や利益を保証するものではありません。",
             "初期RCの選定ルールで、運用成績の検証は未完了です。未評価・欠測は安全を意味しません。", "",
             f"A適格 {counts['a']['eligible_count']}件 / B適格 {counts['b']['eligible_count']}件 / 両枠適格 {counts['overlap_count']}件 / 保留のみ {counts['review_count']}件。上限まで候補を水増ししません。", ""]
    if meta.get("global_issues"):
        lines += ["入力全体の確認事項: " + _escape(_reasons(meta["global_issues"])), ""]
    workflow = meta.get("screening_workflow") or {}
    if meta.get("screening_scope"):
        scope_counts = workflow.get("counts") or {}
        lines += [
            "対象: 一般投資家向け市場の普通株 / TOKYO PRO MARKETは対象外。",
            "収集対象 {}件 / 取得済み {}件 / 未評価 {}件 / 市場による対象外 {}件。".format(
                _escape(scope_counts.get("target_count")), _escape(scope_counts.get("complete_count")),
                _escape(scope_counts.get("unavailable_count")), _escape(scope_counts.get("scope_excluded_count"))),
            "信用取得: 完了 {}件 / 失敗 {}件 / 未処理 {}件。".format(
                _escape(scope_counts.get("credit_completed_count")), _escape(scope_counts.get("credit_failed_count")),
                _escape(scope_counts.get("credit_pending_count"))),
            "財務全履歴の最終取得（対象内の最古）: " + _escape(workflow.get("oldest_last_full_financial_at")),
            "未評価の理由は screening_unavailable.csv、市場による対象外は scope_excluded.csv を参照してください。", ""]
    lines += ["信用取得の情報締切: " + _escape(meta.get("registration_margin_snapshot_cutoff")), ""]
    for lane, title in (("a", "A：割安で財務と成長根拠がある企業"),
                        ("b", "B：高成長と信用売残を確認するスイング候補")):
        candidates = selected[f"top_registration_{lane}.csv"]
        lines += [f"## {title} — 上位{top}件", ""]
        lines += [f"適格 {counts[lane]['eligible_count']}件 / 保留 {counts[lane]['hold_count']}件 / 対象外 {counts[lane]['excluded_count']}件。", ""]
        if not candidates:
            lines += ["候補なし。欠測や条件未達を埋めて推奨することはしません。", ""]
            reasons = sorted(counts[lane]["reason_counts"].items(), key=lambda item: (-item[1], item[0]))[:10]
            if reasons:
                lines += ["主な判定理由: " + _escape("、".join(f"{_label(reason)} {count}件" for reason, count in reasons)), ""]
            else:
                lines += ["判定理由の記録なし。入力件数と採点済みデータを確認してください。", ""]
            continue
        overview = []
        for index, record in enumerate(candidates, 1):
            old = "旧適格" if record.get("eligible_new_entry") is True else "旧非適格" if record.get("eligible_new_entry") is False else "旧未評価"
            overview.append([
                record.get(f"registration_{lane}_rank") or index, record.get("code"), record.get("name"),
                _fmt(record.get(f"registration_{lane}_score")), _growth(record, lane),
                _fmt(record.get("ps")) + " / " + _fmt(record.get("per")),
                _label(record.get("registration_price_state")), _label(record.get("registration_supply_state")),
                old + " / 点 " + _fmt(record.get("recommendation_score")) + " / " + _label(record.get("candidate_lane")),
            ])
        _table(lines, ["順位", "コード", "銘柄", "登録候補点", "成長根拠", "PS / PER", "価格局面", "信用需給", "旧方式との比較"], overview)
        lines += ["### 採用根拠と懸念点", ""]
        for record in candidates:
            warnings = []
            for field in (f"registration_{lane}_warnings", "registration_supply_reasons", "registration_supply_warnings", "registration_price_warnings"):
                warnings.extend(str(item) for item in _items(record.get(field)))
            warnings = list(dict.fromkeys(warnings))
            components = record.get(f"registration_{lane}_components") or {}
            component_text = "、".join(f"{_label(key)} {_fmt(value)}" for key, value in components.items()) if isinstance(components, dict) else str(components)
            grounds = ("財務・成長・割安性の条件を通過" if lane == "a" else "財務・成長・信用売残・モメンタムの条件を通過")
            if record.get(f"registration_{lane}_reasons"):
                grounds += " / " + _reasons(record[f"registration_{lane}_reasons"])
            lines += [f"- **{_escape(record.get('code'))} {_escape(record.get('name'))}**：採用根拠 {_escape(grounds)}。"
                      f" 財務品質 {_escape(record.get('summary_quality_score'))}/7（確認項目 {_escape(record.get('summary_quality_available'))}/7）。"
                      f" 懸念・未評価: {_escape(_reasons(warnings))}。"]
            if component_text:
                lines += [f"  点数内訳: {_escape(component_text)}。"]
        lines += ["", "### 信用買残・売残の観測根拠", "",
                  "変化は株数差分（変化率）。買残・売残の多さだけで方向や安全性を判断しません。", ""]
        margin_rows = []
        for record in candidates:
            margin = _margin(record)
            ratio = _metric(margin, "margin_ratio", digits=2) if margin.get("margin_ratio_state") == "finite" else _label(margin.get("margin_ratio_state"))
            freshness = (f"{_label(margin.get('margin_diagnostic_status'))} / {_label(margin.get('margin_frequency'))} / "
                         f"申込から {_fmt(margin.get('margin_application_age_bd'), 0)}営業日")
            if margin.get("margin_after_price_asof") is True:
                freshness += " / 価格基準日より後の残高・判定使用不可"
            margin_rows.append([
                record.get("code"), _metric(margin, "margin_long", suffix="株"), _metric(margin, "margin_short", suffix="株"), ratio,
                "買 " + _change(margin, 5, "long") + " / 売 " + _change(margin, 5, "short"),
                "買 " + _change(margin, 20, "long") + " / 売 " + _change(margin, 20, "short"),
                "買 " + _metric(margin, "margin_long_to_volume_20bd", digits=2) + " / 売 " + _metric(margin, "margin_short_to_volume_20bd", digits=2),
                "申込 " + str(margin.get("margin_reference_date") or "未評価") + " / 公表 " + str(margin.get("margin_pub_date") or "未評価")
                + " / 観測 " + str(margin.get("margin_available_at") or "未評価"), freshness,
            ])
        _table(lines, ["コード", "買残", "売残", "買残 / 売残", "5営業日変化", "20営業日変化", "20日平均出来高との比（日）", "残高の時点", "鮮度・確認状態"], margin_rows)
    lines += ["## 保留と読み方", "",
              f"A・Bとも不適格で、少なくとも一方が保留の銘柄: {counts['review_count']}件。詳細は registration_review.csv を参照してください。",
              "旧方式の適格・推奨点は比較用です。新しい登録候補点と同じ尺度ではありません。",
              "信用の時点不整合、履歴不足、株数調整、取得失敗は、それぞれの未評価理由を残します。", ""]
    return "\n".join(lines)
