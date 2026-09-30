"""Small default menu and read-only display of saved screening evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

WORKFLOW_STATUS_FILE = "screening_workflow_status.json"
REPORT_FILE = "report_registration_top10.md"
_STATUS_LABELS = {"running": "実行中", "complete": "完了", "partial": "一部未完了",
                  "failed": "失敗", "interrupted": "中断"}
_STAGE_LABELS = {"prepare": "準備・接続確認", "full_history": "価格・財務の履歴更新",
                 "margin": "信用買残・売残の更新", "daily_update": "新しい価格・財務開示・決算予定の取得",
                 "verify": "データの時点・欠測の確認", "analyze": "業種比較とA・B候補の採点", "done": "終了"}
_COUNT_LABELS = {"target_count": "対象", "complete_count": "取得済", "unavailable_count": "未評価",
                 "financial_unavailable_count": "財務データなし", "collection_failed_count": "取得失敗",
                 "scope_excluded_count": "対象外", "credit_completed_count": "信用取得済",
                 "credit_failed_count": "信用取得失敗", "credit_pending_count": "信用未完了",
                 "credit_empty_count": "信用データなし"}


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("JSONの形式を確認してください: " + str(path))
    return value


def read_workflow_status(cache_dir: Path | str) -> dict | None:
    """Read only; absence must not create a cache directory or contact an API."""
    path = Path(cache_dir) / WORKFLOW_STATUS_FILE
    return _read_object(path) if path.is_file() else None


def print_workflow_status(cache_dir: Path | str) -> dict | None:
    status = read_workflow_status(cache_dir)
    if status is None:
        print("[一括更新] まだ実行記録がありません。保存済み結果があればメニュー2で確認できます。", flush=True)
        return None
    state = status.get("status", "unknown")
    print("[一括更新] 状態={} / 開始={} / 段階={}".format(
        _STATUS_LABELS.get(state, "未確認(" + str(state) + ")"),
        status.get("started_at") or "未記録", _STAGE_LABELS.get(status.get("stage"), "未確認")), flush=True)
    print("[更新の時点] 価格基準日={} / 情報締切={} / run={}".format(
        status.get("price_asof_date") or "未確定", status.get("information_cutoff") or "未確定",
        status.get("run_id") or "未公開"), flush=True)
    counts = status.get("counts")
    if isinstance(counts, dict):
        display = [label + "=" + str(counts[key]) + "件" for key, label in _COUNT_LABELS.items() if key in counts]
        if display:
            print("[更新の件数] " + " / ".join(display), flush=True)
    if "oldest_last_full_financial_at" in status or "last_full_financial_at_unknown_count" in status:
        print("[財務全履歴の最終取得] 最古={} / 不明={}件".format(
            status.get("oldest_last_full_financial_at") or "未記録",
            status.get("last_full_financial_at_unknown_count", "未確認")), flush=True)
    if status.get("reason"):
        reasons = {"some_inputs_unavailable": "一部の入力が取得できていません。候補の未評価理由を確認してください。",
                   "user_interrupted": "利用者の操作で中断しました。メニュー1で再開できます。"}
        print("[確認事項] " + reasons.get(status["reason"], str(status["reason"])), flush=True)
    if state in {"failed", "interrupted", "running"}:
        print("[注意] 今回の更新は完了していません。保存済みレポートを今回の最新結果として扱わないでください。", flush=True)
    elif state == "partial":
        print("[注意] 一部の更新に未完了があります。レポートの対象件数・保留理由を確認してください。", flush=True)
    return status


def _saved_run(output_dir: Path | str) -> tuple[Path, dict] | None:
    output = Path(output_dir).resolve()
    pointer_path = output / "latest_run.json"
    if not pointer_path.is_file():
        return None
    pointer = _read_object(pointer_path)
    run_id, raw_path = pointer.get("run_id"), pointer.get("history_path")
    if (not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id)
            or not isinstance(raw_path, str) or not raw_path):
        raise ValueError("保存結果の参照が不正です。latest_run.jsonを確認してください。")
    history = (output / "history").resolve()
    try:
        history.relative_to(output)
    except ValueError:
        raise ValueError("履歴フォルダーが出力先の外を参照しています。") from None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = output / candidate
    root = candidate.resolve()
    # A symlink/junction or path traversal must not turn display into an arbitrary read.
    if root.parent != history or root.name != run_id:
        raise ValueError("保存結果は出力先のhistory/<run_id>内である必要があります。")
    manifest_path = root / "run_manifest.json"
    if manifest_path.resolve().parent != root:
        raise ValueError("実行記録が保存履歴の外を参照しています。")
    manifest = _read_object(manifest_path)
    if manifest.get("run_id") != run_id:
        raise ValueError("保存結果と実行記録のrun_idが一致しません。")
    if manifest.get("run_status") in {"incomplete", "failed", "running", "interrupted"}:
        raise ValueError("未完了の実行結果は保存済みレポートとして表示できません。")
    return root, manifest


def _print_saved_context(root: Path, manifest: dict, cache_dir: Path | str) -> dict | None:
    workflow = None
    try:
        workflow = read_workflow_status(cache_dir)
    except (OSError, ValueError):
        print("[注意] 一括更新の状況を読み取れません。以下は保存済み結果で、今回更新できたことは確認できません。", flush=True)
    state = workflow.get("status") if workflow else None
    same_run = workflow is not None and workflow.get("run_id") == manifest.get("run_id")
    if state == "complete" and same_run:
        print("[保存済み結果] 直近に完了した一括更新の結果です。", flush=True)
    elif state == "partial" and same_run:
        print("[今回の部分結果] 更新に未完了があります。有効な入力の範囲で公開された結果です。", flush=True)
    elif workflow:
        print("[前回保存分] 直近の一括更新は{}です。以下を今回の最新結果として扱わないでください。".format(
            _STATUS_LABELS.get(state, "状況未確認")), flush=True)
    else:
        print("[保存済み結果] 一括更新との対応記録がありません。以下の価格基準日・情報締切を確認してください。", flush=True)
    print("[結果の時点] 価格基準日={} / 情報締切={} / run={}".format(
        manifest.get("price_asof_date") or "未記録", manifest.get("information_cutoff") or "未記録",
        manifest.get("run_id") or root.name), flush=True)
    print("[結果の状態] " + str(manifest.get("run_status") or "未記録"), flush=True)
    return workflow


def show_saved_report(output_dir: Path | str, cache_dir: Path | str) -> int:
    """Print an existing report with its dates; no collection, scoring or writes."""
    saved = _saved_run(output_dir)
    if saved is None:
        print("[保存済み結果なし] メニュー1で更新・スクリーニングを実行してください。", flush=True)
        return 2
    root, manifest = saved
    path = root / REPORT_FILE
    if path.resolve().parent != root:
        raise ValueError("レポートが保存履歴の外を参照しています。")
    _print_saved_context(root, manifest, cache_dir)
    if not path.is_file():
        print("[A・Bレポートなし] 保存先: " + str(root), flush=True)
        print("この履歴にはA・B形式のレポートがありません。メニュー1で新しい結果を作成できます。", flush=True)
        return 2
    raw = path.read_bytes()
    outputs = manifest.get("outputs") or {}
    expected = outputs.get(REPORT_FILE) if isinstance(outputs, dict) else None
    if expected is not None and expected != hashlib.sha256(raw).hexdigest():
        raise ValueError("保存レポートの検証値が一致しません。履歴を確認してください。")
    print("[A・Bレポート] " + str(path), flush=True)
    print(raw.decode("utf-8-sig"), flush=True)
    return 0


_CLEAR_VALUES = {
    "asof": None, "information_cutoff": None, "snapshot_id": None, "margin_snapshot_id": None,
    "run_id": None, "code": None, "codes": None, "update_from": None, "refresh_days": None,
    "force_refresh": False, "force_full": False, "force": False, "reset_pending": False,
    "once": False, "max_batches": None,
}


def _normal_args(args: Any, phase: str) -> Any:
    result = copy.copy(args)
    result.phase = phase
    result._menu_child = True
    for key, value in _CLEAR_VALUES.items():
        setattr(result, key, value)
    return result


def run_interactive(bot: Any, args: Any, cfg: Any) -> int:
    print("キャッシュ: " + str(Path(args.cache_dir).resolve()), flush=True)
    print("出力先: " + str(Path(args.output_dir).resolve()), flush=True)
    print("通常は1を選ぶだけで、価格・財務・信用残を更新してA・B候補を作成します。", flush=True)
    print("新しい財務開示は毎回取得し、全履歴は最終取得から30日を目安に再取得します。初回や不足分は必要な履歴を収集します。", flush=True)
    print("保存結果の表示・状況確認はAPI通信を行いません。銘柄登録・資金配分はご自身で管理します。", flush=True)
    print("Ctrl+C / Ctrl+Zで中断できます。再開は1、個別操作は詳細メニューを使います。", flush=True)
    try:
        print_workflow_status(args.cache_dir)
    except (ValueError, OSError) as exc:
        print("[確認] 更新状況を読み取れません: " + str(exc), flush=True)
    errors = (getattr(bot, "DataError", ValueError), getattr(bot, "CollectionError", RuntimeError),
              OSError, ValueError, KeyError)
    ask = getattr(bot, "_interactive_input", lambda prompt: input(prompt).strip())
    while True:
        print("\n=== スクリーニング ===", flush=True)
        print("1) データを更新してスクリーニング（通常はこちら・API通信あり）")
        print("2) 保存済みのA・Bレポートを見る（通信なし）")
        print("3) 単銘柄の保存済み結果を見る（通信なし）")
        print("4) 更新・収集状況を見る（通信なし）")
        print("5) 詳細メニュー（従来の個別操作）")
        print("q) 終了", flush=True)
        choice = ask("選択: ").strip().lower()
        if choice == "q":
            print("[終了] 終了しました。", flush=True)
            return 0
        if choice not in {"1", "2", "3", "4", "5"}:
            print("1〜5 または q を入力してください。", flush=True)
            continue
        try:
            if choice == "5":
                result = bot.run_advanced_interactive(args, cfg)
                if result == 130:
                    raise KeyboardInterrupt()
                print("通常メニューに戻りました。", flush=True)
                continue
            phase = {"1": "screen", "2": "show-report", "3": "single", "4": "status"}[choice]
            nxt = _normal_args(args, phase)
            if choice == "1":
                if any(getattr(args, key, value) != value for key, value in _CLEAR_VALUES.items()):
                    print("[通常更新] 銘柄限定・過去日・履歴ID・強制再収集などの個別指定は使わず、通常の一括更新を実行します。", flush=True)
                print("[開始] データ更新からA・B候補の作成まで続けて実行します。進捗は定期表示されます。", flush=True)
            elif choice == "3":
                saved = _saved_run(args.output_dir)
                if saved is None:
                    print("[保存済み結果なし] 先にメニュー1を実行してください。", flush=True)
                    continue
                root, manifest = saved
                _print_saved_context(root, manifest, args.cache_dir)
                nxt.code = ask("銘柄コード（4文字・英字可、qで戻る）: ").strip()
                if not nxt.code or nxt.code.lower() == "q":
                    continue
                bot.canonical_code(nxt.code)
                nxt.run_id = manifest["run_id"]
            result = bot.dispatch(nxt, cfg)
            if result == 130:
                raise KeyboardInterrupt()
            if choice == "1":
                status = print_workflow_status(args.cache_dir)
                if result == 0:
                    show_saved_report(args.output_dir, args.cache_dir)
                elif result == 2:
                    saved = _saved_run(args.output_dir)
                    if (status and status.get("status") == "partial" and saved
                            and status.get("run_id") == saved[1].get("run_id")):
                        show_saved_report(args.output_dir, args.cache_dir)
                    else:
                        print("[要確認] 今回の結果は公開されていません。保存済み結果はメニュー2で確認できます。", flush=True)
                else:
                    print("[更新未完了] 今回の最新結果は作成できていません。直前のログと状況を確認してください。", flush=True)
            elif result == 1:
                print("[確認] 処理できませんでした。直前のログを確認してください。", flush=True)
        except errors as exc:
            print("[確認] " + str(exc), flush=True)
