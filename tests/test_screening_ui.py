"""Default-menu and saved-display contracts; no live API or user cache access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import jq_screening_ui as ui


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def saved_run(tmp_path, run_id="r_current", *, workflow=None):
    output, cache = tmp_path / "out", tmp_path / "cache"
    root = output / "history" / run_id
    root.mkdir(parents=True)
    report = "# A・B候補\n\nA候補の根拠。Bは信用条件の確認待ち。\n"
    raw = report.encode("utf-8")
    (root / ui.REPORT_FILE).write_bytes(raw)
    manifest = {"run_id": run_id, "run_status": "complete", "price_asof_date": "2026-09-30",
                "information_cutoff": "2026-09-30T19:00:00+09:00",
                "outputs": {ui.REPORT_FILE: hashlib.sha256(raw).hexdigest()}}
    write_json(root / "run_manifest.json", manifest)
    write_json(output / "latest_run.json", {"run_id": run_id, "history_path": str(root.resolve())})
    if workflow is not None:
        write_json(cache / ui.WORKFLOW_STATUS_FILE, workflow)
    return output, cache, root, manifest


def arguments(tmp_path, **overrides):
    values = dict(phase="interactive", cache_dir=tmp_path / "cache", output_dir=tmp_path / "out",
                  config=None, top=10, budget=400, max_retries=3, retry_delay=0, api_ini=tmp_path / "never-read.ini")
    values.update(ui._CLEAR_VALUES)
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_bot(choices, calls, dispatch=None):
    answers = iter(choices)
    return SimpleNamespace(
        DataError=ValueError, CollectionError=RuntimeError,
        _interactive_input=lambda prompt: next(answers),
        canonical_code=lambda code: code,
        dispatch=dispatch or (lambda args, cfg: calls.append(args) or 0),
        run_advanced_interactive=lambda args, cfg: calls.append(("advanced", args)) or 0,
    )


def test_missing_status_and_report_do_not_create_files_or_cache(tmp_path, capsys):
    assert ui.read_workflow_status(tmp_path / "cache") is None
    assert ui.print_workflow_status(tmp_path / "cache") is None
    assert ui.show_saved_report(tmp_path / "out", tmp_path / "cache") == 2
    assert not list(tmp_path.iterdir())
    assert "保存済み結果なし" in capsys.readouterr().out


def test_status_labels_explain_stage_counts_and_oldest_full_financial_refresh(tmp_path, capsys):
    cache = tmp_path / "cache"
    write_json(cache / ui.WORKFLOW_STATUS_FILE, {
        "status": "running", "stage": "daily_update", "started_at": "2026-09-30T10:00:00+09:00",
        "counts": {"target_count": 100, "complete_count": 98, "unavailable_count": 2,
                   "financial_unavailable_count": 1, "collection_failed_count": 1, "scope_excluded_count": 4,
                   "credit_completed_count": 95, "credit_failed_count": 3, "credit_pending_count": 2, "credit_empty_count": 4,
                   "internal_test_counter": 999},
        "oldest_last_full_financial_at": "2026-09-02T12:00:00+09:00", "last_full_financial_at_unknown_count": 2,
    })
    ui.print_workflow_status(cache)
    text = capsys.readouterr().out
    assert "新しい価格・財務開示・決算予定の取得" in text
    assert "対象=100件" in text
    assert "取得済=98件" in text
    assert "財務データなし=1件" in text
    assert "信用取得失敗=3件" in text
    assert "信用データなし=4件" in text
    assert "財務全履歴の最終取得" in text
    assert "最古=2026-09-02T12:00:00+09:00 / 不明=2件" in text
    assert "daily_update" not in text
    assert "target_count" not in text
    assert "internal_test_counter" not in text


def test_saved_report_display_is_read_only_and_prints_dates_and_both_routes(tmp_path, capsys):
    output, cache, _, _ = saved_run(tmp_path, workflow={"status": "complete", "run_id": "r_current"})
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert ui.show_saved_report(output, cache) == 0
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert before == after
    text = capsys.readouterr().out
    assert "直近に完了した一括更新" in text
    assert "2026-09-30" in text
    assert "2026-09-30T19:00:00+09:00" in text
    assert "A候補の根拠" in text
    assert "Bは信用条件の確認待ち" in text


@pytest.mark.parametrize("status", ["failed", "interrupted", "running", "partial"])
def test_failed_or_running_update_cannot_label_previous_report_as_current(tmp_path, capsys, status):
    output, cache, _, _ = saved_run(tmp_path, workflow={
        "status": status, "run_id": None, "started_at": "2026-10-01T10:00:00+09:00"})
    assert ui.show_saved_report(output, cache) == 0
    text = capsys.readouterr().out
    assert "前回保存分" in text
    assert "今回の最新結果として扱わない" in text
    assert "2026-09-30" in text


def test_success_with_different_pointer_is_also_labeled_previous(tmp_path, capsys):
    output, cache, _, _ = saved_run(tmp_path, workflow={"status": "complete", "run_id": "r_other"})
    ui.show_saved_report(output, cache)
    assert "前回保存分" in capsys.readouterr().out


def test_matching_partial_result_is_distinguished_from_stale_previous_result(tmp_path, capsys):
    output, cache, _, _ = saved_run(tmp_path, workflow={"status": "partial", "run_id": "r_current"})
    assert ui.show_saved_report(output, cache) == 0
    text = capsys.readouterr().out
    assert "今回の部分結果" in text
    assert "前回保存分" not in text


@pytest.mark.parametrize("path_kind", ["outside", "sibling_prefix", "wrong_run", "traversal"])
def test_pointer_cannot_read_outside_exact_history_run(tmp_path, path_kind):
    output, cache, root, _ = saved_run(tmp_path)
    targets = {"outside": tmp_path / "outside", "sibling_prefix": output / "history_evil" / "r_current",
               "wrong_run": output / "history" / "r_other", "traversal": "../secret"}
    write_json(output / "latest_run.json", {"run_id": "r_current", "history_path": str(targets[path_kind])})
    with pytest.raises(ValueError, match="history/<run_id>"):
        ui.show_saved_report(output, cache)


def test_relative_history_pointer_is_supported_within_output(tmp_path, capsys):
    output, cache, _, _ = saved_run(tmp_path)
    write_json(output / "latest_run.json", {"run_id": "r_current", "history_path": "history/r_current"})
    assert ui.show_saved_report(output, cache) == 0


@pytest.mark.parametrize("change", [{"run_id": "r_wrong"}, {"run_status": "incomplete"}])
def test_mismatched_or_unfinished_manifest_is_not_a_published_result(tmp_path, change):
    output, cache, root, manifest = saved_run(tmp_path)
    write_json(root / "run_manifest.json", {**manifest, **change})
    with pytest.raises(ValueError):
        ui.show_saved_report(output, cache)


def test_report_hash_mismatch_is_not_silently_displayed(tmp_path, capsys):
    output, cache, root, _ = saved_run(tmp_path)
    (root / ui.REPORT_FILE).write_text("tampered contents", encoding="utf-8")
    with pytest.raises(ValueError, match="検証値"):
        ui.show_saved_report(output, cache)
    assert "tampered contents" not in capsys.readouterr().out


def test_corrupt_workflow_status_does_not_claim_saved_report_is_current(tmp_path, capsys):
    output, cache, _, _ = saved_run(tmp_path)
    cache.mkdir()
    (cache / ui.WORKFLOW_STATUS_FILE).write_text("{invalid", encoding="utf-8")
    assert ui.show_saved_report(output, cache) == 0
    text = capsys.readouterr().out
    assert "今回更新できたことは確認できません" in text
    assert "直近に完了した一括更新" not in text


def test_menu_launch_and_quit_never_dispatch_collection(tmp_path, capsys):
    calls = []
    assert ui.run_interactive(fake_bot(["q"], calls), arguments(tmp_path), object()) == 0
    assert calls == []
    assert not list(tmp_path.iterdir())
    text = capsys.readouterr().out
    assert "価格・財務・信用残" in text
    assert "新しい財務開示は毎回取得" in text
    assert "全履歴は最終取得から30日" in text
    assert "API通信あり" in text


def test_normal_screen_discards_advanced_overrides_without_mutating_original_args(tmp_path, capsys):
    calls = []
    custom = {key: True if value is False else "advanced-value" for key, value in ui._CLEAR_VALUES.items()}
    args = arguments(tmp_path, **custom)
    original = vars(args).copy()
    bot = fake_bot(["1", "q"], calls)
    assert ui.run_interactive(bot, args, object()) == 0
    assert len(calls) == 1
    selected = calls[0]
    assert selected.phase == "screen"
    assert selected._menu_child is True
    assert all(getattr(selected, key) == value for key, value in ui._CLEAR_VALUES.items())
    assert selected.cache_dir == args.cache_dir
    assert selected.budget == 400
    assert vars(args) == original
    assert "個別指定は使わず" in capsys.readouterr().out


def test_display_status_and_advanced_route_without_screen_dispatch(tmp_path):
    calls = []
    args = arguments(tmp_path)
    bot = fake_bot(["2", "4", "5", "q"], calls)
    assert ui.run_interactive(bot, args, object()) == 0
    assert calls[0].phase == "show-report"
    assert calls[1].phase == "status"
    assert calls[2] == ("advanced", args)
    assert not any(getattr(value, "phase", None) == "screen" for value in calls)


def test_single_uses_validated_saved_run_and_warns_about_prior_result(tmp_path, capsys):
    saved_run(tmp_path, workflow={"status": "failed", "run_id": None})
    calls = []
    bot = fake_bot(["3", "7203", "q"], calls)
    assert ui.run_interactive(bot, arguments(tmp_path, run_id="r_ignored"), object()) == 0
    assert len(calls) == 1
    assert calls[0].phase == "single"
    assert calls[0].run_id == "r_current"
    assert calls[0].code == "7203"
    assert "前回保存分" in capsys.readouterr().out


def test_single_missing_report_does_not_collect_or_prompt_for_code(tmp_path, capsys):
    calls = []
    assert ui.run_interactive(fake_bot(["3", "q"], calls), arguments(tmp_path), object()) == 0
    assert not calls
    assert "保存済み結果なし" in capsys.readouterr().out


@pytest.mark.parametrize("status,run_id,expected", [
    ("partial", "r_current", "今回の部分結果"),
    ("partial", None, "今回の結果は公開されていません"),
    ("failed", None, "今回の結果は公開されていません"),
])
def test_screen_partial_exit_prints_new_partial_only_if_workflow_run_matches(tmp_path, capsys, status, run_id, expected):
    saved_run(tmp_path, workflow={"status": status, "run_id": run_id})
    calls = []
    bot = fake_bot(["1", "q"], calls, dispatch=lambda args, cfg: calls.append(args.phase) or 2)
    assert ui.run_interactive(bot, arguments(tmp_path), object()) == 0
    assert calls == ["screen"]
    text = capsys.readouterr().out
    assert expected in text
    assert ("A候補の根拠" in text) == (run_id == "r_current")


def test_screen_interrupt_propagates_without_showing_old_report(tmp_path, capsys):
    saved_run(tmp_path)
    bot = fake_bot(["1"], [], dispatch=lambda args, cfg: 130)
    with pytest.raises(KeyboardInterrupt):
        ui.run_interactive(bot, arguments(tmp_path), object())
    assert "A候補の根拠" not in capsys.readouterr().out


def runner_script(source):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", "-c", source], cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_runner_default_launch_is_interactive_not_automatic_api_or_analysis():
    output = runner_script("import runner; print(runner.DEFAULT_PHASE)")
    assert output == "interactive"


def test_runner_exposes_screen_saved_report_status_and_update_start():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-B", str(root / "runner.py"), "--help"], cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    for choice in ("screen", "show-report", "status", "update", "--update-from"):
        assert choice in result.stdout


def test_runner_forwards_update_from_without_starting_main_or_network():
    source = ("import json,sys,runner; "
              "runner.invoke_jquants_screening_bot=lambda args: print(json.dumps(runner._build_bot_argv(args))); "
              "sys.argv=['runner.py','--phase','update','--update-from','2026-09-01']; runner.main()")
    argv = json.loads(runner_script(source))
    assert argv[:3] == ["JQuamtsScreeningBot.py", "--phase", "update"]
    assert argv[argv.index("--update-from") + 1] == "2026-09-01"
