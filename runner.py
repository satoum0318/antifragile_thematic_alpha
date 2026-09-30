"""Helper launcher for JQuamtsScreeningBot.py.

VS Codeなどで実行ボタンを押すだけで既定のオフライン分析を走らせたい場合に利用する。
必要に応じて引数を付ければ、JQuamtsScreeningBot.py CLI と同じ動作を切り替え可能。
"""
from __future__ import annotations

import argparse
import sys
import io
from typing import List

# WindowsでUTF-8出力を有効化
if sys.platform == "win32":
    try:
        # Python 3.7+ の場合
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # 古いPythonバージョンの場合
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import JQuamtsScreeningBot as bot

DEFAULT_PHASE = "analyze"
DEFAULT_TOP = 10
DEFAULT_BUDGET = None  # Let the bot's active config select its checkpoint size.


def _build_bot_argv(args: argparse.Namespace) -> List[str]:
    argv = ["JQuamtsScreeningBot.py", "--phase", args.phase]
    if args.phase == "analyze":
        argv += ["--top", str(args.top)]
    if args.phase in {"collect", "collect_all", "fields-audit", "margin_refresh"}:
        if args.budget is not None:
            argv += ["--budget", str(args.budget)]
        if args.refresh_days is not None:
            argv += ["--refresh-days", str(args.refresh_days)]
        if args.force_full:
            argv.append("--force-full")
        if args.reset_pending:
            argv.append("--reset-pending")
    if args.code:
        argv += ["--code", args.code]
    for name in ("snapshot_id", "margin_snapshot_id", "cache_dir", "output_dir", "config", "asof", "information_cutoff"):
        value = getattr(args, name, None)
        if value is not None:
            argv += ["--" + name.replace("_", "-"), str(value)]
    if getattr(args, "once", False):
        argv.append("--once")
    return argv


def invoke_jquants_screening_bot(args: argparse.Namespace) -> int:
    new_argv = _build_bot_argv(args)
    backup = list(sys.argv)
    try:
        sys.argv = new_argv
        return bot.main()
    finally:
        sys.argv = backup


def main():
    parser = argparse.ArgumentParser(description="Convenience runner for JQuamtsScreeningBot.py")
    parser.add_argument("--phase",
                        choices=["collect", "collect_all", "analyze", "single", "interactive", "fields-audit",
                                 "margin_refresh", "margin_analyze"],
                        default=DEFAULT_PHASE)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--code")
    parser.add_argument("--refresh-days", type=int)
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--reset-pending", action="store_true")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--margin-snapshot-id")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--asof")
    parser.add_argument("--information-cutoff")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    return invoke_jquants_screening_bot(args)


if __name__ == "__main__":
    raise SystemExit(main())
