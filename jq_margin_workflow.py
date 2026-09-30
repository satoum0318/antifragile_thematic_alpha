"""Independent margin collection and offline reports; no screening mutations.

The screening module is injected by callers. Importing this module never loads
credentials, opens a session, or imports the CLI a second time.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path

from jq_margin import adapt_margin_bundle, margin_diagnostic


MARGIN_SNAPSHOT_SCHEMA = "jq-margin-snapshot-v1"
MARGIN_RESUME_SCHEMA = "jq-margin-resume-v1"


def _identifier(bot, value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise bot.DataError("invalid_margin_snapshot_id")
    return value


def load_margin_snapshot(bot, store, snapshot_id=None):
    """Read a diagnostic snapshot without changing either latest pointer."""
    if snapshot_id is None:
        pointer = store.root / "latest_margin_snapshot.json"
        if not pointer.exists():
            return {}
        snapshot_id = json.loads(pointer.read_text("utf-8"))["snapshot_id"]
    path = store.root / "margin_snapshots" / (_identifier(bot, snapshot_id) + ".json")
    if not path.is_file():
        raise bot.DataError("margin_snapshot_not_found:" + snapshot_id)
    snapshot = json.loads(path.read_text("utf-8"))
    if snapshot.get("schema") != MARGIN_SNAPSHOT_SCHEMA:
        raise bot.DataError("margin_snapshot_schema_mismatch")
    return snapshot


def _save_snapshot(bot, store, snapshot):
    path = store.root / "margin_snapshots" / (snapshot["snapshot_id"] + ".json")
    if path.exists():
        raise bot.DataError("immutable_margin_snapshot_already_exists")
    bot.atomic_json(path, snapshot)
    bot.atomic_json(store.root / "latest_margin_snapshot.json",
                    {"snapshot_id": snapshot["snapshot_id"]})


def _calendar_covers(bot, bundle, start, end):
    if bundle.get("status") != "complete":
        return False
    try:
        bot.trading_sessions(bundle.get("rows", []))
        days = {bot.date(row.get("Date")) for row in bundle.get("rows", [])}
        return all(start + dt.timedelta(days=n) in days for n in range((end-start).days+1))
    except (bot.DataError, ValueError, TypeError, KeyError):
        return False


def _read_optional_bundle(bot, store, ref):
    if not ref:
        return {"status": "not_collected", "rows": []}
    try:
        bundle = store.get(ref)
        if not isinstance(bundle, dict):
            return {"status": "unsupported_schema", "rows": [], "reason": "cache_object_not_mapping", "unavailable_ref": ref}
        return bundle
    except (bot.DataError, OSError, ValueError, KeyError) as exc:
        return {"status": "cache_unavailable", "rows": [], "reason": str(exc), "unavailable_ref": ref}


def _previous_bundle(bot, store, source, previous, code, api_code):
    item = previous.get("observation_index", previous.get("symbols", {})).get(code, {})
    if item.get("api_code") not in (None, api_code):
        item = {}
    old_ref = item.get("margin_ref") or item.get("history_ref")
    old = _read_optional_bundle(bot, store, old_ref) if old_ref else None
    source_ref = source.get("symbols", {}).get(code, {}).get("margin")
    if source_ref:
        old = adapt_margin_bundle(_read_optional_bundle(bot, store, source_ref), api_code, old)
    return old, item


def _nonempty_success_ref(bot, store, item):
    for key in ("last_success_ref", "previous_observation_ref"):
        ref = item.get(key)
        if ref:
            blob = _read_optional_bundle(bot, store, ref)
            if blob.get("status") == "complete" and blob.get("rows"):
                return ref
    return None


def margin_refresh(bot, client, source_snapshot_id=None, budget=800,
                   only_codes=None, max_retries=3, reset_pending=False):
    """Collect one bounded checkpoint, preserving a fixed source and date window.

    Call again while pending_codes is nonempty. A fresh run always attempts all
    selected ordinary shares, irrespective of their screening complete flag.
    Interrupted and failed attempts remain visible in the diagnostic snapshot.
    """
    if budget < 1 or max_retries < 1:
        raise bot.DataError("margin_budget_and_retries_must_be_positive")
    store = client.store
    resume_path = store.root / "margin_collection_resume.json"
    requested = sorted({bot.canonical_code(c) for c in only_codes}) if only_codes else None
    journal = None
    if reset_pending:
        resume_path.unlink(missing_ok=True)
    elif resume_path.exists():
        journal = json.loads(resume_path.read_text("utf-8"))
        if journal.get("schema") != MARGIN_RESUME_SCHEMA:
            raise bot.DataError("margin_resume_schema_mismatch")
        if ((source_snapshot_id is not None and source_snapshot_id != journal.get("source_snapshot_id"))
                or (requested is not None and requested != journal.get("selected_codes"))
                or max_retries != journal.get("max_retries")):
            raise bot.DataError("margin_resume_request_mismatch:use_reset_pending")
        source_snapshot_id = journal["source_snapshot_id"]

    source = store.snapshot(source_snapshot_id)
    if not source:
        raise bot.DataError("margin_requires_source_snapshot")
    source_id = source["snapshot_id"]
    price_date = bot.date(source.get("price_asof_date"))
    source_cutoff = bot.timestamp(source.get("information_cutoff"))
    if price_date is None or source_cutoff is None:
        raise bot.DataError("margin_source_missing_dates")
    if price_date > bot.now_jst().date():
        raise bot.DataError("margin_source_price_after_collection_date")

    if journal:
        previous = load_margin_snapshot(bot, store, journal["snapshot_id"])
        if previous.get("source_snapshot_id") != source_id:
            raise bot.DataError("margin_resume_source_mismatch")
        snapshot = copy.deepcopy(previous)
        pending = list(journal["pending_codes"])
        retry_counts = dict(journal.get("retry_counts", {}))
        selected = list(journal["selected_codes"])
        targets = snapshot["targets"]
    else:
        master_ref = source.get("globals", {}).get("master")
        master = bot.master_records(bot.bundle_rows(store, master_ref))
        selected = requested or [m["code"] for m in master if m["instrument_type"] == "stock"]
        if requested:
            selected = [bot.resolve_code(c, master) for c in requested]
        selected = sorted(selected)
        if not selected:
            raise bot.DataError("margin_source_has_no_ordinary_shares")
        targets = {m["code"]: copy.deepcopy(m) for m in master if m["code"] in selected}
        previous = load_margin_snapshot(bot, store)
        today = bot.now_jst().date()
        snapshot = {
            "schema": MARGIN_SNAPSHOT_SCHEMA, "source_snapshot_id": source_id,
            "source_price_asof_date": str(price_date), "price_asof_date": str(price_date),
            "source_information_cutoff": source_cutoff.isoformat(),
            "window_from": str(today-dt.timedelta(days=89)), "window_to": str(today),
            "started_at": bot.now_jst().isoformat(), "targets": targets,
            "selected_codes": selected, "symbols": {}, "terminal_failed_codes": [],
            "calendar_ref": source.get("globals", {}).get("calendar"),
            "observation_index": copy.deepcopy(previous.get("observation_index", previous.get("symbols", {}))),
        }
        for code in selected:
            old, old_item = _previous_bundle(bot, store, source, previous, code, targets[code]["api_code"])
            history_ref = store.put(old) if old else None
            snapshot["symbols"][code] = {
                "api_code": targets[code]["api_code"], "last_attempt_status": "not_collected",
                "history_ref": history_ref, "last_success_ref": old_item.get("last_success_ref"),
                "previous_observation_ref": _nonempty_success_ref(bot, store, old_item),
            }
            if old and old.get("status") == "complete" and not old_item.get("last_success_ref"):
                snapshot["symbols"][code]["last_success_ref"] = history_ref
                if old.get("rows"):
                    snapshot["symbols"][code]["previous_observation_ref"] = history_ref
        pending, retry_counts = list(selected), {}

    snapshot.update(snapshot_id=bot.new_id("ms_"), status="incomplete",
                    code_version=bot.CODE_VERSION, logic_version=bot.LOGIC_VERSION,
                    parent_margin_snapshot_id=previous.get("snapshot_id"),
                    max_retries=max_retries, errors=[], batch_success_count=0,
                    batch_incomplete_count=0, attempted_codes=[])
    queue = list(pending)
    terminal = list(snapshot.get("terminal_failed_codes", []))
    interrupted = False
    try:
        cal_ref = snapshot.get("calendar_ref")
        cal = _read_optional_bundle(bot, store, cal_ref)
        start = bot.date(snapshot["window_from"])
        end = max(bot.date(snapshot["window_to"]), bot.now_jst().date())
        if not _calendar_covers(bot, cal, start, end):
            try:
                cal = client.fetch("markets/calendar", {"from": str(start), "to": str(end)})
                snapshot["calendar_ref"] = store.put(cal)
                snapshot["calendar_status"] = "complete" if _calendar_covers(bot, cal, start, end) else "calendar_incomplete"
            except (bot.CollectionError, bot.DataError, OSError, ValueError) as exc:
                snapshot["calendar_status"] = "permission_unavailable" if isinstance(exc, bot.AuthError) else "collection_failed"
                snapshot["calendar_reason"] = str(exc)
                snapshot["errors"].append({"kind": "calendar", "reason": str(exc)})
                if (isinstance(exc, bot.AuthError) and "401" in str(exc)) or "daily_self_quota_reached" in str(exc):
                    raise
        else:
            snapshot["calendar_status"] = "complete"
            snapshot.pop("calendar_reason", None)

        for code in pending[:budget]:
            item = snapshot["symbols"][code]
            api_code = targets[code]["api_code"]
            params = {"code": api_code, "from": snapshot["window_from"], "to": snapshot["window_to"]}
            old_ref = item.get("margin_ref") or item.get("history_ref")
            old = _read_optional_bundle(bot, store, old_ref) if old_ref else None
            item["previous_observation_ref"] = _nonempty_success_ref(bot, store, item)
            snapshot["attempted_codes"].append(code)
            abort = None
            try:
                raw = client.fetch("markets/margin-interest", params)
                if raw.get("status") != "complete":
                    raise bot.CollectionError("margin_response_not_complete")
                raw_ref = store.put(raw)
            except (KeyboardInterrupt, SystemExit) as exc:
                abort = exc
                raw = {"status": "collection_interrupted", "rows": [], "endpoint": "markets/margin-interest",
                       "params": params, "collected_at": bot.now_jst().isoformat(), "reason": type(exc).__name__}
                raw_ref = getattr(exc, "raw_cache_ref", None) or store.put(raw)
            except (bot.CollectionError, bot.DataError, OSError, ValueError, KeyError) as exc:
                status = "collection_failed"
                if isinstance(exc, bot.AuthError):
                    status = "authentication_failed" if "401" in str(exc) else "permission_unavailable"
                    if "401" in str(exc):
                        abort = exc
                elif "unexpected_response_schema" in str(exc):
                    status = "schema_unsupported"
                elif "daily_self_quota_reached" in str(exc):
                    status = "quota_exhausted"
                    abort = exc
                raw = {"status": status, "rows": [], "endpoint": "markets/margin-interest",
                       "params": params, "collected_at": bot.now_jst().isoformat(), "reason": str(exc)}
                raw_ref = getattr(exc, "raw_cache_ref", None) or store.put(raw)

            adapted = adapt_margin_bundle(raw, api_code, old)
            accepted = raw["status"] == "complete" and adapted.get("adapter_status") in {"valid", "no_rows"}
            attempt_status = "complete" if accepted else (adapted.get("adapter_status") or raw["status"])
            item.update(margin_ref=store.put(adapted), raw_ref=raw_ref,
                        last_attempt_status=attempt_status, transport_status=raw["status"], collected_at=raw.get("collected_at"),
                        adapter_status=adapted.get("adapter_status"),
                        adapter_version=adapted.get("adapter_version"))
            if accepted:
                item["last_success_ref"] = item["margin_ref"]
                retry_counts.pop(code, None)
                queue.remove(code)
                snapshot["batch_success_count"] += 1
            else:
                snapshot["batch_incomplete_count"] += 1
                snapshot["errors"].append({"code": code, "status": attempt_status,
                                           "reason": raw.get("reason") or adapted.get("adapter_reasons")})
                if not isinstance(abort, (KeyboardInterrupt, SystemExit)) and raw["status"] != "quota_exhausted":
                    retry_counts[code] = retry_counts.get(code, 0)+1
                if retry_counts.get(code, 0) >= max_retries:
                    queue.remove(code)
                    if code not in terminal:
                        terminal.append(code)
            if abort is not None:
                raise abort
        snapshot["status"] = "partial" if queue or terminal or snapshot.get("calendar_status") != "complete" else "complete"
        return snapshot
    except BaseException:
        interrupted = True
        snapshot["status"] = "incomplete"
        raise
    finally:
        observed = bot.now_jst().isoformat()
        snapshot.update(information_cutoff=observed, created_at=observed,
                        pending_codes=queue, pending_count=len(queue), requested_pending_count=len(queue),
                        unattempted_count=sum(snapshot["symbols"][c].get("last_attempt_status") == "not_collected" for c in queue),
                        retry_counts=retry_counts, terminal_failed_codes=sorted(terminal),
                        attempted_count=len(snapshot["attempted_codes"]),
                        completed_count=sum(i.get("last_attempt_status") == "complete" for i in snapshot["symbols"].values()))
        if interrupted:
            snapshot["failure_at"] = observed
        snapshot.setdefault("observation_index", {}).update(copy.deepcopy(snapshot["symbols"]))
        _save_snapshot(bot, store, snapshot)
        if queue:
            bot.atomic_json(resume_path, {
                "schema": MARGIN_RESUME_SCHEMA, "snapshot_id": snapshot["snapshot_id"],
                "source_snapshot_id": source_id, "selected_codes": selected,
                "window_from": snapshot["window_from"], "window_to": snapshot["window_to"],
                "pending_codes": queue, "retry_counts": retry_counts, "max_retries": max_retries,
            })
        else:
            resume_path.unlink(missing_ok=True)


def margin_analyze(bot, store, margin_snapshot_id=None, cutoff=None):
    """Reproduce diagnostics from saved evidence only; never rescore screening."""
    snapshot = load_margin_snapshot(bot, store, margin_snapshot_id)
    if not snapshot:
        raise bot.DataError("margin_snapshot_required")
    source = store.snapshot(snapshot["source_snapshot_id"])
    verified = bot.timestamp(snapshot.get("information_cutoff"))
    information = bot.timestamp(cutoff) if cutoff is not None else verified
    price_date = bot.date(source.get("price_asof_date"))
    if information is None or verified is None or price_date is None:
        raise bot.DataError("margin_snapshot_missing_dates")
    if information > verified:
        raise bot.DataError("margin_cutoff_after_snapshot_verification")
    if price_date > information.date():
        raise bot.DataError("margin_price_after_information_cutoff")
    cal = _read_optional_bundle(bot, store, snapshot.get("calendar_ref"))
    calendar_rows = cal.get("rows", []) if snapshot.get("calendar_status") == "complete" else []
    records = []
    for code in snapshot["selected_codes"]:
        target = snapshot["targets"][code]
        item = snapshot["symbols"].get(code, {})
        bundle = _read_optional_bundle(bot, store, item.get("margin_ref"))
        price_ref = source.get("symbols", {}).get(code, {}).get("prices")
        price_blob = _read_optional_bundle(bot, store, price_ref)
        prices = price_blob.get("rows", []) if price_blob.get("status") == "complete" else []
        options = dict(expected_code=target["api_code"], price_asof_date=price_date,
                       calendar_rows=calendar_rows, price_rows=prices)
        record = {"code": code, "api_code": target["api_code"], "name": target.get("name", ""),
                  "source_snapshot_id": source["snapshot_id"], "price_asof_date": str(price_date),
                  "source_information_cutoff": source.get("information_cutoff"),
                  "information_cutoff": information.isoformat(), "margin_snapshot_id": snapshot["snapshot_id"],
                  "margin_raw_ref": item.get("raw_ref"), "margin_bundle_ref": item.get("margin_ref"),
                  "margin_price_cache_status": price_blob.get("status"), "margin_price_cache_reason": price_blob.get("reason"),
                  "margin_last_attempt_status": item.get("last_attempt_status", "not_collected")}
        record.update(margin_diagnostic(bundle, information, **options))
        previous_ref = item.get("previous_observation_ref")
        record["margin_previous_observation_ref"] = previous_ref
        record["margin_previous_observation"] = (
            margin_diagnostic(_read_optional_bundle(bot, store, previous_ref), information, **options) if previous_ref else None)
        records.append(record)
    meta = {"margin_snapshot_id": snapshot["snapshot_id"], "source_snapshot_id": source["snapshot_id"],
            "source_price_asof_date": str(price_date), "price_asof_date": str(price_date),
            "source_information_cutoff": source.get("information_cutoff"),
            "information_cutoff": information.isoformat(), "snapshot_information_cutoff": verified.isoformat(),
            "run_status": snapshot["status"], "mode": "margin_diagnostic_only",
            "screening_recomputed": False, "ranking_reproduced": False,
            "operational_validation": "not_certified_by_diagnostic_run",
            "code_version": bot.CODE_VERSION, "logic_version": bot.LOGIC_VERSION,
            "window_from": snapshot["window_from"], "window_to": snapshot["window_to"],
            "calendar_status": snapshot.get("calendar_status") if cal.get("status") == "complete" else cal.get("status"),
            "calendar_ref": snapshot.get("calendar_ref"),
            "selected_codes": snapshot["selected_codes"], "pending_codes": snapshot["pending_codes"],
            "terminal_failed_codes": snapshot["terminal_failed_codes"],
            "adapter_versions": sorted({i["adapter_version"] for i in snapshot["symbols"].values() if i.get("adapter_version")}),
            "pit_quality": "saved_observed_versions_only_not_guaranteed_original_publication_vintage"}
    return records, meta


def write_margin_run(bot, records, meta, output):
    """Publish an immutable diagnostic artifact without touching screening runs."""
    output = Path(output)
    run_id = bot.new_id("mr_")
    rows, manifest = copy.deepcopy(records), copy.deepcopy(meta)
    manifest.update(run_id=run_id, generated_at=bot.now_jst().isoformat(),
                    result_hash=bot.digest(records), record_count=len(records))
    history = output / "margin_history"
    history.mkdir(parents=True, exist_ok=True)
    final, stage = history / run_id, history / (".incomplete_"+run_id)
    if final.exists() or stage.exists():
        raise bot.DataError("immutable_margin_run_already_exists")
    stage.mkdir()
    leading = ["code", "name", "price_asof_date", "source_information_cutoff", "information_cutoff",
               "margin_diagnostic_status", "margin_reference_date", "margin_pub_date", "margin_observed_at",
               "margin_ratio", "margin_ratio_state", "margin_short"]
    columns = leading + sorted({key for row in rows for key in row}-set(leading))
    lines = ["# 信用取引残高の診断", "",
             f"元snapshot: {manifest['source_snapshot_id']}",
             f"参照価格日: {manifest['price_asof_date']}",
             f"元スクリーニング情報締切: {manifest['source_information_cutoff']}",
             f"信用診断情報締切: {manifest['information_cutoff']}",
             f"取得窓: {manifest['window_from']} ～ {manifest['window_to']} / 状態: {manifest['run_status']}", "",
             "現在の信用観測を表示します。元順位・推奨点・新規買い可否は再計算していません。",
             "空欄・未取得・対象外・失敗は安全を意味しません。金額はAPI原値・単位未確認です。",
             "開発検証と実データによる運用確認は別です。この出力は運用監査完了を証明しません。", "",
             "|コード|銘柄|状態|申込日|公表日|取得確認時刻|信用倍率|倍率状態・空欄理由|信用売残株数|価格日より後|5営業日変化・理由|20営業日変化・理由|前回観測日|",
             "|---|---|---|---|---|---|---:|---|---:|---|---|---|---|"]
    for row in rows:
        previous = row.get("margin_previous_observation") or {}
        reasons = row.get("margin_metric_reasons") or {}
        def change_text(days):
            key = f"margin_short_change_{days}bd_pct"
            return bot.fmt(row.get(key))+"%" if row.get(key) is not None else reasons.get(key) or "N/A"
        values = [row.get("code"), row.get("name"), row.get("margin_diagnostic_status"),
                  row.get("margin_reference_date"), row.get("margin_pub_date"), row.get("margin_observed_at"),
                  bot.fmt(row.get("margin_ratio")), reasons.get("margin_ratio") or row.get("margin_ratio_state"),
                  bot.fmt(row.get("margin_short")), row.get("margin_after_price_asof"), change_text(5), change_text(20),
                  previous.get("margin_reference_date")]
        lines.append("|"+"|".join(bot.md_escape(v) for v in values)+"|")
    try:
        bot.atomic_json(stage / "margin_diagnostics.json", rows)
        bot.atomic_bytes(stage / "margin_diagnostics.csv", bot.csv_bytes(rows, columns))
        bot.atomic_bytes(stage / "report_margin.md", ("\n".join(lines)+"\n").encode("utf-8"))
        manifest["outputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir() if p.is_file()}
        manifest["source_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     (Path(__file__), Path(__file__).with_name("jq_margin.py"), Path(bot.__file__)) if p.is_file()}
        bot.atomic_json(stage / "manifest.json", manifest)
        os.replace(stage, final)
        bot.atomic_json(output / "latest_margin_run.json", {"run_id": run_id, "history_path": str(final.resolve())})
        return final
    except BaseException:
        if stage.exists():
            bot.atomic_json(stage / "manifest.json", {**manifest, "run_status": "incomplete"})
        raise
