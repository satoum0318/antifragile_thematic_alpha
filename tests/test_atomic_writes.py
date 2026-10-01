"""Atomic saves must tolerate brief Windows locks without losing old data."""
import errno
import os
from pathlib import Path

import pytest

import JQuamtsScreeningBot as bot


def windows_lock_error(code):
    error = PermissionError(errno.EACCES, 'simulated_windows_lock')
    error.winerror = code
    return error


@pytest.mark.parametrize('winerror', [5, 32, 33])
@pytest.mark.parametrize('existing', [False, True], ids=['new-file', 'replace-file'])
def test_transient_windows_lock_retries_with_same_complete_temp_file(tmp_path, monkeypatch, winerror, existing):
    path = tmp_path/'state.json'
    old, new = b'{"value":"old"}', b'{"value":"new"}'
    if existing:
        path.write_bytes(old)
    replace, sources, waits = bot.os.replace, [], []

    def locked_replace(source, destination):
        sources.append(source)
        assert Path(source).read_bytes() == new
        if existing:
            assert path.read_bytes() == old
        else:
            assert not path.exists()
        if len(sources) <= 2:
            raise windows_lock_error(winerror)
        replace(source, destination)

    monkeypatch.setattr(bot.os, 'replace', locked_replace)
    monkeypatch.setattr(bot.time, 'sleep', waits.append)
    bot.atomic_bytes(path, new)
    assert path.read_bytes() == new
    assert len(waits) == 2 and all(delay > 0 for delay in waits)
    assert len(set(sources)) == 1
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_windows_lock_stops_with_old_file_intact(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    path.write_bytes(b'old')
    error, calls, waits = windows_lock_error(5), [], []

    def denied(source, destination):
        calls.append(source)
        assert path.read_bytes() == b'old'
        raise error

    monkeypatch.setattr(bot.os, 'replace', denied)
    monkeypatch.setattr(bot.time, 'sleep', waits.append)
    with pytest.raises(PermissionError) as caught:
        bot.atomic_bytes(path, b'new')
    assert caught.value is error
    assert 1 < len(calls) <= 10
    assert len(waits) == len(calls)-1 and sum(waits) <= 5
    assert path.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('error', [OSError(errno.ENOSPC, 'disk_full'),
                                 PermissionError(errno.EACCES, 'non_windows_denied')])
def test_unrelated_errors_fail_without_retry(tmp_path, monkeypatch, error):
    path = tmp_path/'state.json'
    path.write_bytes(b'old')

    def fail_replace(*args):
        raise error

    monkeypatch.setattr(bot.os, 'replace', fail_replace)
    monkeypatch.setattr(bot.time, 'sleep', lambda _: pytest.fail('must not retry this error'))
    with pytest.raises(OSError) as caught:
        bot.atomic_bytes(path, b'new')
    assert caught.value is error
    assert path.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [path]


def test_interrupt_during_retry_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    path.write_bytes(b'old')

    def fail_replace(*args):
        raise windows_lock_error(32)

    def interrupt(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(bot.os, 'replace', fail_replace)
    monkeypatch.setattr(bot.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt):
        bot.atomic_bytes(path, b'new')
    assert path.read_bytes() == b'old'
    assert list(tmp_path.iterdir()) == [path]


def test_cleanup_error_does_not_mask_original_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path/'state.json'
    path.write_bytes(b'old')
    error = OSError(errno.ENOSPC, 'original_disk_full')
    unlink = bot.os.unlink

    def fail_replace(*args):
        raise error

    def fail_cleanup(*args, **kwargs):
        raise windows_lock_error(5)

    monkeypatch.setattr(bot.os, 'replace', fail_replace)
    monkeypatch.setattr(bot.os, 'unlink', fail_cleanup)
    try:
        with pytest.raises(OSError) as caught:
            bot.atomic_bytes(path, b'new')
        assert caught.value is error
        assert path.read_bytes() == b'old'
        assert '一時ファイル削除失敗' in caplog.text
    finally:
        for scratch in tmp_path.glob('.state.json.*'):
            unlink(scratch)


@pytest.mark.skipif(os.name != 'nt', reason='requires Windows file sharing semantics')
def test_real_windows_reader_lock_recovers_after_handle_closes(tmp_path, monkeypatch):
    path = tmp_path/'state.json'
    path.write_bytes(b'old')
    waits = []
    with path.open('rb') as reader:
        def release_reader(delay):
            waits.append(delay)
            assert reader.read() == b'old'
            reader.close()

        monkeypatch.setattr(bot.time, 'sleep', release_reader)
        bot.atomic_bytes(path, b'new')
    assert len(waits) == 1
    assert path.read_bytes() == b'new'
    assert list(tmp_path.iterdir()) == [path]
