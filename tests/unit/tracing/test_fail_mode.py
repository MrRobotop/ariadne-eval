"""Fail-mode policy: env-var resolution and unattached-record handling."""

from __future__ import annotations

import importlib
import warnings

import pytest

from ariadne_eval.tracing import _fail_mode as fm


@pytest.mark.fast
def test_fail_mode_enum_values():
    assert fm.FailMode.STRICT.value == "strict"
    assert fm.FailMode.WARN.value == "warn"
    assert fm.FailMode.SILENT.value == "silent"


@pytest.mark.fast
def test_resolve_default_is_strict(monkeypatch):
    monkeypatch.delenv("ARIADNE_FAIL_MODE", raising=False)
    assert fm._resolve_fail_mode() == fm.FailMode.STRICT


@pytest.mark.fast
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("strict", fm.FailMode.STRICT),
        ("WARN", fm.FailMode.WARN),
        ("silent", fm.FailMode.SILENT),
        ("Silent", fm.FailMode.SILENT),
    ],
)
def test_resolve_from_env(monkeypatch, raw, expected):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", raw)
    assert fm._resolve_fail_mode() == expected


@pytest.mark.fast
def test_resolve_invalid_env_raises(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "banana")
    with pytest.raises(ValueError) as exc:
        fm._resolve_fail_mode()
    assert "banana" in str(exc.value)


@pytest.mark.fast
def test_unattached_warning_class():
    assert issubclass(fm.UnattachedTracingWarning, UserWarning)


@pytest.mark.fast
def test_handle_unattached_strict_raises(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "strict")
    importlib.reload(fm)
    with pytest.raises(RuntimeError) as exc:
        fm.handle_unattached("record_llm_call")
    assert "no active trajectory" in str(exc.value).lower()
    assert "record_llm_call" in str(exc.value)


@pytest.mark.fast
def test_handle_unattached_warn_logs_once(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "warn")
    importlib.reload(fm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fm.UnattachedTracingWarning)
        fm.handle_unattached("record_llm_call")
        fm.handle_unattached("record_llm_call")
        fm.handle_unattached("record_tool_call")
    types = [w.category for w in caught if issubclass(w.category, fm.UnattachedTracingWarning)]
    assert len(types) == 1


@pytest.mark.fast
def test_handle_unattached_silent_returns_quietly(monkeypatch):
    monkeypatch.setenv("ARIADNE_FAIL_MODE", "silent")
    importlib.reload(fm)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fm.handle_unattached("record_llm_call")
    assert caught == []
