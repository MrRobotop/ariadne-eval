"""End-to-end DuckDBStore tests."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ariadne_eval.storage.duckdb_store import DuckDBStore


@pytest.mark.fast
async def test_init_creates_file_and_runs_migrations(tmp_path):
    db = tmp_path / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        conn = duckdb.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT version, name FROM _meta ORDER BY version"
            ).fetchall()
            assert (1, "initial") in rows
        finally:
            conn.close()
    finally:
        await store.close()


@pytest.mark.fast
async def test_init_creates_parent_directory(tmp_path):
    db = tmp_path / "nested" / "subdir" / "store.duckdb"
    store = DuckDBStore(path=db)
    try:
        assert db.exists()
        assert db.parent.is_dir()
    finally:
        await store.close()


@pytest.mark.fast
async def test_constructor_path_overrides_env_var(tmp_path, monkeypatch):
    env_db = tmp_path / "env.duckdb"
    arg_db = tmp_path / "arg.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(env_db))

    store = DuckDBStore(path=arg_db)
    try:
        assert arg_db.exists()
        assert not env_db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_env_var_used_when_no_arg(tmp_path, monkeypatch):
    db = tmp_path / "env.duckdb"
    monkeypatch.setenv("ARIADNE_STORE_PATH", str(db))
    store = DuckDBStore()
    try:
        assert db.exists()
    finally:
        await store.close()


@pytest.mark.fast
async def test_default_path_is_under_dot_ariadne(monkeypatch, tmp_path):
    """When no path or env var is set, default is ~/.ariadne/store.duckdb."""
    monkeypatch.delenv("ARIADNE_STORE_PATH", raising=False)
    fake_home = tmp_path / "fake_home"
    monkeypatch.setenv("HOME", str(fake_home))

    store = DuckDBStore()
    try:
        expected = fake_home / ".ariadne" / "store.duckdb"
        assert expected.exists()
    finally:
        await store.close()
