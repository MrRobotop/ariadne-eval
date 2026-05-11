"""Migration discovery, ordering, application, and _meta tracking."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from ariadne_eval.storage.migrations import (
    Migration,
    apply_pending,
    discover_migrations,
)


@pytest.mark.fast
def test_discover_returns_sorted_by_version(tmp_path):
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "010_ten.sql").write_text("CREATE TABLE ten (id INTEGER);")

    migs = discover_migrations(tmp_path)
    assert [m.version for m in migs] == [1, 2, 10]
    assert migs[0].name == "one"


@pytest.mark.fast
def test_discover_ignores_non_sql_files(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "README.md").write_text("not sql")
    (tmp_path / "002_two.txt").write_text("not sql")

    migs = discover_migrations(tmp_path)
    assert len(migs) == 1


@pytest.mark.fast
def test_discover_rejects_malformed_filename(tmp_path):
    (tmp_path / "bad-name.sql").write_text("--")
    with pytest.raises(ValueError):
        discover_migrations(tmp_path)


@pytest.mark.fast
def test_apply_pending_creates_meta_table_and_runs_all_on_empty(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")

    db = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db))
    try:
        applied = apply_pending(conn, tmp_path)
        assert applied == 2

        rows = conn.execute("SELECT version, name FROM _meta ORDER BY version").fetchall()
        assert rows == [(1, "one"), (2, "two")]

        names = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        assert {"one", "two", "_meta"}.issubset(names)
    finally:
        conn.close()


@pytest.mark.fast
def test_apply_pending_only_runs_missing_versions(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    db = tmp_path / "test.duckdb"

    # First pass: only v1 exists
    conn = duckdb.connect(str(db))
    apply_pending(conn, tmp_path)
    conn.close()

    # Add v2 to the directory
    (tmp_path / "002_two.sql").write_text("CREATE TABLE two (id INTEGER);")

    conn = duckdb.connect(str(db))
    try:
        applied = apply_pending(conn, tmp_path)
        assert applied == 1  # only v2

        rows = conn.execute("SELECT version FROM _meta ORDER BY version").fetchall()
        assert rows == [(1,), (2,)]
    finally:
        conn.close()


@pytest.mark.fast
def test_apply_pending_returns_zero_when_up_to_date(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    db = tmp_path / "test.duckdb"

    conn = duckdb.connect(str(db))
    apply_pending(conn, tmp_path)
    assert apply_pending(conn, tmp_path) == 0
    conn.close()


@pytest.mark.fast
def test_failed_migration_aborts_and_raises(tmp_path):
    (tmp_path / "001_one.sql").write_text("CREATE TABLE one (id INTEGER);")
    (tmp_path / "002_bad.sql").write_text("THIS IS NOT VALID SQL;")

    db = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db))
    try:
        from ariadne_eval.storage.base import StoreError

        with pytest.raises(StoreError) as exc:
            apply_pending(conn, tmp_path)
        assert "002" in str(exc.value)

        # v1 still applied (committed before failure)
        rows = conn.execute("SELECT version FROM _meta ORDER BY version").fetchall()
        assert rows == [(1,)]
    finally:
        conn.close()


@pytest.mark.fast
def test_initial_schema_is_discoverable():
    """The bundled 001_initial.sql is found by importlib.resources."""
    from importlib.resources import files

    bundled_dir = files("ariadne_eval.storage") / "migrations_sql"
    migs = discover_migrations(Path(str(bundled_dir)))
    assert any(m.version == 1 and m.name == "initial" for m in migs)


@pytest.mark.fast
def test_migration_dataclass_fields():
    m = Migration(version=42, name="hello", sql="SELECT 1;")
    assert m.version == 42
    assert m.name == "hello"
    assert m.sql == "SELECT 1;"
