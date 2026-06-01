"""Smoke test: package imports and CLI entrypoint is registered.

Marked ``fast`` so it runs in the default pytest selection. The CLI invocation
goes through ``click.testing.CliRunner`` rather than spawning a subprocess so
the test stays fast and works inside ``uv run pytest`` without depending on
PATH wiring of the installed console script.
"""

from __future__ import annotations

import subprocess
import sys
from shutil import which

import pytest
from click.testing import CliRunner

import ariadne_eval
from ariadne_eval.cli.main import cli


@pytest.mark.fast
def test_package_version_is_pinned():
    assert ariadne_eval.__version__ == "0.0.9-alpha"


@pytest.mark.fast
def test_cli_version_via_click_runner():
    """The cheap path: CLI works in-process."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.0.9-alpha" in result.output


@pytest.mark.fast
def test_cli_hello_subcommand():
    runner = CliRunner()
    result = runner.invoke(cli, ["hello", "--name", "ariadne"])
    assert result.exit_code == 0
    assert "ariadne" in result.output
    assert "0.0.9-alpha" in result.output


@pytest.mark.fast
def test_public_api_exports_core_types():
    """Pin the public surface so accidental removals are caught early."""
    import ariadne_eval

    expected = {
        "__version__",
        "Trajectory",
        "Step",
        "Message",
        "ContentBlock",
        "TextBlock",
        "ToolCallRef",
        "LLMCallPayload",
        "ToolCallPayload",
        "UserInputPayload",
        "InternalPayload",
        "StepError",
        "StepStatus",
        "TrajectoryStatus",
        "JsonValue",
        "new_id",
        "is_valid_id",
        # Storage
        "Store",
        "DuckDBStore",
        "StoreError",
        "TrajectoryNotFoundError",
        "MetadataTooLargeError",
        "export_jsonl",
        "import_jsonl",
        # Tracing
        "start_trajectory",
        "current_trajectory",
        "current_step",
        "trace_step",
        "record_llm_call",
        "record_tool_call",
        "Sampler",
        "AlwaysSampler",
        "RateSampler",
        "TaskFilterSampler",
        "enable_litellm_autotrace",
        "TrajectoryHandle",
        "FailMode",
        "UnattachedTracingWarning",
    }
    missing = expected - set(ariadne_eval.__all__)
    assert not missing, f"Missing from public API: {missing}"
    for name in expected:
        assert hasattr(ariadne_eval, name), f"ariadne_eval.{name} not importable"


@pytest.mark.fast
def test_installed_console_script_runs():
    """Verifies the entry point is wired up by ``uv sync`` / ``pip install``.

    Skipped if ``ariadne`` isn't on PATH (e.g. before ``uv sync`` has run);
    CI and local dev environments will have it after the standard setup.
    """
    if which("ariadne") is None:
        pytest.skip("ariadne console script not on PATH; run `uv sync` first")
    result = subprocess.run(
        ["ariadne", "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "0.0.9-alpha" in result.stdout

    # And via `python -m` as a belt-and-braces check.
    result_module = subprocess.run(
        [sys.executable, "-m", "ariadne_eval.cli.main", "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result_module.returncode == 0
    assert "0.0.9-alpha" in result_module.stdout
