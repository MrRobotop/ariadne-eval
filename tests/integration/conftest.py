"""Shared fixtures for the integration test suite.

VCR config: redact auth headers, refuse to make real HTTP calls
(``record_mode='none'``), and match on method + URL only so the
cassette is robust to minor request-body differences.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    """pytest-recording reads this fixture for the ``@pytest.mark.vcr`` config."""
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-organization", "REDACTED"),
            ("anthropic-version", None),
        ],
        "record_mode": "none",
        "match_on": ["method", "scheme", "host", "port", "path"],
    }


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    """Provide dummy API keys + disable upstream startup HTTP calls.

    Dummy keys: client-side validation in upstream SDKs (litellm, openai,
    anthropic) raises before VCR can intercept if no key is set. The
    cassette serves the actual HTTP response; the keys never reach the
    network.

    ``LITELLM_LOCAL_MODEL_COST_MAP``: stops litellm from fetching the
    model-price JSON from GitHub at startup. Without this, every test
    would need the GitHub host in its cassette.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy-not-real")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
