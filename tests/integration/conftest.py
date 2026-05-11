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
