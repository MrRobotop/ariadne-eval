"""ULID-based time-sortable string IDs.

We use ``python-ulid`` (already pinned in pyproject.toml). ULIDs are 26-char
Crockford base32 strings, sortable lexicographically by the millisecond at
which they were minted. Crockford base32 excludes ``I``, ``L``, ``O``, and
``U`` to avoid visual ambiguity.
"""

from __future__ import annotations

from typing import Any, Final

from ulid import ULID

__all__ = ["is_valid_id", "new_id"]


_VALID_CHARS: Final[frozenset[str]] = frozenset(
    "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
)
"""Crockford base32 alphabet: digits + uppercase letters minus I, L, O, U."""


def new_id() -> str:
    """Mint a new time-sortable string ID."""
    return str(ULID())


def is_valid_id(value: Any) -> bool:  # noqa: ANN401 - public API takes arbitrary input
    """Return ``True`` iff ``value`` is a syntactically valid ULID string.

    Does not raise on non-string input; callers can pass JSON-decoded
    values directly without a type-check.
    """
    if not isinstance(value, str):
        return False
    if len(value) != 26:
        return False
    return all(ch in _VALID_CHARS for ch in value)
