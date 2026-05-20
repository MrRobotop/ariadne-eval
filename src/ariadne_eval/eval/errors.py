"""Errors and warnings raised by the evaluation layer."""

from __future__ import annotations

__all__ = ["BootstrapInsufficientDataWarning", "MissingReferenceError"]


class MissingReferenceError(ValueError):
    """A metric required a Case field that was not provided.

    Subclass of ``ValueError`` so callers who broadly catch validation
    errors still see it; ``Runner`` catches it explicitly to honor
    ``on_missing_reference``.
    """

    def __init__(self, field: str, *, case_id: str) -> None:
        """Construct with the missing field name and the case it belongs to."""
        super().__init__(f"Case {case_id!r} is missing required reference field {field!r}")
        self.field = field
        self.case_id = case_id


class BootstrapInsufficientDataWarning(UserWarning):
    """Emitted when ``bootstrap_mean_ci`` cannot produce a meaningful CI.

    Raised for ``n == 0`` (NaN result) and ``n == 1`` (degenerate CI equal
    to the single value).
    """
