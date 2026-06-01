"""Single source of truth for the ariadne-eval version string.

The value here mirrors the ``version`` field in ``pyproject.toml`` and is
read by ``ariadne_eval.__init__`` so users can do ``ariadne_eval.__version__``.

Bumping the version is intentionally a two-line edit (this file plus
``pyproject.toml``) so version drift is visible in code review.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.0.8-alpha"
