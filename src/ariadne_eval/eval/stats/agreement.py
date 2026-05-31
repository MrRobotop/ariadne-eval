"""Inter-rater agreement statistics — Cohen's kappa."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ariadne_eval.eval.errors import KappaInsufficientDataWarning

__all__ = ["KappaResult", "cohens_kappa"]


Interpretation = Literal["poor", "slight", "fair", "moderate", "substantial", "almost_perfect"]


class KappaResult(BaseModel):
    """Result of a Cohen's kappa computation."""

    model_config = {"frozen": True}

    kappa: float
    n: int
    label_set: tuple[str, ...]
    interpretation: Interpretation


def _interpret(kappa: float) -> Interpretation:
    """Landis-Koch (1977) interpretation bands.

    Half-open intervals on the right; ``< 0`` is poor, ``[0.0, 0.2)`` slight,
    ``[0.2, 0.4)`` fair, ``[0.4, 0.6)`` moderate, ``[0.6, 0.8)`` substantial,
    ``[0.8, 1.0]`` almost_perfect.
    """
    if math.isnan(kappa) or kappa < 0.0:
        return "poor"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost_perfect"


def cohens_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    *,
    labels: tuple[str, ...] | None = None,
) -> KappaResult:
    """Cohen's kappa for two raters over categorical labels.

    ``labels`` optionally fixes the label set (useful when the two raters
    do not span every category in the small sample). If omitted, the union
    of observed labels is used.

    Edge cases:

    - ``len(rater_a) != len(rater_b)``  → raises ``ValueError``.
    - ``n == 0``                         → ``kappa = NaN``, label_set ``()``,
      interpretation ``"poor"``, and a ``KappaInsufficientDataWarning``.
    - All entries equal AND single label → ``kappa = 1.0`` (perfect agreement).
    - Chance-agreement denominator zero  → ``kappa = NaN`` with a warning.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"rater_a and rater_b must have the same length ({len(rater_a)} vs {len(rater_b)})"
        )
    n = len(rater_a)
    if n == 0:
        warnings.warn(
            "cohens_kappa called with n=0; returning NaN",
            KappaInsufficientDataWarning,
            stacklevel=2,
        )
        return KappaResult(kappa=math.nan, n=0, label_set=(), interpretation="poor")

    if labels is None:
        label_set: tuple[str, ...] = tuple(sorted(set(rater_a) | set(rater_b)))
    else:
        label_set = labels

    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    p_o = float((a == b).mean())
    p_a = np.array([float(np.sum(a == lbl)) / n for lbl in label_set])
    p_b = np.array([float(np.sum(b == lbl)) / n for lbl in label_set])
    p_e = float(np.sum(p_a * p_b))

    if math.isclose(p_e, 1.0):
        if p_o == 1.0:
            kappa = 1.0
        else:
            warnings.warn(
                "cohens_kappa: degenerate distribution (p_e == 1); returning NaN",
                KappaInsufficientDataWarning,
                stacklevel=2,
            )
            kappa = math.nan
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return KappaResult(
        kappa=kappa,
        n=n,
        label_set=label_set,
        interpretation=_interpret(kappa),
    )
