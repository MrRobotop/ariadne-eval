"""Opt-in redact hook on Trajectory."""

from datetime import UTC, datetime

import pytest

from ariadne_eval.core.ids import new_id
from ariadne_eval.core.status import TrajectoryStatus
from ariadne_eval.core.trajectory import Trajectory


def _t() -> Trajectory:
    return Trajectory(
        id=new_id(),
        task="compute 2+2",
        agent_name="react",
        agent_version="0.1",
        model_id="claude-sonnet",
        started_at=datetime.now(tz=UTC),
        finished_at=None,
        final_status=TrajectoryStatus.RUNNING,
        metadata={"user_email": "alice@example.com"},
    )


@pytest.mark.fast
def test_redact_with_noop_returns_equal_copy():
    t = _t()
    redacted = t.redact(lambda x: x)
    assert redacted == t
    assert redacted is not t  # returns a new instance


@pytest.mark.fast
def test_redact_with_user_function_modifies_metadata():
    t = _t()

    def scrub(traj: Trajectory) -> Trajectory:
        new_meta = {**traj.metadata, "user_email": "[REDACTED]"}
        return traj.model_copy(update={"metadata": new_meta})

    redacted = t.redact(scrub)
    assert redacted.metadata["user_email"] == "[REDACTED]"
    assert t.metadata["user_email"] == "alice@example.com"  # original untouched
