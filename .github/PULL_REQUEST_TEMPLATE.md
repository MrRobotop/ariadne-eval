<!--
Thanks for the PR. Please make sure your branch has all of the following
green locally before requesting review.

If this is a phase-execution PR, link the spec and plan from
`docs/superpowers/`.
-->

## Summary

<!-- 1-3 bullets describing the change and the why. -->

## Phase reference (if applicable)

<!--
e.g. "Implements Phase 6.1 per docs/superpowers/specs/2026-05-31-calibration-set-design.md
and docs/superpowers/plans/2026-05-31-calibration-set.md."
Delete this section for repo-polish / dependency / chore PRs.
-->

## Verification

- [ ] `uv run pytest -m "fast and not integration" -q` is green
- [ ] `uv run pytest -m integration -q` is green
- [ ] `uv run mypy --strict src/ariadne_eval` is clean
- [ ] `uv run ruff check src tests examples scripts` is clean
- [ ] `uv run ruff format --check src tests examples scripts` is clean
- [ ] `uv run mkdocs build --strict` succeeds
- [ ] `CHANGELOG.md` `[Unreleased]` updated if the public API changed
- [ ] Docs page added/updated for any new public symbol

## Notes for reviewers

<!--
Anything non-obvious: a deliberate scope cut, a known follow-up,
an integration gotcha worth flagging.
-->
