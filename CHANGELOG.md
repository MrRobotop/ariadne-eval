# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core trajectory data model: `Trajectory`, `Step`, `Message`, four payload
  variants (`LLMCallPayload`, `ToolCallPayload`, `UserInputPayload`,
  `InternalPayload`), `StepError`, `StepStatus`, `TrajectoryStatus`,
  `JsonValue`, `new_id`, `is_valid_id`. Validators: tz-aware datetimes,
  ULID format, no self-parenting, failed-step requires error. Truncation on
  `completion` and `result` above 64K chars. Opt-in `Trajectory.redact()`
  hook. Hypothesis round-trip property tests (200 examples each).

## [0.0.1] - 2026-05-10

### Added
- Bootstrap: project scaffold, Apache-2.0 license, ruff/mypy/pytest configuration,
  pre-commit hooks, README skeleton, mkdocs site skeleton, no-op CLI entrypoint
  (`ariadne --version`), and a smoke test asserting the package imports and the
  CLI is registered.

[Unreleased]: https://github.com/MrRobotop/ariadne-eval/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/MrRobotop/ariadne-eval/releases/tag/v0.0.1
