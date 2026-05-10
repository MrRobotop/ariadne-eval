# Contributing to ariadne-eval

Thanks for considering a contribution. `ariadne-eval` is an open-source library
for trajectory-level observability and evaluation of LLM agents — pull requests,
issues, and discussions are all welcome.

## Quick start

```bash
git clone https://github.com/MrRobotop/ariadne-eval.git
cd ariadne-eval
uv sync --all-extras
uv run pre-commit install
uv run pytest -m fast
```

If those four commands succeed you have a working dev environment.

## Workflow

1. **Open an issue first** for non-trivial changes. A 5-minute discussion is
   cheaper than a 5-hour PR that needs to be redesigned.
2. **Branch from `main`**. Use a descriptive name: `feat/drift-cusum`,
   `fix/duckdb-lock`, `docs/quickstart-typo`.
3. **Write tests before fixing or adding behavior.** This project uses
   test-driven development; the test suite is the spec.
4. **Keep PRs focused.** One topic per PR. Refactors and feature work go in
   separate PRs.
5. **Update `CHANGELOG.md`** under `## [Unreleased]` with one line describing
   the user-visible change.

## Code style

- `ruff format` and `ruff check` must be clean. Pre-commit enforces both.
- `mypy --strict src/` must pass. Tests can be looser; library code cannot.
- Type hints are required on every public function and method.
- Docstrings use Google style. Public symbols must have a docstring; private
  helpers may skip them.
- Line length: 100.

## Testing tiers

- `pytest -m fast` — pure-Python, no network, runs in <10 s. Default.
- `pytest -m integration` — uses recorded HTTP cassettes (pytest-recording).
  Re-record with a real key via `--record-mode=rewrite`.
- `pytest -m slow` — real benchmark runs and the overhead benchmark. Manual
  trigger only.

A PR that adds a feature without unit tests will not be merged.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/) format. Examples:

- `feat(tracing): add RateSampler with seed support`
- `fix(storage): DuckDB write lock now releases on cancellation`
- `docs(quickstart): correct litellm import path`
- `perf(metrics): vectorize bootstrap inner loop`

The git log is part of the project's public artifact — write commit messages
that are useful when read months later.

## Reporting bugs

Open an issue with a minimal reproduction, the version of `ariadne-eval`,
your Python version, and the relevant model provider. If the bug involves a
trajectory, attach the JSONL export (`ariadne export jsonl`).

## Be kind

Be respectful, assume good faith, and remember that strangers on the internet
have their own context. Disagreements about code are healthy; personal attacks
are not. Maintainers will moderate threads that go off the rails.
