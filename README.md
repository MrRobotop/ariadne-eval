# ariadne-eval

> Trajectory-level observability and evaluation for LLM agents. Open source, self-hosted, framework-agnostic.

[![PyPI version](https://img.shields.io/pypi/v/ariadne-eval.svg)](https://pypi.org/project/ariadne-eval/)
[![Python versions](https://img.shields.io/pypi/pyversions/ariadne-eval.svg)](https://pypi.org/project/ariadne-eval/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/MrRobotop/ariadne-eval/actions/workflows/tests.yml/badge.svg)](https://github.com/MrRobotop/ariadne-eval/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://MrRobotop.github.io/ariadne-eval/)

> **Status: pre-alpha (v0.0.1).** Bootstrap scaffold only. The public API,
> tracing primitives, replay UI, and benchmark integrations land in subsequent
> releases. See [CHANGELOG.md](./CHANGELOG.md) for what's shipped.

---

## Why ariadne-eval?

Most LLM observability tools were designed for chat: they treat an agent run
as a flat sequence of API calls and lose the structure — plans, tool calls,
recovery from errors, decision branches — that *is* the agent. The few tools
that handle agents are SaaS-only, vendor-locked, and rarely ship with rigorous
evaluation built in. `ariadne-eval` is open source, self-hosted, framework-
agnostic, and treats trajectory-level evaluation (not just final-answer
accuracy) as a first-class concern.

The metaphor: Ariadne gave Theseus a thread to find his way back through the
labyrinth. When your agent gets lost twelve steps into a task, `ariadne-eval`
is the thread that lets you trace what happened, score the trajectory, and
catch regressions before users do.

---

## Install

```bash
pip install ariadne-eval
```

Optional integrations:

```bash
pip install "ariadne-eval[langgraph]"        # LangGraph adapter
pip install "ariadne-eval[crewai]"           # CrewAI adapter
pip install "ariadne-eval[openai-assistants]"
pip install "ariadne-eval[all]"              # everything plus dev tooling
```

---

## Quickstart

> The quickstart below is a **placeholder** for the v0.0.1 scaffold. It will
> be filled in once the tracing API ships in v0.0.3. Track progress in the
> [CHANGELOG](./CHANGELOG.md).

```python
# Coming in v0.0.3:
# import ariadne_eval as ae
#
# with ae.start_trajectory("compute 17*23 / len('banana')",
#                          agent_name="react",
#                          model_id="claude-sonnet"):
#     ...  # run your agent
#
# Then: `ariadne ui` to view the trajectory.
```

For now, `ariadne --version` prints `0.0.1`.

---

## Screenshots & headline benchmarks

> Placeholders. The replay UI lands in v0.0.9; the headline tau-bench results
> with bootstrapped 95% CIs land in v0.0.7.

| Agent | Benchmark | Success rate (95% CI) | Mean cost / task |
| --- | --- | --- | --- |
| _tbd_ | _tbd_ | _tbd_ | _tbd_ |

---

## What's planned

| Phase | Milestone | Status |
| --- | --- | --- |
| 0 | Bootstrap, license, CI scaffold | shipped (v0.0.1) |
| 1 | Trajectory data model | planned |
| 2 | DuckDB storage layer | planned |
| 3 | `@trace` decorator + context API | planned |
| 4 | Reference ReAct agent + E2E test | planned |
| 5 | Programmatic trajectory metrics | planned |
| 6 | Calibrated LLM judge (with kappa) | planned |
| 7 | tau-bench runner | planned |
| 8 | CLI polish | planned |
| 9 | Streamlit replay UI | planned |
| 10 | Drift detection | planned |
| 11 | PyPI release + docs | planned |

---

## Documentation

- [Full documentation](https://MrRobotop.github.io/ariadne-eval/) (deploys with v0.1.0)
- [Contributing guide](./CONTRIBUTING.md)
- [Changelog](./CHANGELOG.md)
- Methodology (lands with v0.0.5): see `METHODOLOGY.md`

---

## License

[Apache License 2.0](./LICENSE). Includes explicit patent grants — friendly
to enterprise adoption.

---

## Citation

A `CITATION.cff` file ships with v0.1.0. Until then, cite as:

> Patil, R. (2026). *ariadne-eval: trajectory-level observability and
> evaluation for LLM agents* (Version 0.0.1) [Computer software].
> https://github.com/MrRobotop/ariadne-eval
