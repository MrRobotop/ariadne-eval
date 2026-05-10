# CLAUDE.md

This file is the persistent context for Claude Code working on this repository. Read it at the start of every session.

---

## Project: `ariadne-eval` — Trajectory-Level Observability and Evaluation for LLM Agents

**One-line description:** An open-source, self-hosted Python library and replay UI for tracing, evaluating, and detecting regressions in LLM agent trajectories — purpose-built for multi-step, tool-using agents.

**The metaphor:** Ariadne gave Theseus a thread to find his way back through the labyrinth. When your agent gets lost twelve steps into a task, `ariadne-eval` is the thread that lets you trace what happened, score the trajectory, and catch regressions before users do.

**Why this exists:** Most LLM observability tools (LangSmith, Helicone, Langfuse) treat agent runs as flat sequences of API calls. They're designed for chat. They miss the structure — plans, tool calls, recovery from errors, decision branches — that *is* the agent. The few that handle agents are SaaS-only, vendor-locked, and don't have rigorous evaluation built in. `ariadne-eval` is open source, self-hosted, framework-agnostic, and treats trajectory evaluation as a first-class concern.

**Audience:**
- **Primary:** Engineers running LLM agents in production who want to understand and improve them. The library must be `pip install ariadne-eval` and useful within 10 minutes of reading the README.
- **Secondary:** Hiring managers and senior engineers reviewing the GitHub repo. The README, METHODOLOGY.md, and demo deployment are first-class deliverables.

---

## Goals (ordered)

1. **Useful in 10 minutes.** A new user should be able to `pip install ariadne-eval`, decorate one function, run an agent, and see a trajectory in the replay UI within 10 minutes. If that flow breaks, everything else is irrelevant.
2. **Framework-agnostic.** Must work with LangGraph, CrewAI, AutoGen, OpenAI Assistants, and hand-rolled agents. Achieved by tracing at the LLM-call and tool-call boundaries, not by wrapping any specific framework.
3. **Statistically honest.** All evaluation results carry confidence intervals. Judge agreement is reported, not hidden. Drift detection has documented false-positive rates.
4. **Production-grade.** Low overhead (target <2% added latency on traced agents), sampling support, async-first, no global state pollution.
5. **Extensible.** Custom metrics, custom judges, custom storage backends. The core abstractions are the product.
6. **Reproducible.** Benchmark runs are deterministic given a seed. Trace data is portable (export to JSON/Parquet).

---

## Tech Stack (all open source)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Type system, performance, ecosystem |
| Package manager | `uv` | Fast, reproducible |
| Data validation | `pydantic` v2 | Trajectory schemas |
| Storage | `duckdb` | Embedded, columnar, excellent for trace queries |
| Async runtime | `asyncio` + `anyio` | Standard, framework-neutral |
| LLM gateway | `litellm` | One interface for OpenAI / Anthropic / open-weights |
| Tracing primitive | OpenTelemetry-compatible spans | Interop without dependency |
| Telemetry export | Optional OTLP exporter | For users with existing OTel infra |
| CLI | `click` + `rich` | Standard, good UX |
| Replay UI | `streamlit` | Fast iteration, multipage |
| Graph visualization | `pyvis` or `networkx` + d3 export | Tree/DAG of agent decisions |
| Stats | `scipy`, `numpy`, `statsmodels` | CIs, drift detection |
| Benchmark targets | `tau-bench` (Sierra), `swe-bench-lite`, custom | Recognized public benchmarks |
| Testing | `pytest`, `pytest-asyncio`, `hypothesis`, `pytest-recording` | Property tests + VCR for HTTP |
| Linting/formatting | `ruff` | Fast, opinionated |
| Type checking | `mypy --strict` | Library code must be strict-typed |
| CI | GitHub Actions | Matrix testing across Python versions |
| Docs | `mkdocs-material` | Search, code copy, API ref |
| License | Apache 2.0 | Permissive enough for enterprise adoption |
| Distribution | PyPI (`ariadne-eval`) | Standard Python packaging |

**Models used in evaluation and judging:**
- Claude Sonnet (current generation) — primary judge model
- GPT-4o, GPT-4o-mini — secondary judge / cross-validation
- Llama 3.3 70B via Groq or Ollama — open-weights option for sensitive use cases
- Reference agents driven by the user's choice of model

The user supplies API keys via `.env`. The library never makes network calls without an explicit user action (model adapter init, eval run, judge invocation).

---

## Repository Layout

```
ariadne-eval/
├── CLAUDE.md
├── README.md                       # Public-facing, results- and quickstart-first
├── METHODOLOGY.md                  # Trajectory metrics, judge calibration, drift methodology
├── CHANGELOG.md
├── LICENSE                         # Apache 2.0
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/
│   │   ├── tests.yml               # Matrix: 3.11, 3.12, 3.13
│   │   ├── publish-pypi.yml        # On tag push
│   │   ├── docs.yml                # Build & deploy mkdocs to gh-pages
│   │   └── benchmark-tracking.yml  # Track agent benchmark scores over time
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/                           # mkdocs site
│   ├── index.md
│   ├── quickstart.md
│   ├── concepts/
│   ├── tutorials/
│   ├── reference/                  # API reference (mkdocstrings)
│   └── screenshots/
├── src/ariadne_eval/
│   ├── __init__.py                 # Public API surface
│   ├── _version.py
│   ├── core/
│   │   ├── trajectory.py           # Trajectory, Step, ToolCall, LLMCall (Pydantic)
│   │   ├── status.py               # StepStatus enum
│   │   └── ids.py                  # ULID-based IDs
│   ├── tracing/
│   │   ├── decorator.py            # @trace decorators
│   │   ├── context.py              # Context manager API
│   │   ├── otel_bridge.py          # Optional OpenTelemetry export
│   │   └── sampler.py              # Production sampling
│   ├── storage/
│   │   ├── base.py                 # AbstractStore protocol
│   │   ├── duckdb_store.py         # Default backend
│   │   ├── jsonl_store.py          # Portable / archival format
│   │   └── migrations.py           # Schema versioning
│   ├── adapters/                   # Framework-specific helpers
│   │   ├── litellm.py              # Auto-trace LLM calls
│   │   ├── langgraph.py            # Optional, lazy-imported
│   │   ├── crewai.py               # Optional, lazy-imported
│   │   └── openai_assistants.py    # Optional, lazy-imported
│   ├── eval/
│   │   ├── runner.py               # Trajectory evaluation orchestrator
│   │   ├── metrics/
│   │   │   ├── final_answer.py     # Task-success metric
│   │   │   ├── tool_accuracy.py    # Per-call correctness
│   │   │   ├── plan_quality.py     # Pre-execution plan score
│   │   │   ├── recovery.py         # Recovery-from-error rate
│   │   │   ├── efficiency.py       # Steps-to-completion vs gold
│   │   │   └── base.py
│   │   ├── judges/
│   │   │   ├── base.py
│   │   │   ├── trajectory_judge.py # LLM-as-judge over full trajectory
│   │   │   ├── stepwise_judge.py   # Per-step judgment
│   │   │   ├── prompts.py
│   │   │   └── calibration.py
│   │   └── stats/
│   │       ├── bootstrap.py
│   │       ├── agreement.py        # Cohen's kappa, Krippendorff's alpha
│   │       └── drift.py            # CUSUM, ADWIN-style drift detection
│   ├── benchmarks/
│   │   ├── base.py                 # AbstractBenchmark
│   │   ├── tau_bench.py            # tau-bench integration
│   │   ├── swe_bench_lite.py       # SWE-Bench Lite integration
│   │   └── synthetic.py            # Generated tool-use benchmarks
│   ├── ui/
│   │   ├── app.py                  # Streamlit entry point
│   │   ├── pages/
│   │   │   ├── 01_runs.py
│   │   │   ├── 02_trajectory.py    # Drill-down with graph view
│   │   │   ├── 03_compare.py
│   │   │   ├── 04_metrics.py
│   │   │   ├── 05_drift.py
│   │   │   └── 06_calibration.py
│   │   └── components/
│   │       ├── graph.py            # Trajectory graph rendering
│   │       └── diff.py             # Trajectory diff viewer
│   ├── cli/
│   │   ├── main.py                 # `ariadne` entry point
│   │   ├── eval.py                 # `ariadne eval`
│   │   ├── ui.py                   # `ariadne ui`
│   │   ├── bench.py                # `ariadne bench`
│   │   └── export.py               # `ariadne export`
│   └── examples/                   # Importable reference agents
│       ├── react_agent.py
│       └── tool_use_agent.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── property/
│   └── fixtures/
├── examples/                       # End-user examples (separate from src/examples)
│   ├── 01_quickstart/
│   ├── 02_langgraph_integration/
│   ├── 03_custom_metric/
│   ├── 04_ci_regression/
│   └── 05_production_sampling/
├── benchmarks/                     # Performance benchmarks (overhead, throughput)
│   └── overhead.py
└── scripts/
    ├── build_calibration_set.py
    └── publish.sh
```

---

## Conventions

**Public API discipline.** Anything in `ariadne_eval/__init__.py` is public. Anything else is private and may change without warning. Document the public surface in `docs/reference/`.

**Strict typing.** `mypy --strict` must pass on `src/`. Tests can be looser.

**Code style.** `ruff format` + `ruff check`. Line length 100. Type hints everywhere in `src/`.

**Testing tiers.**
- `pytest -m fast` — pure-Python, no network, <10s. Default in pre-commit.
- `pytest -m integration` — uses recorded HTTP cassettes (pytest-recording). Re-record with `--record-mode=rewrite` and a real key.
- `pytest -m slow` — runs real benchmarks. Manual trigger only.

**Async-first.** Public APIs are async by default with sync wrappers via `asyncio.run` for the CLI.

**No global state.** Tracing context is propagated via `contextvars`, never via module-level globals. This is critical for async correctness and for use inside frameworks that own the event loop.

**Performance budget.** `@trace` overhead must stay under 2% on a representative workload (measured in `benchmarks/overhead.py`, run in CI on a fixed runner). Regressions fail the build.

**Storage migrations.** Schema changes go through `migrations.py` with a version number. Old DuckDB files must remain readable.

**Determinism.** Every entry point that produces evaluations takes a `seed`. Bootstrap runs are seeded. Judge sampling is seeded.

**Secrets.** Never logged. `.env` gitignored. CI uses repo secrets.

**Commits.** Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`, `chore:`, `test:`, `refactor:`).

**Versioning.** SemVer. Breaking changes only at major versions. v0.x is "API may change" but we still write changelog entries.

---

## Skills to Use

This repo is built with [Superpowers](https://github.com/obra/superpowers) (open source, by Jesse Vincent). The following skills should activate automatically — when they don't, invoke them explicitly:

- **`brainstorming`** — Before any non-trivial feature, refine the design through questions before writing code.
- **`writing-plans`** — Break work into 2–5 minute tasks with exact file paths and verification steps.
- **`test-driven-development`** — RED → GREEN → REFACTOR. Failing test before any implementation.
- **`systematic-debugging`** — Four-phase root cause process when something breaks.
- **`verification-before-completion`** — Don't claim "done" without running the verification step.
- **`using-git-worktrees`** — One worktree per phase to keep main green.
- **`subagent-driven-development`** — Dispatch independent tasks to subagents for parallel progress.

Slash commands:
- `/superpowers:brainstorm`
- `/superpowers:write-plan`
- `/superpowers:execute-plan`

Also useful in this project:
- The built-in **frontend-design** skill (`/mnt/skills/public/frontend-design/SKILL.md`) for the Streamlit UI components, especially the trajectory graph view.
- The built-in **product-self-knowledge** skill for any references to Anthropic API features in the docs.

---

## Hard Rules ("Don'ts")

1. **Never depend on a SaaS service.** If a SaaS-only tool is the easiest option, write the open-source equivalent. Self-hostable is a non-negotiable property.
2. **Never break the public API in a minor release.** SemVer is real here.
3. **Never add a framework-specific dependency to the core.** LangGraph, CrewAI, etc. integrations are optional extras (`pip install ariadne-eval[langgraph]`), lazy-imported, and absence does not break core.
4. **Never report a metric without a confidence interval.** Single numbers lie.
5. **Never ship an LLM-as-judge without calibration data.** Judge agreement vs. human labels is shown in the docs and reproducible via `scripts/build_calibration_set.py`.
6. **Never silently swallow tracing errors.** Tracing must fail loudly in dev, fail safe in prod (configurable via `ARIADNE_FAIL_MODE`).
7. **Never block the user's event loop.** Tracing writes are async; blocking writes are an opt-in escape hatch only.
8. **Never log raw prompts or completions by default.** Privacy matters. Provide a `redact` hook and an explicit `capture_payloads` flag, default off in production-mode storage.
9. **Don't generalize prematurely.** Build for the reference ReAct agent first; abstractions emerge from the second integration, not the first.
10. **Don't ship without docs.** A feature without a docs page in `docs/` is unfinished.

---

## Definition of Done (per phase)

A phase is complete only when **all** of the following hold:

- [ ] All new code has unit tests (>90% line coverage on touched files in `src/`).
- [ ] `mypy --strict src/` is clean.
- [ ] `ruff check` and `ruff format --check` are clean.
- [ ] `pytest -m fast` passes.
- [ ] If the phase produces a user-facing artifact, there's a docs page added under `docs/` and a screenshot or recorded run in the PR description.
- [ ] `README.md` is updated if the public API or quickstart changed.
- [ ] A `CHANGELOG.md` entry is appended under `## [Unreleased]`.
- [ ] If the phase touches the public API, there's an entry in `docs/reference/`.

---

## Headline Deliverables (the "is this impressive" gate)

By the time the project is ready for a 1.0.0 release, the following must exist and be linked from the README:

1. **A 60-second quickstart** that takes a user from zero to a viewable trajectory in the UI. Recorded as an animated GIF, embedded in the README.
2. **Headline benchmark numbers**: scores for ≥3 reference agents on tau-bench (or equivalent) with bootstrapped 95% CIs.
3. **Judge calibration table**: kappa values for the trajectory judge against a hand-labeled gold set of ≥50 examples.
4. **Performance overhead measurement**: a chart showing tracing overhead is <2% on a representative workload.
5. **A live demo deployment** of the UI on a hosted Streamlit (or fly.io) instance, populated with anonymized example trajectories.
6. **A blog post** in `docs/blog/` that walks through one non-obvious finding from building the project (e.g., "Why per-step judges disagree more than full-trajectory judges, and what we did about it").
7. **PyPI release** of v0.1.0+ with installation instructions in the README that actually work.

If any of those is missing, the project is not done.

---

## How to Talk to the User

The user (the maintainer) is building this both as a useful open-source library and as a portfolio piece demonstrating LLMOps competence. Optimize for:
- **Honesty over enthusiasm.** If a metric is noisy, say so. If a benchmark is gameable, document the gaming.
- **Engineering depth over breadth.** Better to do five things rigorously than fifteen sloppily.
- **Adoption-readability.** Every PR should produce something a user could `pip install` from a pre-release and try.
- **Recruiter-readability.** Every PR should produce something a hiring manager could open and understand in 30 seconds.
