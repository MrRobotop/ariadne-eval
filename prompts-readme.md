# prompts-readme.md

How to use `Prompts.md` and `CLAUDE.md` to build the `ariadne-eval` project end-to-end with Claude Code, ship it to PyPI, and publish it on GitHub. Read this **before** opening Claude Code.

---

## What these three files do

| File | Purpose | Who reads it |
|---|---|---|
| `CLAUDE.md` | Persistent project context: goals, stack, conventions, hard rules. | Claude Code, every session. |
| `Prompts.md` | 12 sequential build prompts, one per phase. | You — copy/paste one phase at a time. |
| `prompts-readme.md` | This file. Setup, workflow, troubleshooting. | You — once, before starting. |

You will not paste `CLAUDE.md` or `prompts-readme.md` into Claude. You will paste *prompts from `Prompts.md`*, one phase at a time, into a Claude Code session that already has `CLAUDE.md` checked into the repo.

---

## What you're building

`ariadne-eval` is an open-source Python library and self-hosted UI for trajectory-level observability and evaluation of LLM agents. It traces multi-step agent runs, scores them on five trajectory metrics plus a calibrated LLM judge, detects drift, and ships with a Streamlit replay UI.

**Why this is a strong second project after the eval harness:**
- Different problem space (multi-step agents, not single-turn).
- Different deliverable shape (a `pip install` library + UI, not just a benchmark).
- Demonstrates end-to-end ownership: design, implementation, packaging, release, deployment.
- Fills a genuine gap in the open-source LLM tooling ecosystem.

---

## Prerequisites

1. **Claude Code installed.** Latest version. See `https://docs.claude.com`.
2. **Superpowers plugin installed.** Open-source skills framework by Jesse Vincent (obra) at `https://github.com/obra/superpowers`. Provides the brainstorming, planning, TDD, debugging, and verification workflows the prompts assume.
3. **Python 3.11+** and **`uv`** (`https://docs.astral.sh/uv/`).
4. **Git**, **`gh` CLI** (for the GitHub release flow), **Docker** (optional, for benchmark reproducibility).
5. **API keys** for at least Anthropic and OpenAI. Optionally Groq.
6. **PyPI account** (for Phase 11). Set up Trusted Publisher with GitHub Actions OIDC — no API tokens required if you do this right.
7. **A GitHub repo** named `ariadne-eval` (create it empty before Phase 0; the bootstrap will populate it).

---

## Installing Superpowers

Inside Claude Code, run:

```
/plugin install superpowers@claude-plugins-official
```

Or via the community marketplace:

```
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```

Quit and restart Claude Code. At session start you should see a hook injection mentioning Superpowers — that confirms it's active.

Verify with `/help`. You should see:
- `/superpowers:brainstorm`
- `/superpowers:write-plan`
- `/superpowers:execute-plan`

The skills the prompts depend on:
- `brainstorming` — design refinement.
- `writing-plans` — task breakdown.
- `subagent-driven-development` / `executing-plans` — execution.
- `test-driven-development` — RED → GREEN → REFACTOR.
- `systematic-debugging` — root cause when things break.
- `verification-before-completion` — proof of done.
- `using-git-worktrees` — one branch per phase.

These activate automatically; the prompts mention them so you know what to expect.

---

## Other useful skills

- **`obra/superpowers-lab`** — experimental skills (`https://github.com/obra/superpowers-lab`).
- The skills shipped with Claude Code at `/mnt/skills/public/`:
  - `frontend-design` — used in Phase 9 (Streamlit UI design).
  - `product-self-knowledge` — used in Phase 11 (any docs that reference Anthropic API specifics).

All open source.

---

## One-time setup

```bash
# 1. Create a GitHub repo (empty) named "ariadne-eval" via the web or:
gh repo create ariadne-eval --public --description "Trajectory-level observability and evaluation for LLM agents"

# 2. Clone it locally
git clone https://github.com/YOURUSER/ariadne-eval.git
cd ariadne-eval

# 3. Place CLAUDE.md, Prompts.md, and prompts-readme.md in the repo root.
#    (CLAUDE.md is the only one Claude Code reads automatically.)

# 4. Open Claude Code in this directory
claude
```

When Claude Code starts, it reads `CLAUDE.md` automatically. Confirm by asking:
"What's the project goal, the public API surface, and the headline deliverables?"

If the answer isn't grounded in `CLAUDE.md`, stop and check that the file is in the repo root.

---

## The build workflow

**One phase per session.** This is non-negotiable for projects of this size — sessions accumulate context that confuses later turns, and the project will end up worse than necessary.

For each phase, in order:

1. **Open `Prompts.md` and copy the prompt for the next phase.**
2. **Start a new Claude Code session in the project directory.** Fresh context per phase.
3. **Paste the prompt.** Claude usually starts with `brainstorming` if the phase calls for it.
4. **Engage with the brainstorm honestly.** Push back on assumptions. This is the cheapest place to catch design errors.
5. **Approve the plan.** The `writing-plans` skill produces a task list. Read it. Reject it if a task is wrong.
6. **Let it execute.** Subagent-driven development dispatches tasks, runs tests, reports. Watch but don't micromanage.
7. **Run the verification step yourself.** Trust but verify. The phase's "Acceptance" section in `Prompts.md` is your checklist.
8. **Commit.** Conventional Commits format. The git log is part of the portfolio.
9. **Update CHANGELOG.md.** One line per phase under `## [Unreleased]`.
10. **Tag intermediate alpha releases** from Phase 7 onward (e.g., `v0.0.7-alpha`). GitHub Releases pages become a visible portfolio artifact.
11. **Close the session.** Open a fresh one for the next phase.

---

## When something fails

**If a test fails during execution:**
- Don't ask Claude to "just make it pass". That's how bad code ships.
- Invoke `systematic-debugging` explicitly: "Use systematic-debugging to find the root cause."
- Once the root cause is identified, decide whether to fix the code or fix the test. Both are valid. Document the choice.

**If Claude generates plausible-but-wrong code:**
- Stop the execution.
- Ask Claude to explain its assumptions in plain English.
- This usually surfaces a missing piece of context. Add it to `CLAUDE.md` so future sessions don't repeat the mistake.

**If the public API drifts during a phase:**
- This is serious for this project — `ariadne-eval` is a library that other people will depend on.
- Open `docs/reference/` and `src/ariadne_eval/__init__.py` and check what's exported.
- Any change to a public symbol must be in CHANGELOG.md under "Changed" or "Removed."
- After v0.1.0, breaking changes only at major versions. Be strict.

**If a phase's verification step fails:**
- Don't move on. Bad foundations compound.
- Open a new session, re-read `CLAUDE.md`, paste the failing test output, ask for a fix.

**If you find a real design flaw** (not an implementation issue):
- Update `CLAUDE.md` and `METHODOLOGY.md`.
- Add an `decisions/` directory with an ADR (architecture decision record) explaining what changed and why.
- Recruiters read decision records.

---

## Keeping context clean

Long Claude Code sessions accumulate context that confuses later turns. Mitigations:

- **One phase per session.** Already covered.
- **Use `using-git-worktrees`** so each phase is on its own branch — the repo state itself acts as context.
- **Don't paste large files into prompts** unless asked. Reference paths instead; Claude can read them.
- **If a session is going off the rails**, end it and start fresh. The next session will read `CLAUDE.md` and pick up from the committed state.

---

## How long this takes

Rough estimates with focused work and decent API throughput:

| Phase | Time |
|---|---|
| 0 — Bootstrap | 45–90 min |
| 1 — Trajectory data model | 1.5–2.5 hours |
| 2 — Storage layer | 2.5–4 hours |
| 3 — Tracing instrumentation | 4–6 hours (this is the hardest phase) |
| 4 — Reference agent + E2E | 2–3 hours |
| 5 — Programmatic metrics | 3–4 hours |
| 6 — Calibrated judge | 5–7 hours (includes manual labeling) |
| 7 — Benchmark runner | 4–6 hours |
| 8 — CLI polish | 2–3 hours |
| 9 — Streamlit UI | 5–8 hours |
| 10 — Drift detection | 3–4 hours |
| 11 — Docs + PyPI release | 5–7 hours |
| 12 — Reference deployment (optional) | 6–10 hours |

**Total: ~50–70 hours over 3–5 weeks** at a sustainable pace. Phase 3 is the hardest; budget extra. Phase 6 has unavoidable manual labeling work that you should not skip.

---

## Cost expectations

Most cost is in Phase 6 (judge calibration: multiple judge configs × calibration set) and Phase 7 (real benchmark runs). Estimates:

- One full tau-bench run with 50 tasks × 3 agents: **~$5–15**.
- One judge calibration run (50 trajectories × 3 prompts × 2 judges): **~$3–8**.
- CI runs (cached, mostly cassettes): negligible.
- Total budget over the project: **$80–200** if you cache aggressively. Caching is built in from Phase 3.

Set a hard cap in your provider dashboards before starting.

---

## What "done" looks like

You're done when a hiring manager (or potential user) opens the GitHub repo for 90 seconds and:

1. Sees the GIF of the replay UI in the README.
2. Sees the headline benchmark table with confidence intervals.
3. Sees the judge calibration table with kappa values.
4. Can `pip install ariadne-eval` and run the quickstart in 5 minutes.
5. Can click into the docs and find the API reference.
6. Can read METHODOLOGY.md and tell that you understand statistics.
7. Can see CI passing, PyPI version published, docs deployed.
8. (Optional) Can click a live demo URL and see the UI running with real data.

If any of those is missing, you're not done.

---

## What this signals to recruiters

In rough order of how impressive each component is:

1. **You built a library, not a notebook.** PyPI packaging, public API discipline, SemVer — these are senior signals.
2. **You take agent evaluation seriously at the trajectory level.** Most candidates measure final answers and call it done. Trajectory metrics + calibrated judges + drift detection is rare.
3. **You ship infrastructure with statistical rigor.** Bootstrap CIs, kappa, FPR-calibrated drift detection. Most candidates don't.
4. **You can deliver a UI.** The Streamlit app and animated GIF prove this isn't just backend code.
5. **You write maintainable code.** Strict typing, tests at multiple tiers, async-correctness, performance budgets enforced in CI.
6. **You communicate.** README, methodology doc, blog post, comparison page.
7. **You operate it.** Phase 12's live deployment is the rarest combination of skills in this list.

The combination is rare. Be honest in the write-ups about what's hard and what you'd do differently — false confidence reads worse than acknowledged limits.

---

## Notes specific to this project

**Naming.** The PyPI package is `ariadne-eval`, the import is `ariadne_eval`, the CLI is `ariadne`. Verify the PyPI name is available before Phase 0 — if it's taken, choose a variant (`ariadne-trace`, `ariadne-agent`) and update CLAUDE.md and Prompts.md before starting.

**License choice.** Apache 2.0, not MIT. Apache adds explicit patent grants which matter for enterprise adoption. The first project (eval harness) was MIT because that's a benchmark-style project; this is a library others will integrate into commercial products.

**The litellm dependency.** litellm is an active project; pin a minor version in pyproject.toml and bump deliberately. Don't track main.

**The tau-bench dependency.** Sierra's tau-bench is what we benchmark against in Phase 7. It's open source but check its license terms and credit it prominently.

**Privacy by default.** Phase 3 has a strict rule: payloads are not captured by default in production-mode storage. This is non-negotiable; don't relax it. Users opt in.

---

## Final note

This project is different from a portfolio demo because real people will use it. That changes the bar:

- A breaking API change in v0.2.0 inconveniences strangers who depend on you.
- A poorly-documented feature wastes their time.
- A flaky test in CI annoys contributors.
- A bad METHODOLOGY.md misleads people who trust your numbers.

Treat the library like a product. The code, the docs, the issues you respond to, the releases you cut, and the discussions you have on GitHub — all of it is the portfolio. The README and the GitHub stars are the headline, but the substance is everything underneath.

Good luck. Build it well — and ship it.
