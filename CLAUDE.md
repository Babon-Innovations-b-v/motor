# CLAUDE.md

MOTOR: one standardised solver between physical AI and biomechanics. Any physical-AI body in, MyoFullBody as the one
model, biomechanics' conventions out (ISB angles, joint moments, ground reaction forces, muscle forces), on Newton and
MuJoCo Warp on the GPU; and its paper. Public repository, MIT. Owner: `@JoeyKardolus`. SCORE
(`Babon-Innovations-b-v/score`) uses it as a pinned dependency for its sim body.

## Persona

Read [`.claude/soul.md`](.claude/soul.md) for voice, hard rules, and working rhythm. Match it.

## Workflow protocol — every conversation in this repo

**On the first user message in any conversation, invoke the [`grill`](.claude/skills/grill/) skill before generating
any response.** The flow is SCORE's: grill → [`to-prd`](.claude/skills/to-prd/) → code →
[`close`](.claude/skills/close/), with [`brain-dump`](.claude/skills/brain-dump/) for a page of thinking and
[`wayfinder`](.claude/skills/wayfinder/) for an effort too foggy for one session. A first message with `#N` or an issue
URL resumes that issue. Term definitions: [`CONTEXT.md`](CONTEXT.md); a new word goes there first.

**The harness is SCORE's.** Skills, the shared hooks, `.claude/settings.json`, `.claude/soul.md` and `.claude/index/`
are copied from SCORE and locked in `.claude/harness.lock` (the SCORE commit and each file's sha256).
`python3 .claude/scripts/harness-sync.py` brings SCORE's main in; an edit of a shared file made here is carried to
SCORE's main by the commit that makes it (the commit-msg hook), or refused and lands nowhere. A shared file names no
fact of this repository. This repository's own hooks live in `.claude/hooks/local/` (`hooks.json`, run by
`.claude/hooks/dispatch.py`); git runs `.claude/hooks/` (`core.hooksPath`).

### Bible chapter map

| Topic | Chapter ID |
|---|---|
| What MOTOR is, its input, model, output and stack | `architecture/system-context` |
| The solver: the fit, the whole-trial solve, why GPU | `architecture/solver` |
| The calibration: what is solved once over many bodies, per person, per frame | `architecture/calibration` |
| The validation: LabValidation against OpenCap, every measure | `architecture/validation` |
| Build, test and release; how SCORE pins MOTOR | `build/release` |
| The loop, the module standard, the test rule | `workflow/loop`, `workflow/module-standard` |
| Glossary (canonical terms + `_Avoid_` aliases) | `CONTEXT.md` |

Per-area overlays: `<area>/CLAUDE.md` loads when files in its directory are touched; read the closest one.

## Agent tools

- **Summaries, not dumps.** A tool prints one summary line and writes detail to a file; `output_cap.py` saves any
  Bash output over 6,000 characters to a file and shows its head, tail and errors.
- **Never wait in the foreground.** Anything that takes minutes (a GPU solve, a cloud run, the full suite) runs with
  the Bash tool's `run_in_background`, or detached with one result line at the end; `wait_guard.py` blocks foreground
  sleeps and poll loops.
- **Code index**: the `index` MCP server (`search_graph`, `query_graph`, `search_code`); ask it before a repo-wide grep.

Fresh box: `bash .claude/scripts/bootstrap.sh` links Claude's memory folder, sets the repo's git hooks, and builds the
Python.

## Response Style

Concise. Bullets and short summaries; state results and decisions; an end-of-turn summary is one or two sentences.
Plans and reports for agents stay on the issue; the owner gets only the decisions he must make, in a few plain lines.

## Rules

- **Branch**: `main` only. No feature branches, no PRs. Durable authorization for every session: push directly to
  `origin/main` (from a worktree: `git push origin HEAD:main`). Leftover remote branches get deleted.
- **Public from the first commit.** Nothing from the owner's company or its business (code, names, clients, data, people),
  no secrets, keys, tokens, or cloud account and project ids, ever: not in code, docs, the paper or its comments.
- **Patient data never.** No recording, image, measure or derived file of a patient or study participant goes into this
  repository, its history, its issues or its paper's evidence. Public datasets are referenced by name and fetched under
  their own licence; the files stay out of git.
- **Never the owner's patented method** (arXiv:2607.17639, the owner's patent application NL4001444; MOTOR is public
  MIT): no swing-twist split into calibrated axes, no per-joint calibration table of axes, weights and offsets, no
  regression of joint angles against a reference. MOTOR's angles are MyoFullBody's own coordinates after its fit.
- **No fake data.** If something doesn't work, say so. No mock or synthetic output presented as a result; every result
  in the paper is a measured number or a marked gap.
- **The test rule (hard rule).** One test file a module, holding ONE test function (parametrized cases of one check are
  one), named `test_<invariant>.py`, its docstring naming the break that turns it red; small fixtures. A test file runs
  in at most 60 seconds (`TEST_SECONDS`), the whole suite in at most 10 minutes (`SUITE_SECONDS`), both in the
  Makefile. `make tests` runs the guard (`tools/test_rule/check.py`) first; its lists only shrink. Speed never comes
  from a dropped or loosened check. `docs/bible.md#workflowmodule-standard`.
- **The push gate.** A push to main runs the test rule over the tree and every touched module's test file
  (`.claude/hooks/local/pre-push`); a red refuses the push.
- **Minimal changes.** Don't refactor, rename or "improve" beyond what was asked; scope first (every caller, test and
  config a change reaches). A bug fix fixes the root once.
- **No from-scratch solutions.** Search this codebase, Newton's, MuJoCo's and Warp's own tools and docs first, then the
  standard library; ask before building custom.
- **One function, one purpose.** A docstring that needs "and" between two concerns is two functions.
- **No single-letter variable names** in production code (comprehensions and lambdas exempt).
- **Delete dead code.** New non-trivial logic gets a test, in its module's test file.
- **A review finding becomes an issue**, never a comment in the code.
- **Every GitHub issue gets an assignee at creation** (a PRD the owner; a sub-issue its parent's). Address people by
  `@handle`.
- **A new `.md` file needs the owner's go-ahead** (`md_guard.py`); a decision goes on its issue and into its bible
  chapter. A new code directory's `CLAUDE.md` overlay needs no ask.
- **No loose files.** A directory holds subdirectories plus framework files only (`__init__.py`, `conftest.py`,
  `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `CLAUDE.md`, `CONTEXT.md`, `LICENSE`, `docs/bible.md`).
- **Commits**: imperative, scoped (`module: description`).
- **Quality is never a cost lever.** No lower settings, resolutions or iteration counts for anything that ships;
  savings come from waste only.

## Deployment

Nothing is deployed. A push to `main` publishes the source and the paper. SCORE pins a MOTOR commit and its images
that run the sim body install MOTOR at that pin; a MOTOR change reaches SCORE only when SCORE moves its pin.

**Claude's memory stays private.** `.claude/memory/` is git-ignored. Anything a later session must know goes into the
bible, `CONTEXT.md` or an issue.

## Autonomy boundaries

- Don't continue into adjacent work after the requested task is done.
- Never ship unverified claims (measured numbers, benchmark figures, compatibility).
- Paid runs, machines and model calls wait for the owner's yes; his yes in a session is recorded where SCORE records
  it (SCORE's rules govern spend; MOTOR has no money of its own).

## Verification

- On "u sure?" or any audit, recount from scratch.
- For research and citations, check current sources; do not answer from memory.
- Failures propagate. No silent `except: pass` around a write.
