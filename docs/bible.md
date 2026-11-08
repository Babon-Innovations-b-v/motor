# The bible

The single source of truth for how MOTOR works. One file, chaptered. A chapter ID is the slug after the `##` (for
example `architecture/solver`); root [`CLAUDE.md`](../CLAUDE.md)'s chapter map points at it, and a link reads
`docs/bible.md#architecturesolver`.

Read the chapter the work touches, never the whole file. When you learn something load-bearing that a future reader
would otherwise re-derive from code, write it into its chapter in the same session. When a chapter contradicts the
code, the code wins and the chapter is fixed on the spot. Vocabulary is in [`CONTEXT.md`](../CONTEXT.md), the state of
ongoing work on its GitHub issue, history in git. Each directory's own rules are in its overlay.

---

## architecture/system-context

There are two ecosystems, physical AI and biomechanics, and each is fragmented inside too: many body models,
skeletons and solvers. MOTOR is a standardised solver between them (the paper's story, agreed with the owner on
2026-11-08):

- **One input**: any physical-AI body (SOMA, SMPL, MHR, Kimodo, a video estimator), read as a mesh, segment rotations
  and joint centres.
- **One model**: MyoFullBody.
- **One output**: biomechanics' conventions (ISB angles, joint moments, ground reaction forces, muscle forces), readable
  by OpenSim users.
- **One stack**: it runs on the physical-AI stack, Newton and MuJoCo Warp.

Why it matters: it closes the gap between the two ecosystems. Physical AI gets bodies with real biomechanics, and
biomechanics gets physical AI's scale.

**Who uses it.** SCORE's sim body stage is the first caller: it pins a MOTOR commit and hands MOTOR each character's
physical-AI body and the scene's ground; the row wiring SCORE's graph needs stays in SCORE as a thin stage. MOTOR knows
nothing of SCORE: no store, no rows, no spend.

**State (2026-11-08).** The repository holds the shared harness, the rules and the empty `motor/` package. The solver
(SCORE's `score/stages/sim_body`), its validation (`tools/labval`) and its paper (`paper/motor`) move here with their
history in the same release as the whole-trial solver enters SCORE's sim body stage (SCORE #101), because that release
re-keys SCORE's sim-body rows anyway.

## architecture/solver

The method has two parts.

1. **The fit.** MyoFullBody inside the skin, with the segment rotations as the main target and about 100 ISB landmarks
   (the landmark set) placing the joint centres and lengths. Each SOMA segment is matched by its bone group's combined
   rotation. The bones are kept inside the skin, an idea from SKEL. The fit is solved in three levels (the
   calibration, below).
2. **The whole-trial solve.** One solve over the whole recording finds the movement, the ground reaction forces and the
   muscle forces together, so each agrees with the others and with the physics. The step-by-step chain (the fit,
   residual reduction, forces frame by frame) is only its warm start. Residuals are measured and recorded, never
   applied.

**Why GPU.** It is unavoidable for simulation solvers: the more complex what is simulated, the more parallel the solver
must be, as computing has shown again and again (graphics, deep learning, robot simulation). The shared stack with
physical AI, the scale (every frame, every person at once) and gradients through the physics follow from it. Newton
and Warp's own tools (`newton.ik`, Warp's autodiff, MuJoCo Warp) come before any loop of our own.

**Never the owner's patented method** (arXiv:2607.17639, patent application NL4001444): no swing-twist split into
calibrated axes, no per-joint calibration table of axes, weights and offsets, no regression of joint angles against a
reference. The angles MOTOR reports are MyoFullBody's own coordinates after the fit.

## architecture/calibration

What the fit solves once and then holds, at three levels:

- **Over many bodies**: what every body shares, solved over a set of bodies together.
- **Per person**: the person constants, small and bounded landmark offsets and the places of the joints inside each
  bone group, solved from all frames of all that person's recordings together. A person's lengths come from their
  physical-AI body's sizing, never unknowns of the fit.
- **Per frame**: the pose, with the person constants held.

The detail of each level arrives with the code (SCORE's `inside/calibration`, `inside/tuning` and `inside/fit` today)
and is written here when it moves.

## architecture/validation

The proof: MOTOR run as a single-camera video pipeline on OpenCap's LabValidation dataset, against OpenCap (monocular
and two-camera), the only comparable state of the art. Every measure is reported, never a chosen subset:

- every joint angle (every degree of freedom MOTOR and the lab both have), against the lab's own inverse kinematics;
- ground reaction forces on all three axes, against the force plates;
- joint moments, against the lab's inverse dynamics;
- muscle activity against EMG, every channel;
- the solve time.

There is no separate held-out section: the comparison is the proof. Until the solver is validated, every result in
the paper is a gap. LabValidation is fetched from its source under its own licence and never committed; no recording
of a patient or participant enters this repository.

## build/release

- **Python 3.12**, the pins in `pyproject.toml`, exactly SCORE's newton image's (newton 1.7.0, mujoco and mujoco-warp
  3.15.0, warp-lang 1.18.0, usd-core 26.8 and their set). `uv.lock` is committed; `make env` builds `.venv` from it. A
  pin changes here and in SCORE's newton image together.
- **The package** is `motor/` (built by `uv_build`), installable by commit:
  `motor @ git+https://github.com/Babon-Innovations-b-v/motor@<sha>`. SCORE pins a commit, the way it pins its library.
- **Tests**: `make tests` runs the test rule's guard over every module, then every test file side by side, each killed
  at `TEST_SECONDS` (60); the whole suite stays inside `SUITE_SECONDS` (600). The push gate
  (`.claude/hooks/local/pre-push`) runs the guard and every touched module's test file on a push to main. A test that
  needs a GPU is named in the guard's `MAY_SKIP` with that reason and runs where there is a card.
- **Releases**: none yet. A push to main publishes the source and, once it is here, the paper.

**The move from SCORE (SCORE #101).** In the release where the whole-trial solver enters SCORE's sim body stage:

1. `score/stages/sim_body` (its solver part, `inside/`), `tools/labval` and `paper/motor` come in with their history
   (a subdirectory filter of SCORE's history): the solver's path rewritten to `motor/`, `tools/labval/` and
   `paper/motor/` kept at their paths, so their overlays and build scripts read the same.
2. The solver's modules import each other by bare name with the script's folder on `sys.path`; in `motor/` they become
   package imports.
3. The guard's lists (`MAY_SKIP`, `MAIN_ONLY`, `FUNCTION_DEBT`) take the moved files' entries as SCORE holds them that
   day, at their new paths, and nothing else.
4. SCORE pins the MOTOR commit, its sim body stage calls MOTOR, its images that run the sim body (newton, labval)
   install MOTOR at the pin, and no folder is left in both repositories.

## workflow/loop

SCORE's loop, through the shared skills: grill → to-prd → code → close. A conversation is a task; its plan is a PRD
issue on this repository, a grabbable piece a sub-issue under it. Commits are imperative and scoped
(`module: description`) and go straight to main. One agent per area at a time; an agent runs its own module's test
locally, pushes and stops.

## workflow/module-standard

SCORE's house standard, kept the same here so a reader of either can predict the other:

- **One interface, one directory.** A module is a directory with an obvious way in (`__init__.py`); files outside the
  interface are private (`_prefix`).
- **Deep over shallow**; **one function, one purpose**; **every directory carries its overlay** once it has rules.
- **Tests sit with the module**, one file a module (a `tests/` folder counting as its parent's), named for the invariant
  (`test_<invariant>.py`), ONE test function, driving the public interface. A test gives an input whose answer is known
  apart from the code (by hand, from the data, from a published value) or checks a property that must hold. No shape
  tests, no echo tests. Every test can fail: its docstring names each break, applied once, seen red and reverted.
- **Budgets**: a test file at most 60 seconds, the whole suite at most 10 minutes (owner, 2026-11-07). Speed never
  comes from a dropped or loosened check.
- **The guard** (`tools/test_rule/check.py`) enforces all of it; its lists only shrink.
- **Failures propagate.** A failure the caller can act on raises a typed error; one nothing can act on crashes loudly.
