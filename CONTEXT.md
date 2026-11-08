# MOTOR — Domain Context

Canonical vocabulary for this codebase: the terms that mean something specific here, each with the aliases to avoid
(`_Avoid_:`). [`grill`](.claude/skills/grill/) challenges new terms against it and writes new ones here inline.
Glossary only; how things work lives in [`docs/bible.md`](docs/bible.md). Format:
[`.claude/skills/grill/CONTEXT-FORMAT.md`](.claude/skills/grill/CONTEXT-FORMAT.md).

## Language

- **soul** — the checked-in voice spec [`.claude/soul.md`](.claude/soul.md), shared with SCORE. _Avoid_: style guide.
- **bible** — [`docs/bible.md`](docs/bible.md), one file, chaptered: the single source of truth for how MOTOR works.
  _Avoid_: docs, wiki, handbook.
- **shared harness** — the skills, hooks, settings, soul and code index this repository copies from SCORE and locks in
  `.claude/harness.lock`. _Avoid_: template, boilerplate.

## The two ecosystems

- **physical AI** — the field that makes and moves digital bodies for robots, games and film at scale (body models,
  motion generators, video estimators, GPU simulators). _Avoid_: AI, graphics.
- **biomechanics** — the field that measures and explains human movement as forces on a skeleton, by its conventions
  and tools (ISB, OpenSim). _Avoid_: kinesiology, gait analysis (one use of it).
- **physical-AI body** — any body from physical AI that MOTOR takes as input (SOMA, SMPL, MHR, a Kimodo clip, a video
  estimator's output), read as a mesh, segment rotations and joint centres. _Avoid_: avatar, rig, character.
- **SOMA** — the physical-AI body MOTOR was first built against; its segments and skin are the fit's truth.
  _Avoid_: the skin model.

## The model

- **MyoFullBody** — the one musculoskeletal model MOTOR solves on: bones, joints and 416 muscles, sized to each person.
  _Avoid_: the skeleton, OpenSim model.
- **sim body** — one person's MyoFullBody, sized and fitted inside their physical-AI body. _Avoid_: twin, digital
  human.
- **bone group** — the bones of MyoFullBody that together cover the same stretch of body as one SOMA segment, from the
  same joint to the same joint. _Avoid_: segment (that is SOMA's).
- **landmark set** — MOTOR's own list of about 100 bony points defined from standard anatomy and the ISB conventions,
  never copied from a licensed dataset's marker set. _Avoid_: markers, keypoints.
- **ISB conventions** — the International Society of Biomechanics' joint coordinate systems and angle orders that
  MOTOR's output follows. _Avoid_: standard angles.

## The method

- **the fit** — placing MyoFullBody inside the skin: the segment rotations its main target, the landmark set placing
  the joint centres and lengths, the bones kept inside the skin. _Avoid_: IK, retargeting.
- **calibration** — the part of the fit solved once and held: over many bodies (what every body shares), per person
  (the person constants), per frame (the pose). _Avoid_: tuning (a calibration is not a search over settings), setup.
- **person constants** — what the calibration solves once for one person and then holds fixed: small, bounded landmark
  offsets and the places of the joints inside each bone group. _Avoid_: per-trial constants.
- **whole-trial solve** — one solve over a whole recording that finds the movement, the ground reaction forces and the
  muscle forces together, so each agrees with the others and with the physics. _Avoid_: per-frame chain, pipeline.
- **warm start** — the step-by-step chain (fit, residual reduction, per-frame forces) used only to start the
  whole-trial solve. _Avoid_: the solver (the whole-trial solve is the solver).
- **residual** — the force a movement needs that neither the ground nor the muscles can supply; recorded, never
  applied. _Avoid_: helper force, leftover force.
- **residual reduction** (RRA) — the warm start's step that nudges the path and the segment masses so the ground
  reaction forces and gravity explain the motion. _Avoid_: correction step.
- **ground reaction force** (GRF) — the force the floor gives back on each foot while it touches. _Avoid_: ground push.
- **joint moment** — the net turning force about a joint that the movement needs, in ISB axes. _Avoid_: torque (in
  prose), joint force.
- **muscle force** — the pull of one muscle, with its **activation** (0 to 1) compared against EMG. _Avoid_: muscle
  power.

## The proof

- **LabValidation** — OpenCap's public dataset (on SimTK) of lab recordings (motion capture, force plates, EMG, phone
  video) MOTOR is validated on; fetched under its own licence, never committed. _Avoid_: the lab data.
- **OpenCap** — the video-to-biomechanics pipeline MOTOR is compared against, monocular and two-camera: the only
  comparable state of the art. _Avoid_: the baseline.
- **gap** — a result the paper cannot state yet, shown as missing, never estimated. _Avoid_: TBD, placeholder value.
