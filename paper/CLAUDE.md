# paper/

The MOTOR paper, moved in from SCORE's `paper/motor` with its history (a subtree split at SCORE 8d34af91), kept at
its path so its build line and evidence paths read the same. Its `#N` and comment ids are SCORE's issues.

```
(cd motor && tectonic -X compile --keep-intermediates main.tex)   # rebuild motor/main.pdf and main.bbl, then delete main.aux, main.out, main.log
```

## Layout

- `motor/`: written directly in LaTeX: `main.tex`, `references.bib`, `figures/` (with `figures/teaser/`, the
  scripts that make Figure 1 again from a run's trial folder: `prepare.py` (in a Python with mujoco), Blender's
  `render.py`, `compose.py`), the built `main.pdf` and `main.bbl`, and `evidence/`, the run records its numbers are
  read from. Same class, margins and natbib style as SCORE's paper; gap boxes are `\gapbox`, numbers still moving
  are `\interim`.
- Its story is the one agreed with the owner (SCORE #98 comment 6723992362): SOMA is physical AI's standard body,
  MyoFullBody a candidate standard musculoskeletal model, MOTOR the solver between the two standards. The main body
  after the abstract is four sections, related work, method, results, conclusion (owner, 2026-11-18; the discussion removed 2026-11-23), and
  with its figures **at most 4 pages**: it ends at `\label{endofmain}` after a `\FloatBarrier`, and its page in `main.aux` is at most
  4. Method detail and full tables go to the appendices. MOTOR is the per-frame solver: its results are the run of
  `evidence/labval-final-2026-11-07.txt`; the whole-trial solve is future work, its numbers from
  `evidence/motor-labval-2026-11-09.txt`. OpenCap's numbers are its published ones; PBL is not rerun (licence).
- `motor/figures/demo/`: the demo clips at the top of the README (owner, 2026-11-16), one a motion: a squat (the
  teaser's subject2_squats1_Cam1), walking (subject3_walking2_Cam1) and a drop jump (subject4_DJ2_Cam1), the walking
  and jump trials chosen where MOTOR's vertical GRF follows the plates best (peaks and timing). Three windows side by
  side in time: the input body, MyoFullBody in its skin with the activation colour bar, the floor's vertical GRF with
  a moving marker. `prepare.py` with every frame, then Blender's `render_clip.py` (the teaser's look, one fixed camera
  over the clip), then `compose_clip.py` (H.264 and a looping GIF, 440 px high). The videos are `motor/demos/`, made
  2026-11-18 on this PC's card from a run of 4799331 (each foot its share of the contacts' timeline, #1) on the
  single-camera LabValidation jobs, every window from that one run: bodies, activations and GRF; named `-timeline` so
  GitHub shows them, not its cached earlier clips.
- `motor/demos/sizing-2026-11-26/`: MyoFullBody sized and fitted to three generated SOMA bodies (tall, heavy, short)
  on one Kimodo walk, for the MyoSuite discussion (owner, 2026-11-26): the clips, each clip's job and SOMA data, the
  bodies, the scripts, and its `README.md`, the log to recreate it and the fit's three errors.

## Rules

- **Every number is read from a file** in `evidence/`, a commit or an issue comment, named in a comment beside it.
- **A change to the draft is rebuilt and committed in the same commit** as its PDF and `main.bbl`.
- **The register of the owner's published paper** (arXiv:2607.17639): full, connected academic prose; each
  paragraph makes one point and leads into the next; no project jargon, no stacked lists in running text.
- **Pictures:** our own outputs only, from public data under its licence; never a recording of a patient or participant.
