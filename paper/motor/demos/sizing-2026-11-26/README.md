# MOTOR fit demo, 2026-11-26

MyoFullBody sized and fitted by MOTOR to three SOMA bodies of different builds, each walking. Each clip shows the
input SOMA body (left) and the fitted MyoFullBody inside its skin with its muscle activations (right). Nothing is
filtered or retouched; every frame is the solve's own output.

## What is here

| Folder | Contents |
|---|---|
| `clips/` | MP4 (1080 px high) and looping GIF (540 px high) of each clip |
| `data/<clip>/` | the clip's MOTOR job (`job.json`) and its SOMA data (`targets.npz`: every segment's frame per frame, the SOMA mesh at its bind and its skin weights) |
| `bodies/` | the three SOMA bodies (`build.npz`: SOMA-X identity coefficients, bind, mesh) and `designs.json` (their measures) |
| `scripts/` | the scripts that made the bodies, the jobs and the clips' layout |

## The clips

| Clip | Height | Mass | Leg ratio | Volume BMI |
|---|---|---|---|---|
| `tall_walking` | 1.93 m | 91.3 kg | 0.511 | 24.5 |
| `heavy_walking` | 1.69 m | 100.0 kg | 0.511 | 35.0 |
| `short_walking` | 1.50 m | 55.1 kg | 0.511 | 24.5 |

**The bodies.** SOMA-X identity coefficients (128) solved by Newton steps so the body's stature, leg ratio (hip joint
height over stature) and volume BMI (mesh volume at 1000 kg/m3 over stature squared) hit their targets
(`scripts/sample.py`, `scripts/design.py`). Their mass is that volume at 1000 kg/m3: generated bodies have no measured
mass.

**The motion.** One Kimodo walk (Kimodo-SOMA-RP-v1.1, seed 1), the same clip on every body, its joint angles played on
each body's own bones with Kimodo's foot contacts (`scripts/jobs.py`, which uses the sim body helpers of SCORE,
https://github.com/Babon-Innovations-b-v/score). The resulting jobs are in `data/`, so this step need not be rerun.

**How MOTOR sizes MyoFullBody.** Each bone group is scaled so its length matches the SOMA body's; limb widths follow
the SOMA body's girth; segment masses, centres of mass and inertias come from the SOMA body's segment volumes, brought
to the person's mass. There is no separate sex switch: sex enters through the SOMA body's shape.

## Recreate

MOTOR is at https://github.com/Babon-Innovations-b-v/motor (MIT). An NVIDIA card is needed.

```
git clone https://github.com/Babon-Innovations-b-v/motor && cd motor
uv sync --frozen
cd paper/motor/demos/sizing-2026-11-26/data/tall_walking
../../../../../../.venv/bin/python -m motor.run job.json out.json
```

`out.json` holds the joint angles, joint moments, ground reaction forces and muscle forces and activations per frame;
`out.npz` the bone poses and muscle paths. Each job uses MOTOR's vendored MyoFullBody. One solve took about 60 s on an
RTX 5080; `tall_walking` was rerun from this folder exactly as above to check these steps. The clips were made
with MOTOR 82da288.

**The pictures**, from a solved folder (Blender 5.0.1, Cycles, 64 samples):

```
python <motor>/paper/motor/figures/teaser/prepare.py <clip folder> clip.npz 0 1 2 ... 149
blender -b --python <motor>/paper/motor/figures/demo/render_clip.py -- clip.npz frames/ 3600 1200 64
MOTOR_REPO=<motor> python scripts/compose_follow.py frames/ <clip folder> - clip.mp4 clip.gif
```

The camera is fixed over the whole walk; `compose_follow.py` cuts each frame to a window centred on the body, so the
body fills its window.

## How well it fits

The fit's own three errors (the MOTOR paper, Equation 1): the rotation error, each bone group's main bone against its
SOMA segment after the rest alignment; the joint error, each SOMA joint against MyoFullBody's joint centre (the
segment ends the markers include); and the skin error, the share of bone vertices outside the SOMA skin. MOTOR writes
them for every run in `out.json` (`people/<clip>/model/fit/following`); these are over all 150 frames and every group
or joint, median and largest, split by whether the joint sits at its limit. They say how closely MyoFullBody follows
the SOMA body; generated bodies have no ground truth, so they say nothing about accuracy against a real person.

| Clip | Rotation error, median / largest (free) | At a limit | Joint error, median / largest (free) | Bones outside the skin |
|---|---|---|---|---|
| `tall_walking` | 0.99° / 7.99° | 3.55° / 6.61° | 12.3 / 42.9 mm | 2.35% (feet 21.9%) |
| `heavy_walking` | 1.32° / 8.03° | 3.92° / 6.67° | 13.9 / 62.6 mm | 2.91% (feet 23.6%) |
| `short_walking` | 1.41° / 8.06° | 4.26° / 6.67° | 12.8 / 56.7 mm | 3.60% (feet 33.1%) |

The largest rotation errors are the arms and forearms, about 6° median on every body. The largest joint errors are
the hips, 37 to 39 mm median on the tall body, 52 to 58 mm on the short and heavy ones. The feet's bones are the ones
most outside the skin. The same measures on the path after the force step (`following, the path the physics takes`)
differ by at most 0.3° and 0.3 mm in the medians.

Beside the fit: the lowest point of the body stays within 0.8 cm of the floor, and the vertical ground reaction force
is 98 to 101% of body weight in the median, 126 to 130% at its peak.

## Squats: the body's limits hold

A squat generated by Kimodo (three prompts tried: deep, thighs parallel, half squat) asks for more knee and ankle bend
than a human body makes. MOTOR keeps MyoFullBody inside its anatomical joint ranges, so the fit stops at those limits
rather than follow the input. Measured against the standing pose on the heavy body:

| Joint | Kimodo asks | MOTOR's fit | MyoFullBody's range |
|---|---|---|---|
| hip | 93 to 99° | 79 to 85° | up to 120° |
| knee | 156 to 160° | 109 to 115° | up to 120° |
| ankle | 64 to 78° | 24 to 29° | up to 33° |

With the legs held at their limits, the feet end 8 to 12 cm below the input's floor at the bottom of the squat. The
squat is left out of the clips because its input is not a human motion. MOTOR's fitted pose, inside the limits, can be
fed back to Kimodo as full-body keyframe constraints (`kimodo_gen --constraints`) to generate a squat a human body can
make.

## Licences

MOTOR: MIT. MyoFullBody and its parts: their own licences, in MOTOR's `vendor/myofullbody/licenses`. SOMA and Kimodo:
NVIDIA's licences.
