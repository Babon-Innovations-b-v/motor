# MOTOR

One standardised solver between physical AI and biomechanics.

Physical AI and biomechanics are two ecosystems, and each is fragmented inside: many body models, skeletons and
solvers. MOTOR stands between them:

- **one input**: any physical-AI body (SOMA, SMPL, MHR, Kimodo, a video estimator), read as a mesh, segment rotations
  and joint centres;
- **one model**: MyoFullBody;
- **one output**: biomechanics' conventions (ISB joint angles, joint moments, ground reaction forces, muscle forces),
  readable by OpenSim users;
- **one stack**: it runs where physical AI runs, on Newton and MuJoCo Warp, on the GPU.

The method has two parts. **The fit** places MyoFullBody inside the skin, the segment rotations its main target and
about 100 ISB landmarks placing the joint centres and lengths, solved over many bodies, per person and per frame.
**The whole-trial solve** then finds the movement, the ground reaction forces and the muscle forces together, so each
agrees with the others and with the physics.

The proof is a single-camera video pipeline on the LabValidation dataset against OpenCap (monocular and two-camera):
every joint angle, ground reaction forces on all three axes, joint moments, muscle activity against EMG, and the solve
time.

## Status

Being set up. The solver, its validation and its paper are still built inside
[SCORE](https://github.com/Babon-Innovations-b-v/score) and move here, with their history, together with the
whole-trial solver. Until then this repository holds the harness, the rules and the empty package.

## Use

```bash
bash .claude/scripts/bootstrap.sh   # hooks, memory link, the Python from uv.lock
make tests                          # the test rule's guard, then every test file
```

Python 3.12 and the pins in `pyproject.toml` (Newton 1.7.0, MuJoCo and MuJoCo Warp 3.15.0, Warp 1.18.0). The solve
is written for the GPU (Warp); a test that needs a card says so and runs where there is one.

## Licence

MIT. See [LICENSE](LICENSE).
