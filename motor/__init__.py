"""MOTOR: a GPU solver that puts physical-AI bodies (SOMA's form) onto MyoFullBody and returns joint angles, joint
moments, ground reaction forces and muscle forces, on Newton and MuJoCo Warp (README.md, docs/bible.md).

`motor.run` is the run (`python -m motor.run <job.json> <out.json>`); each step is a module beside it, each with its
interface in its `__init__.py` and its one test file: `bone_groups`, `ground`, `solvers`, `landmarks`,
`calibration`, `volumes`, `model`, `batched`, `sizing`, `resting`, `myo`, `contact`, `fit`, `muscles`, `rra`,
`records`, `following` (the measures of how closely the fit follows the body) and `trial` (the whole-trial solve).
"""
