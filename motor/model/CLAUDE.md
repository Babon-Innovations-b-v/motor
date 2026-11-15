# motor/model/

The person's MyoFullBody as Newton builds it (`built`: SolverMuJoCo's MuJoCo model; with a CUDA card Newton builds it there and the MuJoCo Warp model its
SolverMuJoCo puts on the card is kept, `on_card`, for `batched`; else MuJoCo's CPU backend, `DEVICE`;
no meshes given to Newton, nothing stepped; its SolverMuJoCo made inside `listed_once`: a spec's `bodies`
listed once until a body is added, attached or deleted, since Newton reads that list, new at each read, twice for each
of MyoFullBody's 11319 excluded pairs: 4 s of the 4.5 s build on this PC's CPU, the model the same field for field) and what every later step reads off it: `body_index` (a body by the
MyoFullBody name, Newton naming it by its path), `Couplings` (MuJoCo's joint couplings: the leading dofs and each
follower's polynomial), `limited`, `own_pose` and `TO_OWN` (MuJoCo's Z up to the person's Y up), `default_pose`
(MyoSuite's `stand`), `posed_path` (each SOMA segment riding its bone), `filtered` (the zero-lag low-pass) and `rates`.

- `limited` does not clamp today: its `model.jnt_type[joint] in (mujoco.mjtJoint.mjJNT_HINGE, ...)` is False for
  numpy's int32 against MuJoCo's enum (measured 2026-10-28; `==` is True), and `muscles.Physics.stops` reads joint
  types the same way. Kept as it was by the #70 move (behaviour unchanged first); fixing it changes every body's
  numbers, so it is a change of its own, measured on the two walkers.
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
