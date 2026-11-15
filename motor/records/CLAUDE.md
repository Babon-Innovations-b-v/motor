# motor/records/

What the run writes for one person, the contract the stage reads (`../../__init__.py` `measured`, `write_layer`):
`frames` (each frame's record and the drawn arrays: poses, muscle paths; its `touches` steadied over the clip,
`steadied`: each part's push low-passed as the path is over each span it touches, its centre on its ground, and
measured, `push_measures`, out.json's `pushes`: quiet stance's sum against the weight, centres off the ground or
outside their touching skin, frame to frame jumps; `push_faults` judges a quiet stance by them), `arrays` (<out>.npz's
names), `entry` (out.json's), `bones_of` (the bone meshes from MuJoCo's own compile, merged within `BONE_CELL`) and
`adjustment_record` (RRA's `adjusted`). A field renamed or dropped here is a contract change: the stage's `CONTRACT`
version moves with it.

- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
