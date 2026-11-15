# motor/bone_groups/

The bone groups (`GROUPS`: each SOMA segment's frame body is the body it rides), the feet (`FEET`), the lumbar leaders
RRA moves (`TRUNK`) and what MyoFullBody lacks (`LACKS`, recorded with each body); `sided` and `sided_map` write a
`{Side}` name out for both sides. Pure Python: no MuJoCo, no Newton, so its test runs in the score venv.

- `GROUPS` is the bone-group table (#70; CONTEXT.md "bone group"): each SOMA segment's stretch, start and end joint,
  on SOMA and on MyoFullBody (the centres are `landmarks.CENTRES`), its frame body and its bodies; `groups()` writes
  it out for both sides. `NO_COUNTERPART` and `NO_TARGET` list the bodies with no segment or no target of their own.
  `RIDES`, `TARGETS` and `HIPS` (the old fit's) went with it on 2026-10-28. The spine is paired by height share,
  measured 2026-10-28 (the table's comment). The test checks each group spans its segment's joints, joint to joint.
- These are SCORE's own tables (#70: the bone-group and static calibration tables land here as code with their test).
  Never a per-joint table of axes, weights and offsets, nor anything fitted to reference data (the parent overlay:
  the owner's own method is never used).
