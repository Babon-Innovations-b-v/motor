"""fit: on a small model posed by hand (a free pelvis, a left leg of thigh and shin, a right leg whose ankle follows its
hip by a coupling, one thigh's calibration turned 30 degrees), with the person's constants set by hand (one pelvis
marker 1 cm off its link, the knee, a joint inside the left leg's group, 1 cm off its place), the solver gives back from
the bone groups' turns and the markers the marker's move (within its pull to its link) and the knee's (within its pull
to its place), and at those constants every frame's angles; a frame asking a hip past its range stops it at the range's
end, its coupled follower on its leader by the coupling, though the follower's own range is narrower; each marker weighs
its trust among the markers, the markers' goal (group ends included, one goal) MARKER_GOAL in all against the turns' 1;
a group end that is the middle of two bodies' centres (the hips') sits on their common holder (the pelvis) as one body's
sits on its own, one rule; the constants are the body's, sampled from every one of its recordings; SOMA's targets (its
segments' turns, its skin's landmarks and rig joints) leave both solves unchanged (#70: SOMA is never moved); a frame's
fit ends on a step that moves no task's error, though the step itself is not small; level 2 solves only the inside joint
axes its Jacobian can tell apart from the other unknowns (a pair fixed to the root: their common shift is the root's
place; a marked body welded under a markerless hinged middle: the hinge's own axis), and every other axis it keeps,
the other unknowns' span taken by their Gram matrix exactly as by the pivoted QR; the constants are found the same
with no card (CUDA hidden: mink's frames and scipy's solve, a CPU-only machine's way). It runs in Newton's own Python (mink
pinned there; MOTOR's own Python, the pins in pyproject.toml).

Level 1's skin residual (fit._skin.marks, depth_mm) is a bone point's distance to the skin's closest point when
outside, 0 inside, by the winding test; the region's root mean square. RRA's frame check (`FrameSkins`) marks each
frame's bone vertices against that frame's own posed skin.

Breaks (each applied once with the env linked into a sandbox home, seen red, reverted):
- (2026-11-01) _skin.apart_of's ring `face = ring_faces[slot]` made `face = found.face` (the float32 query's face
  alone): 9 of 600 depths off the nearest face's, by up to 1.31e-6 m
  (the_depth_is_the_nearest_faces_where_the_query_flips red).
- (2026-11-01) _skin.closest_on_triangle's edge `bend = _away(along)` made a zero matrix: the offsets' projectors 0.496
  off their central differences (the_depth_is_the_nearest_faces_where_the_query_flips red).
- (2026-11-01) _skin_rows' point rows `jacp - crossed(...) @ jacr` made `jacp + 0 * crossed(...) @ jacr` (the turn's
  lever dropped): a frame's fit does not converge in 300 steps (NotConverged).
- (2026-11-01) _skin_rows' `rows = bends @ moved` made `moved` (no projector): NotConverged, its last step 1.5e-5.
- (2026-11-01) _frames.SkinTask's objective `pull` made `-pull`: the bones 183.8 mm outside held against 7.07 mm free,
  the held frames' cost (turns, markers and skin rows) 170.2 over the free 0.25
  (the_frames_hold_the_bones_inside_the_skin red).
- (2026-11-01) _frames.Frames.fitted's fit without the skin dropped (its `self.frame(qpos, frame_turns,
  frame_points)` run only when `skins is None`): on the first frame, from the standing start the skin holds the shin
  bone in the other part's box, 153.6 mm outside its own skin, the knee 1.033 rad off its target (a_frame_is_not_held_in_another_parts_skin red).
- (2026-11-04) _frames.Frames.fitted's every frame fitted without the skin first made the first frame's only (the code
  before 2026-11-04): the frame after the swing held in the other part's box, the bone 189.1 mm outside its own skin,
  the knee 0.693 rad off (a_frame_is_not_held_in_another_parts_skin red).
- (2026-11-01) _skin.unfolded's `<= FOLD_MOST` made `<= 2.0`: the overlapping cubes' frame is kept and its point reads
  1 m deep (a_skin_folding_into_itself_gives_no_rows red).
- (2026-11-01) _skin.joined_faces' `| (parents[:, None] == joints[None, :]) | (joints[:, None] == parents[None, :])`
  dropped (each segment joined to itself alone): the parent and child cubes fold 0, not over 0.4; made every segment
  joined to every other: the sibling cubes fold 0.5 (a_skin_folding_into_itself_gives_no_rows red).
- (2026-11-01) _skin.apart_kernel's `if kept[mesh] != 0:` made `... or True`: the folded frame's point reads 1 m deep
  (a_skin_folding_into_itself_gives_no_rows red).
- (2026-11-05) _skin._blended's `weights[vert, segment] * moved` made `moved`: the card's skin metres off the CPU's;
  card_folded's `>= FOLD_OWN` made `> 2.0`: no frame folds on the card (a_skin_folding_into_itself_gives_no_rows red).
- (2026-11-01) PosedSkins.__getitem__ dropping `self.kept[frames]`: indexed skins read every frame kept.
- (2026-11-01) _skin.envelope's `for verts in pockets(rig):` made `pockets(rig)[:0]`, no pocket closed: the point in
  the cube's mouth reads outside the envelope (the_heads_pockets_are_closed_for_the_inside_test red).
- (2026-11-06) _skin.POCKET_REACH `0.08` made `0.03` (before): the 6 cm cavity stays open, its point reads outside
  (the_heads_pockets_are_closed_for_the_inside_test red).
- (2026-11-01) `_card/_measure.turn_errors`' default `table=cpu.judged_turn` made `frozen_turn`, or
  `evaluation._measure.judged_turn` returning `frozen_turn`: level 1 scores the turns on main's frozen table while
  the stage's follows targets ride the table read again from the moved centres.
- (2026-11-03) posed_paths' `card_paths if on_card() else mink_paths` made `card_paths`: with the card hidden the
  solve raises "Invalid device identifier: cuda:0" (a CPU-only machine's fault; the_solver_gives_back_the_angles_and_the_
  constants (no card) red). constants_on's `wide=card` made `wide=True` stays green: level 2's card solve
  (`_card._level2_lm`, `_level2_wide`) runs on Warp's default device, the CPU with no card.
- (2026-11-06) _card._level3.angles' settle rule `still[final - 1 - frames[0]]` made `True` (every frame final
  after one sweep, its seed the chain's, not the frame before's answer): 1 sweep over 17 frames; the skin pass
  dropped from `_level3.sweep` (`return free`): angles 0.502 deg, sites 10.1 mm, bones 10.0 mm off mink's
  (level_3_on_the_card_is_minks_frames red). The toy has one minimum a frame, so a sweep seeded from each frame's own
  last answer, or a chain from the start every time, stays green: the recurrence's history is the real bodies'
  measure (the shoulder's flat elevation-plane valley), not this check's.
- (2026-11-06) _card._level3.sweep's fit without the skin `exact=True` dropped (Newton's float32-built joint places
  read again): "2 of level 3's solves read Newton's joint places"; _forward.Exact's body places rounded to float32:
  the card's frames 9.35e-09 off MuJoCo's; _solve.solved's `exact.drive(optimizer)` dropped: "0 of level 3's 4
  solves on MuJoCo's twists"; _forward._subspace's hinge `cross(anchor, axis)` made `cross(axis, anchor)`: the twists
  2.51 off, the card 62.97 deg off mink (level_3_on_the_card_reads_minks_own_model_at_the_shoulder red). The toy's
  angles and sites stay within the gate either way (sites 0.0002 mm on Newton's places, 0.0003 on MuJoCo's): its
  chain is too short for float32's rounding to reach 0.01 mm, mean's jump showed 0.024 mm.
- (2026-10-28) Fitting's site `Rotation.from_matrix(calibration.T)` made `Rotation.from_matrix(calibration)`: the turned
  thigh's target is met by a wrong pose.
- (2026-10-28) limited_model's `found.jnt_limited[self.followers] = 0` dropped: the follower's own range stops its
  leader short.
- (2026-10-30) _card._clip.for_fitting's `moved_inside(builder, fitting)` dropped: Newton poses the knee where the
  file has it, not where the fit moved it, and the card's sampled frames are refused, sites 10.0000 mm off MuJoCo's.
- (2026-10-30) card_paths' `for fit in fits` made `for fit in fits[:1] * len(fits)`: every set posed on set 0's
  model, the second set 0.003047 rad off its card_path alone.
- (2026-10-30) constants_across's `zip(fits, sampled_sets, paths)` made `zip(fits, sampled_sets, paths[::-1])`: each
  set solved on the other's frames, 2.3e-7 m off its constants alone (green 3.9e-8: the card's float32 frames;
  a thin margin, the toy's two sets' frames differ little).
- (2026-10-28) Frames' constraints `[mink.EqualityConstraintTask(...)]` made `[]`: the follower leaves its leader.
- (2026-10-29) Fitting.apply's `self.model.site_pos[self.mark_sites] += marks` dropped: the marker's move is not found.
- (2026-10-29) _frames.weights' `np.sqrt(trusts / max(trusts.sum(), 1e-12))` made `np.full(len(trusts), 1.0 /
  np.sqrt(len(trusts)))`: the low-trust marker weighs as much as the others.
- (2026-10-29) _frames.weights' `trusts.sum()` made `trusts[:-1].sum()`: a group end weighs on top of the markers' goal,
  not inside it.
- (2026-10-29) _frames.weights' `fitting.marker_goal * trusts` made `2 * trusts`: the markers' goal outweighs the turns'.
- (2026-10-30) _frames.weights' `fitting.marker_goal` made the module's `MARKER_GOAL`: a set solved across others weighs
  its markers by the last set's goal (the_place_goal_sums_to_the_marker_goal red).
- (2026-10-29) sampled's `for turns, points in recordings` made `for turns, points in recordings[:1]`: the body's second
  recording is left out of its constants.
- (2026-10-29) Fitting.apply's `self.model.body_pos[self.inside] += joints` dropped: the knee's move is not found.
- (2026-10-29) angles' `Frames(fit).fitted(turns, points)` after `points[...] = points + 0.0`: SOMA's targets are moved
  by the solve.
- (2026-10-29) end_site's holder `next(body for body in _above(...) if all(...))` made `bodies[0]`: the hips' middle
  rides the left thigh.
- (2026-10-31) end_site's `holder = _between(model, holder, bodies)` dropped: the hip and knee's middle rides the
  pelvis, not the thigh moving between them.
- (2026-10-29) _frames.solved's `or moved < STEP_DONE` dropped: the hinge whose mass sits 1e-6 m off its axis drifts
  1e-3 rad a step and raises NotConverged.
- (2026-10-29) _constants.told's kept `np.abs(np.diag(upper)) >= TOLD` made `>= 0`: every inside axis is solved, the
  pair's common shift and the hinge's axis among them.
- (2026-10-31) _skin.marks' mesh built without `support_winding_number=True`: Warp's sign query misjudges points
  of a skin that crosses itself (four overlapping spheres: 7 to 143 of 4,000 points on the wrong side).
- (2026-10-31) _constants.told's QR pivoted again (`pivoting=True`, kept by its order): of the pair's two bodies the
  one whose marks' trust is doubled is kept (on tall a 5 % change flipped humphant_l's y and humphant1_l's, level 2's
  constants a centimetre apart between level 1's nudged sets).
- (2026-10-29) _constants.off_span's `scaled.T @ columns` made `scaled.T @ columns * 0.5`: the Gram path's projection
  is not the QR's.
- (2026-10-30) _skin.marks_kernel's depth `warp.where(found.sign >= 0.0, ..., 0.0)` made the distance alone: a point
  inside the skin counts 0.8 m deep.
- (2026-10-30) _constants.staged's ftol/xtol/gtol made scipy's defaults (1e-8): level 2 stops on the cost's change at
  4.79e-5 of its start's gradient (settled 1.26e-5).
- (2026-10-30) _constants.steadied's `< np.abs(gradient).max()` made `> np.abs(gradient).max()`: the step that lowers
  the free gradient is refused and the one turned back taken (mean_kimodo's polish stood at 1.59e-10 with no
  gradient test).
- (2026-10-30) _constants.descended's `< cost` made `<= cost`: a step the cost cannot see is taken by `descended`
  and never judged by `steadied` (heavy_kimodo's polish stood at 1.48e-10 for its last 10 rounds, halvings 1e-13 long).

- (2026-10-31) _constants.polished's `lowest < FLOOR` made `lowest < np.inf`: a polish stalled far from settled is
  taken as settled.
- (2026-11-01) shifted_end's `local - turn.T @ ...` made `local + turn.T @ ...`: SOMA's joint pulls the bones the
  shift's way twice over, not off it (an_end_read_in_the_skin_holds_its_bones_off_somas_joint red).
- (2026-11-01) shifted_end's `turn.T @ (TO_OWN.T @ shift)` made `turn.T @ shift`: the shift taken in MuJoCo's frame,
  not the person's (Y up), the bones held off sideways.
- (2026-11-05) FrameSkins.outside's meshes `[mesh.id for mesh in self.meshes]` made the first frame's for every frame:
  each frame's bones judged against one frame's skin (a_frames_skin_marks_the_bones_out_of_it_frame_by_frame red).
- (2026-11-05) _frames.SkinTask.set_skin's `self.seen = {}` dropped: a new skin at the same pose reads the last
  skin's measure (a_pose_measured_once_gives_the_same_fit red).
- (2026-11-05) _frames.SkinTask._apart's key `placed.tobytes()` made `placed.round(2).tobytes()`: nearby poses read
  one another's measure, the angles off measuring every ask (a_pose_measured_once_gives_the_same_fit red).
- (2026-11-07) bone_meshes' `and KNEECAP not in ...` dropped: the kneecap's 4 vertices held in the skin too
  (a_frames_skin_marks_the_bones_out_of_it_frame_by_frame red).

Run: .venv/bin/python motor/fit/test_the_solver_gives_back_the_pose_and_constants_that_made_its_targets.py
"""
import os
import pathlib
import subprocess
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[1]
NO_CARD = "SIM_BODY_INSIDE_NO_CARD"  # the solve's check again with the card hidden (CUDA_VISIBLE_DEVICES empty)
TOY = """<mujoco model="toy">
  <compiler angle="radian"/>
  <asset><mesh name="shinbone" vertex="-0.015 -0.015 -0.05  0.015 -0.015 -0.05  -0.015 0.015 -0.05  0.015 0.015 -0.05
    -0.015 -0.015 -0.35  0.015 -0.015 -0.35  -0.015 0.015 -0.35  0.015 0.015 -0.35"/></asset>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/><geom type="box" size="0.15 0.1 0.1" mass="10"/>
      <body name="leg_l" pos="0.1 0 -0.1">
        <joint name="hip_l" type="hinge" axis="1 0 0" range="-1 1"/><geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
        <body name="shin_l" pos="0 0 -0.4">
          <joint name="knee_l" type="hinge" axis="1 0 0" range="-2 0"/><geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
          <geom type="mesh" mesh="shinbone" contype="0" conaffinity="0" mass="0.001"/>
        </body></body>
      <body name="leg_r" pos="-0.1 0 -0.1">
        <joint name="hip_r" type="hinge" axis="0 1 0" range="-1 1"/><geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
        <body name="foot_r" pos="0 0 -0.4">
          <joint name="ankle_r" type="hinge" axis="0 1 0" range="-0.4 0.4"/><geom type="box" size="0.05 0.1 0.02" mass="1"/>
        </body></body>
    </body></worldbody>
  <equality><joint joint1="ankle_r" joint2="hip_r" polycoef="0 0.5 0 0 0"/></equality>
</mujoco>
"""
MARKS = {"pelvis": [(0.15, 0.1, 0.0), (-0.15, 0.1, 0.0), (0.0, -0.1, 0.1)],
         "leg_l": [(0.05, 0.0, -0.2), (0.0, 0.05, -0.35)],
         "shin_l": [(0.04, 0.0, -0.2), (0.0, 0.04, -0.38)], "leg_r": [(-0.05, 0.0, -0.2), (0.0, 0.05, -0.35)],
         "foot_r": [(0.0, 0.1, -0.02)]}
BODIES = ("pelvis", "leg_l", "shin_l", "leg_r", "foot_r")  # each a turn site
TRUE_MOVE = ("pelvis1", (0.0, 0.01, 0.0))  # a marker 1 cm off its link, on its body
TRUE_JOINT = ("shin_l", (0.0, 0.01, 0.0))  # the knee 1 cm off its place in the thigh (a joint inside a group)
LEG_GROUPS = [{"leg_l", "shin_l"}, {"leg_r", "foot_r"}]  # the toy's two leg groups: the knee and ankle inside them
LOW_TRUST = ("pelvis2", 0.25)
# Each frame's root place, then hip_l, knee_l, hip_r (radians); the last asks hip_l past its range's end, 1.
POSES = [((0.0, 0.0, 1.0), 0.2, -0.4, 0.1), ((0.05, 0.0, 1.0), 0.4, -0.8, -0.3), ((0.1, 0.02, 0.98), 0.1, -1.2, 0.6),
         ((0.15, 0.03, 0.99), -0.3, -0.3, 0.9), ((0.2, 0.03, 1.0), -0.5, -0.1, 0.3),
         ((0.25, 0.03, 1.0), 1.3, -0.5, 0.2)]
# The toy shin's skin (each frame posed with the shin at its true angles): a box about the bone's middle (0, 0, -0.2)
# in the shin's frame, half sizes 0.03, 0.03, 0.18, moved SKIN_SHIFT along y, so the bone's four -y corners stand
# 1 cm outside it at the true pose and the leg must swing about x to hold them in.
SKIN_SHIFT = 0.025
CUBE_FACES = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
              [0, 6, 4], [1, 5, 7], [1, 7, 3]]  # a box's outward faces over its corners in x, y, z order
# Another part's skin beside the shin bone at the standing start: a box of half width DECOY_HALF along the bone,
# DECOY_SHIFT along the shin's y off it (the bone 1.5 cm outside it); FOUND_MM the bone's depth outside its own skin
# once held (2026-11-01: 0.07 mm from the fit without the skin, 154 mm from the standing start).
DECOY_HALF, DECOY_SHIFT, FOUND_MM = 0.03, 0.06, 1.0
# The card's posed skins against the CPU's (`card_posed_as_the_cpu`): float64 both, the card's sums in another order
# and fused (2026-11-05, mean's pickup on Warp's CPU device: the vertices within 6.7e-16 m, the shares 1.0e-17).
CARD_SKIN_M, CARD_SHARE = 1e-12, 1e-12
MOST_RADIANS = 1e-4
# level 3 on the card against mink's frames (`level_3_on_the_card_is_minks_frames`): the plan's gate on a clip of
# LEVEL3_FRAMES frames
LEVEL3_FRAMES, LEVEL3_DEG, LEVEL3_MM = 17, 0.01, 0.01
SETTLED_GRADIENT = 1e-10  # the free gradient level 2 ends at, absolute (2026-10-30: scipy's trf/lsmr alone left 2.5e-7)
# The marker's move is pulled to its link by one row at its own weight against the five frames' (found 2026-10-29: 7.1 mm
# of the true 10 mm, 10.0 with the pull dropped): along the true move at least MOVE_SHARE of it and no more than all of
# it, off that line by under MOST_METRES.
MOVE_SHARE = 0.5
MOST_METRES = 2e-3


def calibration(body):
    """A body's calibration turn: the left thigh's 30 degrees about its long axis, the others none."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("z", 30 if body == "leg_l" else 0, degrees=True).as_matrix()


def toy_fitting(folder):
    """The toy's `Fitting`: a turn site on every body (the left thigh's calibration 30 degrees about its long axis), its
    markers, one of them trusted a quarter."""
    import numpy as np
    from motor.fit import Fitting
    path = pathlib.Path(folder) / "toy.xml"
    path.write_text(TOY)
    frames = [(body, body, calibration(body)) for body in BODIES]
    marks = [(f"{body}{number}", body, np.array(point)) for body, points in MARKS.items()
             for number, point in enumerate(points)]
    trusts = [LOW_TRUST[1] if name == LOW_TRUST[0] else 1.0 for name, _, _ in marks]
    return Fitting(path, frames, marks, (), trusts, LEG_GROUPS)


def posed_targets(fitting, truth, poses):
    """Each pose's group turns (each body's turn times its calibration's transpose, R_segment = R_body R_cal^T) and
    marker places on the toy at the constants `truth` (the model left neutral)."""
    import mujoco
    import numpy as np
    model, data = fitting.model, mujoco.MjData(fitting.model)
    fitting.apply(truth)
    turns, points, angles = [], [], []
    for place, hip_l, knee_l, hip_r in poses:
        qpos = model.qpos0.copy()
        qpos[:3] = place
        qpos[7:11] = hip_l, knee_l, hip_r, 0.5 * hip_r
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        turns.append([data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)].reshape(3, 3)
                      @ calibration(body).T for body in fitting.groups])
        points.append(data.site_xpos[fitting.mark_sites].copy())
        angles.append(qpos[7:11])
    fitting.apply(fitting.neutral())
    return np.array(turns), np.array(points), np.array(angles)


def posed_qpos(model, pose):
    """The toy's qpos at one of POSES' rows (the ankle on its coupling)."""
    place, hip_l, knee_l, hip_r = pose
    qpos = model.qpos0.copy()
    qpos[:3] = place
    qpos[7:11] = hip_l, knee_l, hip_r, 0.5 * hip_r
    return qpos


def shin_skins(fitting, truth, poses):
    """Each pose's skin around the toy's shin bone (`SKIN_SHIFT`), posed with the shin at the constants `truth`, as
    `_skin.PosedSkins` in MuJoCo's frame; the model left neutral."""
    import mujoco
    import numpy as np
    from motor.fit._skin import PosedSkins
    model, data = fitting.model, mujoco.MjData(fitting.model)
    corners = np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-0.18, 0.18)])
    corners += [0.0, SKIN_SHIFT, -0.2]
    fitting.apply(truth)
    verts = []
    for pose in poses:
        data.qpos[:] = posed_qpos(model, pose)
        mujoco.mj_kinematics(model, data)
        shin = model.body("shin_l").id
        verts.append(corners @ data.xmat[shin].reshape(3, 3).T + data.xpos[shin])
    fitting.apply(fitting.neutral())
    return PosedSkins(np.array(verts), CUBE_FACES)


def bone_depths(model, path, skins):
    """Every frame's bone points' depth outside its skin (frames x points, m) by `_skin.marks`, MuJoCo's frame."""
    import mujoco
    import numpy as np
    from motor.fit import _skin
    data, found = mujoco.MjData(model), []
    bodies, places = _skin.bone_bodies(model)
    for qpos, verts in zip(path, skins.verts):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        points = data.xpos[bodies] + np.einsum("pij,pj->pi", data.xmat[bodies].reshape(-1, 3, 3), places)
        found.append(_skin.marks(points, verts, skins.faces)[1])
    return np.array(found)


def truth_of(fitting):
    """The constants set by hand: TRUE_MOVE's marker and TRUE_JOINT's joint moved."""
    found = fitting.neutral()
    at = 3 * fitting.marks.index(TRUE_MOVE[0])
    found[at:at + 3] = TRUE_MOVE[1]
    at = 3 * (len(fitting.marks) + fitting.inside_names.index(TRUE_JOINT[0]))
    found[at:at + 3] = TRUE_JOINT[1]
    return found

TRADED = """<mujoco><worldbody><body name="base"><freejoint/><geom size="0.1" mass="1"/>
  <body name="hip" pos="0 0 -0.1"><geom size="0.05" mass="1"/></body>
  <body name="back" pos="0 0 0.1"><geom size="0.05" mass="1"/>
    <body name="mid" pos="0 0 0.1"><joint axis="1 0 0"/><geom size="0.05" mass="1"/>
      <body name="top" pos="0 0 0.1"><geom size="0.05" mass="1"/></body></body></body>
</body></worldbody></mujoco>"""


def only_what_the_data_tells_is_an_inside_unknown():
    """On a free markerless root holding a hip and a back with no joint, the back holding a markerless middle on a
    hinge about x with a marked top fixed on it (the markers' own moves held): of the inside joints' twelve axes the
    Jacobian cannot tell three (the hip's and back's common shift, the root's place) and one (the middle's and top's
    along the hinge); the other eight are solved."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motor.fit import Fitting
    from motor.fit._card._level2 import jacobian
    from scipy.sparse import csc_matrix
    from motor.fit._constants import Problem, off_span, qr_off_span, told
    def told_at(trusts):
        """(the Problem, its Jacobian at the start, `told`'s kept) of the traded toy with the marks' `trusts`."""
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "traded.xml"
            path.write_text(TRADED)
            marks = [(f"{body}{axis}", body, np.eye(3)[axis] * 0.05) for body in ("hip", "back", "top")
                     for axis in range(3)]
            fitting = Fitting(path, [("base", "base", np.eye(3))], marks, (), trusts,
                              [{"base", "hip", "back", "mid", "top"}])
        poses = np.tile(fitting.model.qpos0, (4, 1))
        poses[:, 3:7] = Rotation.from_rotvec([[0, 0, 0], [0.3, 0, 0], [0, 0.4, 0.1], [0.1, -0.2, 0.5]]).as_quat(
            scalar_first=True)
        poses[:, 7] = [0.0, 0.5, -0.4, 1.0]
        turns, points, data = [], [], mujoco.MjData(fitting.model)
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_kinematics(fitting.model, data)
            turns.append(data.site_xmat[fitting.frame_sites].reshape(-1, 3, 3))
            points.append(data.site_xpos[fitting.mark_sites].copy())
        problem = Problem(fitting, poses, np.array(turns), np.array(points), fitting.neutral())
        matrix = jacobian(problem, problem.start())
        return problem, matrix, told(problem, matrix)
    # the pair's common shift is told by either body's axes: which is kept must not follow the trusts (one body's doubled)
    picks = [told_at([1.0 + 1.0 * (body == heavier) for body in ("hip", "back", "top") for _ in range(3)])[2]
             for heavier in ("hip", "back")]
    if not np.array_equal(*picks):
        return [f"the kept axes follow the trusts: {sorted(set(picks[0]) ^ set(picks[1]))}"]
    problem, matrix, kept = told_at(None)
    fitting = problem.fitting
    data = csc_matrix(matrix)[:matrix.shape[0] - len(problem.slots)]
    inside = np.arange(3 * len(problem.free), len(problem.slots))
    others, columns = data[:, np.setdiff1d(np.arange(data.shape[1]), inside)], data[:, inside].toarray()
    if np.abs(off_span(others, columns) - qr_off_span(others.toarray(), columns)).max() > 1e-12:
        return ["the Gram matrix's projection is not the QR's"]
    untold = sorted(set(range(len(problem.slots))) - set(kept))
    names = [f"{fitting.inside_names[slot // 3]}.{'xyz'[slot % 3]}" for slot in untold]
    pair = [name for name in names if name[:3] in ("hip", "bac")]
    hinge = [name for name in names if name in ("mid.x", "top.x")]
    return [] if (len(pair), len(hinge), len(names)) == (3, 1, 4) else [f"untold {names}"]


def the_solver_gives_back_the_angles_and_the_constants():
    """TRUE_MOVE's marker and TRUE_JOINT's joint moved along their true moves by MOVE_SHARE to all of it, within MOST_METRES of that line; at the true constants every frame within reach (all but the last) within
    MOST_RADIANS of its angles; the constants so on the card when there is one and with none (`fit.on_card` False:
    mink's frames and scipy's solve, a CPU-only machine's way), within MOST_METRES of each other."""
    from motor import fit
    import numpy as np
    seen = fit.on_card
    found = {}
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        truth = truth_of(fitting)
        turns, points, angles = posed_targets(fitting, truth, POSES[:-1])
        for card in (True, False) if seen() else (False,):
            fit.on_card = lambda card=card: card
            try:
                found[card] = fit.constants(fitting, [(turns[:2], points[:2]), (turns[2:], points[2:])])
            finally:
                fit.on_card = seen
        fitting.apply(truth)
        path = fit.angles(fitting, turns, points)
    problems = []
    off = np.abs(path[:, 7:11] - angles).max()
    if off > MOST_RADIANS:
        problems.append(f"angles off by up to {off:.6f} rad")
    for card, constants in found.items():
        for kind, (name, wanted) in (("marks", TRUE_MOVE), ("joints", TRUE_JOINT)):
            move, true = np.array(constants[kind][name]), np.array(wanted)
            along = move @ true / (true @ true)
            if not MOVE_SHARE <= along <= 1.0 or np.linalg.norm(move - along * true) > MOST_METRES:
                problems.append(f"{'card' if card else 'no card'}: {name} moved {move}, not {wanted}")
            if card and np.linalg.norm(move - np.array(found[False][kind][name])) > MOST_METRES:
                problems.append(f"{name} moved {move} on the card, {found[False][kind][name]} with none")
    return problems


def every_recording_of_the_body_is_sampled():
    """The constants' frames from two recordings of the body (two frames and three): all five, in order, each posed
    from its own recording's start (at the true constants within MOST_RADIANS of its angles)."""
    from motor import fit
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        truth = truth_of(fitting)
        turns, points, angles = posed_targets(fitting, truth, POSES[:-1])
        fitting.apply(truth)
        found, _, path = fit.sampled(fitting, [(turns[:2], points[:2]), (turns[2:], points[2:])])
        with tempfile.TemporaryDirectory() as other_folder:
            neutral = toy_fitting(other_folder)
            neutral.marker_goal = 2.0  # another set's level 1 marker goal
            both = fit.card_paths([fitting, neutral], [turns] * 2, [points] * 2, [2, len(turns) - 2])
            alone = fit.card_path(neutral, turns, points, [2, len(turns) - 2])
            recordings = [(turns[:2], points[:2]), (turns[2:], points[2:])]
            across = fit.constants_across([fitting, neutral], [recordings] * 2)
            apart = fit.constants(neutral, recordings)
    off = max(abs(across[1]["marks"][name][axis] - move[axis])
              for name, move in apart["marks"].items() for axis in range(3))
    if off > 1e-7:
        return [f"constants_across's second set {off:.9f} m off its constants alone"]
    miss = max(np.abs(both[0] - path).max(), np.abs(both[1] - alone).max())
    if miss > 1e-5:
        return [f"card_paths across two sets {miss:.6f} rad off each set alone"]
    if len(found) != len(turns) or not np.allclose(found, turns):
        return [f"{len(found)} frames sampled of the recordings' {len(turns)}"]
    off = np.abs(path[:, 7:11] - angles).max()
    return [] if off <= MOST_RADIANS else [f"the sampled frames' angles off by up to {off:.6f} rad"]


def somas_targets_leave_the_solves_unchanged():
    """The targets (SOMA's turns and points) read-only through the constants' and the angles' solves: no write to them
    (numpy refuses one) and the same values after."""
    from motor import fit
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        turns, points, _ = posed_targets(fitting, truth_of(fitting), POSES[:-1])
        kept = turns.copy(), points.copy()
        for target in (turns, points):
            target.setflags(write=False)
        try:
            fit.constants(fitting, [(turns[:2], points[:2]), (turns[2:], points[2:])])
            fit.angles(fitting, turns, points)
        except ValueError as refused:
            return [f"a solve wrote to SOMA's targets: {refused}"]
    return [] if np.array_equal(turns, kept[0]) and np.array_equal(points, kept[1]) else ["SOMA's targets changed"]


def a_hip_past_its_range_stops_at_its_end_and_its_follower_follows():
    """The last frame's left hip at its range's end (1 rad) within MOST_RADIANS; on every frame the right ankle 0.5
    times the right hip within MOST_RADIANS (its own range, 0.4, narrower than 0.5 times the hip's 0.9)."""
    from motor import fit
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        turns, points, _ = posed_targets(fitting, truth_of(fitting), POSES)
        fitting.apply(truth_of(fitting))
        path = fit.angles(fitting, turns, points)
    problems = [] if abs(path[-1, 7] - 1.0) < MOST_RADIANS else [f"the hip past its range stands at {path[-1, 7]}"]
    follower = np.abs(path[:, 10] - 0.5 * path[:, 9]).max()
    if follower > MOST_RADIANS:
        problems.append(f"the ankle leaves its coupling by up to {follower:.6f} rad")
    reached = np.abs(path[3, 9] - POSES[3][3])
    return problems + ([] if reached < MOST_RADIANS else [f"the right hip stopped {reached:.4f} rad short of 0.9"])


def each_marker_weighs_its_trust_and_the_markers_goal_the_turns_goal():
    """The toy's ten markers, one trusted 0.25: each place cost the square root of its trust over 9.25, each of the
    five turns one over the square root of 5; the markers' goal, every marker's (a group end's too) squared cost
    summed, is the fitting's marker goal (the toy's 1; the stage's `MARKER_GOAL` is level 1's), the turns' goal's 1."""
    import numpy as np
    from motor.fit._frames import weights
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        turn_cost, place_costs = weights(fitting)
    wanted = np.array([np.sqrt(fitting.marker_goal * (LOW_TRUST[1] if name == LOW_TRUST[0] else 1.0) / 9.25)
                       for name in fitting.marks])
    problems = [] if np.allclose(place_costs, wanted) else [f"place costs {place_costs.tolist()}"]
    problems += [] if np.isclose(turn_cost, 1 / np.sqrt(5)) else [f"turn cost {turn_cost}"]
    marker_sum, turn_sum = float(np.sum(place_costs ** 2)), turn_cost ** 2 * len(fitting.frame_sites)
    good = np.isclose(marker_sum, fitting.marker_goal) and np.isclose(turn_sum, 1.0)
    return problems + ([] if good else [f"markers' goal {marker_sum}, turns' {turn_sum}"])


def the_place_goal_sums_to_the_marker_goal():
    """With the fitting's own marker goal set to 4 (the module's MARKER_GOAL left at 1), the ten markers' squared costs
    sum to 4 and the five turns' to 1: one balance for every landmark and group, each set's own."""
    import numpy as np
    from motor.fit import _frames
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        fitting.marker_goal = 4.0
        turn_cost, place_costs = _frames.weights(fitting)
    found, turns = float(np.sum(place_costs ** 2)), 5 * turn_cost ** 2
    return [] if np.isclose(found, 4.0) and np.isclose(turns, 1.0) else [f"places {found:.4f}, turns {turns:.4f}"]


def a_middle_end_sits_on_the_common_holder():
    """The toy's hips' middle (each thigh's origin, half each) lands on the pelvis between them; one body's point stays
    on that body as it is; the left hip and knee's middle (the thigh's origin on the pelvis, the shin's) rides the
    thigh, the body moving between them, at its middle."""
    import mujoco
    import numpy as np
    from motor.fit import end_site
    model = mujoco.MjModel.from_xml_string(TOY)
    found = [end_site(model, [("leg_l", np.zeros(3), 0.5), ("leg_r", np.zeros(3), 0.5)]),
             end_site(model, [("shin_l", np.array([0.0, 0.01, 0.0]), 1.0)]),
             end_site(model, [("pelvis", np.array([0.1, 0.0, -0.1]), 0.5), ("shin_l", np.zeros(3), 0.5)])]
    wanted = [("pelvis", (0.0, 0.0, -0.1)), ("shin_l", (0.0, 0.01, 0.0)), ("leg_l", (0.0, 0.0, -0.2))]
    return [f"{name} at {np.round(local, 4)}, not {body} {place}" for (name, local), (body, place) in zip(found, wanted)
            if name != body or not np.allclose(local, place, atol=1e-9)]


def an_end_read_in_the_skin_holds_its_bones_off_somas_joint():
    """An end's marker moved by `shifted_end` stands -shift (the person's frame, Y up) off its place at rest, on a
    holder turned at rest: SOMA's joint, pulling the marker, holds the bones `shift` off it."""
    import mujoco
    import numpy as np
    from motor.fit._model import shifted_end
    from motor.model import TO_OWN
    model = mujoco.MjModel.from_xml_string(TOY.replace('<body name="shin_l"', '<body name="shin_l" euler="0 0 90"', 1))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    body, local, shift = model.body("shin_l").id, np.array([0.0, 0.01, 0.0]), np.array([0.0, 0.006, 0.008])
    turn = data.xmat[body].reshape(3, 3)
    moved = data.xpos[body] + turn @ shifted_end(model, "shin_l", local, shift)
    wanted = data.xpos[body] + turn @ local - TO_OWN.T @ shift
    problems = [] if np.allclose(moved, wanted, atol=1e-12) else [f"the end stands at {np.round(moved, 4)}, not {np.round(wanted, 4)}"]
    if np.allclose(turn, np.eye(3)):
        problems.append("the toy's shin is not turned at rest: the frame of the shift is not tested")
    return problems


def a_step_no_task_sees_ends_the_fit():
    """A hinge whose mass sits 1e-6 m off its axis, pulled 1 m away: each step turns it 1e-3 rad and moves the mass 1e-9
    m, a direction its task barely sees (#70's arms_raise gimbal); the fit ends, its error as it was within a step."""
    import mink
    import mujoco
    import numpy as np
    from motor.fit import _frames
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type="hinge" axis="0 0 1"/><inertial '
                                           'pos="1e-6 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>')
    configuration, task = mink.Configuration(model), mink.ComTask(cost=1.0, lm_damping=_frames.LM_DAMPING)
    task.set_target(np.array([0.0, 1.0, 0.0]))
    before = task.compute_error(configuration)
    try:
        _frames.solved(configuration, [task], [], [])
    except _frames.NotConverged as error:
        return [str(error)]
    moved = np.linalg.norm(task.compute_error(configuration) - before)
    return [] if moved < 1e-6 else [f"its error moved {moved}"]


def a_bone_point_outside_the_skin_lies_as_deep_as_its_distance():
    """Level 1's smooth skin residual: on a 2 m cube, a point inside is 0 deep and not outside, one 1 m off a face 1 m
    deep, one off an edge its distance (sqrt 2) deep; the region's residual their root mean square in mm."""
    import numpy as np
    from motor.fit import _skin
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
             [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    out, depths = _skin.marks(np.array([[0.2, 0.1, 0.0], [2.0, 0.3, 0.0], [2.0, 2.0, 0.5]]), corners, faces)
    expected = np.array([0.0, 1.0, np.sqrt(2.0)])
    problems = [] if list(out) == [False, True, True] else [f"outside {list(out)}"]
    problems += [] if np.allclose(depths, expected, atol=1e-6) else [f"depths {depths} against {expected}"]
    rms = _skin.depth_mm(depths)
    return problems + ([] if abs(rms - 1000 * np.sqrt(1.0)) < 1e-3 else [f"root mean square {rms} mm"])


def a_turn_is_judged_on_the_calibration_table_as_it_stands():
    """Level 1's turn rows are the stage's: with the calibration table read again 20 degrees off main's frozen row
    (as a moved centre turns it), a thigh standing as SOMA's segment through that table reads 0 degrees on the card's
    turn errors and the CPU's `judged_turn`, and 20 on main's frozen table beside them."""
    import numpy as np
    import mujoco
    import warp
    from scipy.spatial.transform import Rotation
    from motor import calibration
    from motor import following as cpu
    from motor.fit._card import _measure as card
    from motor.fit._frozen import frozen_turn
    from motor.model import TO_OWN
    kept = dict(calibration.TABLE)
    moved = Rotation.from_matrix(frozen_turn("LeftLeg")) * Rotation.from_rotvec([0.0, 0.0, np.radians(20.0)])
    calibration.TABLE["LeftLeg"] = tuple(moved.as_quat())
    try:
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="femur_l"><geom size="0.1"/></body>'
                                              '</worldbody></mujoco>')
        body = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()  # the thigh's turn in MuJoCo's frame
        xmat = np.repeat(np.stack([np.eye(3), body])[None], 2, axis=0)
        data = type("Posed", (), {"xmat": warp.array(xmat, dtype=warp.mat33d, device=card.DEVICE)})()
        world = np.tile(np.eye(4), (2, 1, 1, 1))
        world[:, 0, :3, :3] = TO_OWN @ body @ calibration.turn("LeftLeg").T  # SOMA's segment through the table
        judged = card.turn_errors(model, data, world, ["LeftLeg"])["LeftLeg"]
        frozen = card.turn_errors(model, data, world, ["LeftLeg"], frozen_turn)["LeftLeg"]
        problems = [] if np.allclose(judged, 0.0, atol=1e-5) else [f"card on the table as it stands: {judged}"]
        problems += [] if np.allclose(frozen, 20.0, atol=1e-4) else [f"card on main's frozen table: {frozen}"]
        cpu_turn = cpu.judged_turn("LeftLeg")
        problems += [] if np.allclose(cpu_turn, calibration.turn("LeftLeg")) else ["the CPU judges on another table"]
        return problems
    finally:
        calibration.TABLE.clear()
        calibration.TABLE.update(kept)


def a_skin_crossing_itself_tells_inside_by_its_winding_number():
    """A posed skin crosses itself where limbs meet (2026-10-31: a forearm point 5.7 mm off mean_running's skin read
    outside, then inside after a 0.003 mm move, though its exact winding number is 1 both times): on four overlapping
    coarse spheres as one mesh, `_skin.marks`' side of 4,000 points is the exact winding number's (solid angles)."""
    import numpy as np
    from motor.fit import _skin
    rng = np.random.default_rng(4)
    verts, faces = [], []
    for _ in range(4):
        turn, tilt = np.meshgrid(np.linspace(0, 2 * np.pi, 10, endpoint=False), np.linspace(0.15, np.pi - 0.15, 8),
                                 indexing="ij")
        ring = np.stack([np.cos(turn) * np.sin(tilt), np.sin(turn) * np.sin(tilt), np.cos(tilt)], -1).reshape(-1, 3)
        start = sum(len(part) for part in verts)
        verts.append(np.concatenate([ring, [[0, 0, 1], [0, 0, -1]]]) * rng.uniform(0.04, 0.08) + rng.normal(0, 0.05, 3))
        for around in range(10):
            ahead = (around + 1) % 10
            for down in range(7):
                faces += [[start + around * 8 + down, start + around * 8 + down + 1, start + ahead * 8 + down + 1],
                          [start + around * 8 + down, start + ahead * 8 + down + 1, start + ahead * 8 + down]]
            faces += [[start + 80, start + around * 8, start + ahead * 8],
                      [start + 81, start + ahead * 8 + 7, start + around * 8 + 7]]
    verts, faces = np.concatenate(verts), np.array(faces)
    points = rng.normal(0, 0.07, (4000, 3))

    def winding(point):
        """The exact winding number of the mesh about `point` (each triangle's solid angle)."""
        a, b, c = (verts[faces[:, corner]] - point for corner in range(3))
        la, lb, lc = (np.linalg.norm(side, axis=1) for side in (a, b, c))
        top = np.einsum("ij,ij->i", a, np.cross(b, c))
        bottom = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", b, c) * la
                  + np.einsum("ij,ij->i", c, a) * lb)
        return np.sum(2 * np.arctan2(top, bottom)) / (4 * np.pi)
    inside = np.array([winding(point) for point in points]) >= 0.5
    out, _ = _skin.marks(points, verts, faces)
    wrong = int(np.sum(out == inside))
    return [] if wrong == 0 else [f"{wrong} of {len(points)} points on the wrong side of the self-crossing skin"]


def the_frames_hold_the_bones_inside_the_skin():
    """mink's frames on the toy at the true constants: given the skins (1 cm off at the true pose), the bones' depth
    outside (root mean square over the frames) falls to under a quarter of the frames' without them, and the
    frames' whole cost (`_constants.Problem`'s turns and markers, plus the skin rows: `_skin.skin_cost` times each
    point's depth) is lower there than at the frames fitted without the skin: the skin task moves the pose to the
    joint optimum, never away from it."""
    import numpy as np
    from motor import fit
    from motor.fit import _skin
    from motor.fit._constants import Problem
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        truth = truth_of(fitting)
        turns, points, _ = posed_targets(fitting, truth, POSES[:-1])
        skins = shin_skins(fitting, truth, POSES[:-1])
        fitting.apply(truth)
        held = fit.angles(fitting, turns, points, skins)
        free = fit.angles(fitting, turns, points)
        depths = [bone_depths(fitting.model, path, skins) for path in (held, free)]
        inside, outside = (float(np.sqrt(np.mean(found ** 2))) for found in depths)
        problems_at = [Problem(fitting, path, turns, points, truth, ()) for path in (held, free)]
        costs = [float(np.sum(problem.residuals(problem.start()) ** 2))
                 + _skin.skin_cost(found.shape[1]) ** 2 * float(np.sum(found ** 2))
                 for problem, found in zip(problems_at, depths)]
    problems = [] if inside < 0.25 * outside else [f"bones {1000 * inside:.2f} mm outside held, {1000 * outside:.2f} free"]
    return problems + ([] if costs[0] < costs[1] else [f"the held frames' cost {costs[0]:.6f} over the free {costs[1]:.6f}"])


def a_pose_measured_once_gives_the_same_fit():
    """mink's frames on the toy held in the skin with each pose's skin measure kept (`_frames.SEEN`) give the same
    angles, bit for bit, as with every ask measured again (SEEN 0), in under half the closest-point launches; and a new
    skin set at the same pose is measured again, never read off the last skin's."""
    import numpy as np
    from motor.fit import _frames, _skin
    calls, measure = [0], _skin.SkinMeshes.apart

    def counted(meshes, points, frames):
        calls[0] += 1
        return measure(meshes, points, frames)

    found, problems, saved = [], [], _frames.SEEN
    _skin.SkinMeshes.apart = counted
    try:
        with tempfile.TemporaryDirectory() as folder:
            fitting = toy_fitting(folder)
            truth = truth_of(fitting)
            turns, points, _ = posed_targets(fitting, truth, POSES[:-1])
            skins = shin_skins(fitting, truth, POSES[:-1])
            fitting.apply(truth)
            for seen in (0, saved):
                _frames.SEEN, calls[0] = seen, 0
                found.append((_frames.Frames(fitting).fitted(turns, points, skins=skins), calls[0]))
            task, configuration = _frames.SkinTask(fitting.model), _frames.mink.Configuration(fitting.model)
            configuration.update(found[1][0][0])
            task.set_skin(skins[0:1])
            first = task.compute_error(configuration)
            task.set_skin(skins[1:2])
            second = task.compute_error(configuration)
    finally:
        _skin.SkinMeshes.apart = measure
        _frames.SEEN = saved
    (again, every), (kept, fewer) = found
    if again.tobytes() != kept.tobytes():
        problems.append(f"the kept measure's angles {np.abs(again - kept).max():.3g} rad off measuring every ask")
    if not fewer < 0.5 * every:
        problems.append(f"{fewer} closest-point launches kept against {every} measuring every ask")
    if np.array_equal(first, second):
        problems.append("a new skin at the same pose read off the last skin's measure")
    return problems


def toy_sites(model, path):
    """Every frame's sites (frames x sites x 3, m) on the toy at `path`'s angles."""
    import mujoco
    import numpy as np
    data, found = mujoco.MjData(model), []
    for qpos in path:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        found.append(data.site_xpos.copy())
    return np.array(found)


def level_3_on_the_card_is_minks_frames():
    """Level 3 on the card, every frame at once (`fit.LEVEL3_CARD`, `_card._level3`'s sweeps of mink's recurrence),
    on the toy held in its shin's skin over a clip of LEVEL3_FRAMES frames between POSES' first five: every frame's
    angles within LEVEL3_DEG of mink's (`Frames.fitted`), its sites within LEVEL3_MM, its bone points' depth outside
    the skin within LEVEL3_MM, in more than one sweep and fewer than one a frame; with the switch off, mink's own."""
    import numpy as np
    from motor import fit
    from motor.fit._card import _level3
    if not fit.on_card():
        return []
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        truth = truth_of(fitting)
        rows = np.array([[*place, *joints] for place, *joints in POSES[:-1]])
        between = np.linspace(0, len(rows) - 1, LEVEL3_FRAMES)
        flat = np.stack([np.interp(between, range(len(rows)), column) for column in rows.T], axis=1)
        poses = [(tuple(row[:3]), *row[3:]) for row in flat]
        turns, points, _ = posed_targets(fitting, truth, poses)
        skins = shin_skins(fitting, truth, poses)
        fitting.apply(truth)
        mink = fit.angles(fitting, turns, points, skins)
        card, sweeps = fit.card_angles(fitting, turns, points, skins)
        kept, fit.LEVEL3_CARD = fit.LEVEL3_CARD, True
        try:
            switched = fit.angles(fitting, turns, points, skins)
        finally:
            fit.LEVEL3_CARD = kept
        off = fit.angles(fitting, turns, points, skins)
        model = fitting.model
        degrees = float(np.degrees(np.abs(card[:, 7:] - mink[:, 7:]).max()))
        sites = 1000 * float(np.linalg.norm(toy_sites(model, card) - toy_sites(model, mink), axis=-1).max())
        depths = 1000 * float(np.abs(bone_depths(model, card, skins) - bone_depths(model, mink, skins)).max())
    problems = [] if degrees <= LEVEL3_DEG else [f"the card's angles {degrees:.5f} deg off mink's"]
    problems += [] if sites <= LEVEL3_MM else [f"the card's sites {sites:.5f} mm off mink's"]
    problems += [] if depths <= LEVEL3_MM else [f"the card's bones {depths:.5f} mm deeper or shallower than mink's"]
    problems += [] if 1 < sweeps < len(poses) else [f"{sweeps} sweeps over {len(poses)} frames"]
    problems += [] if np.array_equal(switched, card) else ["the switch on does not give the card's angles"]
    problems += [] if off.tobytes() == mink.tobytes() else ["the switch off does not give mink's angles"]
    return problems + ([] if _level3.STRIDE < len(poses) else ["the seeding chain holds every frame"])


# A toy shoulder girdle as MyoFullBody's (its own six-digit places and turns): the clavicle's leader hinge and its
# unrotscap follower at -1 a body further out about the same axis, so the arm keeps its turn and the clavicle moves only
# its place, the girdle's weak direction; a thorax, an arm, a forearm and a hand beyond.
SHOULDER = """<mujoco model="shoulder">
  <compiler angle="radian"/>
  <worldbody>
    <body name="thorax" pos="0 0 1.2">
      <freejoint name="root"/><geom type="box" size="0.12 0.08 0.2" mass="20"/>
      <body name="attachment" pos="0.0173133 0.328944 0.000413982" quat="0.500398 -0.5 0.499602 0.5">
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="clavicle" pos="0.0409701 -0.0165714 0.0233532" quat="0.500398 0.5 -0.5 -0.499602">
          <joint name="clavicle_r2" axis="0.0153 0.989299 -0.1451" range="-0.75 0.349066"/>
          <geom type="capsule" fromto="0 0 0 -0.0154411 0.0176525 0.145429" size="0.01" mass="0.2"/>
          <body name="phantom" pos="-0.0154411 0.0176525 0.145429">
            <joint name="unrot_r2" axis="0.0153 0.989299 -0.1451" range="-0.349066 0.75"/>
            <geom type="sphere" size="0.01" mass="0.01"/>
            <body name="humerus" pos="-0.00955 -0.034 0.009">
              <joint name="elevation" axis="-0.0581 0.226 0.9724" range="-1.5 2.5"/>
              <geom type="capsule" fromto="0 0 0 0.00574177 -0.272512 0.0115777" size="0.03" mass="2"/>
              <body name="forearm" pos="0.00574177 -0.272512 0.0115777">
                <joint name="elbow" axis="0.226047 0.022269 0.973862" range="0 2.6"/>
                <geom type="capsule" fromto="0 0 0 0.0173589 -0.233584 -0.0241096" size="0.02" mass="1"/>
                <body name="hand" pos="0.0173589 -0.233584 -0.0241096">
                  <geom type="box" size="0.04 0.08 0.015" mass="0.4"/>
                </body></body></body></body></body></body>
    </body></worldbody>
  <equality><joint joint1="unrot_r2" joint2="clavicle_r2" polycoef="0 -1 0 0 0"/></equality>
</mujoco>
"""
SHOULDER_MARKS = {"thorax": [(0.1, 0.05, 0.15), (-0.1, 0.05, 0.15), (0.0, -0.08, -0.1)],
                  "humerus": [(0.02, -0.05, 0.0), (0.0, -0.2, 0.03)],
                  "forearm": [(0.03, -0.1, 0.0), (0.0, -0.22, -0.02)],
                  "hand": [(0.03, -0.07, 0.0), (-0.03, -0.07, 0.0)]}
SHOULDER_FRAMES, SHOULDER_NUDGE = 9, 0.004  # the clip's frames; its markers nudged off (m), as no skin fits exactly
FRAMES_EXACT = 1e-12  # m and rad: level 3's card frames against MuJoCo's own mj_kinematics (float64 both)
TWIST_EXACT = 1e-6  # its hinges' twists (float32, Newton's analytic Jacobian's) against MuJoCo's anchors and axes


def shoulder_clip(fitting):
    """The toy shoulder's clip: (turns, points, qpos) of SHOULDER_FRAMES poses, the root walking and turning, the
    clavicle on its coupling, the arm rising and the elbow bending; the markers nudged SHOULDER_NUDGE off."""
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    model, data = fitting.model, mujoco.MjData(fitting.model)
    at = {name: int(model.jnt_qposadr[model.joint(name).id])
          for name in ("clavicle_r2", "unrot_r2", "elevation", "elbow")}
    nudge, turns, points, poses = np.random.default_rng(7), [], [], []
    for frame in range(SHOULDER_FRAMES):
        share = frame / (SHOULDER_FRAMES - 1)
        qpos = model.qpos0.copy()
        qpos[:3] = 0.3 * share, 0.05 * share, 1.2
        qpos[3:7] = Rotation.from_euler("z", 0.4 * share).as_quat(scalar_first=True)
        qpos[at["clavicle_r2"]], qpos[at["unrot_r2"]] = -0.2 + 0.3 * share, 0.2 - 0.3 * share
        qpos[at["elevation"]], qpos[at["elbow"]] = 0.3 + 1.2 * share, 0.2 + 1.5 * share
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        turns.append([data.site_xmat[site].reshape(3, 3) for site in fitting.frame_sites])
        points.append(data.site_xpos[fitting.mark_sites] + nudge.normal(0, SHOULDER_NUDGE, (len(fitting.marks), 3)))
        poses.append(qpos)
    return np.array(turns), np.array(points), np.array(poses)


def level_3_on_the_card_reads_minks_own_model_at_the_shoulder():
    """Level 3 on the card at a coupled shoulder girdle (SHOULDER): the residuals' and the skin's link frames are
    MuJoCo's own float64 (`_card._forward.Exact`, mj_kinematics on the compiled model) within FRAMES_EXACT at every
    pose, its Jacobian's hinge twists MuJoCo's anchors and axes within TWIST_EXACT, every solve `_card._level3` makes
    reads both (never Newton's float32-built joint places, `_forward.Forward`:
    on mean's jump they stood 7.3 um off at a hand and the card stopped 0.0073 deg off mink in the girdle, sites
    0.024 mm, 2026-11-06), and its answer is mink's within LEVEL3_DEG and LEVEL3_MM on every frame, the follower on
    its leader by the coupling."""
    import mujoco
    import numpy as np
    import warp as wp
    from scipy.spatial.transform import Rotation
    from motor import fit
    from motor.fit import Fitting
    from motor.fit._card import _clip, _forward, _level3, _solve
    from motor.fit._card._seeds import newton_order
    if not fit.on_card():
        return []
    with tempfile.TemporaryDirectory() as folder:
        path = pathlib.Path(folder) / "shoulder.xml"
        path.write_text(SHOULDER)
        marks = [(f"{body}{number}", body, np.array(point)) for body, points in SHOULDER_MARKS.items()
                 for number, point in enumerate(points)]
        fitting = Fitting(path, [(body, body, np.eye(3)) for body in SHOULDER_MARKS], marks)
        turns, points, poses = shoulder_clip(fitting)
        clip = _clip.for_fitting(fitting, path)
        model, data = fitting.model, mujoco.MjData(fitting.model)
        body_q = wp.empty((len(poses), clip.model.body_count), dtype=wp.transformd, device=clip.model.device)
        _forward.frames(clip.exact(), wp.array(newton_order(poses).astype(np.float32), dtype=wp.float32,
                                               device=clip.model.device), body_q)
        laid, gaps = body_q.numpy(), []
        for row, qpos in enumerate(poses.astype(np.float32).astype(float)):
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            for label, link in clip.body.items():
                body = model.body(label).id
                turned = Rotation.from_quat(laid[row, link, 3:7]).as_matrix() - data.xmat[body].reshape(3, 3)
                placed = np.abs(laid[row, link, :3] - data.xpos[body]).max()
                gaps.append(max(float(placed), float(np.abs(turned).max())))
        optimizer = clip.exact().drive(clip.optimizer_for(turns[-1:], points[-1:])(0, 1))
        context = optimizer._ctx_solver(wp.array(newton_order(poses[-1:]).astype(np.float32), dtype=wp.float32,
                                                 device=clip.model.device))
        optimizer._residuals_analytic(context)
        optimizer._jacobian_at(context)
        twists, joints = optimizer.joint_S_s.numpy()[0], clip.exact().dof_joint.numpy()
        data.qpos[:] = poses[-1].astype(np.float32).astype(float)
        mujoco.mj_kinematics(model, data)
        hinges = [np.concatenate([np.cross(data.xanchor[joint], data.xaxis[joint]), data.xaxis[joint]])
                  for joint in joints if joint >= 0]
        twisted = float(np.abs(twists[joints >= 0] - np.array(hinges)).max())
        mink = fit.angles(fitting, turns, points)
        kept, read, drive, driven = _solve.Forward, [], _forward.Exact.drive, []
        _solve.Forward = lambda *args, **kwargs: read.append(args) or kept(*args, **kwargs)
        _forward.Exact.drive = lambda self, optimizer: driven.append(optimizer) or drive(self, optimizer)
        try:
            card, sweeps = _level3.angles(clip, turns, points, None, fit.Frames(fitting).start(turns[0], points[0]))
        finally:
            _solve.Forward, _forward.Exact.drive = kept, drive
        sites = list(fitting.frame_sites) + list(fitting.place_sites)
        apart = 1000 * float(np.linalg.norm(toy_sites(model, card)[:, sites] - toy_sites(model, mink)[:, sites],
                                            axis=-1).max())
        degrees = float(np.degrees(np.abs(card[:, 7:] - mink[:, 7:]).max()))
        lead, follow = (int(model.jnt_qposadr[model.joint(name).id]) for name in ("clavicle_r2", "unrot_r2"))
        coupled = float(np.abs(card[:, follow] + card[:, lead]).max())
    problems = [] if max(gaps) <= FRAMES_EXACT else [f"the card's frames {max(gaps):.2e} off MuJoCo's"]
    problems += [] if not read else [f"{len(read)} of level 3's solves read Newton's joint places"]
    problems += [] if twisted <= TWIST_EXACT else [f"the card's hinge twists {twisted:.2e} off MuJoCo's"]
    solves = len(range(0, SHOULDER_FRAMES, _level3.STRIDE)) + sweeps  # the chain, then a sweep a solve (no skin)
    problems += [] if len(driven) == solves else [f"{len(driven)} of level 3's {solves} solves on MuJoCo's twists"]
    problems += [] if degrees <= LEVEL3_DEG else [f"the card's angles {degrees:.5f} deg off mink's"]
    problems += [] if apart <= LEVEL3_MM else [f"the card's sites {apart:.5f} mm off mink's"]
    return problems + ([] if coupled <= 1e-6 else [f"the follower {coupled:.2e} rad off its leader's coupling"])


def decoy_box(model, qpos):
    """Another part's skin beside the toy's shin bone where `qpos` puts it, DECOY_SHIFT off ([8 x 3] corners)."""
    import mujoco
    import numpy as np
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    shin = model.body("shin_l").id
    corners = np.array([[x, y, z] for x in (-DECOY_HALF, DECOY_HALF) for y in (-DECOY_HALF, DECOY_HALF)
                        for z in (-0.37, -0.03)]) + [0.0, DECOY_SHIFT, 0.0]
    return corners @ data.xmat[shin].reshape(3, 3).T + data.xpos[shin]


def a_frame_is_not_held_in_another_parts_skin():
    """mink's frames on the toy whose skin also holds another part's (a box beside the shin bone, DECOY_SHIFT off,
    where the frame's start puts it: as env1's seated thighs stood beside his hanging forearms at the standing start,
    and as the mean body's hips and chest stood beside a fast-swinging arm at the frame before's pose): the knee still
    reaches its bent target and the bone its own skin (within FOUND_MM), on a clip's first frame (from the standing
    start) and on a later frame after a swing (the knee -0.4 to -1.2 rad), each frame's skin rows starting from its
    own fit without them; from the start the skin pulled the bone into the other box and held it there."""
    import numpy as np
    from motor import fit
    from motor.fit import _frames, _skin
    problems = []
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        truth = truth_of(fitting)
        model = fitting.model
        # the first frame: the decoy where the standing start puts the shin
        turns, points, angles = posed_targets(fitting, truth, POSES[2:3])
        own = shin_skins(fitting, truth, POSES[2:3])
        fitting.apply(truth)
        decoy = decoy_box(model, _frames.Frames(fitting).start(turns[0], points[0]))
        faces = np.concatenate([own.faces, np.array(CUBE_FACES) + 8])
        held = fit.angles(fitting, turns, points, _skin.PosedSkins(np.concatenate([own.verts[0], decoy])[None], faces))
        cases = [("the first frame", held[0], own.verts[0], angles[0])]
        # a swing: frame 0 at POSES[0] in its own skin (its decoy far off), frame 1 at POSES[2], the decoy where
        # frame 0's fit put the shin
        turns, points, angles = posed_targets(fitting, truth, (POSES[0], POSES[2]))
        own = shin_skins(fitting, truth, (POSES[0], POSES[2]))
        fitting.apply(truth)
        first = fit.angles(fitting, turns[:1], points[:1], _skin.PosedSkins(own.verts[:1], own.faces))
        decoys = np.array([decoy_box(model, first[0]) + [0.0, 0.0, 10.0], decoy_box(model, first[0])])
        held = fit.angles(fitting, turns, points, _skin.PosedSkins(np.concatenate([own.verts, decoys], axis=1), faces))
        cases.append(("the frame after a swing", held[1], own.verts[1], angles[1]))
        for name, qpos, verts, wanted in cases:
            depth = 1000 * float(np.sqrt(np.mean(bone_depths(model, qpos[None], _skin.PosedSkins(verts[None], CUBE_FACES))
                                                 ** 2)))
            knee = abs(float(qpos[8] - wanted[1]))
            if not (depth < FOUND_MM and knee < 0.01):
                problems.append(f"{name}: the bone {depth:.1f} mm outside its own skin, the knee {knee:.3f} rad off")
    return problems


def pocket_head(gap=0.01, deep=0.02):
    """A head-weighted closed mesh: a 10 cm cube whose +x face holds a pocket 2 `gap` wide and `deep` deep (a mouth's
    slit by default), its faces outward; as a rig ({"points", "faces", "weights", "names"})."""
    import numpy as np
    half = 0.05
    corners = [[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)]
    rim = [[half, y, z] for y, z in ((-gap, -gap), (gap, -gap), (gap, gap), (-gap, gap))]
    bottom = [[half - deep, y, z] for y, z in ((-gap, -gap), (gap, -gap), (gap, gap), (-gap, gap))]
    faces = [[0, 1, 3], [0, 3, 2], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    outer = [4, 6, 7, 5]  # the +x face's corners: (-,-), (+,-), (+,+), (-,+) in (y, z)
    for side in range(4):
        ahead = (side + 1) % 4
        faces += [[outer[side], outer[ahead], 8 + ahead], [outer[side], 8 + ahead, 8 + side]]
        faces += [[8 + side, 8 + ahead, 12 + ahead], [8 + side, 12 + ahead, 12 + side]]
    faces += [[12, 13, 14], [12, 14, 15]]
    points = np.array(corners + rim + bottom, dtype=float)
    return {"points": points, "faces": np.array(faces, dtype=np.int32), "weights": np.ones((len(points), 1)),
            "names": np.array(["Head"])}


def a_frames_skin_marks_the_bones_out_of_it_frame_by_frame():
    """`FrameSkins` (RRA's frame check): a 1 cm bone on a femur at the origin, a 10 cm cube of skin on its first frame
    round it and on its second 1 m along x: every bone vertex inside on the first frame and outside on the second, each
    labelled thigh; a kneecap on it (patella_r) is not among the bones held in the skin."""
    import mujoco
    import numpy as np
    from motor.fit import _skin
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><mesh name="bone" vertex="0 0 0 0.01 0 0 0 0.01 0 0 0 0.01"/></asset><worldbody>'
        '<body name="femur_r"><freejoint/><geom type="mesh" mesh="bone"/><body name="patella_r" pos="0 0 0.05">'
        '<geom type="mesh" mesh="bone"/></body></body></worldbody></mujoco>')
    rig = pocket_head()
    rig["bind"] = np.eye(4)[None]
    worlds = np.tile(np.eye(4), (2, 1, 1, 1))
    worlds[1, 0, 0, 3] = 1.0
    skins = _skin.FrameSkins(model, rig, ["Head"], worlds)
    out = skins.outside(np.tile(model.qpos0, (2, 1)))
    problems = [] if out.shape[1] == 4 and not out[0].any() and out[1].all() else [f"outside {out.tolist()}"]
    return problems + ([] if list(skins.labels) == ["thigh"] * 4 else [f"labels {list(skins.labels)}"])


def the_heads_pockets_are_closed_for_the_inside_test():
    """SOMA's head holds pockets (the mouth's slit and the oral cavity behind it, the eye sockets, nostrils, ears) a
    closed face covers: on a head mesh with a mouth-like pocket, a point in the pocket reads outside the skin as it is
    and inside its `envelope` (`_skin.pockets` finds the one pocket), a point in the head inside both, one beside the
    head outside both; in a cavity 6 cm wide and deep (SOMA's mouth, to 103 mm deep) a point reads inside the envelope."""
    import numpy as np
    from motor.fit import _skin
    rig = pocket_head()
    points = np.array([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.08, 0.0, 0.0]])
    plain = _skin.marks(points, rig["points"], rig["faces"])[0]
    closed = _skin.marks(points, rig["points"], _skin.envelope(rig))[0]
    problems = [] if list(plain) == [True, False, True] else [f"the pocket mesh's sides as it is: {list(plain)}"]
    problems += [] if len(_skin.pockets(rig)) == 1 else [f"{len(_skin.pockets(rig))} pockets found, not 1"]
    problems += [] if list(closed) == [False, False, True] else [f"the envelope's sides: {list(closed)}"]
    cavity = pocket_head(gap=0.03, deep=0.06)  # an oral cavity behind the lips: walls 6 cm apart, 6 cm deep
    held = _skin.marks(np.array([[0.02, 0.0, 0.0]]), cavity["points"], _skin.envelope(cavity))[0]
    return problems + ([] if not held.any() else ["a point in the 6 cm cavity reads outside the envelope"])


def nearest_on_faces(points, verts, faces):
    """Each point's distance to the nearest of every face (verts, faces), in float64 by brute force: per face the
    closest point of its plane's projection clamped to its edges (the nearest of the inside projection and every
    edge's clamped segment point)."""
    import numpy as np
    first, second, third = (verts[faces[:, corner]][None] for corner in range(3))
    off = points[:, None] - first
    normal = np.cross(second - first, third - first)
    normal /= np.linalg.norm(normal, axis=2)[..., None]
    flat = points[:, None] - np.einsum("pfi,pfi->pf", off, np.broadcast_to(normal, off.shape))[..., None] * normal
    inside = np.ones(flat.shape[:2], dtype=bool)
    for start, end in ((first, second), (second, third), (third, first)):
        inside &= np.einsum("pfi,pfi->pf", np.cross(end - start, flat - start),
                            np.broadcast_to(normal, flat.shape)) >= 0
    best = np.where(inside, np.linalg.norm(points[:, None] - flat, axis=2), np.inf)
    for start, end in ((first, second), (second, third), (third, first)):
        along = np.clip(np.einsum("pfi,pfi->pf", points[:, None] - start, np.broadcast_to(end - start, flat.shape))
                        / np.einsum("pfi,pfi->pf", end - start, end - start), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points[:, None] - start - along[..., None] * (end - start), axis=2))
    return best.min(axis=1)


def the_depth_is_the_nearest_faces_where_the_query_flips():
    """A point's depth off the skin (`SkinMeshes.apart`) is its distance to the nearest face in float64, though Warp's
    float32 query picks one of near-tied faces (mean's sampled frames, 2026-11-01: a level 2 inside-joint column
    2.95e-4 off its central difference where it flipped): on a 400-point sphere of 0.1 m standing 10 m out (float32's
    step there 1e-6 m), 600 points 0.1 to 5 mm outside lie as deep as brute force says to 1e-9 m (the query's face
    alone: 9 off, by up to 1.3e-6 m), and each offset's projector (`bends`: a face's n n^T, an edge's I - e e^T, a
    corner's I) is its central difference over a 1e-7 m move to 1e-3 of the largest."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from motor.fit import _skin
    generator = np.random.default_rng(1)
    unit = generator.normal(size=(400, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    faces = ConvexHull(unit).simplices.copy()
    outward = np.einsum("ij,ij->i", np.cross(unit[faces[:, 1]] - unit[faces[:, 0]], unit[faces[:, 2]] - unit[faces[:, 0]]),
                        unit[faces].mean(axis=1)) > 0
    faces[~outward] = faces[~outward][:, [0, 2, 1]]
    verts = (0.1 * unit + [10.0, 0.0, 0.0]).astype(np.float32).astype(np.float64)
    away = generator.normal(size=(600, 3))
    away /= np.linalg.norm(away, axis=1)[:, None]
    points = 0.1 * away * (1 + generator.uniform(0.001, 0.05, (600, 1))) + [10.0, 0.0, 0.0]
    meshes = _skin.SkinMeshes(_skin.PosedSkins(verts[None], faces), "cpu")
    offsets, bends = meshes.apart(points, np.zeros(600, np.int32))
    gap = np.abs(np.linalg.norm(offsets, axis=1) - nearest_on_faces(points, verts, faces))
    problems = [] if gap.max() < 1e-9 else [f"{int((gap >= 1e-9).sum())} depths off the nearest face's, by {gap.max():.2e}"]
    move = 1e-7 * generator.normal(size=(600, 3))
    moved = (meshes.apart(points + move, np.zeros(600, np.int32))[0]
             - meshes.apart(points - move, np.zeros(600, np.int32))[0]) / 2
    slope = np.abs(np.einsum("pij,pj->pi", bends, move) - moved).max() / np.abs(moved).max()
    return problems + ([] if slope < 1e-3 else [f"the offsets' projectors {slope:.2e} off their central differences"])


def a_skin_folding_into_itself_gives_no_rows():
    """A posed skin folding into itself across a joint (a deep squat's knee; no skeleton can follow it) is left out of
    the skin rows, a skin pressing into a segment further off (a heavy body's arm on its chest) is not: on two 2 m cubes
    as one mesh, each its own segment, standing apart the `_skin.folded` share is 0 and the frame kept; overlapping by
    half (half of each cube's corners inside the other) as parent and child it folds over 0.4 and is not kept, as two
    siblings it folds 0 and is kept; `PosedSkins` carries which through indexing and joining, and a point 1 m outside
    reads 1 m off the kept frame's skin (`SkinMeshes.apart`) and 0 off the folded one's. Posed every frame at once on
    the card (`card_skins`, `card_folded`; Warp's CPU device with no card), the same rig blended (a corner riding both
    cubes) and turned on three frames gives the CPU's `skin` to CARD_SKIN_M and its `folded` shares to CARD_SHARE, so
    the same frames kept, the folding one among them."""
    import numpy as np
    from motor.fit import _skin
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    faces = np.array(CUBE_FACES + [[corner + 8 for corner in face] for face in CUBE_FACES], dtype=np.int32)
    weights = np.zeros((16, 3))
    weights[:8, 1], weights[8:, 2] = 1.0, 1.0
    joined, siblings = ({"faces": faces, "weights": weights, "parents": parents} for parents in ([-1, 0, 1], [-1, 0, 0]))
    apart, folding = (np.concatenate([corners, corners + [shift, 0.0, 0.0]]) for shift in (5.0, 1.0))
    shares = [_skin.folded(apart, joined), _skin.folded(folding, joined), _skin.folded(folding, siblings)]
    problems = [] if shares[0] == 0.0 and shares[1] > 0.4 and shares[2] == 0.0 else [f"folded shares {shares}"]
    posed = _skin.PosedSkins.joined([_skin.PosedSkins([verts], faces, [_skin.unfolded(verts, joined)])
                                     for verts in (apart, folding)])
    problems += [] if list(posed.kept) == [True, False] and list(posed[::-1].kept) == [False, True] else [
        f"kept frames {list(posed.kept)}"]
    offsets, _ = _skin.SkinMeshes(posed, "cpu").apart(np.array([[0.0, 2.0, 0.0]] * 2), np.array([0, 1]))
    depths = np.linalg.norm(offsets, axis=1)
    problems += [] if np.allclose(depths, [1.0, 0.0], atol=1e-6) else [f"depths off the frames {depths}"]
    return problems + card_posed_as_the_cpu(apart, joined)


def card_posed_as_the_cpu(points, rig):
    """`a_skin_folding_into_itself_gives_no_rows`' card half: the two cubes (`points`, `rig`'s faces and parents) as a
    rig of Root, Hips and Head, one corner riding both, posed on three frames (apart; the Head 4 m back, folding into
    its parent; both turned), every frame at once on `_skin.card_device()` against the CPU's frame by frame."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motor.fit import _skin
    weights = np.zeros((16, 3))
    weights[:, 1:] = rig["weights"][:, 1:]
    weights[0, 1:] = 0.7, 0.3
    posing = {**rig, "points": points, "weights": weights, "names": np.array(["Root", "Hips", "Head"]),
              "bind": np.tile(np.eye(4), (3, 1, 1))}
    worlds = np.tile(np.eye(4), (3, 2, 1, 1))
    worlds[1:, 1, 0, 3] = -4.0
    worlds[2, 0, :3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    worlds[2, 1, :3, :3] = Rotation.from_euler("xz", [15, 35], degrees=True).as_matrix()
    segments, device = ["Hips", "Head"], _skin.card_device()
    cpu = np.stack([_skin.skin(posing, segments, world) for world in worlds])
    shares = np.array([_skin.folded(verts, posing) for verts in cpu])
    card = _skin.card_skins(posing, segments, worlds, device)
    found = _skin.card_folded(card, posing, device)
    gap = float(np.abs(card.numpy() - cpu).max())
    problems = [] if gap < CARD_SKIN_M else [f"card skin {gap:.2e} m off the CPU's on {device}"]
    problems += [] if np.abs(found - shares).max() < CARD_SHARE else [f"card folded {found} against the CPU's {shares}"]
    kept = list(_skin.posed_skins(posing, segments, worlds).kept)
    return problems + ([] if kept == list(shares <= _skin.FOLD_MOST) and not kept[1] else [f"kept frames {kept}"])


def level_2_ends_settled():
    """On noisy markers (the toy's, 2 mm noise, seeded) level 2's solve ends where the cost's gradient on its free
    told unknowns off their bounds is under SETTLED_GRADIENT, float64's floor: not stalled short of it by scipy's
    trf/lsmr step or by float32 coordinates in the residuals."""
    from motor import fit
    import numpy as np
    from motor.fit._card._level2 import jacobian
    from motor.fit._constants import Problem, off_bounds, staged, told
    from motor.landmarks import HELD
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        turns, points, _ = posed_targets(fitting, truth_of(fitting), POSES)
        points = points + np.random.default_rng(7).normal(0.0, 2e-3, points.shape)
        turns, points, path = fit.sampled(fitting, [(turns, points)])
        free = [index for index, name in enumerate(fitting.marks) if name not in HELD and not name.startswith(fit.END)]
        constants, angles, _ = staged(fitting, path, turns, points, fitting.neutral(), free, jacobian)
        fitting.apply(fitting.neutral())
        problem = Problem(fitting, angles, turns, points, constants, free)
        unknowns = problem.start()
        inside = off_bounds(unknowns, told(problem, jacobian(problem, unknowns)), problem.bounds())
        gradient = float(np.abs((jacobian(problem, unknowns).T @ problem.residuals(unknowns))[inside]).max())
    return [] if gradient < SETTLED_GRADIENT else [f"level 2 stopped at a free gradient of {gradient:.2e}"]


def a_step_the_cost_cannot_see_is_judged_by_its_gradient():
    """`steadied` takes level 2's Gauss-Newton step from the toy's start, where it lowers the free gradient, and refuses
    the step turned back, which raises it; `descended` leaves a step the cost cannot see (a zero step) to it."""
    import numpy as np
    from motor.fit._card._level2 import jacobian
    from motor.fit._constants import Problem, descended, off_bounds, steadied, told
    from motor.landmarks import HELD
    from motor import fit
    with tempfile.TemporaryDirectory() as folder:
        fitting = toy_fitting(folder)
        turns, points, _ = posed_targets(fitting, truth_of(fitting), POSES)
        turns, points, path = fit.sampled(fitting, [(turns, points)])
        free = [index for index, name in enumerate(fitting.marks) if name not in HELD and not name.startswith(fit.END)]
        problem = Problem(fitting, path, turns, points, fitting.neutral(), free)
        unknowns = problem.start()
        kept = off_bounds(unknowns, told(problem, jacobian(problem, unknowns)), problem.bounds())
        matrix = jacobian(problem, unknowns)[:, kept].tocsc()
        gradient = matrix.T @ problem.residuals(unknowns)
        step = np.linalg.solve((matrix.T @ matrix).toarray(), -gradient)
        taken = steadied(problem, unknowns, kept, step, problem.bounds(), gradient)
        refused = steadied(problem, unknowns, kept, -step, problem.bounds(), gradient)
        unseen = descended(problem, unknowns, kept, 0 * step, problem.bounds())
    return ([] if taken is not unknowns else ["the step that lowers the gradient is refused"]) + (
        [] if refused is unknowns else ["the step that raises the gradient is taken"]) + (
        [] if unseen is unknowns else ["a step the cost cannot see is taken by descended"])


CHECKS = (an_end_read_in_the_skin_holds_its_bones_off_somas_joint, a_turn_is_judged_on_the_calibration_table_as_it_stands, a_bone_point_outside_the_skin_lies_as_deep_as_its_distance, a_skin_crossing_itself_tells_inside_by_its_winding_number, only_what_the_data_tells_is_an_inside_unknown, a_middle_end_sits_on_the_common_holder, each_marker_weighs_its_trust_and_the_markers_goal_the_turns_goal,
          the_place_goal_sums_to_the_marker_goal, the_solver_gives_back_the_angles_and_the_constants,
          every_recording_of_the_body_is_sampled, somas_targets_leave_the_solves_unchanged, a_hip_past_its_range_stops_at_its_end_and_its_follower_follows,
          a_step_no_task_sees_ends_the_fit, level_2_ends_settled, a_step_the_cost_cannot_see_is_judged_by_its_gradient,
          the_frames_hold_the_bones_inside_the_skin, a_pose_measured_once_gives_the_same_fit,
          a_frame_is_not_held_in_another_parts_skin, the_heads_pockets_are_closed_for_the_inside_test, a_skin_folding_into_itself_gives_no_rows,
          the_depth_is_the_nearest_faces_where_the_query_flips, a_frames_skin_marks_the_bones_out_of_it_frame_by_frame,
          level_3_on_the_card_is_minks_frames, level_3_on_the_card_reads_minks_own_model_at_the_shoulder)


def in_newton():
    """The checks, in Newton's own Python; with NO_CARD set (the card hidden, a CPU-only machine's way) the solve's alone."""
    checks = (the_solver_gives_back_the_angles_and_the_constants,) if os.environ.get(NO_CARD) else CHECKS
    failures = [f"{check.__name__}{' (no card)' if os.environ.get(NO_CARD) else ''}: {problem}"
                for check in checks for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    # the checks with the card, then the solve's alone with the card hidden (a CPU-only machine's way)
    sys.exit(in_newton() if os.environ.get(NO_CARD) else in_newton() or subprocess.run(
        [sys.executable, "-B", __file__], env={**os.environ, NO_CARD: "1", "CUDA_VISIBLE_DEVICES": ""}).returncode)
