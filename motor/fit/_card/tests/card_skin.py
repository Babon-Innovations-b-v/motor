"""The card's bones-in-skin rows on the toy clip (test fixture, test_the_card_step_and_projection_equal_the_cpu_prototype.py):
the toy with a bone mesh on body c and a box skin about it posed on every frame, two corners 5 mm outside at the
true pose, solved by the card (`_skin_step` in the damped step) and by mink with `_frames.SkinTask`."""
from card_toy import TOY, mink_frames, toy_clip, toy_clip_card, toy_errors, toy_poses

BONE = (0.03, 0.03, 0.12)  # body c's bone mesh: a box's half sizes about (0, 0, 0.15) in its frame
SKIN = (0.06, 0.06, 0.25, 0.035, 0.012)  # its skin: a box's half sizes about the bone's middle moved x and y in c's
# frame (two corners 5 mm off, the others in: no flat valley of turns about c's long axis)
TOY_BONED = TOY.replace('<mujoco><compiler angle="radian"/>', '<mujoco><compiler angle="radian"/><asset><mesh name="bone" vertex="'
                        + " ".join(f"{x * BONE[0]} {y * BONE[1]} {0.15 + z * BONE[2]}" for x in (-1, 1) for y in (-1, 1)
                                   for z in (-1, 1)) + '"/></asset>').replace(
    '<body name="c" pos="0 0 0.3"><joint name="h2" axis="0 0 1" range="-1.2 1.2"/><geom size="0.05"/>',
    '<body name="c" pos="0 0 0.3"><joint name="h2" axis="0 0 1" range="-1.2 1.2"/><geom size="0.05"/>'
    '<geom type="mesh" mesh="bone" contype="0" conaffinity="0" mass="0.001"/>')
TOY_GOAL = 10.0  # the skin's goal on the toy's 8 bone points: each point's weight 1.1, the real model's 0.35 (1e3 over
# 8,144); at the real goal's 1e3 the toy's skin outweighs its turns 100 times and leaves a flat valley about c's axis
# (0.018 deg between two stops of the same cost)
CUBE_FACES = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6],
              [0, 6, 4], [1, 5, 7], [1, 7, 3]]


def toy_skins(mjm):
    """The skin box about body c's bone on every frame of the toy clip at its true pose (`toy_poses`)."""
    import mujoco
    import numpy as np
    from motor.fit._skin import PosedSkins
    data, verts = mujoco.MjData(mjm), []
    corners = np.array([[x * SKIN[0] + SKIN[3], y * SKIN[1] + SKIN[4], 0.15 + z * SKIN[2]] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1)])
    for qpos in toy_poses(mjm):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(mjm, data)
        body = mjm.body("c").id
        verts.append(corners @ data.xmat[body].reshape(3, 3).T + data.xpos[body])
    return PosedSkins(np.array(verts), CUBE_FACES)


def depths_rms(mjm, poses, skins):
    """The bone points' root mean square depth outside their frames' skins (m) at `poses`."""
    import mujoco
    import numpy as np
    from motor.fit import _skin
    data, found = mujoco.MjData(mjm), []
    bodies, places = _skin.bone_bodies(mjm)
    for qpos, verts in zip(poses, skins.verts):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(mjm, data)
        points = data.xpos[bodies] + np.einsum("pij,pj->pi", data.xmat[bodies].reshape(-1, 3, 3), places)
        found.append(_skin.marks(points, verts, skins.faces)[1])
    return float(np.sqrt(np.mean(np.square(found))))


def the_card_holds_the_bones_inside_as_mink():
    """The toy clip with the skin, each row seeded as mink's frame was: the card's angles equal mink's with
    `_frames.SkinTask` to 0.01 deg and its sites to 0.01 mm, both by judged steps, and the bones stand under 0.9 of
    their depth outside without the skin (its rows are in the solve, not only beside it; measured 14.9 against 18.3 mm
    at TOY_GOAL)."""
    import mujoco
    import numpy as np
    from motor.fit import _skin
    mjm = mujoco.MjModel.from_xml_string(TOY_BONED)
    turns, places = toy_clip(mjm)
    skins = toy_skins(mjm)
    kept, _skin.SKIN_GOAL = _skin.SKIN_GOAL, TOY_GOAL
    try:
        reference = mink_frames(mjm, turns, places, skins)
        seeds = np.vstack([mjm.qpos0[None], reference[:-1]])
        found, _ = toy_clip_card(mjm).angles(turns[:, None], places, seeds, skins, exact=True)
    finally:
        _skin.SKIN_GOAL = kept
    free = mink_frames(mjm, turns, places)
    miss, sites = toy_errors(mjm, found.astype(float), reference)
    problems = [] if miss <= 0.01 else [f"angles {miss:.4f} deg off mink's with the skin (plan: 0.01)"]
    problems += [] if sites <= 0.01 else [f"sites {sites:.4f} mm off mink's with the skin (plan: 0.01)"]
    held, loose = depths_rms(mjm, reference, skins), depths_rms(mjm, free, skins)
    if not held < 0.9 * loose:
        problems.append(f"the skin holds the bones {1000 * held:.2f} mm outside against {1000 * loose:.2f} without it")
    return problems


def a_folded_frame_has_no_skin_rows_on_the_card():
    """The toy clip with every frame's skin marked folded (`PosedSkins.kept` all False, `fit._skin.unfolded`): the
    card's angles equal its angles with the skin kept at a goal of 0 (no pull, the same judged steps) to 1e-6 deg, and
    the skin kept at TOY_GOAL moves them over 0.01 deg (its rows pull the bones in)."""
    import mujoco
    import numpy as np
    from motor.fit import _skin
    mjm = mujoco.MjModel.from_xml_string(TOY_BONED)
    turns, places = toy_clip(mjm)
    skins = toy_skins(mjm)
    folded = _skin.PosedSkins(skins.verts, skins.faces, np.zeros(len(skins), dtype=bool))
    seeds = np.repeat(mjm.qpos0[None], len(turns), axis=0)
    kept, found = _skin.SKIN_GOAL, {}
    try:
        for name, goal, posed in (("none", 0.0, skins), ("folded", TOY_GOAL, folded), ("held", TOY_GOAL, skins)):
            _skin.SKIN_GOAL = goal
            found[name] = toy_clip_card(mjm).angles(turns[:, None], places, seeds, posed)[0].astype(float)
    finally:
        _skin.SKIN_GOAL = kept
    off, _ = toy_errors(mjm, found["folded"], found["none"])
    pulled, _ = toy_errors(mjm, found["held"], found["none"])
    problems = [] if off <= 1e-6 else [f"the folded frames' angles {off:.2e} deg off the solve with no skin pull"]
    return problems + ([] if pulled > 0.01 else [f"the kept skin moves the angles only {pulled:.4f} deg"])
