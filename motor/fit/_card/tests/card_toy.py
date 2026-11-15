"""The toy clip the loop checks run on: a toy MJCF (a free root, three ranged hinges, a polyval follower in its own
body, a frozen hinge), its 10-frame targets, mink's answer set up as _frames.Frames, the card's setup and the errors
between two answers (test fixture, test_the_card_step_and_projection_equal_the_cpu_prototype.py)."""
import pathlib
import tempfile

TOY = """<mujoco><compiler angle="radian"/><worldbody><body name="root"><freejoint/><geom size="0.05"/>
 <body name="a" pos="0 0 0.3"><joint name="h0" axis="1 0 0" range="-1.2 1.2"/><geom size="0.05"/>
  <body name="b" pos="0 0 0.3"><joint name="h1" axis="0 1 0" range="-1.2 1.2"/><geom size="0.05"/>
   <body name="c" pos="0 0 0.3"><joint name="h2" axis="0 0 1" range="-1.2 1.2"/><geom size="0.05"/>
   <body name="cf"><joint name="f" axis="1 0 0"/><geom size="0.02"/>
    <site name="t" pos="0.05 0 0.2" quat="0.9 0.1 0.3 0.2"/><site name="p2" pos="0 0.1 0.25"/>
    <body name="d" pos="0 0 0.3"><joint name="z" axis="0 1 0"/><geom size="0.05"/><site name="p3" pos="0.1 0 0.2"/>
 </body></body></body></body><site name="p1" pos="0.05 0 0.2"/></body><site name="p0" pos="0.1 0.1 0"/></body></worldbody>
 <equality><joint joint1="f" joint2="h1" polycoef="0 0.5 0.2 0 0"/></equality></mujoco>"""
CLIP_FRAMES, PLACE_SITES, TURN_COST, PLACE_COST = 10, ("p0", "p1", "p2", "p3"), 0.7, 3.0


def toy_poses(mjm):
    """The toy clip's true poses (qpos per frame): one hinge driven past its range, the follower on its coupling."""
    import numpy as np
    found = []
    for moment in np.linspace(0.0, 1.0, CLIP_FRAMES):
        qpos = mjm.qpos0.copy()
        qpos[:3] = (0.1 * moment, 0.0, 0.05 * moment)
        qpos[7:10] = (0.8 * np.sin(2.0 * moment), 1.5 * moment, -0.6 * moment)
        qpos[10] = 0.5 * qpos[8] + 0.2 * qpos[8] ** 2
        found.append(qpos)
    return found


def toy_clip(mjm):
    """The toy's targets: a path of poses (`toy_poses`) through its sites, the places 2 mm off."""
    import mujoco
    import numpy as np
    data, generator = mujoco.MjData(mjm), np.random.default_rng(5)
    turns, places = [], []
    for qpos in toy_poses(mjm):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(mjm, data)
        turns.append(data.site("t").xmat.reshape(3, 3).copy())
        places.append([data.site(name).xpos + generator.normal(0.0, 0.002, 3) for name in PLACE_SITES])
    return np.array(turns), np.array(places)


def mink_frames(mjm, turns, places, skins=None):
    """mink (../_frames.solved) frame by frame, each frame from the one before, its targets set as _frames.Frames sets
    them (a place's target turned as its site turns now, so the error is the plain place error; limits at gain 1), and
    with `skins` (`fit._skin.PosedSkins`, a frame each) the bones held inside them by `_frames.SkinTask`: the
    reference angles."""
    import mink
    import numpy as np
    from motor.fit import _frames
    configuration = mink.Configuration(mjm)
    turn = mink.FrameTask("t", "site", position_cost=0.0, orientation_cost=TURN_COST, lm_damping=_frames.LM_DAMPING)
    marks = [mink.FrameTask(name, "site", position_cost=PLACE_COST, orientation_cost=0.0,
                            lm_damping=_frames.LM_DAMPING) for name in PLACE_SITES]
    frozen = mink.DofFreezingTask(mjm, dof_indices=[int(mjm.jnt_dofadr[mjm.joint("z").id])])
    constraints = [mink.EqualityConstraintTask(mjm, cost=1.0, gain=1.0), frozen]
    def marked(frame):
        for mark, place in zip(marks, places[frame], strict=True):
            now = configuration.get_transform_frame_to_world(mark.frame_name, "site").rotation()
            mark.set_target(mink.SE3.from_rotation_and_translation(now, place))
    found, skin = [], None if skins is None else _frames.SkinTask(mjm)
    for frame in range(len(turns)):
        turn.set_target(mink.SE3.from_rotation(mink.SO3.from_matrix(turns[frame])))
        marked(frame)
        if skin is not None:
            skin.set_skin(skins[frame:frame + 1])
        found.append(_frames.solved(configuration, [turn, *marks, *([skin] if skin else [])],
                                    [mink.ConfigurationLimit(mjm, gain=1.0)], constraints, lambda: marked(frame)))
    return np.array(found)


def site_places(mjm, poses):
    """Every site's world place [poses, sites, 3] by MuJoCo's own kinematics."""
    import mujoco
    import numpy as np
    data, found = mujoco.MjData(mjm), []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(mjm, data)
        found.append(data.site_xpos.copy())
    return np.array(found)


def toy_card(mjm, turns, places):
    """The card loop's parts for the toy's clip, set up by `_clip.Clip` from the toy's own model as `_frames.Frames`
    sets mink up: (optimizer_for(start, stop), reduction, projection)."""
    clip = toy_clip_card(mjm)
    return clip.optimizer_for(turns[:, None], places), clip.reduction, clip.projection


def toy_clip_card(mjm, frozen=("z",)):
    """The toy's `_clip.Clip` (its `frozen` joints' dofs held), checked as `toy_card` uses it."""
    import newton
    import numpy as np
    from motor.fit._card._clip import MAPPED, Clip, coordinate_bounds, followers_free_ranges, site_miss
    from motor.fit._model import followers_free
    ranged = [mjm.joint("h1").id]  # a ranged joint taken as the follower, so the copy's bounds change
    if not np.array_equal(coordinate_bounds(followers_free_ranges(mjm, ranged)),
                          coordinate_bounds(followers_free(mjm, ranged))):
        raise ValueError("followers_free_ranges' bounds are not followers_free's")
    path = pathlib.Path(tempfile.mkdtemp()) / "toy.xml"
    path.write_text(TOY)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
    clip = Clip(builder.finalize(device="cuda:0"), mjm, [mjm.site("t").id], [mjm.site(name).id for name in PLACE_SITES],
                TURN_COST, [PLACE_COST] * len(PLACE_SITES), [int(mjm.jnt_dofadr[mjm.joint(name).id]) for name in frozen],
                followers_free_ranges(mjm, [mjm.joint("f").id]))
    if site_miss(clip) > MAPPED:
        raise ValueError(f"the toy's Newton sites {site_miss(clip)} m off MuJoCo's")
    return clip


def toy_errors(mjm, found, reference):
    """(largest angle miss in degrees, largest site miss in mm) of `found` against `reference` on the toy."""
    import numpy as np
    angles = np.degrees(np.abs(found[:, 7:] - reference[:, 7:])).max()
    return angles, 1000.0 * np.abs(site_places(mjm, found) - site_places(mjm, reference)).max()
