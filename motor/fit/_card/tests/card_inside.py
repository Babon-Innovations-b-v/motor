"""Newton's import of a MuJoCo body carrying several joints: one Newton joint into the body (its joints merged into
one D6 at the first joint's place), so `_clip.moved_inside`'s map of child body to joint names that one joint and
moving it moves the whole body (test fixture, test_the_card_step_and_projection_equal_the_cpu_prototype.py)."""
SEVERAL = """<mujoco><worldbody><body name="root" pos="0 0 1"><freejoint/><geom size="0.1"/>
<body name="arm" pos="0.2 0 0"><joint name="a1" type="hinge" axis="1 0 0"/><joint name="a2" type="hinge" axis="0 1 0"/>
<joint name="a3" type="slide" axis="0 0 1"/><geom size="0.05"/>
<body name="hand" pos="0.3 0 0"><joint name="h1" type="hinge" axis="0 0 1"/><geom size="0.03"/></body></body>
</body></worldbody></mujoco>"""
MOVE = (0.0, 0.01, 0.0)  # m, the arm's inside joint move
TOLERANCE = 1e-6  # m


def the_body_with_several_joints_moves_as_one():
    """The arm's one Newton joint placed at its file place plus the move, the hand's left alone; problems as text."""
    import types
    import mujoco
    import newton
    import numpy as np
    import warp as wp
    from motor.fit._card._clip import moved_inside
    mjm = mujoco.MjModel.from_xml_string(SEVERAL)
    path = written_file(SEVERAL)
    arm = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "arm")
    fitting = types.SimpleNamespace(model=mjm, base=[mjm.body_pos.copy()], inside=np.array([arm]))
    mjm.body_pos[arm] += MOVE
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
    into = {label.split("/")[-1]: [joint for joint, child in enumerate(builder.joint_child) if child == body]
            for body, label in enumerate(builder.body_label)}
    if len(into["arm"]) != 1:
        return [f"the arm has {len(into['arm'])} Newton joints, not one"]
    before = {name: np.asarray(wp.transform_get_translation(builder.joint_X_p[joints[0]])) for name, joints in into.items()}
    moved_inside(builder, fitting)
    after = {name: np.asarray(wp.transform_get_translation(builder.joint_X_p[joints[0]])) for name, joints in into.items()}
    wanted = {name: before[name] + (np.array(MOVE) if name == "arm" else 0.0) for name in before}
    return [f"{name}'s joint place off by {np.abs(after[name] - wanted[name]).max():.6f} m" for name in before
            if np.abs(after[name] - wanted[name]).max() > TOLERANCE]


def written_file(text):
    """A file holding `text` in a fresh temporary directory (Newton's import reads a path)."""
    import pathlib
    import tempfile
    path = pathlib.Path(tempfile.mkdtemp()) / "several.xml"
    path.write_text(text)
    return path


def the_set_places_equal_a_build_per_set():
    """`_clip.set_places` from one build holding set 0's move (MOVE) against a Newton build moved by each set's own
    move (none, MOVE, three times MOVE): every joint place within TOLERANCE; problems as text."""
    import types
    import mujoco
    import newton
    import numpy as np
    from motor.fit._card._clip import inside_joints, moved_inside, set_places
    mjm = mujoco.MjModel.from_xml_string(SEVERAL)
    path = written_file(SEVERAL)
    arm = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "arm")
    fitting = types.SimpleNamespace(model=mjm, base=[mjm.body_pos.copy()], inside=np.array([arm]))

    def built(move):
        mjm.body_pos[arm] = fitting.base[0][arm] + move
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_mjcf(str(path), convert_mjc_equality_constraints=False, parse_meshes=False)
        moved_inside(builder, fitting)
        return builder.finalize(device="cpu")
    base = built(np.array(MOVE))
    joints, problems = inside_joints(base, fitting), []
    for scale in (0.0, 1.0, 3.0):
        move = scale * np.array(MOVE)
        wanted = built(move).joint_X_p.numpy()
        got = set_places(base.joint_X_p.numpy(), joints, [MOVE], [move])
        if np.abs(got - wanted).max() > TOLERANCE:
            problems.append(f"set_places: the set moved {scale:g} x MOVE off its own build by "
                            f"{np.abs(got - wanted).max():.6f} m")
    return problems


class Kept:
    """A fitting stand-in `_clip.newton_model_for` keeps a build for (weak-referenceable, unlike SimpleNamespace)."""

    def __init__(self, model, inside):
        self.model, self.base, self.inside = model, [model.body_pos.copy()], inside


def a_kept_build_moves_to_the_fittings_moves():
    """`_clip.newton_model_for` asked twice for one fitting and file, its arm moved by MOVE between: the same model,
    its joint places within TOLERANCE of a build moved by MOVE; problems as text."""
    import mujoco
    import numpy as np
    from motor.fit._card._clip import newton_model_for
    mjm = mujoco.MjModel.from_xml_string(SEVERAL)
    path = written_file(SEVERAL)
    arm = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "arm")
    kept, fresh = Kept(mjm, np.array([arm])), Kept(mujoco.MjModel.from_xml_string(SEVERAL), np.array([arm]))
    first = newton_model_for(kept, path, "cpu")
    mjm.body_pos[arm] += MOVE
    fresh.model.body_pos[arm] += MOVE
    second, wanted = newton_model_for(kept, path, "cpu"), newton_model_for(fresh, path, "cpu")
    if second is not first:
        return ["the second call built a new model"]
    gap = np.abs(second.joint_X_p.numpy() - wanted.joint_X_p.numpy()).max()
    return [] if gap <= TOLERANCE else [f"the kept build's joint places {gap:.6f} m off a build moved by MOVE"]
