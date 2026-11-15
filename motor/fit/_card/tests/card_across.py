"""The across-sets solve: the toy's clip as three sets (the base and two with their inside joints moved, each set's
marker targets its own), solved as one batch of rows = set x frame (`_solve.solved_in_chunks` with set_places and
row_set) equals each set solved alone on its own model to 0.01 deg and its root to 0.01 mm (test fixture,
test_the_card_step_and_projection_equal_the_cpu_prototype.py)."""
SETS = 3
SHIFT = 0.001  # m, each set's marker targets moved on from the last set's


def moved_places(model):
    """[SETS, joints, 7] joint places: the toy's own, then each set's inside joints (all but the root) moved."""
    import numpy as np
    base = model.joint_X_p.numpy()
    places = np.repeat(base[None], SETS, axis=0)
    places[1:, 1:, :3] += np.random.default_rng(21).normal(0.0, 0.002, size=(SETS - 1, base.shape[0] - 1, 3))
    return places.astype(np.float32)


def the_sets_solved_together_equal_each_set_alone():
    """Each set's angles from the one batch against the same set solved alone; problems as text."""
    import mujoco
    import numpy as np
    from card_toy import TOY, mink_frames, toy_card, toy_clip
    from motor.fit._card import solved_in_chunks
    from motor.fit._card._seeds import newton_order
    mjm = mujoco.MjModel.from_xml_string(TOY)
    turns, places = toy_clip(mjm)
    frames = turns.shape[0]
    targets = [places + index * SHIFT for index in range(SETS)]
    seeds = mink_frames(mjm, turns, places)  # the base set's answer: every set seeded from it
    optimizer_for, reduction, projection = toy_card(mjm, np.concatenate([turns] * SETS), np.concatenate(targets))
    set_places = moved_places(optimizer_for(0, 1).model)
    row_set = np.repeat(np.arange(SETS), frames)
    together, _ = solved_in_chunks(optimizer_for, reduction, projection, newton_order(np.tile(seeds, (SETS, 1))),
                                   np.tile(seeds, (SETS, 1)), set_places=set_places, row_set=row_set)
    problems = []
    for index in range(SETS):
        alone_for, _, _ = toy_card(mjm, turns, targets[index])
        model = alone_for(0, 1).model
        model.joint_X_p.assign(set_places[index])
        alone, _ = solved_in_chunks(alone_for, reduction, projection, newton_order(seeds), seeds)
        rows = slice(index * frames, (index + 1) * frames)
        miss = np.degrees(np.abs(together[rows, 7:] - alone[:, 7:])).max()
        root = 1000.0 * np.abs(together[rows, :3] - alone[:, :3]).max()
        if miss > 0.01 or root > 0.01:
            problems.append(f"set {index}: angles {miss:.4f} deg, root {root:.4f} mm off the set alone (plan: 0.01)")
    return problems
