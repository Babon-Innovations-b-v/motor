"""The level-2 check's problem: fit's own toy (a free pelvis, two legs, a coupled ankle, the knee a joint inside its
group; `../../test_the_solver_...py`'s `toy_fitting`), five of its frames, five markers free, the unknowns moved off
the start so the root's rotation vector and the moves are not zero; and the 2-point central differences of its
residuals, the reference the exact Jacobian is held to. Newton's Python (mujoco, scipy)."""
import importlib.util
import pathlib

import numpy as np

FIT = pathlib.Path(__file__).resolve().parents[2]
FREE_MARKS = 5
SPREAD = 0.05  # the unknowns' move off the start (metres, radians)
STEP = 1e-6


def toy_test():
    """fit's own test module (its toy, its targets)."""
    path = next(FIT.glob("test_*.py"))
    spec = importlib.util.spec_from_file_location("fit_toy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def problem_and_point(folder):
    """The toy's constants problem and a point off its start."""
    from motor.fit._constants import Problem
    toy = toy_test()
    fitting = toy.toy_fitting(folder)
    turns, points, _ = toy.posed_targets(fitting, toy.truth_of(fitting), toy.POSES[:-1])
    path = []
    for place, hip_l, knee_l, hip_r in toy.POSES[:-1]:
        qpos = fitting.model.qpos0.copy()
        qpos[:3] = place
        qpos[7:11] = hip_l, knee_l, hip_r, 0.5 * hip_r
        path.append(qpos)
    problem = Problem(fitting, np.array(path), turns, points, fitting.neutral(), list(range(FREE_MARKS)))
    start = problem.start()
    return problem, start + np.random.default_rng(1).normal(0.0, SPREAD, start.size)


def differences(problem, unknowns):
    """The residuals' central differences, column by column (dense)."""
    columns = []
    for column in range(unknowns.size):
        up, down = unknowns.copy(), unknowns.copy()
        up[column] += STEP
        down[column] -= STEP
        columns.append((problem.residuals(up) - problem.residuals(down)) / (2 * STEP))
    return np.array(columns).T


def moved_given(folder, seed=2):
    """The toy's staged inputs with the path's places and hinges moved SPREAD off the poses (`seed` the move)."""
    problem, _ = problem_and_point(folder)
    path = problem.path.copy()
    unturned = [index for index in range(path.shape[1]) if index not in range(3, 7)]
    path[:, unturned] += np.random.default_rng(seed).normal(0.0, SPREAD, (len(path), len(unturned)))
    return problem.fitting, path, problem.turns, problem.points, problem.given, problem.free


def staged_both(folder):
    """The toy's constants and angles by `_constants.staged` from the same start, the path's places and hinges
    moved SPREAD off the poses so the solve has a way to go: scipy's 2-point differences (the reference), the exact
    Jacobian (`jacobian`, host kinematics) and the card's float64 evaluator (`wide`)."""
    from motor.fit._card import _level2
    from motor.fit._constants import staged
    given = moved_given(folder)
    return staged(*given), staged(*given, jacobian=_level2.jacobian), staged(*given, wide=True, card=False)


WIDE_MOST = 1e-12  # the card's float64 kinematics (_level2_wide) against the host's, of the largest entry (measured 7.8e-16 / 1.2e-15)
LEVEL2_SOLVE_MM = 1e-3  # level 2 with jac=exact against 2-point differences on the exact toy: constants (the
LEVEL2_SOLVE_DEG = 1e-4  # plan's 0.05 mm / 0.01 deg is the real bodies'; a frozen Jacobian is 0.027 mm off here)
LEVEL2_MOST = 1e-6  # a column's largest miss over its largest entry (measured 1.8e-10; the plan allows 1e-2 against 2-point)


def the_level2_jacobian_equals_its_differences():
    import tempfile
    import numpy as np
    from motor.fit._card import _level2
    with tempfile.TemporaryDirectory() as folder:
        problem, unknowns = problem_and_point(folder)
        moved = differences(problem, unknowns)
        outside = ~problem.sparsity().toarray().astype(bool)
        exact = _level2.jacobian(problem, unknowns).toarray()
    worst = max(np.abs(exact[:, column] - moved[:, column]).max() / max(np.abs(moved[:, column]).max(), 1e-12)
                for column in range(unknowns.size))
    return ([f"host Jacobian off its differences by {worst:.2e} of a column"] if worst > LEVEL2_MOST else []) + (
        ["host Jacobian outside the sparse pattern"] if np.abs(exact[outside]).max() > 0 else [])


def the_wide_level2_equals_the_hosts():
    """`_level2_wide.Wide` (every frame's float64 kinematics on the card, the rows in one pass) gives the host's
    residuals and exact Jacobian, its sparse layout too, at the start and at a point moved off it."""
    import tempfile
    import numpy as np
    from motor.fit._card import _level2_wide
    problems = []
    with tempfile.TemporaryDirectory() as folder:
        problem, moved = problem_and_point(folder)
        wide = _level2_wide.Wide(problem)
        for label, point in (("start", problem.start()), ("moved", moved)):
            host_rows, card_rows = problem.residuals(point), wide.residuals(point)
            host, card = problem.exact(point), wide.exact(point)
            off_rows = np.abs(host_rows - card_rows).max() / np.abs(host_rows).max()
            off = abs(host - card).max() / abs(host).max()
            problems += [f"{label}: card residuals off the host's by {off_rows:.2e}"] if off_rows > WIDE_MOST else []
            problems += [f"{label}: card Jacobian off the host's by {off:.2e}"] if off > WIDE_MOST else []
            problems += ([f"{label}: card Jacobian's sparse layout is not the host's"]
                         if not (np.array_equal(host.indptr, card.indptr) and np.array_equal(host.indices, card.indices))
                         else [])
    return problems


def the_level2_solve_with_the_exact_jacobian_equals_differences():
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as folder:
        (moved, moved_angles, _), *exact = staged_both(folder)
    problems = []
    for label, (constants, angles, _) in zip(("the exact Jacobian", "the card's float64 evaluator"), exact):
        off_mm, off_angles = 1000 * np.abs(constants - moved).max(), np.abs(angles - moved_angles).max()
        problems += ([f"level 2 with {label} off differences' constants by {off_mm:.4f} mm"]
                     if off_mm > LEVEL2_SOLVE_MM else []) + (
            [f"level 2 with {label} off differences' angles by {off_angles:.2e}"]
            if off_angles > np.radians(LEVEL2_SOLVE_DEG) else [])
    return problems


def the_batched_inverses_equal_each_row():
    """`_level2.left_jacobian_inverses`, one array pass over every frame and group, equals `left_jacobian_inverse`
    row by row to the float64 floor (2e-15), a near-zero rotation (its small-angle branch) among the rows."""
    from motor.fit._card import _level2
    rotations = np.vstack([np.random.default_rng(5).normal(0.0, 1.0, (40, 3)), [[1e-9, 0.0, -2e-9], [0.0, 0.0, 0.0]]])
    batched = _level2.left_jacobian_inverses(rotations)
    worst = max(np.abs(batched[index] - _level2.left_jacobian_inverse(rotation)).max()
                for index, rotation in enumerate(rotations))
    return [f"batched inverse left Jacobians off each row's by {worst:.2e}"] if worst > 2e-15 else []


CARD_MM, CARD_DEG, CARD_COST = 0.01, 0.01, 1e-8  # the owner's proof tolerances (#70, 2026-11-03), on the toy too


def free_gradient(ready, constants, angles):
    """The polish's free gradient (told unknowns off their bounds) and the cost at a solved (constants, angles), on
    `ready` (`_constants.prepared(..., wide=True)`: problem, start, told columns, bounds)."""
    from motor.fit._constants import off_bounds
    problem, begin, keep, bounds = ready
    unknowns = begin.copy()
    held = len(problem.slots)
    unknowns[:held] = constants[problem.slots]
    for frame, qpos in enumerate(angles):
        start = 0
        if problem.root is not None:
            from scipy.spatial.transform import Rotation
            at = problem.root
            unknowns[held + frame * problem.width:held + frame * problem.width + 3] = qpos[at:at + 3]
            unknowns[held + frame * problem.width + 3:held + frame * problem.width + 6] = (
                Rotation.from_quat(problem.path[frame][at + 3:at + 7], scalar_first=True).inv()
                * Rotation.from_quat(qpos[at + 3:at + 7], scalar_first=True)).as_rotvec()
            start = 6
        unknowns[held + frame * problem.width + start:held + (frame + 1) * problem.width] = qpos[
            problem.fitting.model.jnt_qposadr[problem.leading]]
    residuals = problem.residuals(unknowns)
    free = off_bounds(unknowns, keep, bounds)
    gradient = problem.exact(unknowns)[:, free].T @ residuals
    return np.abs(gradient).max(), float(residuals @ residuals)


def apart(ready, reference, found):
    """Where `found` (constants, angles, _) misses `reference` past the owner's proof tolerances (`ready` as
    `free_gradient` reads it). The cost only above scipy's: below it is scipy's own shortfall (2026-11-03, the
    bounded toy: trf keeps its unknowns strictly inside their bounds and the polish leaves those within BOUND_HELD of
    one, so scipy's cost stood 3.2e-8 of itself over the card's, whose step lands on the bound)."""
    (constants, angles, _), (own_constants, own_angles, _) = reference, found
    off_mm, off_deg = 1000 * np.abs(own_constants - constants).max(), np.degrees(np.abs(own_angles - angles).max())
    gradient, cost = free_gradient(ready, own_constants, own_angles)
    _, reference_cost = free_gradient(ready, constants, angles)
    return ([f"constants {off_mm:.2e} mm off"] if off_mm > CARD_MM else []) + (
        [f"angles {off_deg:.2e} deg off"] if off_deg > CARD_DEG else []) + (
        [f"cost {cost / reference_cost - 1:.2e} relative over scipy's"] if cost / reference_cost - 1 > CARD_COST
        else []) + ([f"free gradient {gradient:.2e}"] if gradient > 1e-10 else [])


STEP_MOST = 1e-9  # the card's Schur step off the dense damped solve, of the step's largest entry


def steps_off(cases):
    """How far `_card._level2_lm.steps` (both cases in one launch, a bound held in the first) is off each set's
    whole damped normal equations solved dense: (H + lambda diag H) d = -g, held unknowns' rows the identity's."""
    from motor.fit._card._level2_lm import Set, steps
    from motor.fit._constants import prepared
    sets = [Set(*prepared(*case, True)) for case in cases]
    for one in sets:
        one.linearised()
    sets[0].arrow.free[1] = False
    lams = [1e-3, 0.37]
    worst = 0.0
    for one, lam, step in zip(sets, lams, steps([one.arrow for one in sets], lams)):
        arrow = one.arrow
        square = (one.matrix.T @ one.matrix).toarray() + lam * np.diag(arrow.diagonal)
        free = arrow.free
        square[~free, :] = 0.0
        square[:, ~free] = 0.0
        square[~free, ~free] = 1.0
        dense = np.linalg.solve(square, -np.where(free, arrow.gradient, 0.0))
        worst = max(worst, np.abs(step - dense).max() / np.abs(dense).max())
    return worst


NARROW = 1e-3  # metres: the toy's constants' bound in the bounded case, so the solve ends with constants on it


def bounded(folder):
    """The toy at its moved path with its constants' bound NARROW (`_constants.MARK_MOST` for the case's
    preparation and solves): (case, prepared, scipy's reference, the card's solve), several constants ending on it."""
    from motor.fit import _constants
    most = _constants.MARK_MOST
    _constants.MARK_MOST = NARROW
    try:
        case = moved_given(folder, 4)
        ready = _constants.prepared(*case, True)
        return (ready, _constants.staged(*case, wide=True, card=False), _constants.staged(*case, wide=True, card=True))
    finally:
        _constants.MARK_MOST = most


def own_stops(cases):
    """How `_card._level2_lm.solved` itself stopped on each case solved together (before any polish)."""
    from motor.fit._card._level2_lm import Set, solved
    from motor.fit._constants import prepared
    sets = [Set(*prepared(*case, True)) for case in cases]
    solved(sets)
    return [one.stop for one in sets]


def the_card_levenberg_marquardt_equals_scipys_level2():
    """Level 2 by `_card._level2_lm` (`staged(card=True)`) equals `staged(wide=True)` (scipy's least_squares and the
    polish) on the toy from the moved path, constants 0.01 mm, angles 0.01 deg, cost 1e-8 relative, its free
    gradient under 1e-10, the card's solve settled there by itself (not by the polish after it), its damped step (two sets in one launch, a
    bound held) the whole normal equations' dense solve to 1e-9; two sets solved at once (`staged_across`, each its own moved path) equal each alone; with the constants' bound
    NARROW, so several end on it, equal too (a bound held only once the step put it there, 2026-11-03: held within
    BOUND_HELD of it, the real bodies' cost stood 1.1e-8 over scipy's)."""
    import tempfile
    from motor.fit._constants import prepared, staged, staged_across
    problems = []
    with tempfile.TemporaryDirectory() as folder:
        cases = [moved_given(folder, seed) for seed in (2, 3)]
        readies = [prepared(*case, True) for case in cases]
        references = [staged(*case, wide=True, card=False) for case in cases]
        alone = staged(*cases[0], wide=True, card=True)
        problems += [f"alone: {miss}" for miss in apart(readies[0], references[0], alone)]
        problems += [f"the card's own solve stopped {stop}, not settled (the polish did its work)"
                     for stop in own_stops(cases) if stop != "settled"]
        off = steps_off(cases)
        problems += [f"the card's Schur step {off:.2e} off the dense damped solve"] if off > STEP_MOST else []
        for number, (ready, reference, found) in enumerate(zip(readies, references, staged_across(cases))):
            problems += [f"set {number} of two: {miss}" for miss in apart(ready, reference, found)]
        ready, reference, found = bounded(folder)
        on_bound = int((np.abs(np.abs(reference[0]) - NARROW) < 1e-9).sum())
        problems += [f"bounded: {miss}" for miss in apart(ready, reference, found)] + (
            [f"bounded: only {on_bound} constants on their bound"] if on_bound < 2 else [])
    return problems
