"""fit/_card: on 12 rows (frames) of a small chain (20 hinge dofs, two frozen, two followers on one leader and one
without a leader, bounds with some rows sitting on them), seeded alike, one step at a time: the card's damped reduced
step equals the CPU prototype's (card/proto/lm.py: mink's damping 1e-2 |r|^2 + 1e-9 on every full dof, so S^T S in
the reduced dofs (a leader's damping times 1 plus its followers' slopes squared), frozen out, followers through
their coupling's slope, three active-set passes holding dofs at a bound that push out, a held one released when its descent points back inside) to 1e-5 rad, and
the card's projection equals the prototype's (ranges, couplings by polyval, frozen) with every follower on polyval of
its leader to 1e-6 rad and every coordinate inside its range. Rows whose dofs are held must exist (else the hold is
not tested). On a six-link hinge chain, the per-problem place and turn objectives (_objective.py) give the residuals
and analytic Jacobian of Newton's own IKObjectivePosition and IKObjectiveRotation made with each row's offset and
weight, the turn's Jacobian through the log's inverse left Jacobian (mink's exact one, Newton's is the spin alone), to
1e-6. Runs in Newton's own Python on a CUDA card; with no Newton env or no card it prints `skipped`. On a toy MJCF (a
free root, three ranged hinges, a follower coupled by polyval in its own body, a frozen hinge), the card loop on a
10-frame clip, each row seeded as mink's frame was, equals mink (../_frames.solved, targets set as _frames.Frames sets
them) to 0.01 deg and its sites to 0.01 mm, with a frame holding h1 at its range; in chunks of 3 rows it equals
itself in one batch. Seeded from that clip's answers (its angles.npz, `_seeds`) with the places 3 mm on, the card
equals mink's answer to the moved clip to 0.01 deg and 0.01 mm (a cold seed only reported); seeds one coordinate
short of the model are refused. On a toy body
(`card_body.py`), the card's measures (_measure.py) equal evaluation._measure's to 1e-9 (joints mm, turns deg) and
its bones-outside shares exactly (every point's count), their depths outside (level 1's skin residual, the same
kernel on both devices) to DEPTH_MM. A second clip after the skin was posed once (`_measure.POSED`, shared across sets) equals the CPU's on its own. On fit's own toy (`card_level2.py`), level 2's exact
Jacobian (_level2.py) equals its residuals' central differences to 1e-6 of each column's largest entry from the
host's kinematics, nothing outside `Problem.sparsity`; the card's float64 evaluator (_level2_wide.py: every frame's
kinematics on the card) gives the host's residuals and Jacobian to 1e-12 of their largest entry, the same sparse
layout, at the start and off it; level 2 solved with the exact Jacobian (`_constants.staged(jacobian=...)`) and
with the card's evaluator (`staged(wide=True)`) from a path moved 0.05 off equals the 2-point solve to 1e-3 mm and
1e-4 deg. Level 2 by the card's own Levenberg-Marquardt (`_level2_lm`, `staged(card=True)`) equals
`staged(wide=True)` with its polish on the toy (0.01 mm, 0.01 deg, cost 1e-8 relative, free gradient under 1e-10), its
own solve settled there before the polish, its Schur step (two sets in one launch, a constant held) the dense damped
solve to 1e-9, two sets solved together (`staged_across`) each equal to its own reference, and with the toy's
constants' bound 1 mm (several end on it) equal too, its cost judged only above scipy's (scipy stops short of a
bound, so the card's may be lower). Level 2's
inverse left Jacobians batched over every frame and group equal the per-row ones to 2e-15. A body carrying several
MuJoCo joints is one Newton joint, and `_clip.moved_inside` moves that one by the body's move (`card_inside.py`).

Breaks (each applied once, seen red, reverted):
- (2026-11-03) _level2_lm.Set.tried's step not clipped into the bounds: the bounded toy's constants 7.1 mm and its
  angles 0.12 deg off scipy's. (Holding an unknown within BOUND_HELD of its bound instead of on it, the real bodies'
  first proof's fault, cost 1.1e-8 of itself over scipy's, is not seen by the toy: its steps land on the bound.)
- (2026-11-03) _level2_lm._schur's sum made `0.0 *` (no Schur correction): the step 0.24 of itself off the dense
  solve (the solve still settled: Levenberg-Marquardt's control hides a wrong step, so the step is checked itself).
- (2026-11-03) _level2_lm.steps' frame step `own[:, :, 0] - own[...] @ constant_step` made `+`: 0.090 off.
- (2026-11-03) _level2_lm._schur's `frame_set[frame] == own` made `== 0`: the second set's Schur sum reads the
  first set's frames, 0.31 off.
- (2026-11-03) _level2_lm.Arrow.damped's held constant's diagonal 1 dropped: the constants' Cholesky fails
  (LinAlgError) on the held one.
- (2026-10-31) _clip.newton_model_for's kept build returned without `set_places`: the clips' Newton model keeps level 2's
  sampled frames' inside joint places, the constants' moves never reach it.
- (2026-10-31) _solve.solved's coordinate check dropped: seeds one coordinate short of the toy are solved, not
  refused: the toy runs to NotConverged (seeds147, made for the old joint layout, broke the card on the movable
  MyoFullBody).
- (2026-10-29) _kept's release `wp.where(inside, 1.0, 0.0)` made `0.0` (holds only added): rows 0 and 1 keep dof 0
  held on its bound though its descent points inside, the step 1 rad off.
- (2026-10-29) _step's `LM_DAMPING * squared + DAMPING` made `DAMPING`: mink's damping is lost, the step differs.
- (2026-10-29) _normal_equations' `* spread` dropped (damping lambda I in the reduced dofs, not mink's S^T S): the
  step differs from the prototype's by 7.7e-5 rad; on heavy_squat cache 3 mm row 148 swung elv_angle_r 1 rad a step.
- (2026-10-29) _reduced_jacobian's `* slopes[row, coupling]` dropped: followers ride their leader at slope 1.
- (2026-10-29) _clip's joint types compared as mjtJoint enums with the model's ints (never equal): no range held,
  the toy's clip 27 deg and 55 mm off mink's.
- (2026-10-29) _step's `HOLD_PASSES = 3` made `0`: a dof at its bound is not held; the clip's rows never stop.
- (2026-10-29) _kept's `keep[row, column]` made `1.0`: a hold from an earlier pass is let go, the step differs.
- (2026-10-29) _turn_jacobian's `rate[axis]` made `motion[3 + axis]` (Newton's spin alone): the turn's
  Jacobian is no longer the log's exact one.
- (2026-10-29) solved_in_chunks' `values[start:stop]` made `values[:stop - start]`: every chunk solves the first rows.
- (2026-10-29) _measure.joint_table's `* scales[bone]` dropped: the joints miss by the unscaled offset.
- (2026-11-01) _skin_step._bodies' `square[3 + first, 3 + second] -= ...far...` made `+=`: 10 rows do not stop in 300
  steps (NotConverged).
- (2026-11-01) _step's `pulled += extra_g[row, first]` made `0.0 * extra_g`: the card's angles 3.15 deg and sites
  8.86 mm off mink's with the skin (the_card_holds_the_bones_inside_as_mink red).
- (2026-11-01) _skin_step._points' `if kept[row] != 0:` made `... or True`: the folded toy clip's angles 3.24 deg off
  the solve with no skin pull (a_folded_frame_has_no_skin_rows_on_the_card red).
- (2026-11-01) _measure.measures' `skin_shares(marks[skins.kept], ...)` made `skin_shares(marks, ...)`: the card counts
  the folded frame in its shares, 77.2 against the CPU's 74.6 % (a_folded_frame_is_measured_apart_as_on_the_cpu red).
- (2026-11-01) _measure.measures' "bones outside all %" `if unfolded_any else None` dropped: with every frame folded
  the card gives NaN (l1h's "tuning score nan") (a_folded_frame_is_measured_apart_as_on_the_cpu red).
- (2026-11-01) evaluation._measure's `if not unfolded(skin, rig):` made `if False:`: the CPU counts the
  folded frame, no folded frame reported (the same check red).
- (2026-10-29) _measure.skin_shares' "hand" merge dropped: the hand's parts are scored apart, the keys differ.
- (2026-10-30) _per_row._per_row_fk's `set_places[row_set[problem_idx]]` made `set_places[0]`: every row takes
  the first set's joint places, sets 1 and 2 off theirs alone by 0.05-0.08 m (card_rows: per-row FK, rows = set x
  frame, equals Newton's eval_fk_batched on each set alone bit for bit).
- (2026-10-30) _per_row.rowwise's `optimizer._compute_motion_subspace = motion_subspace` dropped: the Jacobian
  reads the model's one joint places, sets 1 and 2's columns off theirs alone by 0.025-0.035 (card_rows).
- (2026-10-30) _forward._frames' `joint_x_p[row_set[row], joint]` made `joint_x_p[0, joint]`: the float64 frames take
  the first set's places, sets 1 and 2 off theirs alone by 0.056-0.074 m (card_rows).
- (2026-10-30) _solve.solved's `Forward(optimizer.model, set_places, row_set)` made `Forward(optimizer.model)`: the
  float64 residuals read the base set's places, sets 1 and 2 off theirs alone by 4-9 deg and 4-19 mm (card_across:
  the toy's clip as 3 sets solved in one batch equals each set alone to 0.01 deg and 0.01 mm).
- (2026-10-30) _sets.Sets.optimizer_for's offsets of each set made the base set's (`base.mjm.site_pos`): sets 1
  and 2 solve with set 0's marker offsets, off their Clip alone (card_sets; also: sets whose frozen dofs differ are
  refused, `_sets.refusal` returning None lets them through).
- (2026-10-30) _step._slopes' and _project._projected's `own = row_set[row]` made `own = 0`: every row reads set 0's
  coupling polys, so sets 1 and 2 (their follower's slope 0.05 and 0.10 on) end 4.2 and 10.2 deg off (card_sets).
- (2026-10-30) _measure.skin_depths' `depths[:, ...]` made `depths[:1, ...]`: the card's depths read the first skin
  frame only, off the CPU's.
- (2026-10-29) _projected's `power *= lead` made `power *= 1.0`: a coupling is no longer polyval of its leader.
- (2026-10-29) _turn_jacobian's `weight[problem] * rate[axis]` made `rate[axis]`: the weight is lost.
- (2026-10-29) _turn_residuals' angle `2 atan2(length, w) / length` made `2`: the turn is no longer the log map.
- (2026-10-29) _level2.dof_map's root `right_jacobian(rotation)` made `np.eye(3)`: the root's rotation-vector
  columns are off their differences.
- (2026-10-29) _level2.frame_rows' inside joint `body_turns[parent]` made `body_turns[fitting.inside[joint]]`: the
  knee's move columns are off.
- (2026-10-29, 2026-10-30, retired with _level2_card on 2026-10-31) _level2_card.CardKinematics' constants laid
  wrongly (the base inside joint places kept; one set's constants for every world).
- (2026-10-30) _level2.left_jacobian_inverses' `factor[:, None, None] * cross @ cross` dropped: every bent row off
  its own (card_level2).
- (2026-10-29) _constants.staged's `jacobian(problem, unknowns)` made `jacobian(problem, begin)` (the Jacobian
  frozen at the start): the toy's constants 0.027 mm off the 2-point solve's.
- (2026-10-30) _measure.posed_key without the worlds: seed 5's clip measured against seed 3's skin, its shares
  and depths off the CPU's (79.46 against 79.91 % outside), one posed clip, not 2.
- (2026-10-29) _solve's `CYCLE_STEP = 1e-5` made `0.0`: a seeded toy row cycles at its float32 pose's floor and never
  stops (NotConverged).
- (2026-10-30) _sampled.path_across' restart `index in firsts` made `index == 0`: a second recording chained on
  from the first's last frame, its start never asked (card_sets).
- (2026-10-30) _sampled.path_across reading set 0's points for every set: sets 1 and 2 18.96 and 76.37 deg off
  their clips chained alone (card_sets).
- (2026-10-30) _clip.moved_inside's map keyed by the joint's own name (`joint_label`, Newton's "a1_a2_a3") instead of
  its child body's: a body carrying several MuJoCo joints (one D6 in Newton) is not found, KeyError (card_inside).

Run: .venv/bin/python motor/fit/_card/tests/test_the_card_step_and_projection_equal_the_cpu_prototype.py
"""
import pathlib
import sys
import tempfile

INSIDE = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from card_chain import (LINKS, OBJECTIVE_ROWS, chain_model, log_rate, objective_rows,  # noqa: E402
                        residuals_and_jacobian)
from card_toy import CLIP_FRAMES, TOY, mink_frames, toy_card, toy_clip, toy_errors  # noqa: E402
from card_body import (a_folded_frame_is_measured_apart_as_on_the_cpu, the_measures_equal_the_cpu_measure,  # noqa: E402
                       the_skins_posed_once_follow_each_clip)
from card_level2 import (the_batched_inverses_equal_each_row, the_wide_level2_equals_the_hosts,  # noqa: E402
                         the_level2_jacobian_equals_its_differences,
                         the_level2_solve_with_the_exact_jacobian_equals_differences,
                         the_card_levenberg_marquardt_equals_scipys_level2)
from card_across import the_sets_solved_together_equal_each_set_alone  # noqa: E402
from card_skin import a_folded_frame_has_no_skin_rows_on_the_card, the_card_holds_the_bones_inside_as_mink  # noqa: E402,E501
from card_inside import (a_kept_build_moves_to_the_fittings_moves, the_body_with_several_joints_moves_as_one,  # noqa: E402
                         the_set_places_equal_a_build_per_set)
from card_sets import the_sampled_path_across_sets_equals_each_set_chained, the_sets_batch_equals_each_clip_alone  # noqa: E402
from card_rows import (the_float64_frames_equal_each_set_alone, the_rows_equal_each_set_alone,  # noqa: E402
                       the_rowwise_optimizer_equals_each_set_alone)
from card_step import (BOUNDED, COUPLINGS, FROZEN, ROWS, fixture, prototype_projected,  # noqa: E402
                       prototype_step)


def the_step_equals_the_prototype():
    import numpy as np
    import warp as wp
    from motor.fit._card import Reduction, damped_step
    jacobian, residuals, qpos, rest, _, lower, upper = fixture()
    wanted, wanted_norms, held_rows, released, free = prototype_step(jacobian, residuals, qpos, rest, lower, upper)
    coupled = COUPLINGS
    reduction = Reduction(free, free, lower[free], upper[free], [c[0] for c in coupled], [c[1] for c in coupled],
                          [c[1] for c in coupled], [rest[c[1]] if c[1] >= 0 else 0.0 for c in coupled],
                          np.array([c[2] for c in coupled]), "cuda:0")
    card = lambda values: wp.array(values.astype(np.float32), dtype=wp.float32, device="cuda:0")
    active = wp.array(np.ones(ROWS, np.int32), dtype=wp.int32, device="cuda:0")
    full, norms = damped_step(reduction, card(jacobian), card(residuals), card(qpos), active)
    problems = []
    if held_rows == 0:
        problems.append("no row holds a dof at its bound: the fixture does not test the hold")
    if released == 0:
        problems.append("no held dof is released: the fixture does not test the active set's exit")
    miss = np.abs(full.numpy() - wanted).max()
    if miss > 1e-5:
        problems.append(f"step differs from the prototype's by {miss:.2e} rad (most 1e-5)")
    if np.abs(norms.numpy() - wanted_norms).max() > 1e-5:
        problems.append("step lengths differ from the prototype's")
    return problems


def the_projection_equals_the_prototype():
    import numpy as np
    import warp as wp
    from motor.fit._card import Projection, project
    _, _, qpos, rest, seeds, lower, upper = fixture()
    stepped = qpos + np.random.default_rng(8).normal(scale=0.3, size=qpos.shape)
    stepped = stepped.astype(np.float32).astype(float)
    wanted = prototype_projected(stepped, rest, seeds, lower, upper)
    projection = Projection(lower, upper, [c[0] for c in COUPLINGS], [c[1] for c in COUPLINGS],
                            [rest[c[0]] for c in COUPLINGS], [rest[c[1]] if c[1] >= 0 else 0.0 for c in COUPLINGS],
                            np.array([c[2] for c in COUPLINGS]), list(FROZEN), "cuda:0")
    card = wp.array(stepped.astype(np.float32), dtype=wp.float32, device="cuda:0")
    project(projection, card, wp.array(seeds.astype(np.float32), dtype=wp.float32, device="cuda:0"))
    found = card.numpy().astype(float)
    problems = []
    if np.abs(found - wanted).max() > 1e-6:
        problems.append(f"projection differs from the prototype's by {np.abs(found - wanted).max():.2e} rad")
    for follower, leader, poly in COUPLINGS:
        lead = found[:, leader] - rest[leader] if leader >= 0 else 0.0
        miss = np.abs(found[:, follower] - rest[follower] - sum(poly[p] * lead ** p for p in range(5))).max()
        if miss > 1e-6:
            problems.append(f"follower {follower} off polyval of its leader by {miss:.2e} rad")
    leaders = [dof for dof in BOUNDED if dof not in {c[0] for c in COUPLINGS}]
    if (found[:, leaders] < lower[leaders]).any() or (found[:, leaders] > upper[leaders]).any():
        problems.append("a coordinate left its range")
    if not np.array_equal(found[:, list(FROZEN)], seeds[:, list(FROZEN)].astype(np.float32)):
        problems.append("a frozen coordinate moved off its seed")
    return problems


def the_objectives_equal_newtons_own_row_by_row():
    """Per-problem place and turn equal Newton's built-in objective made with that row's offset and weight, the turn's
    Jacobian Newton's spin rows through the log's inverse left Jacobian (mink's exact one)."""
    import numpy as np
    import warp as wp
    from newton import ik
    from motor.fit._card import PlacePerProblem, TurnPerProblem
    model = chain_model()
    poses, offsets, turns, weights, places, wanted = objective_rows(model.joint_coord_count)
    link = LINKS - 1
    card = lambda values, kind: wp.array(values, dtype=kind, device="cuda:0")
    mine = residuals_and_jacobian(model, poses, [
        PlacePerProblem(link, card(offsets, wp.vec3), card(places, wp.vec3), card(weights, wp.float32)),
        TurnPerProblem(link, card(turns, wp.quat), card(wanted, wp.vec4), card(weights, wp.float32))])
    problems = []
    for row in range(OBJECTIVE_ROWS):
        theirs = residuals_and_jacobian(model, poses, [
            ik.IKObjectivePosition(link, wp.vec3(*offsets[row]), card(places, wp.vec3), weight=float(weights[row])),
            ik.IKObjectiveRotation(link, wp.quat(*turns[row]), card(wanted, wp.vec4), weight=float(weights[row]))])
        exact = theirs[1][row].copy()
        exact[3:] = log_rate(theirs[0][row][3:] / weights[row]) @ exact[3:]
        for name, found, reference in (("residuals", mine[0][row], theirs[0][row]),
                                       ("Jacobian", mine[1][row], exact)):
            if np.abs(found - reference).max() > 1e-6:
                problems.append(f"row {row}: {name} differ from Newton's by {np.abs(found - reference).max():.2e}")
    if np.abs(mine[0][:, 3:]).max() < 1e-3:
        problems.append("the turn residuals are all near zero: the fixture does not test the turn")
    return problems


CHUNK = 3  # solved_in_chunks' rows a chunk on the toy: three full chunks and one of one


def the_clip_equals_mink_frame_by_frame():
    """The card loop on the toy's clip, each row seeded as mink's frame was, equals mink's angles to 0.01 deg and its
    sites to 0.01 mm, and solved in chunks of CHUNK rows it equals itself in one batch to 1e-6 rad."""
    import mujoco
    import numpy as np
    import warp as wp
    from motor.fit._card import solved, solved_in_chunks
    from motor.fit._card._seeds import newton_order
    mjm = mujoco.MjModel.from_xml_string(TOY)
    turns, places = toy_clip(mjm)
    reference = mink_frames(mjm, turns, places)
    seeds = np.vstack([mjm.qpos0[None], reference[:-1]])
    optimizer_for, reduction, projection = toy_card(mjm, turns, places)
    card = lambda values, kind: wp.array(np.ascontiguousarray(values, dtype=np.float32), dtype=kind, device="cuda:0")
    hinge = lambda name: int(mjm.jnt_qposadr[mjm.joint(name).id])
    newton_seeds = newton_order(seeds)
    found, _ = solved(optimizer_for(0, CLIP_FRAMES), reduction, projection, card(newton_seeds, wp.float32),
                      card(seeds, wp.float32))
    found = found.numpy().astype(float)
    chunked, _ = solved_in_chunks(optimizer_for, reduction, projection, newton_seeds, seeds, chunk_rows=CHUNK)
    problems = [] if np.abs(chunked - found).max() <= 1e-6 else ["in chunks of 3 rows the clip differs from one batch"]
    miss, sites = toy_errors(mjm, found, reference)
    if miss > 0.01:
        problems.append(f"angles {miss:.4f} deg off mink's (plan: 0.01)")
    if sites > 0.01:
        problems.append(f"sites {sites:.4f} mm off mink's (plan: 0.01)")
    if not (reference[:, hinge("h1")] >= 1.2 - 1e-6).any():
        problems.append("no frame holds h1 at its range: the clip does not test the limit")
    return problems


def the_next_iteration_from_its_seeds_equals_mink():
    """Level 1's next iteration (the places 3 mm on, the turn cost 1.5 times) seeded from the last iteration's
    angles of each frame (read back from its angles.npz, `_seeds.first`, then `_seeds.kept`) equals mink's
    frame-by-frame answer to it to 0.01 deg and 0.01 mm; a cold seed (the rest pose every frame) is only reported."""
    import mujoco
    import numpy as np
    from motor.fit._card import solved_in_chunks
    from motor.fit._card import _seeds
    mjm = mujoco.MjModel.from_xml_string(TOY)
    turns, places = toy_clip(mjm)
    folder = pathlib.Path(tempfile.mkdtemp()) / "seeds"
    (folder / "toy").mkdir(parents=True, exist_ok=True)
    np.savez(folder / "toy" / _seeds.SEEDS, qpos=mink_frames(mjm, turns, places))
    seeds = {"toy": _seeds.first(folder, "toy")}
    moved = places + np.array([0.003, -0.002, 0.001])
    reference = mink_frames(mjm, turns, moved)
    optimizer_for, reduction, projection = toy_card(mjm, turns, moved)
    found, _ = solved_in_chunks(optimizer_for, reduction, projection, _seeds.newton_order(seeds["toy"]),
                                seeds["toy"])
    seeds = _seeds.kept(seeds, "toy", found)
    miss, sites = toy_errors(mjm, seeds["toy"], reference)
    problems = [f"from its seeds: angles {miss:.4f} deg off mink's (plan: 0.01)"] if miss > 0.01 else []
    if sites > 0.01:
        problems.append(f"from its seeds: sites {sites:.4f} mm off mink's (plan: 0.01)")
    cold = np.tile(mjm.qpos0, (CLIP_FRAMES, 1))
    cold_found, _ = solved_in_chunks(optimizer_for, reduction, projection, _seeds.newton_order(cold), cold)
    print("cold seed (reported, not used): angles %.4f deg, sites %.4f mm off mink's" % toy_errors(
        mjm, cold_found, reference))
    short = seeds["toy"][:, :-1]
    try:
        solved_in_chunks(optimizer_for, reduction, projection, _seeds.newton_order(short), short)
        problems.append("seeds one coordinate short of the model were solved, not refused")
    except ValueError:
        pass
    return problems


def checks():
    """Every check, in order (their helpers are imported only in Newton's own Python)."""
    return (the_rows_equal_each_set_alone, the_rowwise_optimizer_equals_each_set_alone,
            the_float64_frames_equal_each_set_alone, the_sets_solved_together_equal_each_set_alone,
            the_sets_batch_equals_each_clip_alone, the_sampled_path_across_sets_equals_each_set_chained,
            the_wide_level2_equals_the_hosts, the_batched_inverses_equal_each_row,
            the_level2_jacobian_equals_its_differences, the_level2_solve_with_the_exact_jacobian_equals_differences,
            the_card_levenberg_marquardt_equals_scipys_level2,
            the_measures_equal_the_cpu_measure, the_skins_posed_once_follow_each_clip, the_step_equals_the_prototype, the_projection_equals_the_prototype,
            the_objectives_equal_newtons_own_row_by_row, the_clip_equals_mink_frame_by_frame,
            the_next_iteration_from_its_seeds_equals_mink, the_body_with_several_joints_moves_as_one,
            the_set_places_equal_a_build_per_set, a_kept_build_moves_to_the_fittings_moves,
            the_card_holds_the_bones_inside_as_mink, a_folded_frame_has_no_skin_rows_on_the_card,
            a_folded_frame_is_measured_apart_as_on_the_cpu)


def in_newton():
    """The checks, in Newton's own Python, on the card."""
    import warp as wp
    if not wp.is_cuda_available():
        print("skipped: no CUDA card")
        return 0
    failures = [f"{check.__name__}: {problem}" for check in checks() for problem in check()]
    for failure in failures:
        print(failure)
    print("ok" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(in_newton())
