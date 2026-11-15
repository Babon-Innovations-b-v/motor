"""Level 3's loop on the card (plan step 1): mink's steps for every row at once until each row stops.

Per step: Newton's analytic Jacobian (IKOptimizerLM's own buffers, its own solve never launched), the residuals in
float64 from float64 link frames (`_forward`), the damped reduced step (`_step`), Newton's integrate, the exact
projection (`_project`). A row stops by a rule fixed against mink on the proto clip (mink's own 1e-8 on the step
never fires with a float32 Jacobian): its reduced step under STEP_DONE, or its float64 cost moved less than COST_DONE
of itself over two steps (1e-7 left the shoulder's flat elevation plane 0.012 deg off mink, 1e-9 0.0025 on the proto
clip but 0.047 on mean_squat's chain, where a row still creeps down elv_angle_r's valley; 1e-10 reaches mink there, 0.0048 deg, and 1e-11 runs the
proto clip's frame 41 0.0104 mm past mink's stop along the root's flat height), or it
cycles at its float32 pose's floor: its cost back within CYCLE_RETURN of the change between, two steps apart, by a
step under CYCLE_STEP (1e-4: mean_squat's row 148 cycles at mink's angle by 3e-5 rad steps, its float32
Jacobian's floor in that valley; a toy row seeded 0.5 mm off: the float64 step equals the card's, its model gain 3.7e-12 while
the float32 pose's rounding moves the cost 9e-9 either way, so it went A, B, A for good). A monotone fall never
returns, so the cycle rule cannot stop one early. It also stops, as mink's frames do (`../_frames.solved`), when a step
moved its float64 residuals (the weighted errors) by under MOVED_DONE: a drift no objective sees, elv_angle and its
follower cancelling at zero shoulder elevation (#70's arms_raise). A stopped row's step is zero from then on. A row still going at MOST_STEPS
raises NotConverged, as mink's frames do (`../_frames.py`). Given the rows' posed skins, the bones-in-skin rows
(`_skin_step`) are in each step's normal equations and in the stop rule's cost and change, as mink's `SkinTask` is,
and each step is judged (Levenberg-Marquardt's control, as `../_frames.solved` with the skin): a trial pose whose
cost (the float64 residuals' and the skin's) is not under the last kept pose's is turned back and the row's added
damping raised LAM_RAISE times (from LAM_FIRST), a kept one lowers it (to 0 under LAM_FIRST); a row turned back stops
when its step is under STEP_DONE or its damping passes LAM_MOST (no step lowers its cost: its minimum, though a kink),
the stop rule's cost and change read on kept poses only (#70, 2026-11-01: the skin's depth bends where its closest
point jumps, and the plain steps cycled for good on a sampled frame and ran away on another)."""
import numpy as np
import warp as wp

from ._forward import Forward, frames
from ._objective import wide_residuals
from ._per_row import rowwise
from ._project import project, rows_of
from ._skin_step import SKIN_CHUNK_ROWS
from ._step import damped_step

MOST_STEPS = 300
# A clip's rows (`solved_in_chunks`: level 3's chain and sweeps) are given this many: a row whose shoulder stands at
# zero elevation, on its range's end, has its plane of elevation seen only through the scapula's couplings, and under
# mink's damping at the clip's cost (LM_DAMPING times about 0.02, SOMA's fit) its steps crawl along it, the cost falling
# by 1e-11 a step, until the plane's own range or the stop rules end it: of seven video squats and jumps that
# failed at 300 (#98, 2026-11-07, elv_angle and its follower on an L4), four now stop by the rules above (their fits
# up to 457 s against 50); three still crawl past 3000 and fail
CLIP_STEPS = 3000
CHUNK_ROWS = 8000  # the plan's: about 2.9 GB on the card a chunk of MyoFullBody rows, under the 8 GB cap beside env1
STEP_DONE = 1e-6
COST_DONE = 1e-10
MOVED_DONE = 1e-8  # mink's own, `_frames.STEP_DONE` on the change of the weighted errors
CYCLE_STEP = 1e-4  # a row back at its cost of two steps before, by steps this short, is at its float32 pose's floor
CYCLE_RETURN = 1e-2  # ... back within this share of the cost change between the two
ROOT_TURN = 3  # the free root's quaternion starts at coordinate 3 in both layouts
LAM_FIRST = 1e-3  # the step control's first added damping, then LAM_RAISE times each turned-back step
LAM_RAISE = 10.0
LAM_MOST = 1e8  # a row turned back past this damping has no step that lowers its cost: it stops there


class NotConverged(RuntimeError):
    """A row whose fit did not stop in MOST_STEPS."""


@wp.kernel(enable_backward=False)
def _to_mujoco(joint_q: wp.array2d[wp.float32], qpos: wp.array2d[wp.float32]):
    """Newton's coordinates (root turn x, y, z, w) in MuJoCo's (w, x, y, z)."""
    row, coord = wp.tid()
    source = coord
    if coord == ROOT_TURN:
        source = ROOT_TURN + 3
    elif coord > ROOT_TURN and coord < ROOT_TURN + 4:
        source = coord - 1
    qpos[row, coord] = joint_q[row, source]


@wp.kernel(enable_backward=False)
def _to_newton(qpos: wp.array2d[wp.float32], joint_q: wp.array2d[wp.float32]):
    """MuJoCo's coordinates in Newton's."""
    row, coord = wp.tid()
    source = coord
    if coord == ROOT_TURN + 3:
        source = ROOT_TURN
    elif coord >= ROOT_TURN and coord < ROOT_TURN + 3:
        source = coord + 1
    joint_q[row, coord] = qpos[row, source]


@wp.kernel(enable_backward=False)
def _judged(residuals: wp.array2d[wp.float64], skin_cost: wp.array[wp.float64], kept_cost: wp.array[wp.float64],
            lam: wp.array[wp.float32], current: wp.array2d[wp.float32], kept_q: wp.array2d[wp.float32],
            kept_residuals: wp.array2d[wp.float64], judged: wp.array[wp.int32], active: wp.array[wp.int32],
            first: wp.float32, decaying: int):
    """The step control: a row's trial pose (`current`) is kept when its cost is under its last kept one's (which
    lowers its added damping, to 0 under `first`, or with `decaying` LAM_RAISE times and never put back to 0), else
    turned back to it with its residuals (which raises it, from `first`)."""
    row = wp.tid()
    if active[row] == 0:
        judged[row] = 1
        return
    cost = skin_cost[row]
    for residual in range(residuals.shape[1]):
        cost += residuals[row, residual] * residuals[row, residual]
    if cost < kept_cost[row]:
        kept_cost[row] = cost
        for coord in range(current.shape[1]):
            kept_q[row, coord] = current[row, coord]
        for residual in range(residuals.shape[1]):
            kept_residuals[row, residual] = residuals[row, residual]
        judged[row] = 1
        if decaying != 0:
            lam[row] = lam[row] / LAM_RAISE
        else:
            lam[row] = wp.where(lam[row] > first, lam[row] / LAM_RAISE, wp.float32(0.0))
    else:
        for coord in range(current.shape[1]):
            current[row, coord] = kept_q[row, coord]
        for residual in range(residuals.shape[1]):
            residuals[row, residual] = kept_residuals[row, residual]
        judged[row] = 0
        lam[row] = wp.max(lam[row] * LAM_RAISE, first)


@wp.kernel(enable_backward=False)
def _stopped(residuals: wp.array2d[wp.float64], before: wp.array2d[wp.float64], norms: wp.array[wp.float32],
             history: wp.array2d[wp.float64], taken: wp.array[wp.int32], active: wp.array[wp.int32],
             skin_cost: wp.array[wp.float64], skin_change: wp.array[wp.float64], skin: int, judging: int,
             judged: wp.array[wp.int32], lam: wp.array[wp.float32], steps: wp.array[wp.int32]):
    """The stop rule for one step (taken[0] steps before it): history holds each row's cost at the last three poses,
    before its residuals at the last pose; with `skin` the bones-in-skin rows' cost and change (`_skin_step`) too,
    read on kept poses only (`_judged`); with `judging` (the skin, or the caller's judged steps) a turned-back row
    stops on a short step or past LAM_MOST."""
    row = wp.tid()
    step = taken[0]
    if active[row] == 0:
        return
    if judging != 0 and judged[row] == 0:
        steps[row] = step + 1
        if norms[row] < STEP_DONE or lam[row] > LAM_MOST:
            active[row] = 0
        return
    cost = wp.float64(0.0)
    moved = wp.float64(0.0)
    for residual in range(residuals.shape[1]):
        cost += residuals[row, residual] * residuals[row, residual]
        change = residuals[row, residual] - before[row, residual]
        moved += change * change
        before[row, residual] = residuals[row, residual]
    if skin != 0:
        cost += skin_cost[row]
        moved += skin_change[row]
    history[row, 2] = history[row, 1]
    history[row, 1] = history[row, 0]
    history[row, 0] = cost
    steps[row] = step + 1
    settled = step >= 2 and wp.abs(history[row, 2] - cost) <= wp.float64(COST_DONE) * cost
    returned = wp.abs(history[row, 2] - cost) <= wp.float64(CYCLE_RETURN) * wp.abs(history[row, 1] - cost)
    cycling = step >= 2 and norms[row] < CYCLE_STEP and returned
    unseen = step >= 1 and wp.sqrt(moved) < wp.float64(MOVED_DONE)
    if norms[row] < STEP_DONE or settled or cycling or unseen:
        active[row] = 0


@wp.kernel(enable_backward=False)
def _counted(taken: wp.array[wp.int32]):
    """One step more (the step count lives on the card, so a captured step reads it at each replay)."""
    taken[0] = taken[0] + 1


@wp.kernel(enable_backward=False)
def _narrowed(wide: wp.array2d[wp.float64], narrow: wp.array2d[wp.float32]):
    row, residual = wp.tid()
    narrow[row, residual] = wp.float32(wide[row, residual])


def _wide_residuals(optimizer, forward, joint_q, body_q, wide):
    """Every objective's residuals in float64 from joint_q's float64 link frames."""
    frames(forward, joint_q, body_q)
    for objective, start in zip(optimizer.objectives, optimizer.residual_offsets, strict=True):
        wide_residuals(objective, body_q, start, optimizer.problem_idx, wide)


def solved(optimizer, reduction, projection, joint_q, seeds, most_steps=MOST_STEPS, set_places=None, row_set=None,
           skin=None, hold=None, bounds=None, marked=False, exact=None, judged_from=None):
    """Every row's pose fitted from joint_q [rows, coords] (Newton's layout): (qpos [rows, coords] in MuJoCo's
    layout, steps [rows]). optimizer is the batch's IKOptimizerLM (Newton's IKSolver(...)._impl), seeds [rows,
    coords] MuJoCo's layout (the frozen coordinates' values). set_places [sets, joints, 7] and row_set [rows] (host
    arrays) give each row its set's joint places, row_set its set's coupling polys too (the reduction's and the
    projection's [sets, couplings, 5]; rows = set x frame); neither, the model's own. `skin`: the rows' bones in their
    posed skins (`_skin_step.SkinRows`), else none. `hold`: rows held off the Jacobian, its
    `normal(slopes, reduced, qpos)` their (J^T J, J^T r, squared sum) in the reduced dofs (`rra`'s held leg
    directions) and, with `judged_from`, `cost(qpos)` their float64 cost in the judged steps' cost; `bounds`: each
    row's own ranges, (reduced lower, reduced upper [rows, size], lower, upper [rows, coords]) card arrays, in place
    of the reduction's and the projection's (`rra`'s footing). `marked`: a row still going at `most_steps` is given
    steps -1 instead of NotConverged raised, for the caller to solve it otherwise.
    `exact`: the residuals', the skin's and the Jacobian's link frames and twists from MuJoCo's own float64 geometry
    (`_forward.Exact`, its `drive`) instead of Newton's float32-built joint places (one set only). `judged_from`:
    each step judged as with the skin (Levenberg-Marquardt's control), its added damping from this and lowered on
    every kept step, never put back to 0 (`rra`'s footing, as mink's `../_frames.solved(judged_from=...)`); with the
    skin alone, from LAM_FIRST."""
    device = joint_q.device
    coords = optimizer.model.joint_coord_count
    if joint_q.shape[1] != coords or seeds.shape[1] != coords:
        # a seed cache made for another joint layout (2026-10-31: MyoFullBody made movable, 129 -> more coords)
        raise ValueError(f"seeds have {seeds.shape[1]} coordinates, the model {coords}: made for another model")
    sets_of = None if row_set is None else rows_of(row_set, joint_q.shape[0], device)
    for part in (reduction, projection):
        if row_set is not None and 1 < part.poly.shape[0] <= int(np.max(row_set)):
            raise ValueError(f"row_set reaches set {int(np.max(row_set))}, the coupling polys hold {part.poly.shape[0]}")
    if set_places is not None:
        rowwise(optimizer, wp.array(np.asarray(set_places, dtype=np.float32), dtype=wp.transform, device=device),
                wp.array(np.asarray(row_set, dtype=np.int32), dtype=wp.int32, device=device))
    rows, coords = joint_q.shape
    current = wp.clone(joint_q)
    following = wp.empty_like(current)
    velocity = wp.zeros((rows, optimizer.model.joint_dof_count), dtype=wp.float32, device=device)
    qpos = wp.empty_like(current)
    history = wp.zeros((rows, 3), dtype=wp.float64, device=device)
    if exact is not None and set_places is not None:
        raise ValueError("MuJoCo's frames hold one set: no set places")
    forward = exact if exact is not None else Forward(optimizer.model, set_places, row_set)
    if exact is not None:
        exact.drive(optimizer)
    body_q = wp.empty((rows, forward.bodies), dtype=wp.transformd, device=device)
    wide = wp.zeros((rows, optimizer.n_residuals), dtype=wp.float64, device=device)
    before = wp.zeros_like(wide)
    narrow = wp.empty((rows, optimizer.n_residuals), dtype=wp.float32, device=device)
    active = wp.ones(rows, dtype=wp.int32, device=device)
    steps = wp.zeros(rows, dtype=wp.int32, device=device)
    taken = wp.zeros(1, dtype=wp.int32, device=device)

    held_off = None if skin is None else (lambda slopes, *_: skin.normal(optimizer.joint_S_s, slopes))
    if hold is not None:
        held_off = hold.normal
    step_bounds, own_bounds = (None, None) if bounds is None else (bounds[:2], bounds[2:])
    unread = wp.zeros(1, dtype=wp.float64, device=device)  # the stop rule's skin sums when there is none (flag 0)
    skin_sums = (skin.cost, skin.change, 1) if skin is not None else (unread, unread, 0)
    judging = skin is not None or judged_from is not None
    first = wp.float32(LAM_FIRST if judged_from is None else judged_from)
    no_skin = None if skin is not None else wp.zeros(rows, dtype=wp.float64, device=device)
    judged, lam = wp.ones(rows, dtype=wp.int32, device=device), wp.zeros(rows, dtype=wp.float32, device=device)
    kept_q, kept_residuals = wp.clone(current), wp.zeros_like(wide)
    kept_cost = wp.array(np.full(rows, np.inf), dtype=wp.float64, device=device)

    def step():
        """One step for every row: the Jacobian, the float64 residuals, the damped step, integrate, project, stop;
        with the skin or `judged_from` the trial pose judged first (`_judged`), the step taken from the kept one."""
        if judging:
            _wide_residuals(optimizer, forward, current, body_q, wide)
            if skin is not None:
                skin.measure(body_q)
            elif hold is not None:
                wp.launch(_to_mujoco, dim=(rows, coords), inputs=[current], outputs=[qpos], device=device)
            extra = skin.cost if skin is not None else (hold.cost(qpos) if hold is not None else no_skin)
            wp.launch(_judged, dim=rows, inputs=[wide, extra, kept_cost, lam,
                                                 current, kept_q, kept_residuals, judged, active, first,
                                                 int(judged_from is not None)],
                      device=device)
            if skin is not None:
                skin.kept(judged)
        context = optimizer._ctx_solver(current)
        optimizer._residuals_analytic(context)
        jacobian = optimizer._jacobian_at(context)
        if not judging:
            _wide_residuals(optimizer, forward, current, body_q, wide)
        wp.launch(_narrowed, dim=wide.shape, inputs=[wide], outputs=[narrow], device=device)
        wp.launch(_to_mujoco, dim=(rows, coords), inputs=[current], outputs=[qpos], device=device)
        full, norms = damped_step(reduction, jacobian, narrow, qpos, active, sets_of, held_off,
                                  lam if judging else None, step_bounds)
        optimizer._integrate_dq(current, dq_in=full, joint_q_out=following, joint_qd_out=velocity)
        wp.launch(_to_mujoco, dim=(rows, coords), inputs=[following], outputs=[qpos], device=device)
        project(projection, qpos, seeds, sets_of, own_bounds)
        wp.launch(_to_newton, dim=(rows, coords), inputs=[qpos], outputs=[current], device=device)
        wp.launch(_stopped, dim=rows, inputs=[wide, before, norms, history, taken, active, *skin_sums, int(judging),
                                              judged, lam, steps], device=device)
        wp.launch(_counted, dim=1, inputs=[taken], device=device)

    graph = None
    for done in range(most_steps):
        if done == 0 or not device.is_cuda:
            step()  # the first step eagerly: every kernel loaded before the capture
        else:
            if graph is None:
                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    step()
                graph = capture.graph
            wp.capture_launch(graph)
        if done % 10 == 9 and not active.numpy().any():
            break
    going = active.numpy()
    if going.any() and not marked:
        raise NotConverged(f"{int(going.sum())} rows did not stop in {most_steps} steps")
    if going.any():
        steps.assign(np.where(going != 0, -1, steps.numpy()).astype(np.int32))
    wp.launch(_to_mujoco, dim=(rows, coords), inputs=[kept_q if judging else current], outputs=[qpos],
              device=device)
    return qpos, steps


def solved_in_chunks(optimizer_for, reduction, projection, joint_q, seeds, chunk_rows=CHUNK_ROWS, set_places=None,
                     row_set=None, skin=None, exact=None):
    """`solved` over joint_q's rows CHUNK_ROWS at a time, each row given CLIP_STEPS: (qpos [rows, coords] in MuJoCo's
    layout, steps [rows]) as host arrays. optimizer_for(start, stop) gives the IKOptimizerLM for rows start:stop
    (one IKSolver a chunk: its buffers are per row); joint_q and seeds are host arrays [rows, coords]; set_places and row_set as `solved`'s;
    `skin` (bones, skins): the clip's `_skin_step.BoneSkin` and each row's posed skin (`fit._skin.PosedSkins`), the
    chunks then at most SKIN_CHUNK_ROWS rows (a distinct skin a mesh on the card); `exact` as `solved`'s."""
    found, steps = [], []
    if skin is not None:
        chunk_rows = min(chunk_rows, SKIN_CHUNK_ROWS)
    for start in range(0, joint_q.shape[0], chunk_rows):
        stop = min(start + chunk_rows, joint_q.shape[0])
        optimizer = optimizer_for(start, stop)
        device = optimizer.model.device
        chunk = lambda values: wp.array(np.ascontiguousarray(values[start:stop], dtype=np.float32), dtype=wp.float32,
                                        device=device)
        rows = None if skin is None else skin[0].rows(skin[1][start:stop], device)
        qpos, made = solved(optimizer, reduction, projection, chunk(joint_q), chunk(seeds), CLIP_STEPS,
                            set_places=set_places,
                            row_set=None if row_set is None else np.asarray(row_set)[start:stop], skin=rows,
                            exact=exact)
        found.append(qpos.numpy())
        steps.append(made.numpy())
    return np.concatenate(found), np.concatenate(steps)
