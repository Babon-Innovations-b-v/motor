"""The exact projection after each step (plan step 1): ranges clipped, every coupling's follower set to
polyval of its (clipped) leader, the frozen coordinates set back to the row's seed; MuJoCo's qpos layout."""
import functools

import numpy as np
import warp as wp

COUPLING_TERMS = 5


@wp.kernel(enable_backward=False)
def _projected(qpos: wp.array2d[wp.float32], lower: wp.array[wp.float32], upper: wp.array[wp.float32],
               row_lower: wp.array2d[wp.float32], row_upper: wp.array2d[wp.float32], per_row: int,
               follower_coord: wp.array[wp.int32], leader_coord: wp.array[wp.int32],
               follower_rest: wp.array[wp.float32], leader_rest: wp.array[wp.float32], poly: wp.array3d[wp.float32],
               frozen_coord: wp.array[wp.int32], seeds: wp.array2d[wp.float32], row_set: wp.array[wp.int32]):
    """One row projected in place, in the CPU prototype's order: ranges (with `per_row` the row's own, `row_lower`
    and `row_upper` [rows, coords]), couplings (its set's polys), frozen."""
    row = wp.tid()
    own = row_set[row]
    for coord in range(qpos.shape[1]):
        if per_row != 0:
            qpos[row, coord] = wp.clamp(qpos[row, coord], row_lower[row, coord], row_upper[row, coord])
        else:
            qpos[row, coord] = wp.clamp(qpos[row, coord], lower[coord], upper[coord])
    for coupling in range(follower_coord.shape[0]):
        lead = float(0.0)
        if leader_coord[coupling] >= 0:
            lead = qpos[row, leader_coord[coupling]] - leader_rest[coupling]
        value = float(0.0)
        power = float(1.0)
        for term in range(COUPLING_TERMS):
            value += poly[own, coupling, term] * power
            power *= lead
        qpos[row, follower_coord[coupling]] = follower_rest[coupling] + value
    for frozen in range(frozen_coord.shape[0]):
        qpos[row, frozen_coord[frozen]] = seeds[row, frozen_coord[frozen]]


class Projection:
    """Ranges (an unlimited coordinate +-UNBOUNDED), couplings and frozen coordinates on the card; `poly` [couplings,
    5] for one set or [sets, couplings, 5], kept as the latter."""

    UNBOUNDED = 1e30

    def __init__(self, lower, upper, follower_coord, leader_coord, follower_rest, leader_rest, poly, frozen_coord,
                 device):
        self.lower = wp.array(lower, dtype=wp.float32, device=device)
        self.upper = wp.array(upper, dtype=wp.float32, device=device)
        self.follower_coord = wp.array(follower_coord, dtype=wp.int32, device=device)
        self.leader_coord = wp.array(leader_coord, dtype=wp.int32, device=device)
        self.follower_rest = wp.array(follower_rest, dtype=wp.float32, device=device)
        self.leader_rest = wp.array(leader_rest, dtype=wp.float32, device=device)
        self.poly = per_set(poly, device)
        self.frozen_coord = wp.array(frozen_coord, dtype=wp.int32, device=device)


def per_set(poly, device):
    """Coupling polynomials as a [sets, couplings, COUPLING_TERMS] float32 card array (one set's [couplings, 5] as
    [1, couplings, 5])."""
    values = np.asarray(poly, dtype=np.float32)
    if values.ndim < 3:
        values = values.reshape(-1, COUPLING_TERMS)[None]
    return wp.array(np.ascontiguousarray(values), dtype=wp.float32, device=device)


def rows_of(row_set, rows, device):
    """Each row's set as an int32 card array: `row_set` (host or card), all set 0 when None."""
    if row_set is None:
        return wp.zeros(rows, dtype=wp.int32, device=device)
    if isinstance(row_set, wp.array):
        return row_set
    return wp.array(np.asarray(row_set, dtype=np.int32), dtype=wp.int32, device=device)


def poly_rows(poly, row_set, rows, device):
    """Each row's index into `poly` [sets, couplings, 5]: all 0 when it holds one set (every set's polys equal),
    else `row_set`'s (`_solve.solved` refuses a row whose set has none, once, not every step)."""
    if poly.shape[0] == 1 or row_set is None:
        return rows_of(None, rows, device)
    return rows_of(row_set, rows, device)


def project(projection, qpos, seeds, row_set=None, bounds=None):
    """qpos [rows, coords] projected in place onto the ranges (each row's own, `bounds` (lower, upper) [rows, coords],
    when given), couplings (each row its set's, `row_set`, set 0 when None) and the seeds' frozen coordinates."""
    own = (*bounds, 1) if bounds is not None else (*_no_bounds(str(qpos.device)), 0)
    wp.launch(_projected, dim=qpos.shape[0],
              inputs=[qpos, projection.lower, projection.upper, *own, projection.follower_coord,
                      projection.leader_coord,
                      projection.follower_rest, projection.leader_rest, projection.poly, projection.frozen_coord,
                      seeds, poly_rows(projection.poly, row_set, qpos.shape[0], qpos.device)], device=qpos.device)


@functools.cache
def _no_bounds(device):
    """`_projected`'s per-row bounds when there are none (never read: their flag is 0), one pair a device."""
    return wp.zeros((1, 1), dtype=wp.float32, device=device), wp.zeros((1, 1), dtype=wp.float32, device=device)
