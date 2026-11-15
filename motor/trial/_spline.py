"""The whole trial's path as one uniform cubic B-spline a coordinate (MOTOR step 1, #98), pure numpy and scipy.

N nodes (the clip's frames, `step` seconds apart) take N + 2 control points: node n is (c[n] + 4 c[n+1] + c[n+2]) / 6
(`NODE`), and the span from node s to s + 1 has the constant jerk (-c[s] + 3 c[s+1] - 3 c[s+2] + c[s+3]) / step^3
(`JERK`). The normal equations of the whole trial (each node's Gauss-Newton block spread over its three control
points, each span's jerk over its four) are block-banded, three control points either side, and solved banded."""
import numpy as np
from scipy.linalg import solveh_banded

NODE = np.array([1.0, 4.0, 1.0]) / 6.0
SPEED = np.array([-1.0, 0.0, 1.0]) / 2.0  # over the step
PACE = np.array([1.0, -2.0, 1.0])  # over the step squared
JERK = np.array([-1.0, 3.0, -3.0, 1.0])
REACH = 4  # control-point blocks a row's band holds: itself and three after
SLACK = 1e-9  # rad: a control point this near its bound stands on it
NULL = 1e-9  # the control-point fit's jerk weight against its nodes': the two end points' freedom only


def nodes(control):
    """The nodes' values [N, m] of control points [N + 2, m]."""
    return NODE[0] * control[:-2] + NODE[1] * control[1:-1] + NODE[2] * control[2:]


def speeds(control, step):
    """The nodes' first derivatives [N, m] (units a second)."""
    return (control[2:] - control[:-2]) / (2.0 * step)


def paces(control, step):
    """The nodes' second derivatives [N, m] (units a second squared)."""
    return (control[:-2] - 2.0 * control[1:-1] + control[2:]) / step ** 2


def spread(place, speed, pace, step):
    """A node's rows' Jacobians on its three control points [..., 3, rows, m] from their Jacobians on the node's
    value, speed and acceleration (each [..., rows, m])."""
    return np.stack([NODE[point] * place + SPEED[point] / step * speed + PACE[point] / step ** 2 * pace
                     for point in range(3)], axis=-3)


def jerks(control, step):
    """Each span's jerk [N - 1, m] (units a second cubed)."""
    summed = JERK[0] * control[:-3] + JERK[1] * control[1:-2] + JERK[2] * control[2:-1] + JERK[3] * control[3:]
    return summed / step ** 3


def jerk_cost(control, step, weights):
    """The jerk's cost: each span's step x sum of weights x jerk squared (the integral of the weighted square)."""
    return float(step * np.sum(weights * jerks(control, step) ** 2))


class Banded:
    """The whole trial's symmetric normal equations: `blocks[k, d]` the m x m block of control point k + d against k
    (d < REACH), `gradient` [N + 2, m]."""

    def __init__(self, points, size):
        self.blocks = np.zeros((points, REACH, size, size))
        self.gradient = np.zeros((points, size))

    def add_nodes(self, hessians, gradients):
        """Each node's (H [N, m, m], g [N, m]) spread over its three control points by NODE."""
        count = len(hessians)
        for first in range(3):
            self.gradient[first:first + count] += NODE[first] * gradients
            for second in range(first, 3):
                self.blocks[first:first + count, second - first] += NODE[first] * NODE[second] * hessians

    def add_rows(self, jacobians, residuals):
        """Each node's rows (residuals [N, rows], their Jacobians on its three control points [N, 3, rows, m], from
        `spread`): J_b^T J_a into control point b's block against a's, J_a^T r into a's gradient."""
        count = len(residuals)
        for first in range(3):
            self.gradient[first:first + count] += np.einsum("nrm,nr->nm", jacobians[:, first], residuals)
            for second in range(first, 3):
                self.blocks[first:first + count, second - first] += np.einsum(
                    "nrs,nrm->nsm", jacobians[:, second], jacobians[:, first])

    def add_dense(self, hessians, gradients):
        """Each node's own normal equations on its three control points (H [N, 3, 3, m, m], H[n, b, a] point b's rows
        against point a's; g [N, 3, m]): stage B's, its activations eliminated."""
        count = len(hessians)
        for first in range(3):
            self.gradient[first:first + count] += gradients[:, first]
            for second in range(first, 3):
                self.blocks[first:first + count, second - first] += hessians[:, second, first]

    def add_jerk(self, control, step, weights):
        """The jerk cost's (half) Hessian and gradient: each span's four control points, `weights` [m] a coordinate."""
        spans = len(control) - 3
        scale = JERK / step ** 3
        moved = jerks(control, step) * weights
        for first in range(4):
            self.gradient[first:first + spans] += step * scale[first] * moved
            for second in range(first, 4):
                diagonal = step * scale[first] * scale[second] * weights
                self.blocks[first:first + spans, second - first] += diagonal[None, :, None] * np.eye(len(weights))

    def step(self, damping, held):
        """The damped step -H^-1 g [N + 2, m], `damping` times each diagonal entry (plus `damping` itself) added and
        the `held` unknowns [N + 2, m] (bool) fixed at zero."""
        points, _, size, _ = self.blocks.shape
        total = points * size
        upper = REACH * size - 1
        band = np.zeros((upper + 1, total))
        free = ~held.reshape(-1)
        for offset in range(REACH):
            for first in range(points - offset):
                block = self.blocks[first, offset]  # rows: control point first + offset, columns: first
                rows = (first + offset) * size + np.arange(size)
                columns = first * size + np.arange(size)
                row_index, column_index = np.meshgrid(rows, columns, indexing="ij")
                keep = row_index >= column_index
                values = np.where(keep, block, 0.0) * free[row_index] * free[column_index]
                band[(row_index - column_index)[keep], column_index[keep]] += values[keep]
        diagonal = band[0].copy()
        band[0] = np.where(free, diagonal + damping * (diagonal + 1.0), 1.0)
        right = np.where(free, -self.gradient.reshape(-1), 0.0)
        return solveh_banded(band, right, lower=True).reshape(points, size)


def fitted(values, step):
    """Control points [N + 2, m] whose nodes are `values` [N, m] (the least squares fit, the two spare points held
    by a jerk weight NULL against the nodes')."""
    count, size = values.shape
    system = Banded(count + 2, size)
    system.add_nodes(np.broadcast_to(np.eye(size), (count, size, size)), np.zeros((count, size)))
    control = np.zeros((count + 2, size))
    system.add_jerk(control, step, np.full(size, NULL * step ** 5))
    system.gradient[:] = 0.0
    for first in range(3):
        system.gradient[first:first + count] -= NODE[first] * values
    return system.step(0.0, np.zeros((count + 2, size), dtype=bool))


def held(control, gradient, lower, upper):
    """The unknowns standing on a bound whose descent (-gradient) points out [N + 2, m]."""
    return ((control <= lower + SLACK) & (gradient > 0)) | ((control >= upper - SLACK) & (gradient < 0))
