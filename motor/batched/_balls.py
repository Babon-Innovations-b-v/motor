"""`solvers.within_balls` with its big products on the card (#70, 2026-11-05): RRA's capped move, the least squares
over a product of balls by ADMM, the same rounds, penalty and stops in float64, its per-round products with the ball
rows (balls x 3 by every frame's move) and the inverse of its penalised curvature (`hessian + rho rows^T rows`, made on
the processor by its Cholesky factor, again whenever the penalty changes) launched on the card; the ball projection
and the stops read on the host each round, as `within_balls` does."""
import numpy as np
import warp as wp
from scipy.linalg import cho_factor, cho_solve

from motor.solvers import ADMM_RIDGE, ADMM_ROUNDS, ADMM_TOLERANCE, CAP_INSIDE

DEVICE = "cuda:0"


@wp.kernel(enable_backward=False)
def _times(matrix: wp.array2d[wp.float64], vector: wp.array[wp.float64], out: wp.array[wp.float64]):
    """out = matrix vector."""
    row = wp.tid()
    total = wp.float64(0.0)
    for column in range(vector.shape[0]):
        total += matrix[row, column] * vector[column]
    out[row] = total


@wp.kernel(enable_backward=False)
def _times_t(matrix: wp.array2d[wp.float64], vector: wp.array[wp.float64], out: wp.array[wp.float64]):
    """out = matrix^T vector."""
    column = wp.tid()
    total = wp.float64(0.0)
    for row in range(vector.shape[0]):
        total += matrix[row, column] * vector[row]
    out[column] = total


class Products:
    """`rows` on the card and its two products, `ahead` (rows y) and `back` (rows^T v), and the inverse's (`solve`)."""

    def __init__(self, rows):
        self.rows = wp.array(np.ascontiguousarray(rows, dtype=np.float64), dtype=wp.float64, device=DEVICE)
        self.inverse = None
        self.short = wp.zeros(rows.shape[0], dtype=wp.float64, device=DEVICE)
        self.long = wp.zeros(rows.shape[1], dtype=wp.float64, device=DEVICE)
        self.solved = wp.zeros(rows.shape[1], dtype=wp.float64, device=DEVICE)

    def invert(self, matrix):
        """The inverse of `matrix` (symmetric positive definite) by its Cholesky factor, put on the card."""
        found = cho_solve(cho_factor(matrix), np.eye(len(matrix)))
        self.inverse = wp.array(np.ascontiguousarray(found), dtype=wp.float64, device=DEVICE)

    def ahead(self, vector):
        """rows vector."""
        self.long.assign(vector)
        wp.launch(_times, dim=self.short.shape[0], inputs=[self.rows, self.long], outputs=[self.short], device=DEVICE)
        return self.short.numpy()

    def back(self, vector):
        """rows^T vector."""
        self.short.assign(vector)
        wp.launch(_times_t, dim=self.long.shape[0], inputs=[self.rows, self.short], outputs=[self.long],
                  device=DEVICE)
        return self.long.numpy()

    def solve(self, vector):
        """The inverse times vector."""
        self.long.assign(vector)
        wp.launch(_times, dim=self.solved.shape[0], inputs=[self.inverse, self.long], outputs=[self.solved],
                  device=DEVICE)
        return self.solved.numpy()


def within_balls_card(system, wanted, rows, values):
    """`solvers.within_balls` (the y least in |system y - wanted|^2 with every ball of `rows` y + `values` at most
    CAP_INSIDE, and how far its furthest ball is over one), its rounds' products on the card."""
    hessian, linear = system.T @ system, system.T @ wanted
    hessian += ADMM_RIDGE * np.trace(hessian) / len(hessian) * np.eye(len(hessian))  # the filter's null directions
    rho = float(np.trace(hessian) / max(np.trace(rows.T @ rows), 1e-12))
    gram = rows.T @ rows
    products = Products(rows)
    y = np.zeros(system.shape[1])
    z = (rows @ y + values).reshape(-1, 3)
    u = np.zeros_like(z)
    products.invert(hessian + rho * gram)
    shifted = products.back(values)
    for round_ in range(ADMM_ROUNDS):
        target = linear + rho * products.back((z - u).reshape(-1)) - rho * shifted
        y = products.solve(target)
        reached = (products.ahead(y) + values).reshape(-1, 3)
        before = z
        norms = np.linalg.norm(reached + u, axis=1, keepdims=True)
        z = (reached + u) * np.minimum(1.0, CAP_INSIDE / np.maximum(norms, 1e-12))
        u += reached - z
        primal = np.linalg.norm(reached - z)
        dual = rho * np.linalg.norm(products.back((z - before).reshape(-1)))
        if primal < ADMM_TOLERANCE * np.sqrt(len(z)) and dual < ADMM_TOLERANCE * np.sqrt(len(y)):
            break
        if round_ % 25 == 24 and (primal > 10 * dual or dual > 10 * primal):
            change = 2.0 if primal > dual else 0.5
            rho, u = rho * change, u / change
            products.invert(hessian + rho * gram)
    return y, float(np.linalg.norm((rows @ y + values).reshape(-1, 3), axis=1).max())
