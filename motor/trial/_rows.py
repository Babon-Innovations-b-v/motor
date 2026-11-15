"""Every frame's Gauss-Newton block of the fit's own cost on the card (MOTOR step 1, #98): the same residuals level 3
minimises (`fit._card`: the groups' turns and the landmarks' markers on MuJoCo's float64 frames, Newton's analytic
Jacobian driven by them, the bones-in-skin rows of `_skin_step`), all frames as rows at once, given back undamped in
the reduced dofs: (J^T J, J^T r, the float64 cost) a frame. The step is not taken here: the whole trial's spline
takes it (`_spline`)."""
import numpy as np
import warp as wp

from motor.fit._card._project import poly_rows, project
from motor.fit._card._skin_step import SKIN_CHUNK_ROWS
from motor.fit._card._solve import _to_newton, _wide_residuals
from motor.fit._card._step import _reduced_jacobian, _slopes


@wp.kernel(enable_backward=False)
def _normal(reduced: wp.array3d[wp.float32], residuals: wp.array2d[wp.float64], hessian: wp.array3d[wp.float64],
            gradient: wp.array2d[wp.float64]):
    """J_red^T J_red and J_red^T r, undamped, a row's."""
    row, first, second = wp.tid()
    product = wp.float64(0.0)
    for residual in range(reduced.shape[1]):
        product += wp.float64(reduced[row, residual, first]) * wp.float64(reduced[row, residual, second])
    hessian[row, first, second] = product
    if second == 0:
        pulled = wp.float64(0.0)
        for residual in range(reduced.shape[1]):
            pulled += wp.float64(reduced[row, residual, first]) * residuals[row, residual]
        gradient[row, first] = pulled


@wp.kernel(enable_backward=False)
def _squared(residuals: wp.array2d[wp.float64], cost: wp.array[wp.float64]):
    """A row's squared residuals summed."""
    row = wp.tid()
    total = wp.float64(0.0)
    for residual in range(residuals.shape[1]):
        total += residuals[row, residual] * residuals[row, residual]
    cost[row] = total


class Chunk:
    """At most SKIN_CHUNK_ROWS frames' rows on the card: their optimizer (the targets), their skin and buffers."""

    def __init__(self, clip, turns, points, skins, start, stop):
        self.start, self.stop = start, stop
        rows = stop - start
        self.optimizer = clip.exact().drive(clip.optimizer_for(turns, points)(start, stop))
        self.device = self.optimizer.model.device
        skin = clip.skin_of(skins)
        self.skin = None if skin is None else skin[0].rows(skin[1][start:stop], self.device)
        coords = self.optimizer.model.joint_coord_count
        self.qpos = wp.zeros((rows, coords), dtype=wp.float32, device=self.device)
        self.joint_q = wp.zeros_like(self.qpos)
        self.body_q = wp.zeros((rows, clip.exact().bodies), dtype=wp.transformd, device=self.device)
        self.wide = wp.zeros((rows, self.optimizer.n_residuals), dtype=wp.float64, device=self.device)
        self.cost = wp.zeros(rows, dtype=wp.float64, device=self.device)


class Rows:
    """The clip's frames as card rows: `blocks(qpos)` gives every frame's (H [N, size, size], g [N, size], cost [N],
    the root dofs' world twists S [N, 6, 6] as columns (lin, ang)) at the poses qpos [N, nq] (MuJoCo's layout), after
    `project` (couplings, frozen coordinates from `seeds`, ranges)."""

    def __init__(self, clip, turns, points, skins, seeds):
        self.clip, self.seeds = clip, np.asarray(seeds, dtype=float)
        turns, points = np.asarray(turns), np.asarray(points)
        self.chunks = [Chunk(clip, turns, points, skins, start, min(start + SKIN_CHUNK_ROWS, len(turns)))
                       for start in range(0, len(turns), SKIN_CHUNK_ROWS)]
        self.size = clip.reduction.size

    def frames(self, qpos):
        """qpos as float, refused unless it holds a pose for every frame."""
        qpos = np.asarray(qpos, dtype=float)
        if len(qpos) != len(self.seeds):
            raise ValueError(f"{len(qpos)} poses for the clip's {len(self.seeds)} frames")
        return qpos

    def projected(self, qpos):
        """qpos [N, nq] put on the couplings, the seeds' frozen coordinates and the ranges, as level 3's steps are."""
        qpos, found = self.frames(qpos), []
        for chunk in self.chunks:
            chunk.qpos.assign(np.asarray(qpos[chunk.start:chunk.stop], dtype=np.float32))
            seeds = wp.array(self.seeds[chunk.start:chunk.stop].astype(np.float32), dtype=wp.float32,
                             device=chunk.device)
            project(self.clip.projection, chunk.qpos, seeds)
            found.append(chunk.qpos.numpy().astype(float))
        return np.concatenate(found)

    def blocks(self, qpos):
        """Every frame's (H, g, cost, S) at qpos [N, nq] (already projected)."""
        qpos, hessians, gradients, costs, twists = self.frames(qpos), [], [], [], []
        reduction = self.clip.reduction
        for chunk in self.chunks:
            rows = chunk.stop - chunk.start
            chunk.qpos.assign(np.asarray(qpos[chunk.start:chunk.stop], dtype=np.float32))
            wp.launch(_to_newton, dim=chunk.qpos.shape, inputs=[chunk.qpos], outputs=[chunk.joint_q],
                      device=chunk.device)
            optimizer = chunk.optimizer
            _wide_residuals(optimizer, self.clip.exact(), chunk.joint_q, chunk.body_q, chunk.wide)
            wp.launch(_squared, dim=rows, inputs=[chunk.wide], outputs=[chunk.cost], device=chunk.device)
            context = optimizer._ctx_solver(chunk.joint_q)
            optimizer._residuals_analytic(context)
            jacobian = optimizer._jacobian_at(context)
            slopes = wp.empty((rows, reduction.poly.shape[1]), dtype=wp.float32, device=chunk.device)
            wp.launch(_slopes, dim=slopes.shape,
                      inputs=[chunk.qpos, reduction.leader_coord, reduction.leader_rest, reduction.poly,
                              poly_rows(reduction.poly, None, rows, chunk.device)], outputs=[slopes],
                      device=chunk.device)
            reduced = wp.empty((rows, jacobian.shape[1], reduction.size), dtype=wp.float32, device=chunk.device)
            wp.launch(_reduced_jacobian, dim=reduced.shape,
                      inputs=[jacobian, reduction.free_dof, reduction.follower_dof, reduction.leader_reduced, slopes],
                      outputs=[reduced], device=chunk.device)
            hessian = wp.empty((rows, reduction.size, reduction.size), dtype=wp.float64, device=chunk.device)
            gradient = wp.empty((rows, reduction.size), dtype=wp.float64, device=chunk.device)
            wp.launch(_normal, dim=hessian.shape, inputs=[reduced, chunk.wide], outputs=[hessian, gradient],
                      device=chunk.device)
            found_h, found_g, found_cost = hessian.numpy(), gradient.numpy(), chunk.cost.numpy()
            if chunk.skin is not None:
                chunk.skin.measure(chunk.body_q)
                skin_h, skin_g, _ = chunk.skin.normal(optimizer.joint_S_s, slopes)
                found_h = found_h + skin_h.numpy()
                found_g = found_g + skin_g.numpy()
                found_cost = found_cost + chunk.skin.cost.numpy()
            hessians.append(found_h)
            gradients.append(found_g)
            costs.append(found_cost)
            twists.append(np.transpose(optimizer.joint_S_s.numpy()[:, :6].astype(float), (0, 2, 1)))
        return np.concatenate(hessians), np.concatenate(gradients), np.concatenate(costs), np.concatenate(twists)

