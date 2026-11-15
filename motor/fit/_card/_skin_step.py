"""Level 3's bones-in-skin rows on the card (#70, 2026-11-01): mink's `fit._frames.SkinTask` per row of a batch, put
into the damped step's normal equations exactly as its rows would be, never stored as rows (8,144 a frame).

Every kept bone vertex gives three rows r = c o (c `fit._skin.skin_cost`, o its offset off its row's posed skin, zero
inside by the winding number, `fit._skin.apart_of`: the float32 mesh's query, the closest point on its face in
float64) whose Jacobian is c P A S for each dof's world motion S = (lin, ang) (Newton's joint_S_s, as
`_objective._place_jacobian` reads it), A = [I | -[p]x] (the point's velocity lin + ang x p), P the offset's Jacobian
in the point (the projector off the closest feature). So per body its points give M = c^2 sum A^T P A and
m = c^2 sum A^T o (`_bodies`, one thread a row and body, its points in order: no atomics, so a row's sums are the
same every run), and in the reduced dofs
(each column its free dof's motion plus its followers' by the coupling slopes, `_columns`, on the dofs that move the
body) J^T J = sum V^T M V and J^T r = sum V^T m (`_normal`): the same normal equations the point rows give, its
cost c^2 sum d^2 and the change of its rows (for `_solve`'s stop rule) summed per row in float64 (`_sums`)."""
import hashlib

import numpy as np
import warp as wp

from motor.fit._skin import apart_of, bone_bodies, rings, skin_cost

SKIN_CHUNK_ROWS = 128  # rows a chunk with skins: each distinct frame's skin is a mesh of ~3.6 MB on the card
MOMENTS = 42  # a body's M (6 x 6, row by row) then m (6)


@wp.kernel(enable_backward=False)
def _points(body_q: wp.array2d[wp.transformd], meshes: wp.array[wp.uint64], first_verts: wp.array[wp.int32],
            kept: wp.array[wp.int32],
            verts: wp.array[wp.vec3d], faces: wp.array[wp.int32], ring_starts: wp.array[wp.int32],
            ring_faces: wp.array[wp.int32], point_links: wp.array[wp.int32], places: wp.array[wp.vec3d], world: wp.array2d[wp.vec3d], apart: wp.array2d[wp.vec3d],
            bends: wp.array2d[wp.mat33d]):
    """One row's one bone point: its place, its offset off the row's skin and that offset's projector (the skin's
    query, a thread a point: a thread a body's points in turn took 875 s for a body's 31 sampled frames); zero on a
    row whose skin is not `kept` (`fit._skin.unfolded`)."""
    row, point = wp.tid()
    placed = wp.transform_point(body_q[row, point_links[point]], places[point])
    offset = wp.vec3d()
    bend = wp.mat33d()
    if kept[row] != 0:
        offset, bend = apart_of(meshes[row], verts, first_verts[row], faces, ring_starts, ring_faces, placed)
    world[row, point] = placed
    apart[row, point] = offset
    bends[row, point] = bend


@wp.kernel(enable_backward=False)
def _bodies(world: wp.array2d[wp.vec3d], apart: wp.array2d[wp.vec3d], bends: wp.array2d[wp.mat33d],
            starts: wp.array[wp.int32], cost: wp.float64, residual: wp.array2d[wp.float64],
            trial: wp.array2d[wp.float64], moments: wp.array3d[wp.float64], costs: wp.array2d[wp.float64],
            moved: wp.array2d[wp.float64]):
    """One row's one body: its points' M and m, their cost, their rows (`trial`) and their change since the last
    kept (`residual`), its points summed in order."""
    row, used = wp.tid()
    square = wp.spatial_matrixd()
    pull = wp.spatial_vectord()
    total = wp.float64(0.0)
    change = wp.float64(0.0)
    for point in range(starts[used], starts[used + 1]):
        placed = world[row, point]
        offset = apart[row, point]
        bend = bends[row, point]
        depth = wp.length(offset)
        if depth > wp.float64(0.0):
            across = wp.skew(placed)
            corner = bend * across
            far = across * bend * across
            turn = wp.cross(placed, offset)
            for first in range(3):
                for second in range(3):
                    square[first, second] += cost * cost * bend[first, second]
                    square[first, 3 + second] -= cost * cost * corner[first, second]
                    square[3 + first, second] -= cost * cost * corner[second, first]
                    square[3 + first, 3 + second] -= cost * cost * far[first, second]
                pull[first] += cost * cost * offset[first]
                pull[3 + first] += cost * cost * turn[first]
        value = cost * depth
        total += value * value
        change += (value - residual[row, point]) * (value - residual[row, point])
        trial[row, point] = value
    for first in range(6):
        moments[row, used, 36 + first] = pull[first]
        for second in range(6):
            moments[row, used, 6 * first + second] = square[first, second]
    costs[row, used] = total
    moved[row, used] = change


@wp.kernel(enable_backward=False)
def _sums(costs: wp.array2d[wp.float64], moved: wp.array2d[wp.float64], cost: wp.array[wp.float64],
          change: wp.array[wp.float64], squared: wp.array[wp.float32]):
    """Each row's skin cost and its rows' change, summed over its bodies in order."""
    row = wp.tid()
    total = wp.float64(0.0)
    shift = wp.float64(0.0)
    for used in range(costs.shape[1]):
        total += costs[row, used]
        shift += moved[row, used]
    cost[row] = total
    change[row] = shift
    squared[row] = wp.float32(total)


@wp.kernel(enable_backward=False)
def _kept(taken: wp.array[wp.int32], trial: wp.array2d[wp.float64], residual: wp.array2d[wp.float64],
          moments: wp.array3d[wp.float64], moments_kept: wp.array3d[wp.float64], cost: wp.array[wp.float64],
          cost_kept: wp.array[wp.float64], squared: wp.array[wp.float32]):
    """A row whose trial pose was taken keeps its rows, sums and cost; one turned back gets its kept ones again."""
    row = wp.tid()
    if taken[row] != 0:
        for point in range(trial.shape[1]):
            residual[row, point] = trial[row, point]
        for used in range(moments.shape[1]):
            for entry in range(MOMENTS):
                moments_kept[row, used, entry] = moments[row, used, entry]
        cost_kept[row] = cost[row]
    else:
        for used in range(moments.shape[1]):
            for entry in range(MOMENTS):
                moments[row, used, entry] = moments_kept[row, used, entry]
        cost[row] = cost_kept[row]
        squared[row] = wp.float32(cost_kept[row])


@wp.kernel(enable_backward=False)
def _columns(joint_s: wp.array2d[wp.spatial_vector], slopes: wp.array2d[wp.float32], starts: wp.array[wp.int32],
             dofs: wp.array[wp.int32], couplings: wp.array[wp.int32], columns: wp.array2d[wp.spatial_vector]):
    """Each (body, reduced column) pair's world motion: its free dof's and its followers' by their slopes, the dofs
    that move the body only."""
    row, pair = wp.tid()
    found = wp.spatial_vector()
    for entry in range(starts[pair], starts[pair + 1]):
        slope = float(1.0)
        if couplings[entry] >= 0:
            slope = slopes[row, couplings[entry]]
        found += slope * joint_s[row, dofs[entry]]
    columns[row, pair] = found


@wp.kernel(enable_backward=False)
def _normal(columns: wp.array2d[wp.spatial_vector], moments: wp.array3d[wp.float64], pair_of: wp.array2d[wp.int32],
            hessian: wp.array3d[wp.float32], gradient: wp.array2d[wp.float32]):
    """J^T J and J^T r of the skin's rows in the reduced dofs: sum over bodies of V_k^T M V_l and V_k . m."""
    row, first, second = wp.tid()
    product = float(0.0)
    pulled = float(0.0)
    for used in range(pair_of.shape[0]):
        left = pair_of[used, first]
        right = pair_of[used, second]
        if left >= 0 and right >= 0:
            for one in range(6):
                for other in range(6):
                    product += (columns[row, left][one] * float(moments[row, used, 6 * one + other])
                                * columns[row, right][other])
        if second == 0 and left >= 0:
            for one in range(6):
                pulled += columns[row, left][one] * float(moments[row, used, 36 + one])
    hessian[row, first, second] = product
    if second == 0:
        gradient[row, first] = pulled


def ancestors(model):
    """[bodies, dofs] whether each Newton dof moves each body (its joint lies on the body's way to the root)."""
    starts, children, parents = (model.joint_qd_start.numpy(), model.joint_child.numpy(), model.joint_parent.numpy())
    into = np.full(model.body_count, -1)
    into[children] = np.arange(len(children))
    found = np.zeros((model.body_count, model.joint_dof_count), dtype=bool)
    for body in range(model.body_count):
        joint = into[body]
        while joint >= 0:
            found[body, starts[joint]:starts[joint + 1]] = True
            joint = into[parents[joint]] if parents[joint] >= 0 else -1
    return found


class BoneSkin:
    """A `_clip.Clip`'s bone points for the card: sorted by their Newton link (`starts` per used link), their places in
    the link frames (MuJoCo's body frames), their cost, and each (used link, reduced column) pair's dofs."""

    def __init__(self, clip):
        bodies, places = bone_bodies(clip.mjm)
        links = np.array([clip.body[clip.mjm.body(int(body)).name] for body in bodies], dtype=np.int32)
        order = np.argsort(links, kind="stable")
        self.count, self.cost = len(links), skin_cost(len(links))
        used, counts = np.unique(links[order], return_counts=True)
        self.links = used.astype(np.int32)
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        self.places = places[order]
        reduction, moving = clip.reduction, ancestors(clip.model)
        free, followers = reduction.free_dof.numpy(), reduction.follower_dof.numpy()
        leaders = reduction.leader_reduced.numpy()
        pair_of, starts, dofs, couplings = np.full((len(used), len(free)), -1, dtype=np.int32), [0], [], []
        for index, link in enumerate(used):
            for column, dof in enumerate(free):
                entries = [(int(dof), -1)] + [(int(follower), coupling) for coupling, follower in enumerate(followers)
                                              if leaders[coupling] == column]
                entries = [(dof_, coupling) for dof_, coupling in entries if moving[link, dof_]]
                if entries:
                    pair_of[index, column] = len(starts) - 1
                    dofs += [dof_ for dof_, _ in entries]
                    couplings += [coupling for _, coupling in entries]
                    starts.append(len(dofs))
        self.pair_of, self.pair_starts = pair_of, np.array(starts, dtype=np.int32)
        self.pair_dofs, self.pair_couplings = np.array(dofs, dtype=np.int32), np.array(couplings, dtype=np.int32)
        self.size = len(free)

    def rows(self, skins, device):
        """`SkinRows` for a batch whose rows' posed skins are `skins` (`fit._skin.PosedSkins`, a frame a row)."""
        return SkinRows(self, skins, device)


class SkinRows:
    """One batch's skin on the card: each distinct row skin a Warp mesh (exact winding numbers; rows of several sets
    on one frame share it), the sums `measure` makes at a trial pose, `kept` by the step control, and the normal
    equations `normal` gives `_step.damped_step`."""

    def __init__(self, bones, skins, device):
        self.bones, self.device, rows = bones, device, len(skins)
        keys = [hashlib.sha1(np.ascontiguousarray(verts).tobytes()).digest() for verts in skins.verts]
        distinct = {key: index for index, key in reversed(list(enumerate(keys)))}
        faces = wp.array(skins.faces.ravel(), dtype=int, device=device)
        made = {key: wp.Mesh(points=wp.array(skins.verts[index], dtype=wp.vec3, device=device), indices=faces,
                             support_winding_number=True) for key, index in distinct.items()}
        self.meshes = list(made.values())
        self.mesh_of = wp.array(np.array([made[key].id for key in keys], dtype=np.uint64), dtype=wp.uint64,
                                device=device)
        order = {key: number for number, key in enumerate(distinct)}
        count = skins.verts.shape[1]
        self.first_verts = wp.array(np.array([order[key] * count for key in keys], dtype=np.int32), dtype=wp.int32,
                                    device=device)
        self.verts = wp.array(np.concatenate([skins.verts[index] for index in distinct.values()]).astype(np.float64),
                              dtype=wp.vec3d, device=device)
        self.faces = wp.array(skins.faces.ravel(), dtype=wp.int32, device=device)
        self.unfolded = wp.array(skins.kept.astype(np.int32), dtype=wp.int32, device=device)
        ring_starts, ring_faces = rings(skins.faces, count)
        self.ring_starts = wp.array(ring_starts, dtype=wp.int32, device=device)
        self.ring_faces = wp.array(ring_faces, dtype=wp.int32, device=device)
        array = lambda values, kind: wp.array(np.ascontiguousarray(values), dtype=kind, device=device)  # noqa: E731
        self.starts, self.links = array(bones.starts, wp.int32), array(bones.links, wp.int32)
        self.places = array(bones.places.astype(np.float64), wp.vec3d)
        self.point_links = array(np.repeat(bones.links, np.diff(bones.starts)), wp.int32)
        self.world = wp.zeros((rows, bones.count), dtype=wp.vec3d, device=device)
        self.apart = wp.zeros((rows, bones.count), dtype=wp.vec3d, device=device)
        self.bends = wp.zeros((rows, bones.count), dtype=wp.mat33d, device=device)
        self.pair_of, self.pair_starts = array(bones.pair_of, wp.int32), array(bones.pair_starts, wp.int32)
        self.pair_dofs, self.pair_couplings = array(bones.pair_dofs, wp.int32), array(bones.pair_couplings, wp.int32)
        used, pairs = len(bones.links), max(len(bones.pair_starts) - 1, 1)
        self.residual = wp.zeros((rows, bones.count), dtype=wp.float64, device=device)
        self.trial = wp.zeros((rows, bones.count), dtype=wp.float64, device=device)
        self.moments = wp.zeros((rows, used, MOMENTS), dtype=wp.float64, device=device)
        self.moments_kept = wp.zeros((rows, used, MOMENTS), dtype=wp.float64, device=device)
        self.cost_kept = wp.zeros(rows, dtype=wp.float64, device=device)
        self.costs = wp.zeros((rows, used), dtype=wp.float64, device=device)
        self.moved = wp.zeros((rows, used), dtype=wp.float64, device=device)
        self.cost = wp.zeros(rows, dtype=wp.float64, device=device)
        self.change = wp.zeros(rows, dtype=wp.float64, device=device)
        self.squared = wp.zeros(rows, dtype=wp.float32, device=device)
        self.columns = wp.zeros((rows, pairs), dtype=wp.spatial_vector, device=device)
        self.hessian = wp.zeros((rows, bones.size, bones.size), dtype=wp.float32, device=device)
        self.gradient = wp.zeros((rows, bones.size), dtype=wp.float32, device=device)

    def measure(self, body_q):
        """The bodies' sums at the rows' float64 link frames (`_forward.frames`' body_q), then each row's cost."""
        rows, used = self.costs.shape
        wp.launch(_points, dim=(rows, self.bones.count), device=self.device,
                  inputs=[body_q, self.mesh_of, self.first_verts, self.unfolded, self.verts, self.faces,
                          self.ring_starts, self.ring_faces, self.point_links,
                          self.places], outputs=[self.world, self.apart, self.bends])
        wp.launch(_bodies, dim=(rows, used), device=self.device,
                  inputs=[self.world, self.apart, self.bends, self.starts, wp.float64(self.bones.cost)],
                  outputs=[self.residual, self.trial, self.moments, self.costs, self.moved])
        wp.launch(_sums, dim=rows, inputs=[self.costs, self.moved], outputs=[self.cost, self.change, self.squared],
                  device=self.device)

    def kept(self, taken):
        """Each row's trial measure kept where `taken` (the step control's), else its last kept one back."""
        wp.launch(_kept, dim=self.cost.shape[0], device=self.device,
                  inputs=[taken, self.trial, self.residual, self.moments, self.moments_kept, self.cost, self.cost_kept,
                          self.squared])

    def normal(self, joint_s, slopes):
        """(J^T J [rows, size, size], J^T r [rows, size], the rows' squared sum [rows]) in the reduced dofs at the
        last `measure`, the dofs' world motions `joint_s` (Newton's joint_S_s) and the couplings' `slopes`."""
        rows = self.columns.shape[0]
        wp.launch(_columns, dim=(rows, len(self.bones.pair_starts) - 1), device=self.device,
                  inputs=[joint_s, slopes, self.pair_starts, self.pair_dofs, self.pair_couplings],
                  outputs=[self.columns])
        wp.launch(_normal, dim=(rows, self.bones.size, self.bones.size), device=self.device,
                  inputs=[self.columns, self.moments, self.pair_of], outputs=[self.hessian, self.gradient])
        return self.hessian, self.gradient, self.squared
