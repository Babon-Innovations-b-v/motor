"""Level 3's bones-in-skin rows on the host's MuJoCo kinematics (#70, 2026-11-01): each kept bone vertex
(`_skin.bone_bodies`), `_skin.skin_cost` times its offset off its frame's posed SOMA skin (zero inside by the winding
number, its closest point in float64: `_skin.apart_kernel`). The offset's Jacobian in the point is the projector P off
the closest feature (the face's normal, an edge's plane, a corner's every axis), through the point's own
P (jacp - [r]x jacr) at its body (mj_jacBody; r the point off the body's origin). Mink's `fit._frames.SkinTask`
(`objective`) steps on the offset's three rows a point, so it sees the skin's bend at its edges and corners. Level 2
holds no skin rows (2026-11-01: they bent the person's constants 4 to 5 cm and its polish did not settle; the fit
overlay). SOMA never moves with the pose."""
import mujoco
import numpy as np

from motor.fit._skin import bone_bodies, skin_cost

AXES = 3  # rows a point: its offset's three axes


def crossed(vectors):
    """The cross-product matrices of n vectors (n x 3 x 3)."""
    found = np.zeros((len(vectors), 3, 3))
    found[:, 0, 1], found[:, 0, 2], found[:, 1, 2] = -vectors[:, 2], vectors[:, 1], -vectors[:, 0]
    found[:, 1, 0], found[:, 2, 0], found[:, 2, 1] = vectors[:, 2], -vectors[:, 1], vectors[:, 0]
    return found


class BonePoints:
    """A fitting model's kept bone vertices (`bodies`, `places` in their bodies' frames) and their weight `cost`."""

    def __init__(self, model):
        self.bodies, self.places = bone_bodies(model)
        self.count = len(self.bodies)
        self.cost = skin_cost(self.count)

    def placed(self, data):
        """The points in the world (MuJoCo's frame) from a frame's kinematics."""
        turns = data.xmat[self.bodies].reshape(-1, 3, 3)
        return data.xpos[self.bodies] + np.einsum("pij,pj->pi", turns, self.places)

    def depths(self, apart):
        """Each point's cost times its depth outside, from its offset off its skin (`SkinMeshes.apart`)."""
        return self.cost * np.linalg.norm(apart, axis=1)

    def nv_rows(self, model, data, points, apart, bends):
        """(the outside points' indices, their three rows each over the model's dofs (k x 3 x nv)): cost times the
        offset's Jacobian in the point (`bends`, the projectors) times the point's own, jacp - [r]x jacr."""
        outside = np.flatnonzero(np.linalg.norm(apart, axis=1) > 0)
        rows = np.zeros((len(outside), AXES, model.nv))
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        bodies = self.bodies[outside]
        for body in np.unique(bodies):
            mine = np.flatnonzero(bodies == body)
            mujoco.mj_jacBody(model, data, jacp, jacr, int(body))
            moved = jacp[None] - crossed(points[outside[mine]] - data.xpos[body]) @ jacr[None]
            rows[mine] = bends[outside[mine]] @ moved
        return outside, self.cost * rows

    def objective(self, model, data, apart, bends):
        """(J^T J, J^T r) of the rows over the model's dofs at a frame's kinematics: the outside points' alone."""
        outside, rows = self.nv_rows(model, data, self.placed(data), apart, bends)
        flat = rows.reshape(-1, model.nv)
        return flat.T @ flat, flat.T @ (self.cost * apart[outside]).ravel()
