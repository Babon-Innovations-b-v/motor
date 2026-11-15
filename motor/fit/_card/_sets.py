"""Level 3 across sets: one body's sets (each its own `_clip.Clip`, its constants moving the inside joints, its own
marker offsets and weights) solved as one card batch, rows = set x frame, each row reading its own set's joint
places (`_solve.solved_in_chunks` with set_places and row_set) and its own set's coupling polys. Refused when the
sets' frozen, limited or reduction settings differ, or their sites or bodies do: those are shared by every row."""
import copy

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from motor.fit._skin import PosedSkins

from ._clip import card
from ._objective import PlacePerProblem, TurnPerProblem
from ._seeds import newton_order
from ._solve import solved_in_chunks


def settings_of(part):
    """A `_step.Reduction`'s or `_project.Projection`'s fields as host values, warp arrays read back."""
    return {name: value.numpy() if isinstance(value, wp.array) else value for name, value in vars(part).items()
            if not callable(value)}


PER_SET = ("poly",)  # fields each row reads by its set, so sets may differ there


def differing(first, other):
    """The names of the fields where two `settings_of` dicts differ, the PER_SET fields left out."""
    return sorted(name for name in (first.keys() | other.keys()) - set(PER_SET)
                  if name not in first or name not in other or not np.array_equal(first[name], other[name]))


def refusal(clips):
    """Why `clips` cannot share one batch (text), or None: every set's reduction, projection, sites and bodies
    must be the base set's."""
    base = clips[0]
    for index, clip in enumerate(clips[1:], start=1):
        fields = differing(settings_of(base.reduction), settings_of(clip.reduction))
        fields += differing(settings_of(base.projection), settings_of(clip.projection))
        if fields:
            return f"set {index}: its frozen, limited or reduction settings differ from set 0's ({', '.join(fields)})"
        if (clip.frame_sites, clip.place_sites, clip.body) != (base.frame_sites, base.place_sites, base.body):
            return f"set {index}: its sites or bodies differ from set 0's"
    return None


def stacked(per_set, frames):
    """[sum frames, ...]: each set's row value `per_set[set]` repeated over its frames."""
    return np.concatenate([np.repeat(np.asarray(value)[None], count, axis=0) for value, count in zip(per_set, frames)])


def with_set_polys(base, parts):
    """A copy of `base` (a Reduction or Projection) whose poly is every part's stacked, [sets, couplings, 5]."""
    joined = copy.copy(base)
    joined.poly = wp.array(np.concatenate([part.poly.numpy() for part in parts]), dtype=wp.float32,
                           device=base.poly.device)
    return joined


class Sets:
    """Several sets of one body as one batch: `clips` one `_clip.Clip` per set, set 0 the base."""

    def __init__(self, clips):
        why = refusal(clips)
        if why:
            raise ValueError(why)
        self.clips, self.base = list(clips), clips[0]
        self.set_places = np.stack([clip.model.joint_X_p.numpy() for clip in self.clips]).astype(np.float32)
        self.reduction = with_set_polys(self.base.reduction, [clip.reduction for clip in self.clips])
        self.projection = with_set_polys(self.base.projection, [clip.projection for clip in self.clips])

    def optimizer_for(self, turns, points, frames):
        """optimizer_for(start, stop) on the base set's model, every row its own set's offsets and weights."""
        from newton import ik
        base = self.base
        quats = [stacked([clip.mjm.site_quat[site][[1, 2, 3, 0]] for clip in self.clips], frames)
                 for site in base.frame_sites]
        offsets = [stacked([clip.mjm.site_pos[site] for clip in self.clips], frames) for site in base.place_sites]
        turn_costs = stacked([clip.turn_cost for clip in self.clips], frames)
        place_costs = stacked([clip.place_costs for clip in self.clips], frames)

        def made(start, stop):
            rows = slice(start, stop)
            objectives = [TurnPerProblem(base.link(site), card(quats[group][rows], wp.quat),
                                         card(Rotation.from_matrix(turns[rows, group]).as_quat(), wp.vec4),
                                         card(turn_costs[rows], wp.float32))
                          for group, site in enumerate(base.frame_sites)]
            objectives += [PlacePerProblem(base.link(site), card(offsets[mark][rows], wp.vec3),
                                           card(points[rows, mark], wp.vec3), card(place_costs[rows, mark], wp.float32))
                           for mark, site in enumerate(base.place_sites)]
            return ik.IKSolver(base.model, stop - start, objectives, optimizer="lm",
                               jacobian_mode=ik.IKJacobianType.ANALYTIC)._impl
        return made

    def angles(self, turns, points, seeds, skins=None):
        """Each set's angles (its frames x nq, MuJoCo's layout) and steps from one batch: `turns`, `points`, `seeds`
        and `skins` (`fit._skin.PosedSkins`, or None: no skin rows) one per set, as `_clip.Clip.angles` takes them;
        the bone points are the base set's (the sets' bodies are the same)."""
        frames = [len(set_seeds) for set_seeds in seeds]
        joined = np.concatenate([np.asarray(value, dtype=float) for value in seeds])
        row_set = np.repeat(np.arange(len(self.clips)), frames)
        qpos, steps = solved_in_chunks(self.optimizer_for(np.concatenate(turns), np.concatenate(points), frames),
                                       self.reduction, self.projection, newton_order(joined), joined,
                                       set_places=self.set_places, row_set=row_set,
                                       skin=None if skins is None else self.base.skin_of(PosedSkins.joined(skins)))
        bounds = np.cumsum([0] + frames)
        return [(qpos[start:stop], steps[start:stop]) for start, stop in zip(bounds[:-1], bounds[1:])]
