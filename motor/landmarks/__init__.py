"""SCORE's landmark set (#70; CONTEXT.md "landmark set"; pure numpy): each landmark linked once to a point of the SOMA
mesh and to a point on MyoFullBody's bones (`LANDMARKS`), MyoFullBody's own joint centres (`CENTRES`: the group ends'
and the measure's), each landmark found on a person's mesh (`on_skin`) and skinned on the clip (`posed`), and SOMA's own
joints as group ends for the fit, the measure and the masses (`soma_point`, `soma_joints`, `myo_parts`), and each
segment's bind frame at its group's start for the skin the ground pushes (`at_starts`). The links and
trusts are level 1's (`tuning`, over many bodies at once).

How the set was found (landmark study, 2026-10-29, scripts kept with the study, not here): every MyoFullBody point by
its anatomical definition, an extreme-vertex rule on one bone mesh (ISB: Wu et al. 2002, 2005; Leardini 2007 for the
foot; Van Sint Jan 2007's atlas), no dataset's marker list read; each carried into SOMA's bind by its bone group's
aligned anatomical frames and length, then a ray along its outward direction met SOMA's skin (a tip: the nearest skin
vertex). A landmark's link is where the skin stood off the carried bone point on the template, turned into the body's
frame; its trust started at s0^2 / (s0^2 + slide^2 + placement^2), s0 5 mm (slide: its skin's move off a rigid point
when SOMA's skin plays the rig's clips), and both are level 1's to set (`tuning`). Measure-only (`solver`
False): the scapula's angles and trigonum (SOMA has no scapula: its back skin blends Chest and Shoulder), the greater
tubercle, the olecranon and the finger tips (SOMA has no finger joints, MyoFullBody's fingers are held)."""
import numpy as np

from motor.landmarks._table import CENTRES, LANDMARKS

# The share of a hand's own vertices furthest from its joint whose mean is the hand's tip (its middle finger's end).
TIP_SHARE = 0.02
# The landmarks whose markers hold their link (no move of their own, in the fit or in level 1): the plantar pads
# under the heel and the metatarsal heads (17 and 19 mm of SOMA's sole under the bone on the template; freed, a fit can
# drop the foot's bones through the sole); and the skull's, whose SOMA skin rides Head alone (weight 1.000 at every
# corner: no slide), so its link is the scalp's depth on the template (#70, 2026-11-04: freed, level 2 moved them 10 to
# 22 mm forward and down on env1 and the skull sat that far back and low inside SOMA's head, head outside 10.9 / 25.5 %
# standing / seated; held, 7.0 / 19.2 %). The face's nasion and menton too (2026-11-06): the skull is laid by all seven
# (`calibration.LAID_GROUPS`), their links SOMA's skin off the laid bone points on the template.
HELD = tuple(f"{name}_{side}" for name in ("heel_plantar", "MT_plantar") for side in "lr") + (
    "head_vertex", "forehead", "opisthocranion", "head_side_l", "head_side_r", "nasion", "menton")
# MyoFullBody centres that are the middle of two others (the measure's).
MIDDLE_CENTRES = {"hipmid": ("hip_l", "hip_r"), "mid_cervical": ("C7_T1", "C0_C1")}
# SOMA's group ends that are the middle of two rig joints.
SOMA_MIDDLES = {"hipmid": ("LeftLeg", "RightLeg")}


def solver_marks():
    """The landmarks the solver takes, in LANDMARKS' order ([name])."""
    return [name for name, row in LANDMARKS.items() if row[7]]


def trust(name):
    """A landmark's trust (0 to 1): its weight among the solver's markers."""
    return float(LANDMARKS[name][6])


def marker(name, scale):
    """A landmark's marker on its MyoFullBody body (body-local metres): its bone point scaled with the body, then its
    link (not scaled)."""
    row = LANDMARKS[name]
    return np.asarray(row[4], dtype=float) * scale + np.asarray(row[5], dtype=float)


def on_skin(points, faces, names=None):
    """Each landmark (of `names`, in their order; else every one) on a mesh of SOMA's topology ({name: point}, the
    mesh's frame)."""
    points, faces = np.asarray(points, dtype=float), np.asarray(faces)
    return {name: np.asarray(LANDMARKS[name][1]) @ points[faces[LANDMARKS[name][0]]]
            for name in (LANDMARKS if names is None else names)}


def posed(rig, segments, worlds, names=None):
    """Each of `names` (else the solver's landmarks) on the clip's own skin, every frame (frames x landmarks x 3, the
    person's frame): its face's corners skinned as SOMA skins them (linear blend: each corner the weighted sum of its
    joints' moves from the bind, `worlds` frames x segments x 4 x 4; the rig joints not in `segments` carry no
    weight), then its barycentric weights. `rig`: {"names", "bind", "faces", "weights", "points"}."""
    names_of = [str(joint) for joint in rig["names"]]
    rows = [LANDMARKS[name] for name in (solver_marks() if names is None else names)]
    corners = np.asarray(rig["faces"])[[row[0] for row in rows]]  # landmarks x 3
    joints = [names_of.index(segment) for segment in segments]
    weights = np.asarray(rig["weights"], dtype=float)[corners][:, :, joints]  # landmarks x 3 x segments
    bind = np.asarray(rig["bind"], dtype=float)[joints]
    points = np.asarray(rig["points"], dtype=float)[corners]
    moves = np.asarray(worlds, dtype=float) @ np.linalg.inv(bind)  # frames x segments x 4 x 4
    moved = np.einsum("fsij,lcj->flcsi", moves[..., :3, :3], points) + moves[:, None, None, :, :3, 3]
    skinned = np.einsum("lcs,flcsi->flci", weights, moved)
    return np.einsum("lc,flci->fli", np.array([row[1] for row in rows]), skinned)



def soma_joints(name):
    """The SOMA rig joints whose mean is a group end's place (a rig joint's own, a middle's two)."""
    return SOMA_MIDDLES.get(name, (name,))


def soma_point(name, rig):
    """A bone group's end at SOMA's own joints (own frame, its bind; the measure's and the masses'): a rig joint,
    `hipmid` the hip joints' middle, `<Side>Hand tip` the mean of the TIP_SHARE of the hand's own vertices (skin weight
    over a half) furthest from its joint. `rig`: {"names", "bind" (joints x 4 x 4), "points", "weights"}."""
    names = [str(joint) for joint in rig["names"]]
    bind = np.asarray(rig["bind"], dtype=float)
    if name.endswith(" tip"):
        hand = names.index(name[:-len(" tip")])
        own = np.asarray(rig["points"], dtype=float)[np.asarray(rig["weights"])[:, hand] > 0.5]
        away = np.linalg.norm(own - bind[hand, :3, 3], axis=1)
        return own[away >= np.quantile(away, 1.0 - TIP_SHARE)].mean(axis=0)
    return bind[[names.index(joint) for joint in soma_joints(name)], :3, 3].mean(axis=0)


def at_starts(rig, segments, starts):
    """Each segment's frame at SOMA's bind (segments x 4 x 4) standing at its bone group's SOMA start (`soma_point`):
    where `fit.riding` stands the segment on its bone, at that start's MyoFullBody centre, so a skin point carried by
    the segment keeps its place on it. The Hips' start is the hip joints' middle, 8 cm under SOMA's Hips joint: carried
    from the joint, a seated body's skin sat 4 to 5 cm low, into its seat (#70, 2026-11-04)."""
    names = [str(joint) for joint in rig["names"]]
    found = np.array(np.asarray(rig["bind"], dtype=float)[[names.index(segment) for segment in segments]])
    for frame, start in zip(found, starts):
        frame[:3, 3] = soma_point(start, rig)
    return found


def myo_parts(name):
    """A MyoFullBody joint centre as weighted body points ([(body, local point (source model), weight)])."""
    if name in MIDDLE_CENTRES:
        return [(*CENTRES[part], 0.5) for part in MIDDLE_CENTRES[name]]
    return [(*CENTRES[name], 1.0)]
