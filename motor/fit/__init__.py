"""MyoFullBody fitted to the clip's SOMA body (#70: SOMA is the truth), one solver: the person's constants (where each
landmark's marker sits on its bone and each joint inside a bone group sits in its parent; the lengths are SOMA's, from
the sizing) and every frame's joint angles, against one cost of the bone groups' turns (the main term: each SOMA segment's world turn on its group's
calibration site, OpenSense's orientation tracking with the static table, never a fitted offset) and the landmarks'
markers (SCORE's landmark set on the clip's own skin, each weighed by its trust, and SOMA's rig joint at each bone
group's start, `group_ends`), weighed as OpenSim's IK weighs its goals; SOMA's joints are also the measure
(`following`), within the joints' ranges and with the model's couplings held.

- `targets`: the clip's targets in MuJoCo's frame: each group's segment turn, each solver landmark skinned, SOMA's
  skin posed on every frame (`_skin.PosedSkins`: level 3 holds the bones inside it, #70 2026-11-01).
- `constants`: on SAMPLE frames spread over all the body's recordings (Cotton et al. 2026, arXiv:2608.29928, the
  idea only): the whole body placed and posed by mink at the sizing's lengths (`_frames`), then the poses with the
  markers' moves and the joints inside the groups, scipy's bounded `least_squares` over those constants and the
  frames' angles together, warm from mink's (`_constants`).
- `angles`: mink on every frame of the clip, on the person's file written with the found constants (`LEVEL3_CARD`:
  every frame at once on the card, `_card._level3`).
- `following` and `riding` (`_follow`): the judgement (joint mm, group degrees, at a range limit or not) and where each
  SOMA segment rides its group's frame body.
Nothing here moves, scales or grounds SOMA: the run writes no SOMA pose, and MyoFullBody's root follows the markers on
SOMA's skin."""
import mujoco
import numpy as np

from motor.bone_groups import groups
from motor.calibration import END_SHIFTS, turn
from motor.fit import _frames
from motor.fit._constants import Problem, staged
from motor.fit._follow import following, riding
from motor.fit._frames import LM_DAMPING, Frames, NotConverged, solved
from motor.fit._model import Fitting, end_site, filed, followers_free, shifted_end
from motor.fit._skin import FrameSkins, card_device, posed_skins
from motor.landmarks import HELD, LANDMARKS, MIDDLE_CENTRES, marker, myo_parts, posed, soma_joints, solver_marks, trust
from motor.landmarks._table import CENTRES
from motor.model import TO_OWN

SAMPLE = 30  # frames the constants are fitted on, spread over the clip (recipe 2026-10-28: the sparse problem did not
# finish in 3 minutes at 150 frames; 30 frames hold the clip's range)
HAND_TOP = "lunate"  # the hand group's frame body: the finger bones under it have no target and are held

END = "end:"  # a group end's marker's name prefix: MyoFullBody's joint centre there, SOMA's rig joint its target
# A group end's weight among the markers (#70, 2026-10-29: SOMA's rig joints mark where each bone group starts and
# ends): this times the landmarks' mean trust, the same for every end, inside the one markers' goal (`_frames.weights`),
# never on top of it; 1 (a landmark's) unless level 1 finds otherwise, never set by hand to pass a score
END_WEIGHT = 2.1169

METHOD = ("one solver (#70): mink 1.3.0 per frame (the bone groups' turns on the static calibration table, SCORE's "
          "landmark set on the SOMA skin as markers weighed by their trust and SOMA's rig joints at the group ends, "
          "as OpenSim's IK weighs its goals; couplings held, leading joints in range) and scipy's bounded "
          "least_squares over the person's constants (the markers' moves off their links and the joints inside the "
          "bone groups; the lengths SOMA's, from the sizing) with the angles of frames spread over all the body's "
          "recordings, once per body; then mink on every frame, the pose alone")
__all__ = ["Fitting", "FrameSkins",
           "end_site", "Frames", "NotConverged", "LM_DAMPING", "solved", "followers_free", "following", "riding",
           "followed", "group_ends", "targets", "fitting", "sampled", "constants", "constants_record", "angles"]


def followed(segments):
    """The bone groups whose turns are targets, in `groups()` order: every group SOMA has (OpenSense: every segment a
    sensor, weight 1), one segment to one group (owner, 2026-11-01: "the segments to bonegroups actually CAN be one to
    one"): a segment whose group's frame body an earlier segment's group already rides drives none (Neck2: MyoFullBody's
    neck is one rigid body, which Neck1 drives; before, both drove it and it was met as their compromise). One whose
    joints follow another's (the lumbar levels above L5) is met as near as the joints that lead it allow (`LACKS`)."""
    found, ridden = [], set()
    for group, (_, _, frame_body, _) in groups().items():
        if group in segments and frame_body not in ridden:
            found.append(group)
            ridden.add(frame_body)
    return found


def group_ends(segments):
    """The bone groups' ends that are a SOMA rig joint, in `groups()` order ([(SOMA end, MyoFullBody centre)]): each
    group's start (every joint between two groups starts one; a chain's tip is no joint) whose rig joints `segments`
    hold, one rule for every end: a middle of two (the hips', the neck's) is an end like any other (`soma_joints`,
    `myo_parts`). Every group's end sits on SOMA's rig joint the same way (#70: no joint special-cased). An end
    whose SOMA side is only other ends' own joints (the hips' middle: the two hip ends) is left out, so no rig joint
    is counted twice."""
    ends = [(soma[0], myo[0]) for soma, myo, _, _ in groups().values()
            if all(joint in segments for joint in soma_joints(soma[0]))
            and (myo[0] in CENTRES or myo[0] in MIDDLE_CENTRES)]
    own = {soma_end for soma_end, _ in ends}
    return [(soma_end, centre) for soma_end, centre in ends
            if soma_joints(soma_end) == (soma_end,) or not set(soma_joints(soma_end)) <= own]


def targets(rig, segments, worlds, held, skins=True):
    """The clip's targets in MuJoCo's frame: (each frame's turns of the `held` groups, frames x groups x 3 x 3; each
    frame's solver landmarks on the clip's skin, then SOMA's rig joint (a middle's two's mean) at each `group_ends`, frames x points x 3;
    SOMA's skin posed on each frame, `_skin.PosedSkins`, or None when not `skins`: posing it and its fold test are
    most of the targets' time, and level 2 and level 1's level 3 read none)."""
    worlds = np.asarray(worlds, dtype=float)
    turns = np.einsum("ij,fgjk->fgik", TO_OWN.T, worlds[:, [segments.index(group) for group in held], :3, :3])
    joints = np.stack([worlds[:, [segments.index(joint) for joint in soma_joints(soma_end)]][..., :3, 3].mean(axis=1)
                       for soma_end, _ in group_ends(segments)], axis=1)
    return (turns, np.concatenate([posed(rig, segments, worlds), joints], axis=1) @ TO_OWN,
            posed_skins(rig, segments, worlds) if skins else None)


def fitting(path, scales, held, segments, moves=None):
    """The fitting model of the person's file at `path` (`Fitting`): a calibration site per `held` group, a marker
    per solver landmark (`landmarks.marker` on its body sized by `scales`, moved by `moves` {landmark: 3}) weighed by
    its trust, then one at each of `segments`' `group_ends` (MyoFullBody's centre, sized, never moved; `end_site`) weighed
    END_WEIGHT times the landmarks' mean trust, the fingers held and each group's bodies (its
    joints inside, `Fitting.inside`), its markers' goal `_frames.MARKER_GOAL` as it stands now (each set's own)."""
    table, moves = groups(), moves or {}
    frames = [(group, table[group][2], turn(group)) for group in held]
    names = solver_marks()
    marks = [(name, filed(LANDMARKS[name][3]), marker(name, scales[LANDMARKS[name][3]]) + moves.get(name, 0.0))
             for name in names]
    model = mujoco.MjModel.from_xml_path(str(path))
    ends = [(f"{END}{centre}", *end_marker(model, centre, scales)) for _, centre in group_ends(segments)]
    trusts = [trust(name) for name in names]
    fingers = {body for _, _, frame, bodies in table.values() if frame.startswith(HAND_TOP)
               for body in bodies if body != frame}
    inside = [{filed(body) for body in bodies} for _, _, _, bodies in table.values()]
    return Fitting(path, frames, marks + ends, fingers, trusts + [END_WEIGHT * float(np.mean(trusts))] * len(ends),
                   inside, _frames.MARKER_GOAL)


def end_parts(centre, scales):
    """A group end's centre as weighted points on MyoFullBody's sized bodies ([(body as filed, local point, weight)])."""
    return [(filed(body), np.asarray(local, dtype=float) * scales[body], weight) for body, local, weight in myo_parts(centre)]


def end_marker(model, centre, scales):
    """A group end's marker site (body name, body-local point): `end_site` of its centre's sized parts, moved by
    `shifted_end` off it by the end's shift read in SOMA's skin (`calibration.END_SHIFTS`; none for most ends)."""
    holder, local = end_site(model, end_parts(centre, scales))
    return holder, shifted_end(model, holder, local, END_SHIFTS.get(centre, (0.0, 0.0, 0.0)))


def ends_relaid(fit, scales, segments):
    """`fit` (a `Fitting` with its constants laid in place) with its group ends' sites laid again on its model as it now
    stands, as `fitting` lays them on the person's file written with those constants (`end_site` reads the bodies' rest
    places, which the inside joints' moves change): the set's second file and compile are not needed (2026-10-31)."""
    for _, centre in group_ends(segments):
        site = fit.mark_sites[fit.marks.index(f"{END}{centre}")]
        holder, local = end_marker(fit.model, centre, scales)
        if fit.model.body(holder).id != fit.model.site_bodyid[site]:
            raise ValueError(f"{centre}: its end's holder moved from {fit.model.body(int(fit.model.site_bodyid[site])).name} to {holder}")
        fit.model.site_pos[site] = local
        fit.base[1][site] = local
    return fit


def spread(count, most=SAMPLE):
    """Up to `most` frame indices spread evenly over `count` frames, both ends in."""
    return np.unique(np.linspace(0, count - 1, min(count, most)).round().astype(int))


def sampled_targets(recordings):
    """The frames the constants are fitted on, from every recording of the body ([(turns, points, ...)], each as
    `targets` gives them; level 2 holds no skin rows, so a posed skin is not read): SAMPLE spread over them by length,
    at least two a recording. Returns (turns, points, lengths), the
    recordings' sampled frames end to end and how many each recording gave."""
    total = sum(len(turns) for turns, *_ in recordings)
    picked = [(turns, points, spread(len(turns), max(2, round(SAMPLE * len(turns) / total))))
              for turns, points, *_ in recordings]
    return (np.concatenate([turns[sample] for turns, _, sample in picked]),
            np.concatenate([points[sample] for _, points, sample in picked]),
            [len(sample) for _, _, sample in picked])


def on_card():
    """Whether Warp sees a CUDA card: level 2's sampled frames and solve run there when it does, else on the CPU's
    reference (mink per frame, scipy's least_squares on the host's exact Jacobian), as on a CPU-only machine (16 cores, no
    card)."""
    return card_device() != "cpu"


def sampled(fit, recordings):
    """`sampled_targets` with their angles, each recording's posed from its own start (`posed_paths`). Returns
    (turns, points, path)."""
    turns, points, lengths = sampled_targets(recordings)
    return turns, points, posed_paths([fit], [turns], [points], lengths)[0]


def posed_paths(fits, turns, points, lengths):
    """Every set's sampled frames' angles: on the card when there is one (`card_paths`: mink stops on its joint bounds
    where the card does not), else by mink (`mink_paths`, the reference the card was proved against)."""
    return (card_paths if on_card() else mink_paths)(fits, turns, points, lengths)


def mink_paths(fits, turns, points, lengths):
    """`card_paths` by mink on the CPU: each set's recordings (`lengths` frames each, end to end) fitted frame after
    frame, each from its own `Frames.start`."""
    ends = np.cumsum([0, *lengths])
    return [np.concatenate([Frames(fit).fitted(set_turns[first:last], set_points[first:last])
                            for first, last in zip(ends[:-1], ends[1:])])
            for fit, set_turns, set_points in zip(fits, turns, points)]


def card_path(fit, turns, points, lengths):
    """The sampled frames' angles on the card for one set (`card_paths` of that set alone)."""
    return card_paths([fit], [turns], [points], lengths)[0]


def card_paths(fits, turns, points, lengths):
    """Every set's sampled frames' angles on the card, one launch a frame for all sets (`_card._sampled.path_across`):
    `fits` one `Fitting` a set of one body, `turns`, `points` that set's sampled targets, each recording (`lengths`
    frames each, end to end, the same in every set) started from its set's own `Frames.start`."""
    from ._card import _clip, _sampled, _sets
    starts = [Frames(fit) for fit in fits]
    sets = _sets.Sets([_clip.for_fitting(fit, fit.path) for fit in fits])
    return _sampled.path_across(sets, turns, points, _sampled.recording_firsts(lengths),
                                lambda number, index: starts[number].start(turns[number][index],
                                                                           points[number][index]))


def constants(fit, recordings):
    """The person's constants on `fit` (a `Fitting`), once per body from all its `recordings` together ([(turns,
    points)]: CONTEXT.md "person constants"), on the frames `sampled` picks from mink's angles at the start
    constants (the lengths are SOMA's, from sizing: no size unknowns): the markers' moves (all but HELD's) and the
    joints inside the groups (#70's level 2: the vertebrae, shoulder blade, wrist bones and inner foot joints, each
    body's move in its parent, `Fitting.inside`), together:
    {"marks": {landmark: [3] m, its body's frame}, "joints": {body: [3] m, its parent's frame}, "evaluations": n,
    "followed": [group]}."""
    turns, points, path = sampled(fit, recordings)
    return constants_on(fit, turns, points, path)


def constants_across(fits, recordings):
    """`constants` of every set of one body (`fits` one `Fitting` a set, `recordings` that set's [(turns, points)],
    the same recordings' lengths in every set), every set's sampled frames posed in one card launch a frame
    (`posed_paths`: by mink with no card), then each set's constants solved on its own."""
    sampled_sets = [sampled_targets(given) for given in recordings]
    paths = posed_paths(fits, [turns for turns, _, _ in sampled_sets], [points for _, points, _ in sampled_sets],
                        sampled_sets[0][2])
    return [constants_on(fit, turns, points, path)
            for fit, (turns, points, _), path in zip(fits, sampled_sets, paths)]


def constants_on(fit, turns, points, path):
    """`constants`' solve on sampled targets already posed (`path` their angles at the start constants)."""
    free = [index for index, name in enumerate(fit.marks) if name not in HELD and not name.startswith(END)]
    card = on_card()  # no card: scipy's on the host's exact Jacobian, what the card's solve was proved equal to
    found, _, evaluations = staged(fit, path, turns, points, fit.neutral(), free, None if card else Problem.exact,
                                   wide=card)
    marks, joints = fit.unpack(found)
    return {"marks": {name: move.tolist() for name, move in zip(fit.marks, marks) if not name.startswith(END)},
            "joints": {name: move.tolist() for name, move in zip(fit.inside_names, joints)},
            "evaluations": evaluations,
            "followed": list(fit.groups)}


# level 3 on the card, every frame at once (`_card._level3`: mink's recurrence in sweeps), when Warp sees a card; off
# until it equals mink on the measured clips (0.01 deg, 0.01 mm every frame, the bones outside unchanged), mink the
# reference
LEVEL3_CARD = True  # measured equal to mink on every frame of jump and standing on an L4, fit.angles 30 to 15 s (#70, 2026-11-06)


def angles(fit, turns, points, skins=None):
    """Every frame's joint angles on `fit` (frames x nq of its model; MuJoCo's own compile): the pose alone, the
    person's constants laid on `fit` held fixed, the bones held inside `skins` (each frame's posed skin) when given.
    By mink frame after frame (`Frames.fitted`), or with LEVEL3_CARD on a card by `card_angles`."""
    if LEVEL3_CARD and on_card():
        return card_angles(fit, turns, points, skins)[0]
    return Frames(fit).fitted(turns, points, skins=skins)


def card_angles(fit, turns, points, skins=None):
    """`angles` on the card, every frame at once (`_card._level3.angles`, from mink's own start): (angles, sweeps)."""
    from ._card import _clip, _level3
    return _level3.angles(_clip.for_fitting(fit, fit.path), turns, points, skins,
                          Frames(fit).start(turns[0], points[0]))


def constants_record(found):
    """The constants as the model record keeps them: the markers' moves (mm: median and largest), each inside joint's
    move (mm), the solve's evaluations and the groups followed."""
    moves = [float(np.linalg.norm(move)) * 1000 for move in found["marks"].values()]
    joints = {name: round(float(np.linalg.norm(move)) * 1000, 1) for name, move in found.get("joints", {}).items()}
    return {"inside joints moved mm": joints,
            "markers moved mm": [round(float(np.median(moves)), 1), round(max(moves), 1)] if moves else [],
            "evaluations": found["evaluations"], "followed": found["followed"]}
