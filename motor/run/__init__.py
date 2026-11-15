"""The MOTOR run, in a Python that holds Newton with its MuJoCo solver (the pins in pyproject.toml): every
person's sim body of one job, each MyoFullBody sized to the person's SOMA body (`sizing`, `myo`), on its own
copy of the job's ground.

    python -m motor.run <job.json> <out.json>

job.json: {"rate", "gravity", optional "model" (the folder holding musclemimic_models/; default the vendored
MyoFullBody, vendor/myofullbody), "residual"
({"force", "moment"}: the limits' shares of the weight, and of it times the centre's height), "adjust" ({"move most
m", "turn most deg", "follows deg"}: RRA's caps; optional "feet kept", default true: false skips `Footing`, RRA's move
of the pelvis and trunk taken as it is, the legs as the fit stands them: an experiment, #98), "people":
[{"name", "kg", "height" (the person's own standing height, m), "rig" ({"names", "parents", "bind"}: the person's rig),
"segments" ([SOMA segment name], in the order of `targets`), "targets" (an .npz whose `world` is frames x segments x
4 x 4: each segment's frame in the clip, `ground` n x 3 x 3: the scene's triangles it may touch, both in the
person's own frame, Y up, and `points`, `faces`, `weights`: the person's SOMA mesh at its bind and its skin weights,
which `sizing` sizes MyoFullBody to and the fit's markers are skinned on), optional "recordings" ([.npz path, each with
its `world`]: every recording of the person's SOMA body, its person constants solved from all of them together; else
the clip's alone)}]}.

The movement is the clip's (the owner, 2026-10-27: "we have kimodo for the actual movement; and myobody plus mujoco and
newton for the physics"), and SOMA is the truth (#70): MyoFullBody is fitted to it, never the reverse, and nothing
here moves, scales or grounds SOMA.
- One solver (`fit`, #70): the person's constants (each landmark's marker's place on its
  bone and each joint inside a bone group; the lengths are SOMA's, from the sizing) and every frame's joint angles, against the bone groups' turns (the main term: each SOMA
  segment's world turn on its group's site, turned by the static calibration table; OpenSense's orientation tracking,
  Al Borno et al. 2022, doi:10.1186/s12984-022-01001-x) and MOTOR's landmark set skinned on the clip's own skin as
  markers, weighed by their trust as OpenSim's IK weighs its two goals: mink per frame (couplings held exactly, the
  leading joints in their ranges) and scipy's bounded least_squares over the constants, together with the angles of frames spread over all the body's recordings (the person constants,
  once per SOMA body); then the person's file is written again with the found constants and every frame's pose
  fitted on it, the constants held. MyoFullBody's root follows the markers on SOMA's skin; nothing is grounded. How closely the path follows SOMA
  (each joint's mm, each group's degrees, split at a range limit or not) is the model record's `fit`.
- The fitted path carried onto Newton's model by joint name and low-passed at CUTOFF (Winter's 6 Hz, zero lag)
  before it is differentiated; velocities and accelerations its central differences.
- OpenSim's residual reduction (RRA: Delp et al. 2007, doi:10.1109/TBME.2007.901024; opensim-core's RRATool,
  Apache-2.0), in our own stack (`reduced`): RRA_ROUNDS times, the body's mass changed by the mean upward residual and
  the torso's centre of mass moved under the mean residual moment (RRATool::adjustCOMToReduceResiduals), then the
  pelvis' and trunk's path moved so the ground's pushes explain the movement, each frame's move capped, the feet kept
  where they stood by the legs (`Footing`: mink, as the fit); the result kept within the clip's `follows` limit
  (`within_follows`). Nothing is applied to the body; how far it moved is the body's `adjusted` record.
- The physics of every frame, on the MuJoCo model Newton builds (with a CUDA card every frame's state and least
  squares at once there, MuJoCo Warp on the model Newton's SolverMuJoCo puts on it: `batched`; else MuJoCo's CPU
  backend, frame by frame; nothing is stepped): the generalised forces the movement needs (inverse dynamics, armature and passive forces in), the ground's
  contacts (each point of the person's skin, riding the bones, within GAP of the drawn triangle under it or into it, along the triangle's
  normal: a friction pyramid, FRICTION, under SLIDE_MOST, else sliding, its friction against its slip), and the static
  optimisation (`shared`) in OpenSim's two stages: the ground reaction force first, from the pelvis' six balance rows
  alone, then the forces to a joint at an end of its range (`stops`) and to the muscles (activations 0 to 1 on each
  muscle's force-length-velocity, its passive force in) with it fixed, least squared activation; what they cannot give is left
  over: the residual on the pelvis (force and moment, OpenSim's residual actuators) and each joint's reserve. Neither
  is applied to anything: they are measures of how physically possible the movement is.

This file is the run only: each step is a module beside it (#70), imported from this folder as Python puts a script's
folder first: `bone_groups` (the bone groups: which body each segment rides), `ground`, `solvers`, `landmarks`,
`calibration`, `volumes`, `model` (the model Newton builds, its couplings), `sizing` and `myo` (the person's file),
`contact` (the feet on the ground), `fit` (the one solver), `muscles` (the static optimisation), `rra` (the residual
reduction) and `records` (what is written).

out.json: {"engine", "card" (the CUDA card's name, null on the CPU), "rate", "people": {name: {"model" (the person's
model record), "muscles" ([name]), "joints" ([name]
of each driven dof), "com height" (m, frame 0, over its ground), "frames":
[{"angles": [[x, y, z] degrees, rotation vector, per SOMA segment but the first, against its parent, as the body's
segment riding its bone turns], "targets": [...the clip's...], "torques": [N m per joint, its
followers' with it], "own torques": [N m per joint, across that joint alone], "ground": [x, y, z] N
(the ground's push), "lowest": m (the lowest skin
point's height over the ground, null off it), "touches": {part: [the ground's push on it, xyz N, and its centre of
pressure on the ground, xyz m]} (every part the ground pushes: each foot, its calcaneus and toes as one, by its Foot
segment's name, a seat under a thigh, a bed under the back, by its body's; low-passed over each span it touches, as the
path is: `records.steadied`),
"feet": {segment: [its lowest skin point's height over the ground
(null off it), the ground's upward push on it N]} (FEET), "residual": [force xyz N, moment xyz N m], "reserve": N m
(the largest joint reserve), "activations": [0..1 per muscle], "forces": [N per muscle, tension]}],
"adjusted": RRA's moves (`adjustment_record`), "pushes": the pushes' measures against the physics
(`records.push_measures`), "seconds": {step: wall seconds} (`run_person`: the fit, the set up, the footing, RRA's two
parts without it, the checks, the frames' states, the static optimisation's two stages, the records, the arrays and
their "total")}}}, everything in the person's own frame (Y
up); and beside it <out>.npz, per person: `<name>/bones` (each body's pose every frame, frames x bodies x 7),
`<name>/coordinates` and `<name>/coordinate names` (every hinge's angle on the path the physics takes, radians,
frames x hinges, and the hinges' joint names: tools/labval reads them),
`<name>/segments` (each SOMA segment riding its bone every frame, frames x segments x 7: turned by the calibration
table, at its start joint's centre, `fit.riding`), `<name>/paths` (every muscle's path points, all frames end to end),
`<name>/counts` (frames x muscles: each
path's point count) and `<name>/mesh/<body>/points|faces` (each bone in its body's frame)."""
import json
import pathlib
import sys
import time


import mujoco  # noqa: E402
import newton  # noqa: E402
import numpy as np  # noqa: E402
import warp  # noqa: E402

from motor import fit  # noqa: E402
from motor import trial  # noqa: E402
from motor import myo  # noqa: E402
from motor.bone_groups import FEET, groups  # noqa: E402
from motor.contact import Skin, lowests  # noqa: E402
from motor.landmarks import at_starts  # noqa: E402
from motor.model import TO_OWN, body_index, built, carried, filtered, posed_path, rates  # noqa: E402
from motor.muscles import Physics, parts_of  # noqa: E402
from motor.records import adjustment_record, arrays, bones_of, com_height, entry, frames  # noqa: E402
from motor.rra import Footing, off_clip, reduced, within_follows, within_frames  # noqa: E402
from motor.sizing import Size  # noqa: E402

# the step between RRA and the frames' physics: RRA's record, the feet held and the path's following, on MuJoCo's
# own compile, and the rates and names the frames read (`run_person`'s `seconds`)
# MyoFullBody as vendored in this repository (vendor/myofullbody: MuscleMimic Models 1.0.6, under its own licences)
VENDORED = pathlib.Path(__file__).resolve().parents[2] / "vendor" / "myofullbody"
CHECKED = "checks: RRA's record, feet held, following on MuJoCo's compile"

FEET_SIDES = (("LeftFoot", "LeftToeBase"), ("RightFoot", "RightToeBase"))  # a seen stance's columns, left then right
DYNAMICS = "dynamics"  # a job's "motor" for MOTOR's step 1 and step 2's stage A (#98); true for step 1 alone
MUSCLES = "muscles"  # ... and step 4's stage B on stage A's answer: the frames' activations stage B's


def arguments():
    given = sys.argv[1:]
    if len(given) != 2:
        raise SystemExit(__doc__)
    return pathlib.Path(given[0]), pathlib.Path(given[1])


def recordings(person, worlds):
    """Every recording of the person's SOMA body (each frames x segments x 4 x 4): the job's `recordings` (npz paths,
    each with its `world`) when given, else the clip's own `worlds` alone."""
    return [np.load(found)["world"] for found in person["recordings"]] if person.get("recordings") else [worlds]


def person_set_up(job, person, rig, worlds, path, size, made=None):
    """The person's file at `path` (MyoFullBody sized by `size`, `sizing.Size` of the person's rig) with the fit's
    constants (the person constants, solved once from all the body's `recordings`), its record (the fit's method and
    constants in it), each body's scale and the `Fitting` its clips' angles are solved on. `made`: the recordings'
    targets, when the caller made them already (`constants_set_up`). The body is sized once (`myo.sized_body`): the
    fitted file is the sized one with the constants put in (#70: sized twice, about 1.5 s a person on this PC)."""
    body = myo.sized_body(job["model"], size, person["kg"])
    _, scales = myo.written(body, path)
    return constants_set_up(job, person, rig, worlds, path, size, scales, made, body=body)


def constants_set_up(job, person, rig, worlds, path, size, scales, made=None, found=None, body=None):
    """`person_set_up` from the sized file without constants already at `path` (its scales `scales`): the constants
    solved on it, the file written again with them, its record, each body's scale and the `Fitting`. `made`: the
    recordings' `fit.targets` in order, when the caller made them already; `found`: the constants, when the caller
    solved them already (`fit.constants_across`, every set of a body at once); `body`: the sized body
    (`myo.sized_body`) the file was written from, else sized again."""
    segments = person["segments"]
    held = fit.followed(segments)
    if found is None:
        made = made or [fit.targets(rig, segments, clip, held, skins=False) for clip in recordings(person, worlds)]
        found = fit.constants(fit.fitting(path, scales, held, segments), made)
    record, scales = myo.written(body or myo.sized_body(job["model"], size, person["kg"]), path, found)
    fitting = fit.fitting(path, scales, held, segments, found["marks"])
    record["fit"] = {"method": fit.METHOD, "constants": fit.constants_record(found)}
    return record, scales, fitting


def clip_angles(fitting, rig, segments, worlds, made=None):
    """The clip's fitted path on `fitting` (qpos of MuJoCo's own compile of the file: the pose alone per frame, the
    constants held), by `fit.angles`, the bones held inside SOMA's posed skin (the stage's; level 1's evaluation runs
    without them: `evaluation.SKIN_ROWS`, `evaluation._case.fitted_case`). `made`: the clip's `fit.targets`, when the
    caller made them already (`fitted_person`)."""
    turns, points, skins = made or fit.targets(rig, segments, worlds, fit.followed(segments))
    return fit.angles(fitting, turns, points, skins)


def followed_record(record, fitting, angles, worlds, segments, scales, rig):
    """`record`'s fit given its following of SOMA (`fit.following` of the clip's fitted path)."""
    record["fit"]["following"] = fit.following(fitting.model, angles, worlds, segments, scales, rig)
    return record


def fitted_person(job, person, rig, worlds, path):
    """The person's file at `path` with the fit's constants, its record (the fit and its following of SOMA in it),
    each body's scale, the fitted model and the clip's fitted path: set up, angles, record. The clip's targets are
    made once: with no `recordings` the clip is level 2's one recording too (#70: SOMA's skin was posed on every frame
    twice, about 11 s a clip). With the job's "motor" (#98, MOTOR step 1) the path is the whole trial's solve
    warm started from the fit's (`trial.solved`), its record under "motor", and its motion (step 2's warm start,
    `trial.dynamics_solved`) the fifth value (None without)."""
    clip = fit.targets(rig, person["segments"], worlds, fit.followed(person["segments"]))
    made = None if person.get("recordings") else [clip]
    record, scales, fitting = person_set_up(job, person, rig, worlds, path, Size(rig), made)
    angles = clip_angles(fitting, rig, person["segments"], worlds, clip)
    motion = None
    if job.get("motor"):
        angles, record["motor"], motion = trial.solved(fitting, *clip, angles, job["rate"],
                                                       job.get("motor landmarks"))
    record = followed_record(record, fitting, angles, worlds, person["segments"], scales, rig)
    return record, scales, fitting.model, angles, motion


def skin_of(rig, segments, bodies, riding, fitted, worlds):
    """The person's drawn skin the ground pushes (`contact.Skin`): SOMA's mesh at its bind, its weights on the
    segments, each segment riding its bone from its group's start (`landmarks.at_starts`, as `riding` stands it),
    laid on SOMA's skin over the clip (`Skin.registered`: `fitted` each segment's frame on the fitted path,
    `posed_path`; `worlds` SOMA's segment frames, the targets')."""
    names = [str(name) for name in rig["names"]]
    joints = [names.index(segment) for segment in segments]
    table = groups()
    starts = at_starts(rig, segments, [table[segment][0][0] for segment in segments])
    skin = Skin(rig["points"], np.asarray(rig["weights"], dtype=float)[:, joints], starts,
                [bodies[segment] for segment in segments], [riding[segment] for segment in segments])
    soma = np.asarray(worlds, dtype=float) @ np.linalg.inv(np.asarray(rig["bind"], dtype=float)[joints]) @ starts
    return skin.registered(fitted, soma)


def credited(physics, skin, path, stance, bodies):
    """`skin` with each foot seen standing (`stance`, frames x 2: left, right; the targets') pushed from its own
    lowest point on `path` (`Skin.credited`); as it came where the targets carry no stance."""
    if stance is None:
        return skin
    if len(stance) != len(path):
        raise ValueError(f"{len(stance)} frames of seen stance for a path of {len(path)}")
    sides = [[bodies[segment] for segment in side] for side in FEET_SIDES]
    parts = {side: np.flatnonzero(np.isin(skin.owners, mine)) for side, mine in enumerate(sides)}
    data, lowest = mujoco.MjData(physics.model), []
    for qpos in path:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(physics.model, data)
        found = lowests(physics.model, data, skin, physics.ground, parts)[1]
        lowest.append([np.nan if found[side] is None else found[side] for side in range(len(sides))])
    skin.credited(np.array(lowest), np.asarray(stance, dtype=bool), sides)
    return skin


def said(name, step, since):
    """`step` of the person `name` done, its seconds since `since` said on stderr (the run's log): which step of a
    trial takes a job's minutes (#98, 2026-11-07: some jobs passed their 20 minutes); the time now."""
    now = time.perf_counter()
    print(f"{name}: {step} in {now - since:.1f} s", file=sys.stderr, flush=True)
    return now


def run_person(job, person, out_path):
    """One person: its model sized and fitted to SOMA, its path, its physics frame by frame; its entry's `seconds`
    each step's wall seconds (#98: the paper's per-step timing; the steps add up to its "total")."""
    since = started = time.perf_counter()
    seconds = {}
    targets = np.load(person["targets"])
    worlds, ground = targets["world"], targets["ground"]
    segments = person["segments"]
    path = out_path.parent / f"{person['name']}.xml"
    rig = {**person["rig"], "points": targets["points"], "faces": targets["faces"], "weights": targets["weights"]}
    record, scales, own, angles, motion = fitted_person(job, person, rig, worlds, path)
    seconds["fit"] = time.perf_counter() - since
    since = said(person["name"], f"sized and fitted, {len(worlds)} frames", since)
    model = built(path, job["gravity"])
    rides = {segment: groups()[segment][2] for segment in segments}
    bodies = {segment: body_index(model, rides[segment]) for segment in segments}
    riding = fit.riding(model, scales, segments)
    physics = Physics(model, ground, None, parts_of(model, bodies))
    # MOTOR's path is smooth already: its jerk is the chain's low-pass (`trial.jerk_weights`)
    qpath = (np.array([physics.couplings.apply(qpos) for qpos in carried(own, model, angles)]) if job.get("motor")
             else filtered(model, physics.couplings, carried(own, model, angles), job["rate"]))
    fitted_poses = posed_path(model, qpath, segments, bodies, riding)
    skin = skin_of(rig, segments, bodies, riding, fitted_poses, worlds)
    physics.skin = credited(physics, skin, qpath, targets["stance"] if "stance" in targets.files else None, bodies)
    feet = {segment: np.flatnonzero(skin.owners == bodies[segment]) for segment in segments if segment.endswith(FEET)}
    weight = sum(model.body_mass) * job["gravity"]
    standing = com_height(model, qpath[0], ground)
    activations = None
    if job.get("motor") in (DYNAMICS, MUSCLES):
        # MOTOR step 2 (#98): stage A makes the path dynamically consistent on the seen floor; the chain (RRA, the
        # static optimisation) then runs on it as it ran on the fit's, and measures it as it does any path
        seconds["set up before MOTOR"] = time.perf_counter() - since
        since = time.perf_counter()
        sides = [[bodies[segment] for segment in side] for side in FEET_SIDES]
        side_points = [int(np.isin(skin.owners, side).sum()) for side in sides]
        residual = (job["residual"]["force"], job["residual"]["moment"])
        angles, record["motor"]["dynamics"], motion = trial.dynamics_solved(
            motion, physics, own, max(1, int(np.mean(side_points))), sides, residual, standing,
            job.get("motor settings"))
        seconds["MOTOR stage A"] = time.perf_counter() - since
        since = said(person["name"], "MOTOR's stage A", since)
        if job["motor"] == MUSCLES:
            # MOTOR step 4: stage B solves the muscles with the movement and contact; the frames' activations are
            # its, the chain's static optimisation runs on its path beside it
            angles, activations, record["motor"]["muscles"] = trial.muscles_solved(motion, physics, residual,
                                                                                   standing)
            seconds["MOTOR stage B"] = time.perf_counter() - since
            since = said(person["name"], "MOTOR's stage B", since)
        qpath = np.array([physics.couplings.apply(qpos) for qpos in carried(own, model, angles)])
        fitted_poses = posed_path(model, qpath, segments, bodies, riding)
    names = [str(name) for name in person["rig"]["names"]]
    parents = [segments.index(names[parent]) if parent >= 0 and names[parent] in segments else -1
               for parent in (person["rig"]["parents"][names.index(segment)] for segment in segments)]
    limits = (job["residual"]["force"] * weight, job["residual"]["moment"] * weight * standing)
    most = (job["adjust"]["move most m"], np.radians(job["adjust"]["turn most deg"]))

    kept_feet = job["adjust"].get("feet kept", True)
    footing = Footing(model, physics.couplings, [bodies[segment] for segment in feet], path if kept_feet else None)
    seconds["set up: Newton's model, its card, the filtered path"] = (time.perf_counter() - since
                                                                       + seconds.pop("set up before MOTOR", 0.0))
    since = time.perf_counter()

    def land(path):
        """`path`'s feet put back where they stand on the fitted path: the clip's frames, or whole copies of them
        stacked (`rra.cap_rows`' nudged paths), each copy against the clip's; with "feet kept" false, `path` as it is."""
        if len(path) % len(qpath):
            raise ValueError(f"{len(path)} frames to land: not whole copies of the clip's {len(qpath)}")
        if not kept_feet:
            return np.array(path)
        began = time.perf_counter()
        footed = footing.footed(path, np.concatenate([qpath] * (len(path) // len(qpath))))
        seconds["footing"] = seconds.get("footing", 0.0) + time.perf_counter() - began
        return footed

    riders = (segments, bodies, riding, parents)
    adjusted, moves, step = reduced(physics, qpath, job["rate"], limits, most, land, fitted_poses, riders)
    seconds["RRA, rounds"] = time.perf_counter() - since - seconds.get("footing", 0.0)
    footed_before = seconds.get("footing", 0.0)
    since = said(person["name"], "RRA's rounds", since)
    print(f"{person['name']}: off the clip {off_clip(model, qpath, worlds, *riders):.3f} deg fitted, "
          f"{off_clip(model, adjusted, worlds, *riders):.3f} with RRA's whole move", file=sys.stderr, flush=True)
    adjusted, kept = within_follows(physics, qpath, adjusted, moves, land, worlds, riders, job["adjust"]["follows deg"])
    own_model = mujoco.MjModel.from_xml_path(str(path))  # MuJoCo's own compile: its meshes and muscle names
    # its bones measured on MuJoCo's own compile (Newton is given no meshes); its qpos is Newton's in the same order
    skins = fit.FrameSkins(own_model, rig, segments, worlds)
    adjusted, shares = within_frames(physics, qpath, kept * moves, land, fitted_poses, riders, most, job["rate"],
                                     skins)
    seconds["RRA, follows and frame caps"] = (time.perf_counter() - since
                                              - (seconds.get("footing", 0.0) - footed_before))
    since = said(person["name"], "RRA's follows and frame caps", since)
    adjusted_poses = posed_path(model, adjusted, segments, bodies, riding)
    adjustment = adjustment_record((adjusted[:, :3] - qpath[:, :3]) @ TO_OWN.T, kept, shares, step, fitted_poses,
                                   adjusted_poses, segments,
                                   [off_clip(model, walked, worlds, segments, bodies, riding, parents)
                                    for walked in (qpath, adjusted)])
    feet_off = footing.off(adjusted, qpath)
    adjustment["feet held mm, deg"] = [round(1000 * float(feet_off[:, 0].max()), 3),
                                       round(float(np.degrees(feet_off[:, 1].max())), 3)]
    qpath = adjusted
    record["fit"]["following, the path the physics takes"] = fit.following(own_model, qpath, worlds, segments, scales,
                                                                           rig)
    velocities, accelerations = rates(model, qpath, job["rate"])
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[dof]) for dof in
              physics.couplings.leads[6:]]
    named = {mujoco.mj_id2name(own_model, mujoco.mjtObj.mjOBJ_TENDON, own_model.actuator_trnid[actuator, 0]):
             mujoco.mj_id2name(own_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator) for actuator in range(own_model.nu)}
    muscles = [named[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, model.actuator_trnid[actuator, 0])]
               for actuator in physics.muscles]
    meshes = bones_of(own_model)
    drawn = sorted(meshes)
    drawn_ids = [body_index(model, name) for name in drawn]
    seconds[CHECKED] = time.perf_counter() - since
    since = time.perf_counter()
    records, bones, posed, paths, counts, pushed = frames(physics, qpath, velocities, accelerations, worlds, ground,
                                                          riders, feet, drawn_ids, job["rate"], seconds)
    since = said(person["name"], "the physics frame by frame", since)
    if activations is not None:  # MOTOR's stage B: its activations the frames', the static optimisation's mean kept
        record["motor"]["muscles"]["static optimisation activations mean"] = round(float(np.mean(
            [frame["activations"] for frame in records])), 4)
        for frame, mine in zip(records, activations, strict=True):
            frame["activations"] = np.round(mine, 4).tolist()
    mine = arrays(person["name"], bones, posed, paths, counts, meshes)
    # every hinge's angle on the path the physics takes, by joint name (#98: a validation reads the coordinates)
    hinges = [joint for joint in range(model.njnt) if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE]
    mine[f"{person['name']}/coordinates"] = qpath[:, model.jnt_qposadr[hinges]].astype(np.float32)
    mine[f"{person['name']}/coordinate names"] = np.array(
        [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint) for joint in hinges])
    found = entry(model, record, (muscles, joints, drawn), rides, qpath[0], ground, job["gravity"], adjustment,
                  records, pushed)
    seconds["arrays"] = time.perf_counter() - since
    seconds["total"] = time.perf_counter() - started
    found["seconds"] = {step: round(spent, 3) for step, spent in seconds.items()}
    return found, mine


def run(job_path, out_path):
    job = json.loads(job_path.read_text())
    job.setdefault("model", str(VENDORED))
    people, arrays_found = {}, {}
    for person in job["people"]:
        people[person["name"]], mine = run_person(job, person, out_path)
        arrays_found.update(mine)
    np.savez_compressed(out_path.with_suffix(".npz"), **arrays_found)
    where = "every frame at once on MuJoCo Warp, cuda" if warp.is_cuda_available() else "CPU"
    card = warp.get_cuda_device().name if warp.is_cuda_available() else None  # the card the seconds were on
    out_path.write_text(json.dumps({"engine": f"newton {newton.__version__} SolverMuJoCo's model (MuJoCo "
                                              f"{mujoco.__version__}, {where}), inverse dynamics with contact and static "
                                              "optimisation; nothing stepped", "card": card, "rate": job["rate"],
                                "people": people}))

