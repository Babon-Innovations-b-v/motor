"""What the run writes for one person (out.json's entry and <out>.npz's arrays, the contract the stage reads): each
frame's record (`frames`: the body's and the clip's segment turns, the joints' moments, the ground's push, each touching part's push and centre of pressure (a foot one part, steadied over the clip: `steadied`), the feet,
the residual and reserve, the muscles' activations and forces; the poses the skin plays are SOMA's own, the stage's)
with the arrays drawn beside
(each body's and segment's pose, the muscles' paths), the bone meshes (`bones_of`), the arrays by name (`arrays`),
the person's entry (`entry`, its pushes' measures `push_measures`, judged for a quiet stance by `push_faults`) and RRA's
`adjusted` record (`adjustment_record`)."""
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from motor.bone_groups import LACKS
from motor.ground import heights_over
from motor.model import TO_OWN, own_pose
from motor.rra import RRA_ROUNDS, turned_off
from motor.solvers import smoothed

# `frames`' timed steps but the static optimisation's (#98: the paper's per-step timing)
FRAME_STATES = "inverse dynamics, contacts and muscle paths"  # every frame's state on MuJoCo Warp, its problem set up
RECORDED = "records and push measures"  # each frame's share read out, the pushes steadied and measured

BONE_CELL = 0.005  # metres: the drawn bones' vertices merged within this
QUIET = 0.02  # a frame whose movement needs its weight up within 2% of it is quiet stance (`push_measures`)
# A quiet stance's pushes (`push_faults`): their sum within 5% of the weight, each centre of pressure on its ground
# within 1 mm and within 1 cm of the ground under its touching skin, a part's push changing at most 2% of the weight
# a frame and its centre at most 5 mm a frame while the same parts touch.
QUIET_OFF = 5.0  # % of the weight
OFF_GROUND = 1.0  # mm
OUTSIDE = 10.0  # mm
QUIET_JUMP = 2.0  # % of the weight a frame
QUIET_SHIFT = 5.0  # mm a frame


def segment_poses(worlds):
    """Each of several 4 x 4s as [x, y, z, qx, qy, qz, qw] (n x 7), in one rotation call (#70: one call a pose)."""
    worlds = np.asarray(worlds)
    return np.concatenate([worlds[:, :3, 3], Rotation.from_matrix(worlds[:, :3, :3]).as_quat()], axis=1)


def local_turns(worlds, parents):
    """Each segment but the first against its parent, as rotation vectors in degrees (n x 3, one rotation call)."""
    worlds = np.asarray(worlds)
    children = [index for index, parent in enumerate(parents) if parent >= 0]
    if not children:
        return np.zeros((0, 3))
    above = worlds[[parents[index] for index in children], :3, :3]
    return Rotation.from_matrix(np.matmul(above.transpose(0, 2, 1), worlds[children, :3, :3])).as_rotvec(degrees=True)


def rounded(value):
    return None if value is None else round(value, 5)


def paths_of(data, tendons):
    """The tendons' path points now, end to end in `tendons`' order (MuJoCo's frame), and each one's count: its wrap
    points, a pulley's (a break between branches, at the origin) left out; one gather for them all."""
    tendons = np.asarray(tendons, dtype=int)
    starts, sizes = data.ten_wrapadr[tendons], data.ten_wrapnum[tendons]
    owner = np.repeat(np.arange(len(tendons)), sizes)
    at = np.repeat(starts - np.cumsum(sizes) + sizes, sizes) + np.arange(sizes.sum())
    kept = data.wrap_obj.reshape(-1)[at] != -2
    return data.wrap_xpos.reshape(-1, 3)[at[kept]], np.bincount(owner[kept], minlength=len(tendons))


def merged(points, faces):
    """A mesh with its vertices merged within BONE_CELL (vertex clustering) and the faces that fell flat dropped."""
    cells = np.round(points / BONE_CELL).astype(np.int64)
    _, first, inverse = np.unique(cells, axis=0, return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    return points[first], faces


def bones_of(model):
    """The bone meshes of the person's model in their bodies' frames ({body name: (points, faces)}), from MuJoCo's own
    compile of its file (`model`), each mesh's vertices merged within BONE_CELL."""
    found = {}
    for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        mesh = model.geom_dataid[geom]
        points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        turn = np.zeros(9)
        mujoco.mju_quat2Mat(turn, model.geom_quat[geom])
        points = points @ turn.reshape(3, 3).T + model.geom_pos[geom]
        points, faces = merged(points, faces)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
        if name in found:
            before = found[name]
            points, faces = np.concatenate([before[0], points]), np.concatenate([before[1], faces + len(before[0])])
        found[name] = (points, faces)
    return found


def com_height(model, qpos, ground):
    """The centre of mass' height over its ground at a pose (m)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    return float(heights_over((data.subtree_com[model.jnt_bodyid[0]] @ TO_OWN.T)[None], ground)[0])


def adjustment_record(moved_by, kept, shares, step, before, after, segments, follows):
    """How far RRA moved the body (a quality measure, sim_bodies.json `adjusted`): the pelvis (cm) and its turn and each
    segment's turn (degrees) against the fitted path, median and largest over the frames, and the largest of any segment
    on any frame (the `follows` gate's per-frame cap), every body's mass change (made, within MASS_MOST, and RRA's
    recommended one each round, OpenSim's report) and the torso's centre of mass move (cm, its own frame), the share of
    the move kept under the follows cap, the least share a frame kept under the per-frame cap (`within_frames`), and the
    follows measure before and after."""
    turns = turned_off(before, after)
    turned = {segment: turns[:, index] for index, segment in enumerate(segments)}
    pelvis = 100 * np.linalg.norm(moved_by, axis=1)
    return {"method": "OpenSim's residual reduction (RRA; Delp et al. 2007, opensim-core RRATool): the pelvis' and "
                      "trunk's path and the body's masses and torso centre of mass adjusted, capped by follows",
            "rounds": RRA_ROUNDS, "kept share": kept, "least frame share": round(float(shares.min()), 3),
            "pelvis cm": [round(float(np.median(pelvis)), 2), round(float(pelvis.max()), 2)],
            "largest turn deg": round(float(turns.max()), 2),
            "segments deg": {segment: [round(float(np.median(found)), 2), round(float(found.max()), 2)]
                             for segment, found in turned.items()},
            "mass %": round(100 * float(step["mass"]), 3),
            "mass % recommended": [round(100 * float(share), 3) for share in step["mass recommended"]],
            "torso centre of mass cm": np.round(100 * step["torso"], 2).tolist(),
            "follows deg": [round(follows[0], 3), round(follows[1], 3)]}


def touched(touches):
    """Each touching part's push, centre of pressure, push along its normals and its touching skin's ground points
    ({part: (force xyz N, centre xyz m, N, points m)}, the person's frame), from `Physics.shared`'s `touches`."""
    return {part: (TO_OWN @ force, TO_OWN @ centre, size, under @ TO_OWN.T)
            for part, (force, centre, size, under) in touches.items()}


def runs(on):
    """The spans of frames on end ([(start, end)], end past the last) where `on` holds."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(on, int), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def steadied(raw, rate, ground):
    """Each frame's touching parts' pushes and centres of pressure ([{part: [force xyz N, centre xyz m]}], the
    person's frame) from the solve's (`raw`, `touched`'s a frame), low-passed as the path is (`solvers.smoothed`)
    over each span of frames the part touches: its force, its push along its normals and that push's moment about
    the origin, the centre their quotient, set on its ground. A part has a push only on a frame it touches (the solve
    shares the load between a foot's points afresh each frame: their centre moved 3 to 6 cm a frame standing,
    2026-11-05, #70)."""
    found = [{} for _ in raw]
    for part in sorted({part for touches in raw for part in touches}):
        for start, end in runs([part in touches for touches in raw]):
            span = [raw[frame][part] for frame in range(start, end)]
            forces = smoothed(np.array([force for force, _, _, _ in span]), rate)
            sizes = smoothed(np.array([size for _, _, size, _ in span]), rate)
            moments = smoothed(np.array([size * centre for _, centre, size, _ in span]), rate)
            for offset in np.flatnonzero(sizes > 0):
                centre = moments[offset] / sizes[offset]
                centre[1] -= np.nan_to_num(heights_over(centre[None], ground)[0])
                found[start + offset][part] = [*np.round(forces[offset], 3).tolist(), *np.round(centre, 4).tolist()]
    return found


def outside(centre, points):
    """How far a centre of pressure lies outside its touching skin's ground points, across the ground (x, z; m): 0
    within their hull, else to its nearest edge (to the nearest point where they span no area)."""
    flat, at = points[:, [0, 2]], np.asarray(centre)[[0, 2]]
    try:
        hull = ConvexHull(flat)
    except (QhullError, ValueError):
        return float(np.linalg.norm(flat - at, axis=1).min())
    if (hull.equations[:, :2] @ at + hull.equations[:, 2] <= 1e-12).all():
        return 0.0
    corners = flat[hull.vertices]
    ends = np.roll(corners, -1, axis=0)
    along = np.clip(np.einsum("ij,ij->i", at - corners, ends - corners) / np.einsum("ij,ij->i", ends - corners,
                                                                                      ends - corners), 0.0, 1.0)
    return float(np.linalg.norm(corners + along[:, None] * (ends - corners) - at, axis=1).min())


def push_measures(steady, raw, needs, weight, ground):
    """How the record's pushes (`steady`, `steadied`'s) stand against the physics (out.json's `pushes`): on the quiet
    stance frames (`needs`, the movement's needed force on the pelvis, the person's frame, within QUIET of the
    weight up) their sum's largest miss of the weight (%), its frames, and each part's largest change from one quiet
    frame to the next while the same parts touch (% of the weight, mm); on every frame each centre's largest height
    off its ground and distance outside its touching skin's ground points (`raw`'s; mm)."""
    quiet = [frame for frame, need in enumerate(needs) if np.linalg.norm(need - [0.0, weight, 0.0]) <= QUIET * weight]
    sums = [sum(push[1] for push in steady[frame].values()) for frame in quiet]
    jumps, shifts = [0.0], [0.0]
    for frame, after in zip(quiet, quiet[1:]):
        if after == frame + 1 and steady[frame].keys() == steady[after].keys():
            for part, push in steady[after].items():
                jumps.append(np.linalg.norm(np.subtract(push[:3], steady[frame][part][:3])))
                shifts.append(np.linalg.norm(np.subtract(push[3:], steady[frame][part][3:])))
    centres = [(push[3:], raw[frame][part][3]) for frame, touches in enumerate(steady) for part, push in touches.items()]
    heights = heights_over(np.array([centre for centre, _ in centres]), ground) if centres else np.zeros(0)
    return {"quiet frames": len(quiet),
            "quiet off weight %": round(100 * max((abs(found - weight) for found in sums), default=0.0) / weight, 2),
            "quiet jump % weight": round(100 * float(max(jumps)) / weight, 2),
            "quiet centre jump mm": round(1000 * float(max(shifts)), 2),
            "centre off ground mm": round(1000 * float(np.nanmax(np.abs(heights), initial=0.0)), 3),
            "centre outside mm": round(1000 * max((outside(centre, points) for centre, points in centres),
                                                  default=0.0), 2)}


def push_faults(measures):
    """What a quiet stance's push record breaks (`push_measures`'s, against QUIET_OFF, OFF_GROUND, OUTSIDE,
    QUIET_JUMP and QUIET_SHIFT): [] when nothing."""
    limits = {"quiet off weight %": QUIET_OFF, "centre off ground mm": OFF_GROUND, "centre outside mm": OUTSIDE,
              "quiet jump % weight": QUIET_JUMP, "quiet centre jump mm": QUIET_SHIFT}
    return [f"{name} {measures[name]} over {most}" for name, most in limits.items() if measures[name] > most]


def frames(physics, path, velocities, accelerations, worlds, ground, riders, feet, drawn_ids, rate,
           seconds=None):
    """Every frame's record and the drawn arrays beside it: its physics shared out (`Physics.problem`, every frame's
    least squares at once, `Physics.solved_all`, and `Physics.share_of`: `Physics.shared` frame by frame), its segments
    riding their bones (`riders`: segments, bodies, riding, parents), the feet (`feet`: {segment: its skin points}), its
    touching parts' pushes steadied over the clip (`steadied`, at the clip's `rate`); returns the records, each drawn
    body's poses, each segment's, the muscles' paths, their point counts and the pushes' measures (`push_measures`).
    The poses the skin plays are SOMA's own, written by the stage (#70: SOMA is the truth, never moved). `seconds`,
    when given, has the wall seconds of each step added: the frames' states (FRAME_STATES), the static optimisation's
    two stages (`muscles.STAGES`) and the records (RECORDED)."""
    started = time.perf_counter()
    model = physics.model
    segment_names, bodies, riding, parents = riders
    found, bones, segments, paths, counts, raw, needs, problems, posed_all, lowest = [], [], [], [], [], [], [], [], \
        [], []
    for needed in physics.each(path, velocities, accelerations):  # each frame's state: what it reads of it
        problems.append(physics.problem(needed))
        data = physics.data
        posed = [own_pose(model, data, bodies[segment]) @ riding[segment] for segment in segment_names]
        posed_all.append(posed)
        lowest.append(physics.lowests(feet))
        segments.append(segment_poses(posed))
        bones.append(segment_poses([own_pose(model, data, body) for body in drawn_ids]))
        mine, many = paths_of(data, model.actuator_trnid[physics.muscles, 0])
        counts.append(many)
        paths.append(mine @ TO_OWN.T)
    # the frames' static optimisation by `least` on the processor: on the card (`batched.least_all`) the clips' muscles
    # came out 24% more active (pickup, jump; 2026-11-05, #70), the cause not found yet; RRA's balance agrees on it
    solving = time.perf_counter()
    shared = physics.solved_all(problems, card=False, seconds=seconds)
    shared_at = time.perf_counter()
    for frame, (problem, solved) in enumerate(zip(problems, shared)):  # every frame's share
        share = physics.share_of(problem, solved)
        needed, posed, (low, feet_low) = problem["needed"], posed_all[frame], lowest[frame]
        pushes = share["pushes"] @ TO_OWN.T
        raw.append(touched(share["touches"]))
        needs.append(TO_OWN @ needed[:3])
        found.append({
            "angles": np.round(local_turns(posed, parents), 3).tolist(),  # the body's segments' turns
            "targets": np.round(local_turns(worlds[frame], parents), 3).tolist(),
            "torques": np.round(share["moments"], 3).tolist(),
            "own torques": np.round(share["own moments"], 3).tolist(),
            "ground": np.round(pushes.sum(axis=0), 3).tolist(),
            "lowest": rounded(low),
            "touches": None,  # steadied over the clip below
            "feet": {segment: [rounded(feet_low[segment]), round(float(pushes[bodies[segment], 1]), 3)]
                     for segment in feet},
            "residual": np.round(np.concatenate([TO_OWN @ share["residual"][:3], TO_OWN @ share["residual"][3:]]),
                                 3).tolist(),
            "reserve": round(float(np.abs(share["reserves"]).max(initial=0.0)), 3),
            "activations": np.round(share["activations"], 4).tolist(),
            "forces": np.round(share["forces"], 2).tolist()})
    steady = steadied(raw, rate, ground)
    for record, touches in zip(found, steady):
        record["touches"] = touches
    weight = float(sum(model.body_mass) * np.linalg.norm(model.opt.gravity))
    measures = push_measures(steady, raw, needs, weight, ground)
    if seconds is not None:
        seconds[FRAME_STATES] = seconds.get(FRAME_STATES, 0.0) + solving - started
        seconds[RECORDED] = seconds.get(RECORDED, 0.0) + time.perf_counter() - shared_at
    return found, bones, segments, paths, counts, measures


def arrays(name, bones, segments, paths, counts, meshes):
    """One person's arrays in <out>.npz, by name: `<name>/bones`, `/segments`, `/paths`, `/counts` and each bone's
    `/mesh/<body>/points|faces`."""
    found = {f"{name}/bones": np.asarray(bones, dtype=np.float32),
             f"{name}/segments": np.asarray(segments, dtype=np.float32),
             f"{name}/paths": np.concatenate(paths).astype(np.float32),
             f"{name}/counts": np.asarray(counts, dtype=np.int32)}
    for body, (points, faces) in meshes.items():
        found[f"{name}/mesh/{body}/points"] = points.astype(np.float32)
        found[f"{name}/mesh/{body}/faces"] = faces.astype(np.int32)
    return found


def entry(model, record, names, rides, qpos, ground, gravity, adjustment, records, pushed):
    """One person's out.json entry: its model record with what MyoFullBody lacks, its muscles, joints, drawn bones
    and the body each segment rides (`names`: those four), the centre of mass' height at `qpos` over the ground, its
    mass and weight, RRA's `adjusted` record, every frame's record and its pushes' measures (`pushed`,
    `push_measures`)."""
    muscles, joints, drawn = names
    weight = sum(model.body_mass) * gravity
    return {"model": {**record, "lacks": {key: value for key, value in LACKS.items()}},
            "muscles": muscles, "joints": joints, "bones": drawn, "rides": rides,
            "com height": round(com_height(model, qpos, ground), 4),
            "kg": round(float(sum(model.body_mass)), 3),
            "weight N": round(float(weight), 2), "adjusted": adjustment, "pushes": pushed, "frames": records}
