"""MOTOR on public data only: SOMA-X's own template body (NVlabs/SOMA-X, Apache-2.0, assets v0.3.3 on Hugging Face),
standing still in its bind pose on a flat floor, solved onto the vendored MyoFullBody.

    python -m examples.soma_template <out folder> [--frames 30] [--kg 70] [--motor muscles]

The two SOMA-X files it reads (SOMA_template_rig.usda: the rig, its bind and its LOD-low skin with skin weights;
SOMA_neutral.npz: the LOD-low triangles, the topology MOTOR's landmark set is on) are downloaded once into
<out folder>/soma-x at the pinned revision and checked by their sha256; SOMA-X's LICENSE beside them. Nothing of
SOMA-X is kept in this repository.

What it builds is the run's job (`motor.run`'s docstring): the rig's 27 joints (the fingers, jaw, eyes and SOMA's
twist joints folded into their nearest kept joint up the tree, as SOMA's public rig carries them), every segment's
frame each frame the bind's own, the body moved up or down so its lowest skin point is on the floor (y = 0, Y up,
metres), the floor a flat square under it. `--motor` adds the whole-trial solve to the job (`dynamics`: the
movement and ground reaction forces solved together over the trial; `muscles`: and the muscle forces with them);
without it the run is the warm start's chain alone (the fit, RRA, the per-frame forces). The mass is an input (`--kg`): SOMA's template has none. This is a run
of the solver on a public body, not a validation: a still bind pose has no movement to compare against anything.

Written: <out>/job.json, <out>/targets.npz, and the run's <out>/out.json and <out>/out.npz."""
import argparse
import hashlib
import json
import pathlib
import urllib.request

import numpy as np

# SOMA-X's assets at tag v0.3.3 (the revision SOMA-X's own code asks for), each file's sha256 as downloaded.
REVISION = "3648707b69e1b1dcf14c473e30613266b786b9f1"
SOURCE = f"https://huggingface.co/nvidia/SOMA-X/resolve/{REVISION}/"
FILES = {"SOMA_template_rig.usda": "a952cb7bdf4ccd801e355d53122d395b985b69d0d9181439edad5081bb171856",
         "SOMA_neutral.npz": "5829af4a5333abdba5ad449360b868faacee99d395971e62f35c0e7169432b63",
         "LICENSE": "d1a7d615ab8eff4de143b1456f46dabf232f54daf0fcf9a70442bb6f637a9e95"}
SKELETON = "/OUTPUT/c_skeleton_grp/Root"
SKIN = "/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_lo"
CENTIMETRES = 0.01  # the rig's units (its metersPerUnit)
# The joints MOTOR's rig leaves out, folded into what is left above them.
DROPPED = ("Thumb", "Index", "Middle", "Ring", "Pinky", "Jaw", "Eye", "Twist")
# The joints that carry no segment: the ground point under the body and the end sites.
NOT_SEGMENTS = ("Root", "HeadEnd", "LeftToeEnd", "RightToeEnd")
RATE = 30.0  # frames a second
REACH = 1.0  # metres: the floor reaches this far round the body
GRAVITY = 9.81  # m/s2
# The run's limits and caps: the residual's shares of the weight (and of it times the centre's height), RRA's caps.
RESIDUAL = {"force": 0.02, "moment": 0.01}
ADJUST = {"move most m": 0.02, "turn most deg": 5.0, "follows deg": 5.0}


def fetched(folder):
    """The pinned SOMA-X files in `folder`, downloaded once and checked by their sha256."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in FILES.items():
        path = folder / name
        if not path.exists():
            with urllib.request.urlopen(SOURCE + name, timeout=300) as answer:
                path.write_bytes(answer.read())
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != sha:
            path.unlink()
            raise ValueError(f"{name}: sha256 {found}, not the pinned {sha}")
    return folder


def kept_joints(names, parents):
    """(the kept joints' numbers, their parents in the kept numbering, each joint's kept carrier in that numbering)."""
    keep = [not any(mark in name for mark in DROPPED) for name in names]
    number = {old: new for new, old in enumerate(joint for joint, yes in enumerate(keep) if yes)}
    carriers = []
    for joint in range(len(names)):
        walk = joint
        while not keep[walk]:
            walk = parents[walk]
        carriers.append(number[walk])
    joints = sorted(number)
    return joints, [-1 if parents[joint] < 0 else number[parents[joint]] for joint in joints], carriers


def template_rig(folder):
    """SOMA-X's template as MOTOR's rig: names, parents, bind (27 x 4 x 4, m, world), points (the LOD-low skin at
    the bind, m), faces (SOMA_neutral's LOD-low triangles) and weights (points x 27)."""
    from pxr import Usd, UsdSkel, UsdGeom
    stage = Usd.Stage.Open(str(folder / "SOMA_template_rig.usda"))
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath(SKELETON))
    paths = [str(joint) for joint in skeleton.GetJointsAttr().Get()]
    names = [path.rsplit("/", 1)[-1] for path in paths]
    parents = [paths.index(path.rsplit("/", 1)[0]) if "/" in path else -1 for path in paths]
    bind = np.array([np.array(matrix, dtype=float).T for matrix in skeleton.GetBindTransformsAttr().Get()])
    bind[:, :3, 3] *= CENTIMETRES
    skin = stage.GetPrimAtPath(SKIN)
    points = np.array(UsdGeom.Mesh(skin).GetPointsAttr().Get(), dtype=float) * CENTIMETRES
    binding = UsdSkel.BindingAPI(skin)
    own = [str(joint).rsplit("/", 1)[-1] for joint in binding.GetJointsAttr().Get()]
    indices, weights = binding.GetJointIndicesPrimvar(), binding.GetJointWeightsPrimvar()
    size = indices.GetElementSize()
    indices = np.array(indices.Get(), dtype=int).reshape(-1, size)
    weights = np.array(weights.Get(), dtype=float).reshape(-1, size)
    joints, kept_parents, carriers = kept_joints(names, parents)
    folded = np.zeros((len(points), len(joints)))
    for column in range(size):
        carrier = np.array([carriers[names.index(own[index])] for index in indices[:, column]])
        np.add.at(folded, (np.arange(len(points)), carrier), weights[:, column])
    with np.load(folder / "SOMA_neutral.npz") as neutral:
        faces = np.asarray(neutral["triangles_low"], dtype=int)
    if faces.max() >= len(points):
        raise ValueError(f"SOMA_neutral's LOD-low triangles reach point {faces.max()}; the skin has {len(points)}")
    return {"names": [names[joint] for joint in joints], "parents": kept_parents, "bind": bind[joints],
            "points": points, "faces": faces, "weights": folded}


def on_floor(rig):
    """The rig moved up or down so its lowest skin point stands at y = 0."""
    lowest = float(rig["points"][:, 1].min())
    moved = dict(rig, points=rig["points"] - [0.0, lowest, 0.0], bind=rig["bind"].copy())
    moved["bind"][:, 1, 3] -= lowest
    return moved


def floor_under(worlds):
    """A flat square floor (2 x 3 x 3 triangles at y = 0) under the segments' span, grown by REACH."""
    path = np.asarray(worlds)[:, :, [0, 2], 3].reshape(-1, 2)
    (low_x, low_z), (high_x, high_z) = path.min(axis=0) - REACH, path.max(axis=0) + REACH
    corners = np.array([[low_x, 0.0, low_z], [high_x, 0.0, low_z], [high_x, 0.0, high_z], [low_x, 0.0, high_z]])
    return np.array([corners[[0, 1, 2]], corners[[0, 2, 3]]])


def written_job(rig, out, frames, kg, solve=None):
    """job.json and targets.npz in `out`: the still body, `frames` frames of its bind, on the floor, with the whole-trial
    solve `solve` when given; the job's path."""
    segments = [joint for joint, name in enumerate(rig["names"]) if name not in NOT_SEGMENTS]
    worlds = np.repeat(rig["bind"][None, segments], frames, axis=0)
    targets = out / "targets.npz"
    np.savez_compressed(targets, world=worlds, ground=floor_under(worlds), points=rig["points"], faces=rig["faces"],
                        weights=rig["weights"])
    person = {"name": "soma_template", "kg": float(kg),
              "height": float(rig["points"][:, 1].max() - rig["points"][:, 1].min()),
              "rig": {"names": rig["names"], "parents": rig["parents"], "bind": rig["bind"].tolist()},
              "segments": [rig["names"][joint] for joint in segments], "targets": str(targets)}
    job = out / "job.json"
    found = {"rate": RATE, "gravity": GRAVITY, "people": [person], "residual": RESIDUAL, "adjust": ADJUST}
    if solve:
        found["motor"] = solve
    job.write_text(json.dumps(found))
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--kg", type=float, default=70.0)
    parser.add_argument("--motor", choices=("dynamics", "muscles"), help="the whole-trial solve after the warm start")
    given = parser.parse_args()
    given.out.mkdir(parents=True, exist_ok=True)
    rig = on_floor(template_rig(fetched(given.out / "soma-x")))
    job = written_job(rig, given.out, given.frames, given.kg, given.motor)
    from motor import run
    run.run(job, given.out / "out.json")
    found = json.loads((given.out / "out.json").read_text())["people"]["soma_template"]
    print(f"{given.out / 'out.json'}: {len(found['frames'])} frames, {len(found['muscles'])} muscles, "
          f"{len(found['joints'])} joints")
