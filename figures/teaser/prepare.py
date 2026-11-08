"""The MOTOR paper's teaser (and its sim body in its skin): one LabValidation trial's frames as the arrays render.py draws, from the run's
own saved results only (out.npz's bone poses and muscle paths, out.json's activations, targets.npz's SOMA skin posed
on its own segments by linear blend, as the run skins it: fit/_skin.skin). The bone meshes are MyoFullBody's own at
full resolution (MuJoCo's compile of the trial's model file, each geom in its body's frame), placed by the run's
saved pose of that body; nothing is solved or moved here.

    newton_env/bin/python prepare.py <trial folder> <out.npz> <frame> [<frame> ...]

The teaser: trial subject2_squats1_Cam1 of the final LabValidation run (2026-11-07), frame 216.

Run in a Python with mujoco (the sim body's newton_env)."""
import json
import pathlib
import sys

import mujoco
import numpy as np


def quat_matrix(quat):
    """[qx, qy, qz, qw] as a 3 x 3 (scipy's order, records.segment_poses)."""
    qx, qy, qz, qw = quat
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def body_meshes(model):
    """{body name: (points, faces)} of every mesh geom, in its body's frame (records.bones_of, unmerged)."""
    found = {}
    for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start + count].astype(float)
        faces = model.mesh_face[model.mesh_faceadr[mesh]:model.mesh_faceadr[mesh] + model.mesh_facenum[mesh]]
        turn = np.zeros(9)
        mujoco.mju_quat2Mat(turn, model.geom_quat[geom])
        points = points @ turn.reshape(3, 3).T + model.geom_pos[geom]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
        before = found.get(name, (np.zeros((0, 3)), np.zeros((0, 3), dtype=int)))
        found[name] = (np.concatenate([before[0], points]), np.concatenate([before[1], faces + len(before[0])]))
    return found


def skinned(person, targets, frame):
    """SOMA's mesh on one frame, linear blend over the run's segments (fit/_skin.skin)."""
    names = person["rig"]["names"]
    joints = [names.index(segment) for segment in person["segments"]]
    moves = targets["world"][frame] @ np.linalg.inv(np.asarray(person["rig"]["bind"], dtype=float)[joints])
    moved = np.einsum("sij,vj->vsi", moves[:, :3, :3], targets["points"]) + moves[None, :, :3, 3]
    return np.einsum("vs,vsi->vi", targets["weights"][:, joints], moved)


def main(folder, out, frames):
    folder = pathlib.Path(folder)
    name = folder.name
    job = json.loads((folder / "job.json").read_text())
    person = next(entry for entry in job["people"] if entry["name"] == name)
    record = json.loads((folder / "out.json").read_text())["people"][name]
    arrays, targets = np.load(folder / "out.npz"), np.load(folder / "targets.npz")
    model = mujoco.MjModel.from_xml_path(str(folder / f"{name}.xml"))
    meshes = body_meshes(model)
    drawn = [bone for bone in record["bones"] if bone in meshes]
    faces, offset = [], 0
    for bone in drawn:
        faces.append(meshes[bone][1] + offset)
        offset += len(meshes[bone][0])
    poses, counts, paths = arrays[f"{name}/bones"], arrays[f"{name}/counts"], arrays[f"{name}/paths"]
    starts = np.concatenate([[0], np.cumsum(counts.sum(axis=1))])
    found = {"bone_faces": np.concatenate(faces).astype(np.int32), "skin_faces": targets["faces"].astype(np.int32),
             "frames": np.asarray(frames, dtype=np.int32), "ground": targets["ground"]}
    bone_points, skin_points, muscle_points, muscle_counts, activations = [], [], [], [], []
    for frame in frames:
        placed = []
        for bone in drawn:
            pose = poses[frame, record["bones"].index(bone)].astype(float)
            placed.append(meshes[bone][0] @ quat_matrix(pose[3:]).T + pose[:3])
        bone_points.append(np.concatenate(placed))
        skin_points.append(skinned(person, targets, frame))
        muscle_points.append(paths[starts[frame]:starts[frame + 1]])
        muscle_counts.append(counts[frame])
        activations.append(np.clip(record["frames"][frame]["activations"], 0.0, 1.0))
    feet = [record["bones"].index(bone) for bone in ("toes_l", "toes_r", "calcn_l", "calcn_r")]
    ahead = (poses[frames][:, feet[:2], :3].mean(axis=1) - poses[frames][:, feet[2:], :3].mean(axis=1)).mean(axis=0)
    ahead[1] = 0.0
    found["forward"] = ahead / np.linalg.norm(ahead)  # the feet's heel-to-toe direction on the floor, for the camera
    most = max(len(points) for points in muscle_points)
    found.update(bone_points=np.asarray(bone_points, np.float32), skin_points=np.asarray(skin_points, np.float32),
                 muscle_counts=np.asarray(muscle_counts, np.int32), activations=np.asarray(activations, np.float32),
                 muscle_points=np.asarray([np.pad(points, ((0, most - len(points)), (0, 0))) for points in muscle_points],
                                          np.float32))
    np.savez_compressed(out, **found)
    print(f"{name}: frames {frames[0]}..{frames[-1]} ({len(frames)}), {len(drawn)} bones "
          f"{len(found['bone_faces'])} faces, {counts.shape[1]} muscles, skin {targets['points'].shape[0]} points -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(frame) for frame in sys.argv[3:]])
