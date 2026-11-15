"""MyoFullBody's held fingers laid in SOMA's hand (#70, 2026-11-01): SOMA has no finger joints, so its hand is one
rigid shape and MyoFullBody's fingers are held (`fit` freezes them where they stand); at the published zero pose its
fingers stand straight and its thumb opposed, out of SOMA's spread thumb and cupped little finger (on the template,
hand laid by the calibration table, 67 % of the finger bones' vertices outside SOMA's skin, the thumb 25 to 45 mm).
`rested` turns each finger body by its joints' rest turns (`calibration.FINGERS`) so the file's zero pose is that
rest: the body's place and turn take the joints' rest moves, each joint's axis and anchor are carried so every pose
is reached as before, its range moved by its rest turn. On the spec of the pinned source, before sizing (every
finger body is in the hand's bone group, one scale)."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def rested(spec, angles):
    """Each body holding a joint named in `angles` ({joint: rest turn, radians}) turned so the joints' zero is that
    turn (`_rest_moved`), every joint of the body carried (`_carried`), the turned joints' ranges moved."""
    for body in spec.bodies:
        joints = list(body.joints)
        turns = [float(angles.get(joint.name, 0.0)) for joint in joints]
        if not any(turns):
            continue
        moves = [_move(joint, turn) for joint, turn in zip(joints, turns)]
        _rest_moved(body, moves)
        for index, joint in enumerate(joints):
            _carried(joint, moves[index + 1:])
            joint.range = np.asarray(joint.range, dtype=float) - turns[index]


def _move(joint, turn):
    """A hinge's rigid move by `turn` in its body's frame: (rotation, translation), about its axis through its
    anchor."""
    if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"{joint.name}: only a hinge is laid at rest, not {joint.type}")
    axis = np.asarray(joint.axis, dtype=float)
    turn_matrix = Rotation.from_rotvec(axis / np.linalg.norm(axis) * turn).as_matrix()
    anchor = np.asarray(joint.pos, dtype=float)
    return turn_matrix, anchor - turn_matrix @ anchor


def _composed(moves):
    """The moves one after another, in the body's frame (MuJoCo's joint order): (rotation, translation)."""
    rotation, translation = np.eye(3), np.zeros(3)
    for turn_matrix, shift in moves:
        rotation, translation = rotation @ turn_matrix, rotation @ shift + translation
    return rotation, translation


def _rest_moved(body, moves):
    """The body's place and turn in its parent with its joints' rest moves taken in."""
    if body.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
        raise ValueError(f"{body.name}: its turn is given as {body.alt.type}, only a quaternion is laid at rest")
    rotation, translation = _composed(moves)
    turn = Rotation.from_quat(np.roll(np.asarray(body.quat, dtype=float), -1)).as_matrix()
    body.pos = np.asarray(body.pos, dtype=float) + turn @ translation
    body.quat = np.roll(Rotation.from_matrix(turn @ rotation).as_quat(), 1)


def _carried(joint, later):
    """A joint's axis and anchor carried into the rested frame: the joints after it in the body move it by their rest
    moves, so its own move there is theirs undone around it."""
    rotation, translation = _composed(later)
    joint.axis = rotation.T @ np.asarray(joint.axis, dtype=float)
    joint.pos = rotation.T @ (np.asarray(joint.pos, dtype=float) - translation)
