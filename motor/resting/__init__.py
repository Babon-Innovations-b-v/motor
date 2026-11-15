"""Each muscle's tendon slack length calibrated at the standing pose (#70, the owner's option 3 of 2026-10-29: "Adjust
the tendon lengths so the stomach muscles aren't stretched while standing"): a muscle whose fibre stands longer than
REST_MOST of its optimal length at MyoSuite's `stand` (the zero pose, couplings applied) has its tendon slack length
lengthened so it stands at REST_MOST; its optimal fibre length, peak force and curves are kept.

MuJoCo's muscle has no tendon slack length of its own: its optimal fibre length is L0 = (lr1 - lr0) / (range1 - range0)
and its tendon slack length LT = lr0 - range0 L0, both from its `lengthrange` (lr0, lr1) and `range` (gainprm[0:2]).
Moving both ends of the length range by the same distance keeps L0 and moves LT by that distance alone. LT is solved
from the musculotendon length in a reference pose and the optimal fibre length, the fibre's normalised length there
given: a one-pose form of Manal and Buchanan 2004's numerical method (J Appl Biomech 20(2):195-203,
doi:10.1123/jab.20.2.195, which takes two poses); the optimal fibre lengths are not touched: MyoFullBody's own LT is below
zero for 32 of the 74 muscles over REST_MOST standing (MyoConverter fits the curves, not the lengths), and keeping LT +
L0 would need a slack length below zero for 46 of them (rectus abdominis: L0 3.9 times its own, measured 2026-10-29)."""
import mujoco
import numpy as np

from motor.model import Couplings, default_pose

REST_MOST = 1.1  # the longest normalised fibre length a muscle may stand at


def lengths(model):
    """Each actuator's optimal fibre length and tendon slack length (MuJoCo's, from its length range; metres)."""
    ranges, gains = model.actuator_lengthrange, model.actuator_gainprm
    optimal = (ranges[:, 1] - ranges[:, 0]) / (gains[:, 1] - gains[:, 0])
    return optimal, ranges[:, 0] - gains[:, 0] * optimal


def standing(model):
    """Each actuator's normalised fibre length at MyoSuite's `stand` (its path length less its tendon slack length,
    over its optimal fibre length)."""
    data = default_pose(model, Couplings(model))
    mujoco.mj_fwdPosition(model, data)
    optimal, slack = lengths(model)
    return (data.actuator_length - slack) / optimal


def slack_moves(model):
    """How far each muscle's tendon slack length is lengthened (metres; 0 for every other actuator): to stand at
    REST_MOST when it stands longer."""
    optimal, _ = lengths(model)
    muscle = model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
    return np.where(muscle, np.clip(standing(model) - REST_MOST, 0.0, None) * optimal, 0.0)


def passive(model):
    """Each actuator's passive force at MyoSuite's `stand` (N, a pull positive)."""
    data = default_pose(model, Couplings(model))
    mujoco.mj_fwdPosition(model, data)
    return np.array([-mujoco.mju_muscleBias(data.actuator_length[index], model.actuator_lengthrange[index],
                                            model.actuator_acc0[index], model.actuator_biasprm[index, :9])
                     if model.actuator_gaintype[index] == mujoco.mjtGain.mjGAIN_MUSCLE else 0.0
                     for index in range(model.nu)])


def record(before, after, moves):
    """The calibration's record (the model record's `resting`): its rule, the muscles moved, and the passive force at
    `stand` before and after (the total, the largest ten after)."""
    names = [mujoco.mj_id2name(after, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(after.nu)]
    found, start = passive(after), passive(before)
    moved = np.flatnonzero(moves > 0)
    return {"method": f"tendon slack length lengthened so no muscle stands past {REST_MOST} of its optimal fibre "
                      "length at MyoSuite's stand (one pose of Manal and Buchanan 2004); L0, peak force and curves kept",
            "muscles moved": len(moved),
            "slack moved mm": {names[index]: round(1000 * float(moves[index]), 2) for index in moved},
            "passive standing N": [round(float(start.sum()), 1), round(float(found.sum()), 1)],
            "largest passive standing N": {names[index]: round(float(found[index]), 1)
                                           for index in np.argsort(-found)[:10]}}
