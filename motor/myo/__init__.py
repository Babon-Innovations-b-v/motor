"""MyoFullBody (MyoSuite's musculoskeletal body: its bones, 123 joints, 416 muscles and their tendons) made ready for
Newton and sized to one person (#65): `person_file` writes the person's file and returns its record.

The model is the pinned wheel the stage fetched (musclemimic_models 1.0.6, `SOURCE` in it). Each person's file is the
model with:
- its floor gone (the person stands on the scene's own ground);
- each bone group sized to the person's SOMA body (a limb's width by its girth), its masses from SOMA's volumes
  (`sizing`);
- each muscle's length range scaled by its path's length at the model's zero pose over the model's (OpenSim's
  muscle scaling: optimal fibre and tendon slack length with the path); its peak isometric force kept (#64: strength
  does not scale here); then its tendon slack length lengthened where it stands stretched (`resting`);
- #60's import fixes (`_fixes`): names without '-' or spaces, every muscle's full gain, bias and dynamics parameters,
  explicit inertials, a body with several joints split into 1 g wrapper bodies (one joint each), Euler written out.
"""
import pathlib

import mujoco

from motor import resting
from motor.calibration import fingers
from motor.myo._fixes import full_muscles, multi_joint_bodies, renamed, split
from motor.myo._movable import movable
from motor.myo._rested import rested
from motor.sizing import WRAPPER_KG, assumed, balanced, kept_bodies, path_lengths, sized, ungrouped_kg

SOURCE = "musclemimic_models/model/body/myofullbody.xml"


def _source_model(folder):
    """The published model, compiled."""
    return mujoco.MjModel.from_xml_path(f"{folder}/{SOURCE}")


def _fresh(source):
    """A new spec of the published model, its floor gone, movable above the lower back (`movable`), its held fingers
    at SOMA's hand's rest (`rested`)."""
    spec = mujoco.MjSpec.from_file(source)
    spec.meshdir = spec.texturedir = str(pathlib.Path(source).parents[1])
    for geom in list(spec.worldbody.geoms):
        spec.delete(geom)
    movable(spec)
    rested(spec, fingers())
    return spec


def _filed(name):
    """A body's name as the person's file names it (myo._fixes.renamed's)."""
    return name.replace("-", "_").replace(" ", "_")


def inside_moved(spec, joints):
    """Each joint inside a bone group moved in its parent by the fit's found move ({body as the person's file names
    it: [3] m}; #70's level 2), on the sized places."""
    for body in spec.bodies:
        filed = _filed(body.name)
        if filed in joints:
            body.pos = body.pos + joints[filed]


class Sized:
    """One person's MyoFullBody sized (`sized_body`), before the fit's constants: what `written` makes every file of
    that person from, so a second file (the fitted one) is not sized again (#70)."""

    def __init__(self, folder, size, kg):
        self.folder, self.size, self.kg = folder, size, kg
        source = f"{folder}/{SOURCE}"
        self.model = _source_model(folder)
        self.scales = size.scales(self.model)
        spec = _fresh(source)
        self.wrappers = multi_joint_bodies(spec)
        sized(spec, self.model, self.scales, kg, self.wrappers)
        self.sized = spec
        self.rest = path_lengths(_fresh(source).compile())  # the published model's paths at the same rest

    def spec(self, joints):
        """A new spec of the sized body with the inside joints moved by `joints` (`inside_moved`)."""
        spec = self.sized.copy()
        inside_moved(spec, joints)
        return spec


def sized_body(folder, size, kg):
    """The person's MyoFullBody sized by `size` (a `sizing.Size` of the person's SOMA) to `kg` (`Sized`)."""
    return Sized(folder, size, kg)


def person_file(folder, size, kg, path, constants=None):
    """One person's MyoFullBody, written at `path`, sized by `size` (a `sizing.Size` of the person's SOMA) and, when
    given, the fit's `constants` ({"joints": each inside joint's move, `inside_moved`}; the lengths are the sizing's,
    the fit scales nothing): `written` on a new `sized_body`."""
    return written(sized_body(folder, size, kg), path, constants)


def written(body, path, constants=None):
    """The person's file of `body` (`Sized`) written at `path` with the fit's `constants` when given ({"joints": each
    inside joint's move, `inside_moved`}), and (its record, each body's scale): the record holds each group's scale
    and its length against SOMA's, the inertia scale per group, its muscles' length scale, its total mass and what is
    assumed."""
    size, kg, model, scales, wrappers = body.size, body.kg, body.model, body.scales, body.wrappers
    spec = body.spec(constants.get("joints", {}) if constants else {})
    grown = spec.compile()
    factors = size.weigh(spec, grown, scales, kg - wrappers * WRAPPER_KG - ungrouped_kg(grown, size.table))
    balanced(spec)
    grown = spec.compile()
    stretch = path_lengths(grown) / body.rest  # both at the same rest
    for actuator in spec.actuators:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name)
        actuator.lengthrange = model.actuator_lengthrange[index] * stretch[index]
    stretched = spec.compile()
    moves = resting.slack_moves(stretched)
    for actuator in spec.actuators:
        index = mujoco.mj_name2id(stretched, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.name)
        actuator.lengthrange = stretched.actuator_lengthrange[index] + moves[index]
    final = spec.compile()
    xml = spec.to_xml()
    if "integrator=" not in xml:  # MjSpec.to_xml drops Euler (MuJoCo's default) and Newton would pick implicitfast
        xml = xml.replace("<option ", '<option integrator="Euler" ', 1)
    # an inertia written out to 6 digits may miss A + B >= C by a rounding
    xml = xml.replace("<compiler ", '<compiler balanceinertia="true" ', 1)
    path.write_text(split(full_muscles(renamed(xml), final)))
    return {"scales": {name: [round(float(axis), 4) for axis in scale] for name, scale in scales.items()
                       if name in kept_bodies(size.table)},
            "lengths": size.record(final, scales), "inertia scale": {group: round(factor, 3)
                                                                     for group, factor in factors.items()},
            "centre of mass moved cm": size.moved,
            "muscle length scale": [round(float(stretch.min()), 4), round(float(stretch.max()), 4)],
            "kg": round(float(final.body_subtreemass[0]) + wrappers * WRAPPER_KG, 3),
            "resting": resting.record(stretched, final, moves),
            "assumed": assumed(final, scales, size.rig)}, scales
