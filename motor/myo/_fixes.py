"""#60's import fixes, so Newton builds MyoFullBody as MuJoCo compiles it: names without '-' or spaces (`renamed`),
every muscle's full gain, bias and dynamics parameters (`full_muscles`), and a body with several joints split into
1 g wrapper bodies, one joint each (`split`, `multi_joint_bodies`)."""
import re
import xml.etree.ElementTree as ElementTree

import mujoco

from motor.sizing import WRAPPER_KG

WRAPPER_INERTIA = "1e-4 1e-4 1e-4"
NAMED = ("model", "name", "site", "sidesite", "tendon", "joint", "joint1", "joint2", "body", "body1", "body2", "geom",
         "geom1", "geom2", "childclass", "class")


def renamed(xml):
    """Names MuJoCo and Newton both read: '-' and spaces become '_' (Newton cleans tendon site names on one side
    only, #60)."""
    return re.sub(r'\b(' + "|".join(NAMED) + r')="([^"]*)"',
                  lambda found: f'{found.group(1)}="{found.group(2).replace("-", "_").replace(" ", "_")}"', xml)


def full_muscles(xml, model):
    """Every <general> actuator with its compiled gain, bias and dynamics parameters written in full (a partial gainprm
    loses its class defaults in Newton, #60)."""
    def full(found):
        tag = found.group(0)
        name = re.search(r' name="([^"]*)"', tag)
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.group(1)) if name else -1
        if actuator < 0:
            return tag
        for field in ("gainprm", "biasprm", "dynprm"):
            values = " ".join(f"{value:.9g}" for value in getattr(model, "actuator_" + field)[actuator])
            tag = re.sub(f'\\s{field}="[^"]*"', "", tag).replace("<general ", f'<general {field}="{values}" ', 1)
        return tag
    return re.sub(r"<general [^>]*>", full, xml)


def multi_joint_bodies(spec):
    """How many wrapper bodies the split adds: one for each joint past a body's first."""
    return sum(max(0, len(body.joints) - 1) for body in spec.bodies[1:])


def split(xml):
    """Every body with several joints split: its joints but the last move, in order, into new wrapper bodies at its
    frame (the same motion as MuJoCo's in-body joint order), each WRAPPER_KG; Newton builds an equality only on a
    joint that is its body's own (#60)."""
    root = ElementTree.fromstring(xml)

    def visit(parent):
        for body in [child for child in parent if child.tag == "body"]:
            visit(body)
            joints = [child for child in body if child.tag == "joint"]
            if len(joints) < 2:
                continue
            at = list(parent).index(body)
            parent.remove(body)
            holder = parent
            for number, joint in enumerate(joints[:-1]):
                wrapper = ElementTree.Element("body", {"name": f"{body.get('name')}__j{number}"})
                if number == 0:
                    for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
                        if body.get(key) is not None:
                            wrapper.set(key, body.attrib.pop(key))
                if body.get("childclass"):
                    wrapper.set("childclass", body.get("childclass"))
                ElementTree.SubElement(wrapper, "inertial", {"pos": "0 0 0", "mass": str(WRAPPER_KG),
                                                             "diaginertia": WRAPPER_INERTIA})
                body.remove(joint)
                wrapper.append(joint)
                holder.insert(at, wrapper)
                holder, at = wrapper, len(wrapper)
            holder.append(body)

    visit(root.find("worldbody"))
    return ElementTree.tostring(root, encoding="unicode")
