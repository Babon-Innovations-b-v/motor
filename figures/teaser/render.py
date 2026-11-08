"""Runs inside Blender (5.0, Cycles): the teaser's two body panels from prepare.py's arrays of one frame, rendered with
one camera and one set of lights so they share scale and light, and a third pass, the floor's shadow alone (the opaque
skin's, from an overhead light), that compose.py lays under both. `skin` is the SOMA skin alone, opaque;
`inside` is MyoFullBody's bones and muscles in the same skin made translucent, each muscle a tube along its path
coloured by its activation (RAMP; compose.py's colour bar reads the same stops). The skin is the run's own mesh
(SOMA's low template) with Catmull-Clark subdivision for display. A PNG a pass (skin, inside, shadow), transparent behind.

    blender -b --python render.py -- <shot.npz> <out folder> <width> <height> <samples>

FARM_CYCLES_GPU=1 renders on the card (OptiX, else CUDA)."""
import math
import os
import pathlib
import sys

import bpy
import numpy as np
from mathutils import Vector

# Activation 0..1 to colour (linear RGB): a pale blue-grey at rest through amber and orange to red at full activation.
RAMP = ((0.0, (0.50, 0.58, 0.74)), (0.35, (0.85, 0.60, 0.18)), (0.7, (0.92, 0.22, 0.04)), (1.0, (0.62, 0.0, 0.02)))
SKIN = (0.74, 0.47, 0.35)  # linear RGB, both panels
BONE = (0.86, 0.82, 0.72)
MUSCLE_RADIUS = 0.007  # m
VIEW_TURN = 35.0  # degrees from the person's right side toward their front
LENS = 85.0  # mm
SUBDIVISIONS = 2
AMBIENT = 0.35  # the white world's strength behind the panels' lights


def ramp(value):
    stops = np.array([stop for stop, _ in RAMP])
    colours = np.array([colour for _, colour in RAMP])
    return tuple(np.interp(value, stops, colours[:, axis]) for axis in range(3))


def blender_frame(points):
    """The person's frame (y up) to Blender's (z up)."""
    points = np.asarray(points, dtype=float)
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def use_card(scene):
    """Cycles on the card (OptiX, else CUDA)."""
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = kind
        except TypeError:
            continue
        preferences.get_devices()
        if [device for device in preferences.devices if device.type == kind]:
            for device in preferences.devices:
                device.use = device.type == kind
            scene.cycles.device = "GPU"
            return
    raise RuntimeError("FARM_CYCLES_GPU is set but no OptiX or CUDA card was found")


def scene_of(width, height, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    if os.environ.get("FARM_CYCLES_GPU"):
        use_card(scene)
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.transparent_max_bounces = 64
    scene.cycles.max_bounces = 12
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.35
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs[1].default_value = AMBIENT
    return scene


def material(name, colour, roughness):
    made = bpy.data.materials.new(name)
    made.use_nodes = True
    shader = made.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*colour, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    return made


def opaque_skin():
    """Soft skin: a little subsurface scattering, a satin sheen."""
    made = material("skin", SKIN, 0.5)
    shader = made.node_tree.nodes["Principled BSDF"]
    shader.inputs["Subsurface Weight"].default_value = 0.25
    shader.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.3)
    shader.inputs["Subsurface Scale"].default_value = 0.01
    shader.inputs["Sheen Weight"].default_value = 0.15
    return made


def translucent_skin():
    """The same skin nearly clear where it faces the camera and milky where it turns away, so its outline reads and
    what is inside it shows."""
    made = material("glass skin", SKIN, 0.45)
    nodes, links = made.node_tree.nodes, made.node_tree.links
    shader = nodes["Principled BSDF"]
    clear = nodes.new("ShaderNodeBsdfTransparent")
    edge = nodes.new("ShaderNodeLayerWeight")
    edge.inputs["Blend"].default_value = 0.25
    lift = nodes.new("ShaderNodeMath")
    lift.operation = "MULTIPLY_ADD"
    lift.inputs[1].default_value = 0.75
    lift.inputs[2].default_value = 0.06
    lift.use_clamp = True
    links.new(edge.outputs["Facing"], lift.inputs[0])
    mix = nodes.new("ShaderNodeMixShader")
    links.new(lift.outputs[0], mix.inputs["Fac"])
    links.new(clear.outputs[0], mix.inputs[1])
    links.new(shader.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes["Material Output"].inputs["Surface"])
    return made


def mesh_object(name, points, faces, made, subdivide):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(blender_frame(points).tolist(), [], np.asarray(faces).tolist())
    mesh.update()
    mesh.shade_smooth()
    thing = bpy.data.objects.new(name, mesh)
    thing.data.materials.append(made)
    bpy.context.scene.collection.objects.link(thing)
    if subdivide:
        smoothing = thing.modifiers.new("subdivision", "SUBSURF")
        smoothing.levels = smoothing.render_levels = SUBDIVISIONS
    return thing


def muscle_object(points, counts, activations):
    """One curve, a spline a muscle, its material by its activation."""
    curve = bpy.data.curves.new("muscles", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = MUSCLE_RADIUS
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    start = 0
    for index, count in enumerate(counts):
        mine = points[start:start + count]
        start += count
        if count < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(count - 1)
        placed = np.concatenate([blender_frame(mine), np.ones((count, 1))], axis=1)
        spline.points.foreach_set("co", placed.astype(np.float32).ravel())
        spline.material_index = len(curve.materials)
        curve.materials.append(material(f"muscle{index}", ramp(float(activations[index])), 0.4))
    thing = bpy.data.objects.new("muscles", curve)
    bpy.context.scene.collection.objects.link(thing)
    return thing


def camera_on(scene, skin, forward):
    """Three-quarter from the person's right, the skin filling the frame's height."""
    points = blender_frame(skin)
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2
    ahead = Vector(blender_frame(forward)).normalized()
    right = ahead.cross(Vector((0, 0, 1))).normalized()
    turn = math.radians(VIEW_TURN)
    look = (right * math.cos(turn) + ahead * math.sin(turn)).normalized()
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.lens = LENS
    camera.data.sensor_fit = "VERTICAL"
    scene.collection.objects.link(camera)
    scene.camera = camera
    width = scene.render.resolution_x / scene.render.resolution_y
    across = float(np.linalg.norm(high[:2] - low[:2]))
    reach = max((high[2] - low[2]) * 1.15, across * 1.1 / width)
    distance = reach / 2 / math.tan(camera.data.angle_y / 2) + 0.4
    camera.location = Vector(centre) + look * distance + Vector((0, 0, 0.25))
    camera.rotation_euler = (Vector(centre) - camera.location).to_track_quat("-Z", "Y").to_euler()
    return look, centre


def lights(scene, look, centre):
    """A large soft key from the camera's upper left, a fill from its right and a rim from behind light the body; an
    overhead light alone makes the floor's shadow (main's shadow pass). {name: light object}."""
    up = Vector((0, 0, 1))
    side = look.cross(up).normalized()
    made = {}
    for name, energy, where, size in (("key", 260.0, look * 3 + side * 2.2 + up * 3.0, 3.0),
                                      ("fill", 130.0, look * 3 - side * 2.8 + up * 0.8, 3.5),
                                      ("rim", 140.0, -look * 3 - side * 1.0 + up * 2.5, 2.0),
                                      ("overhead", 120.0, look * 0.3 + up * 3.0, 1.5)):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
        light.data.energy, light.data.size = energy, size
        light.location = Vector(centre) + where
        light.rotation_euler = (Vector(centre) - light.location).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)
        made[name] = light
    return made


def floor(centre, ground):
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(centre[0], centre[1], ground))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True
    return plane


def main(shot, folder, width, height, samples):
    data = np.load(shot)
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scene = scene_of(width, height, samples)
    skin_points = data["skin_points"][0]
    bones = mesh_object("bones", data["bone_points"][0], data["bone_faces"], material("bone", BONE, 0.5), False)
    solid = mesh_object("skin", skin_points, data["skin_faces"], opaque_skin(), True)
    clear = mesh_object("glass skin", skin_points, data["skin_faces"], translucent_skin(), True)
    clear.visible_shadow = False
    muscles = muscle_object(data["muscle_points"][0], data["muscle_counts"][0], data["activations"][0])
    look, centre = camera_on(scene, skin_points, data["forward"])
    plane = floor(centre, float(np.min(blender_frame(skin_points)[:, 2])))
    made = lights(scene, look, centre)
    body = (solid, bones, clear, muscles)
    passes = (("skin", (solid,), ("key", "fill", "rim")), ("inside", (bones, clear, muscles), ("key", "fill", "rim")),
              ("shadow", (plane, solid), ("overhead",)))
    for panel, shown, lit in passes:
        for thing in (*body, plane):
            thing.hide_render = thing not in shown
        for name, light in made.items():
            light.hide_render = name not in lit
        solid.visible_camera = panel != "shadow"  # the shadow pass: the skin casts, only the floor is seen
        scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0 if panel == "shadow" else AMBIENT
        scene.render.filepath = str(folder / f"{panel}.png")
        bpy.ops.render.render(write_still=True)
        print(f"teaser render: wrote {scene.render.filepath}", flush=True)

if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:]
    main(arguments[0], arguments[1], int(arguments[2]), int(arguments[3]), int(arguments[4]))
