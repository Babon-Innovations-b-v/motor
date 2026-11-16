"""Runs inside Blender (5.0, Cycles): a demo clip's frames from prepare.py's arrays of every frame of one trial, in the
teaser's look (figures/teaser/render.py: its materials, lights, camera and activation colours). One camera framed on
the skin over the whole clip, so the body moves through a fixed view; per frame the same three passes as the teaser,
`skin`, `inside` and `shadow`, PNG each, transparent behind. A frame already on disk is skipped, so a stopped run
resumes.

    blender -b --python render_clip.py -- <clip.npz> <out folder> <width> <height> <samples> [<part> <parts>]

<part> of <parts> renders every <parts>-th frame from <part>, so several Blenders share one clip.

FARM_CYCLES_GPU=1 renders on the card (OptiX, else CUDA)."""
import pathlib
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "teaser"))
import render as teaser  # noqa: E402

FILL = 0.92  # of the frame's height or width, whichever binds first, that the skin over the whole clip spans
SHADES = 101  # activation steps 0, 0.01, ..., 1: one shared material each, below any visible colour difference


def shade_materials():
    """One muscle material per activation step, coloured by the teaser's RAMP."""
    return [teaser.material(f"muscle{step}", teaser.ramp(step / (SHADES - 1)), 0.4) for step in range(SHADES)]


def muscles_of(points, counts, activations, shades):
    """One curve, a spline a muscle, its material the shade of its activation (teaser.muscle_object's tubes)."""
    curve = bpy.data.curves.new("muscles", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = teaser.MUSCLE_RADIUS
    curve.bevel_resolution = 4
    curve.use_fill_caps = True
    for made in shades:
        curve.materials.append(made)
    start = 0
    for index, count in enumerate(counts):
        mine = points[start:start + count]
        start += count
        if count < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(count - 1)
        placed = np.concatenate([teaser.blender_frame(mine), np.ones((count, 1))], axis=1)
        spline.points.foreach_set("co", placed.astype(np.float32).ravel())
        spline.material_index = int(round(float(np.clip(activations[index], 0, 1)) * (SHADES - 1)))
    thing = bpy.data.objects.new("muscles", curve)
    bpy.context.scene.collection.objects.link(thing)
    return thing


def spread(scene, points):
    """How far the points reach from the frame's centre, as a share of its half-width or half-height (the larger)."""
    seen = np.array([tuple(world_to_camera_view(scene, scene.camera, Vector(point)))[:2] for point in points])
    return float(np.abs(seen - 0.5).max() * 2)


def close_in(scene, skins, centre):
    """Move the camera along its line of sight until the skin over every frame spans FILL of the frame (the teaser
    frames one still; a clip's fixed view holds the whole motion)."""
    points = teaser.blender_frame(skins[::max(len(skins) // 30, 1), ::7].reshape(-1, 3))
    camera = scene.camera
    for _ in range(4):
        bpy.context.view_layer.update()
        reach = spread(scene, points)
        camera.location = Vector(centre) + (camera.location - Vector(centre)) * (reach / FILL)


def move_mesh(thing, points):
    """Set a mesh object's vertices to one frame's points."""
    thing.data.vertices.foreach_set("co", teaser.blender_frame(points).astype(np.float32).ravel())
    thing.data.update()


def drop(thing):
    """Remove a curve object and its data."""
    data = thing.data
    bpy.data.objects.remove(thing)
    bpy.data.curves.remove(data)


def main(clip, folder, width, height, samples, part, parts):
    data = dict(np.load(clip))  # read each array once, not once a frame
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    scene = teaser.scene_of(width, height, samples)
    skins = data["skin_points"]
    bones = teaser.mesh_object("bones", data["bone_points"][0], data["bone_faces"],
                               teaser.material("bone", teaser.BONE, 0.5), False)
    solid = teaser.mesh_object("skin", skins[0], data["skin_faces"], teaser.opaque_skin(), True)
    clear = teaser.mesh_object("glass skin", skins[0], data["skin_faces"], teaser.translucent_skin(), True)
    clear.visible_shadow = False
    shades = shade_materials()
    look, centre = teaser.camera_on(scene, skins.reshape(-1, 3), data["forward"])
    close_in(scene, skins, centre)
    plane = teaser.floor(centre, float(np.min(teaser.blender_frame(skins.reshape(-1, 3))[:, 2])))
    lights = teaser.lights(scene, look, centre)
    passes = (("skin", ("solid",), ("key", "fill", "rim")),
              ("inside", ("bones", "clear", "muscles"), ("key", "fill", "rim")),
              ("shadow", ("plane", "solid"), ("overhead",)))
    for index in range(part, len(skins), parts):
        if all((folder / f"{panel}_{index:04d}.png").exists() for panel, _, _ in passes):
            continue
        move_mesh(bones, data["bone_points"][index])
        move_mesh(solid, skins[index])
        move_mesh(clear, skins[index])
        muscles = muscles_of(data["muscle_points"][index], data["muscle_counts"][index], data["activations"][index],
                             shades)
        things = {"solid": solid, "bones": bones, "clear": clear, "muscles": muscles, "plane": plane}
        for panel, shown, lit in passes:
            for name, thing in things.items():
                thing.hide_render = name not in shown
            for name, light in lights.items():
                light.hide_render = name not in lit
            solid.visible_camera = panel != "shadow"
            scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.0 if panel == "shadow" else \
                teaser.AMBIENT
            scene.render.filepath = str(folder / f"{panel}_{index:04d}.png")
            bpy.ops.render.render(write_still=True)
        drop(muscles)
        print(f"clip render: frame {index + 1} of {len(skins)}", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1:]
    main(arguments[0], arguments[1], int(arguments[2]), int(arguments[3]), int(arguments[4]),
         *(int(argument) for argument in (arguments[5:7] or (0, 1))))
