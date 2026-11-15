"""Bones inside SOMA's skin (#70's judgement, the audit's item 2): every EVERY-th frame, each bone mesh's vertices
(at most VERTS a mesh) tested against SOMA's skin posed on that frame (linear blend, as SOMA skins it) by the
generalised winding number (inside at 0.5 or more: warp's BVH winding query on the CPU, the rule main's harness
measures with), the share outside kept per region (REGIONS, the hand's parts (HAND) one region). The stage's record
and level 1's score (evaluation._measure) both measure through this one module. Level 1's least squares reads the
depth outside instead (`marks_kernel`: each point's distance to the skin's closest point when outside, 0 inside), smooth
where the share is a step (#70, 2026-10-30): the share stays the judged measure. Levels 2 and 3 hold the bones inside
too (#70, 2026-11-01, the owner: "we just know bones need to be inside the skin"): each kept bone vertex's depth outside
the frame's posed skin (`PosedSkins`, MuJoCo's frame) times `skin_cost` is the residual of both solves, as its offset
off the skin's closest point (`apart_kernel`: the same winding query, the closest point on the face it finds in
float64), whose Jacobian in the point is the projector off the closest feature (the face's normal, an edge's plane, a
corner's every axis): so a solve sees the skin's bend at its edges and corners, where a depth row's own Gauss-Newton
square misses it. Nothing here moves SOMA."""
import hashlib

import mujoco
import numpy as np
import warp

from motor.model import TO_OWN

EVERY = 10  # frames between the skin measures
VERTS = 80  # bone vertices kept per mesh
# The kneecap is not held in the skin (#98, 2026-11-07): it rides the quadriceps' tendon where MyoFullBody's couplings
# put it (sizing), and SOMA's linear-blend skin caves up to 17 mm in at the front of a bent knee; held there, the fit
# turned the knee to keep it in (the leg angles worse against a lab reference).
KNEECAP = "patella"
REGIONS = {"shoulder": ("clavicle", "scapula", "humerus"), "forearm": ("ulna", "radius"),
           "fingers": ("ph_", "thumb"), "metacarpals": ("mc",),
           "carpals": ("lunate", "scaphoid", "pisiform", "triquetrum", "capitate", "trapezium", "trapezoid", "hamate"),
           "pelvis": ("pelvis", "sacrum"), "spine": ("lumbar", "thorax", "torso", "cervical", "neck", "rib"),
           "head": ("head", "skull", "jaw", "mandible"), "thigh": ("femur", "patella"), "shin": ("tibia", "fibula"),
           "foot": ("talus", "calcn", "toes")}
HAND = ("fingers", "metacarpals", "carpals")  # the hand's parts, scored together as "hand"
# m: a face whose outward normal meets the skin again this near is a pocket's wall. 0.08 since 2026-11-06 (#70): at 0.03
# the mouth was its lips' slit alone (64 vertices, 7 mm tall) and SOMA's oral cavity behind it (to 103 mm deep) stood
# open, so the teeth, anatomically inside it, read outside; the head's inside read and level 3's skin rows held the
# skull 25 mm high and 10 degrees chin up to keep them out. From 0.08 to 0.10 the mouth is one pocket of 158 vertices
# and the eyes, nostrils and ears as at 0.06.
POCKET_REACH = 0.08
POCKET_SEGMENT = "Head"  # the segment whose pockets a closed face covers (lips, eyelids and eyes, nostrils, ears)
# The bones-in-skin goal of level 3 against the turns' 1 (`_frames.weights`): its rows sum to SKIN_GOAL times the
# mean over the kept bone vertices of their squared depth outside (m), each goal over its own count as OpenSim weighs
# its goals. Chosen from measurement (#70, 2026-11-01), never set to pass a score: on mean's 15 clips with the skin rows
# in level 3 only, bones outside 11.45 % (none) -> 9.57 / 6.95 / 4.88 % at 1e2 / 1e3 / 1e4, the turns 4.662 ->
# 4.61 / 4.603 / 4.735 deg: 1e3 the largest goal whose turns are no worse; env1's seated and standing astronauts read
# fewer bones outside at it too (fit/CLAUDE.md). Level 2 holds no skin rows: its polish did not settle with them.
SKIN_GOAL = 1.0e3
# A frame whose posed skin folds into itself over FOLD_MOST of its area (a deep squat's or a kneel's skin folding into
# itself, which no skeleton can follow) is left out of the skin rows of levels 1 and 3 and out of the bones-outside
# share, and reported apart (the owner, 2026-11-01). A skin vertex lies inside itself when the point FOLD_IN in from it
# along its normal winds 1.5 or more; it folds when that point winds FOLD_OWN or more on the skin across its own
# segment's joints alone (`joined_faces`: its segment, its parent and its children; a knee's shin into its thigh), not
# where it presses into a segment further off: a heavy body's upper arm into its chest past the shoulder, a hand on the
# thigh (#70, 2026-11-01: by the whole skin l1h's five heavy standing clips read 0.5 to 2.8 % inside itself on every
# frame, their rest pose 0.007 % at most, so every frame folded and their bones-outside share was NaN; by the joined
# skin 0.00 to 0.17 %, none folded). On the mean body's 15 clips (every 10th frame, main's fit) 81 of 281 frames
# folded by the whole skin (their bones 24.8 % outside against 16.0 % on the rest), 52 by the joined skin: every squat
# and kneel frame over 0.7 % still folds (the three dropped, 0.55 to 0.68 %, are their clips' standing ends), the
# arms-down frames of arms_raise and reach no longer (their upper arms into the chest).
FOLD_MOST = 0.005
FOLD_IN = 0.001  # m
FOLD_OWN = 1.5  # the winding on the joined skin at which a doubly inside point folds rather than presses

warp.config.quiet = True


@warp.kernel
def marks_kernel(meshes: warp.array(dtype=warp.uint64), points: warp.array2d(dtype=warp.vec3),
           signs: warp.array2d(dtype=float), depths: warp.array2d(dtype=float)):
    """Per (skin frame, point): the winding sign (outside at 0 or more) and the depth outside (m, 0 inside); the card's
    measure launches this same kernel."""
    shot, point = warp.tid()
    found = warp.mesh_query_point_sign_winding_number(meshes[shot], points[shot, point], 1.0e6, 2.0, 0.5)
    closest = warp.mesh_eval_position(meshes[shot], found.face, found.u, found.v)
    signs[shot, point] = found.sign
    depths[shot, point] = warp.where(found.sign >= 0.0, warp.length(points[shot, point] - closest), 0.0)


@warp.func
def _away(direction: warp.vec3d):
    """The projector off a feature along `direction` (an edge's): I - e e^T."""
    unit = warp.normalize(direction)
    return warp.identity(n=3, dtype=warp.float64) - warp.outer(unit, unit)


@warp.func
def closest_on_triangle(point: warp.vec3d, first: warp.vec3d, second: warp.vec3d, third: warp.vec3d):
    """The triangle's closest point to `point` in float64 (Ericson, Real-Time Collision Detection, 5.1.5) and the
    Hessian of half the squared distance to it there: the projector off the closest feature's span (n n^T on the face,
    I - e e^T on an edge, I at a corner)."""
    along = second - first
    across = third - first
    off = point - first
    d1 = warp.dot(along, off)
    d2 = warp.dot(across, off)
    found = first
    bend = warp.identity(n=3, dtype=warp.float64)
    if d1 > warp.float64(0.0) or d2 > warp.float64(0.0):
        off_second = point - second
        d3 = warp.dot(along, off_second)
        d4 = warp.dot(across, off_second)
        off_third = point - third
        d5 = warp.dot(along, off_third)
        d6 = warp.dot(across, off_third)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        if d3 >= warp.float64(0.0) and d4 <= d3:
            found = second
        elif vc <= warp.float64(0.0) and d1 >= warp.float64(0.0) and d3 <= warp.float64(0.0):
            found = first + along * (d1 / (d1 - d3))
            bend = _away(along)
        elif d6 >= warp.float64(0.0) and d5 <= d6:
            found = third
        elif vb <= warp.float64(0.0) and d2 >= warp.float64(0.0) and d6 <= warp.float64(0.0):
            found = first + across * (d2 / (d2 - d6))
            bend = _away(across)
        elif va <= warp.float64(0.0) and d4 - d3 >= warp.float64(0.0) and d5 - d6 >= warp.float64(0.0):
            found = second + (third - second) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
            bend = _away(third - second)
        else:
            scale = warp.float64(1.0) / (va + vb + vc)
            found = first + along * (vb * scale) + across * (vc * scale)
            normal = warp.normalize(warp.cross(along, across))
            bend = warp.outer(normal, normal)
    return found, bend


@warp.func
def apart_of(mesh: warp.uint64, verts: warp.array(dtype=warp.vec3d), first_vert: int,
             faces: warp.array(dtype=warp.int32), ring_starts: warp.array(dtype=warp.int32),
             ring_faces: warp.array(dtype=warp.int32), point: warp.vec3d):
    """A point's offset off its skin (zero inside by the winding number, `marks_kernel`'s query on the float32 mesh)
    and the offset's Jacobian in the point (`closest_on_triangle`'s projector, zero inside). The closest point is the
    nearest, in float64, of the face the query finds and every face touching its corners (`rings`: the faces at vertex
    v are `ring_faces[ring_starts[v]:ring_starts[v + 1]]`; `verts` the mesh's float64 points from `first_vert`): the
    query's float32 face flips between near neighbours, so the depth to its face alone jumped there (mean's sampled
    frames, 2026-11-01: a lumbar inside-joint column 2.95e-4 off its central difference at a 1e-6 step, 1.4e-8 with
    the ring). What stays is the depth's kink where a point crosses the skin: a residual max(0, d), whose square, the
    cost, is smooth."""
    near = warp.vec3(warp.float32(point[0]), warp.float32(point[1]), warp.float32(point[2]))
    found = warp.mesh_query_point_sign_winding_number(mesh, near, 1.0e6, 2.0, 0.5)
    offset = warp.vec3d()
    bend = warp.mat33d()
    if found.sign >= 0.0:
        best = warp.float64(1.0e30)
        for corner in range(3):
            vert = faces[3 * found.face + corner]
            for slot in range(ring_starts[vert], ring_starts[vert + 1]):
                face = ring_faces[slot]
                closest, projector = closest_on_triangle(point, verts[first_vert + faces[3 * face]],
                                                         verts[first_vert + faces[3 * face + 1]],
                                                         verts[first_vert + faces[3 * face + 2]])
                apart = point - closest
                length = warp.dot(apart, apart)
                if length < best:
                    best = length
                    offset = apart
                    bend = projector
    return offset, bend


def rings(faces, count):
    """Each of `count` vertices' faces (`faces` faces x 3) as (starts count + 1, faces), int32: `apart_of`'s rings."""
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    order = np.argsort(faces.ravel(), kind="stable")
    starts = np.searchsorted(faces.ravel()[order], np.arange(count + 1))
    return starts.astype(np.int32), (order // 3).astype(np.int32)


@warp.kernel
def apart_kernel(meshes: warp.array(dtype=warp.uint64), first_verts: warp.array(dtype=warp.int32),
                 kept: warp.array(dtype=warp.int32), verts: warp.array(dtype=warp.vec3d),
                 faces: warp.array(dtype=warp.int32),
                 ring_starts: warp.array(dtype=warp.int32), ring_faces: warp.array(dtype=warp.int32),
                 mesh_of: warp.array(dtype=warp.int32), points: warp.array(dtype=warp.vec3d),
                 apart: warp.array(dtype=warp.vec3d), bends: warp.array(dtype=warp.mat33d)):
    """Per point: its offset off its skin (`apart_of`; `mesh_of` its mesh), its length the depth, and that offset's
    Jacobian in the point (the projector off the closest feature); zero on a skin not `kept` (`unfolded`)."""
    point = warp.tid()
    mesh = mesh_of[point]
    if kept[mesh] != 0:
        offset, bend = apart_of(meshes[mesh], verts, first_verts[mesh], faces, ring_starts, ring_faces,
                                points[point])
        apart[point] = offset
        bends[point] = bend


def region(body):
    """The region a body's bone belongs to (the hand's parts each their own), "other" when none."""
    return next((label for label, parts in REGIONS.items() if any(part in body for part in parts)), "other")


def scored(part):
    """The region a part is scored in: the hand's parts as "hand"."""
    return "hand" if part in HAND else part


def bone_meshes(model):
    """[(geom, vertex indices kept, its body's name)] of every bone mesh but the kneecaps' (KNEECAP)."""
    found = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and KNEECAP not in model.body(model.geom_bodyid[geom]).name:
            count = int(model.mesh_vertnum[model.geom_dataid[geom]])
            kept = np.linspace(0, count - 1, VERTS).astype(int) if count > VERTS else np.arange(count)
            found.append((geom, kept, model.body(model.geom_bodyid[geom]).name))
    return found


def bone_points(model, data, meshes):
    """The kept bone vertices in the person's frame (Y up) and each one's body name."""
    points, labels = [], []
    for geom, kept, label in meshes:
        start = model.mesh_vertadr[model.geom_dataid[geom]]
        verts = model.mesh_vert[start + kept]
        points.append((verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]) @ TO_OWN.T)
        labels += [label] * len(kept)
    return np.concatenate(points), np.array(labels)


def skin(rig, segments, world):
    """SOMA's mesh posed on one frame (`world` segments x 4 x 4), linear blend over the segments' weights."""
    names = [str(joint) for joint in rig["names"]]
    joints = [names.index(segment) for segment in segments]
    weights = np.asarray(rig["weights"], dtype=float)[:, joints]
    moves = world @ np.linalg.inv(np.asarray(rig["bind"], dtype=float)[joints])
    points = np.asarray(rig["points"], dtype=float)
    moved = np.einsum("sij,vj->vsi", moves[:, :3, :3], points) + moves[None, :, :3, 3]
    return np.einsum("vs,vsi->vi", weights, moved)


def marks(points, verts, faces):
    """(whether each point lies outside the closed mesh (verts, faces) by the winding number, its depth outside in m)."""
    return marks_in(points, closed_mesh(verts, faces))


def closed_mesh(verts, faces):
    """The closed mesh (verts, faces) as `marks_in` queries it (Warp's, its winding number on, on the processor): built
    once for every query of the same mesh (#70: sizing's kneecap built it again for each of a bend's passes)."""
    return warp.Mesh(points=warp.array(np.asarray(verts, dtype=np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(np.asarray(faces, dtype=np.int32).ravel(), dtype=int, device="cpu"),
                     support_winding_number=True)


def marks_in(points, mesh):
    """`marks` of `points` against `mesh` (`closed_mesh`)."""
    signs, depths = (warp.zeros((1, len(points)), dtype=float, device="cpu") for _ in range(2))
    warp.launch(marks_kernel, dim=signs.shape, device="cpu",
                inputs=[warp.array([mesh.id], dtype=warp.uint64, device="cpu"),
                        warp.array(np.asarray(points, dtype=np.float32)[None], dtype=warp.vec3, device="cpu"),
                        signs, depths])
    return signs.numpy()[0] >= 0, depths.numpy()[0].astype(float)


def depth_mm(depths):
    """Level 1's skin residual of one region: the root mean square of its points' depths outside (m given, mm out)."""
    return 1000 * float(np.sqrt(np.mean(np.square(depths))))


def outside(points, verts, faces):
    """Whether each point lies outside the closed mesh (verts, faces) by the winding number."""
    return marks(points, verts, faces)[0]


def pockets(rig):
    """The head's pockets in SOMA's mesh at its bind ([vertex indices] each): connected faces whose outward normal meets
    the mesh again within POCKET_REACH (a pocket's walls face each other), most of whose vertices ride POCKET_SEGMENT
    by their largest skin weight. On the template (2026-11-06): the mouth (its lips' slit and the oral cavity behind
    it, 71 x 71 x 103 mm, 158 vertices), each eye socket (28 x 21 x 24 mm), nostril and ear; the webs between fingers and toes face each other too, but no head."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points, faces = np.asarray(rig["points"], dtype=np.float64), np.asarray(rig["faces"], dtype=np.int32)
    first, second, third = (points[faces[:, corner]] for corner in range(3))
    normals = np.cross(second - first, third - first)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    middles = (first + second + third) / 3
    mesh = warp.Mesh(points=warp.array(points.astype(np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(faces.ravel(), dtype=int, device="cpu"))
    hits = warp.zeros(len(faces), dtype=float, device="cpu")
    warp.launch(_pocket_rays, dim=len(faces), device="cpu",
                inputs=[mesh.id, warp.array((middles + 1e-4 * normals).astype(np.float32), dtype=warp.vec3, device="cpu"),
                        warp.array(normals.astype(np.float32), dtype=warp.vec3, device="cpu"), hits])
    walls = np.flatnonzero(hits.numpy() < POCKET_REACH)
    shared = coo_matrix((np.ones(3 * len(walls)), (np.repeat(walls, 3), faces[walls].ravel())),
                        shape=(len(faces), len(points))).tocsr()
    touching = (shared @ shared.T).tocoo()
    _, label = connected_components(touching, directed=False)
    head = [str(name) for name in rig["names"]].index(POCKET_SEGMENT)
    riders = np.argmax(np.asarray(rig["weights"]), axis=1) == head
    found = []
    for part in np.unique(label[walls]):
        verts = np.unique(faces[walls[label[walls] == part]])
        if len(verts) >= 4 and riders[verts].mean() > 0.5:
            found.append(verts)
    return found


@warp.kernel
def _pocket_rays(mesh: warp.uint64, starts: warp.array(dtype=warp.vec3), directions: warp.array(dtype=warp.vec3),
                 hits: warp.array(dtype=float)):
    face = warp.tid()
    found = warp.mesh_query_ray(mesh, starts[face], directions[face], 1.0)
    hits[face] = warp.where(found.result, found.t, 1.0e6)


ENVELOPES = {}  # {the rig's points and faces' digest: the faces with the head's pockets closed}


def envelope(rig):
    """SOMA's faces with each of the head's `pockets` closed by its convex hull as a shell of its own (its faces after
    the skin's, outward, over the pocket's own vertices, so it moves with them; the skin's face indices unchanged): the
    winding number in a closed pocket is then 1, inside, as under a closed face (lips, eyelids and eyes, nostrils,
    ears), and a point's depth outside is to the face's skin or the pocket's lid. The inside test's faces
    (owner, 2026-11-01: the teeth read outside in every frame; 49 of a frame's 827 outside jaw and skull vertices sat in
    the mouth's slit, 22 in the eye sockets, the other 756 out of the face itself)."""
    from scipy.spatial import ConvexHull
    faces = np.asarray(rig["faces"], dtype=np.int32)
    key = hashlib.sha1(np.ascontiguousarray(rig["points"], dtype=np.float64).tobytes() + faces.tobytes()).hexdigest()
    if key not in ENVELOPES:
        points, lids = np.asarray(rig["points"], dtype=np.float64), [faces]
        for verts in pockets(rig):
            hull = ConvexHull(points[verts])
            shell = verts[hull.simplices]
            outward = np.einsum("ij,ij->i", np.cross(points[shell[:, 1]] - points[shell[:, 0]],
                                                     points[shell[:, 2]] - points[shell[:, 0]]), hull.equations[:, :3])
            shell[outward < 0] = shell[outward < 0][:, [0, 2, 1]]
            lids.append(shell)
        ENVELOPES[key] = np.concatenate(lids).astype(np.int32)
    return ENVELOPES[key]


@warp.kernel
def _doubled(mesh: warp.uint64, points: warp.array(dtype=warp.vec3), found: warp.array(dtype=float)):
    """1 where a point winds 1.5 or more (inside the skin twice), else 0."""
    point = warp.tid()
    query = warp.mesh_query_point_sign_winding_number(mesh, points[point], 1.0e6, 2.0, 1.5)
    found[point] = warp.where(query.sign < 0.0, 1.0, 0.0)


def doubled(verts, faces):
    """Whether each vertex of a posed skin (`verts`, its own `faces`, no pockets closed) lies inside itself: the point
    FOLD_IN in along its normal winds 1.5 or more; and each vertex's area (a third of its faces')."""
    verts, faces = np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)
    normals = np.zeros_like(verts)
    crossed = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], crossed)
    area = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(area, 1e-30)[:, None]
    mesh = warp.Mesh(points=warp.array(verts.astype(np.float32), dtype=warp.vec3, device="cpu"),
                     indices=warp.array(faces.ravel(), dtype=int, device="cpu"), support_winding_number=True)
    found = warp.zeros(len(verts), dtype=float, device="cpu")
    warp.launch(_doubled, dim=len(verts), device="cpu",
                inputs=[mesh.id, warp.array((verts - FOLD_IN * normals).astype(np.float32), dtype=warp.vec3,
                                            device="cpu"), found])
    return found.numpy() > 0, area, verts - FOLD_IN * normals


def winding(points, triangles):
    """Each point's generalised winding number on the triangles (n x 3 x 3), exact in float64 (Van Oosterom and
    Strackee's solid angle, summed): open sheets give fractions."""
    total = np.zeros(len(points))
    for start in range(0, len(points), 64):
        corners = triangles[None, :, :, :] - points[start:start + 64, None, None, :]
        first, second, third = corners[:, :, 0], corners[:, :, 1], corners[:, :, 2]
        lengths = [np.linalg.norm(corner, axis=-1) for corner in (first, second, third)]
        turned = np.einsum("pfi,pfi->pf", first, np.cross(second, third))
        under = (lengths[0] * lengths[1] * lengths[2] + np.einsum("pfi,pfi->pf", first, second) * lengths[2]
                 + np.einsum("pfi,pfi->pf", second, third) * lengths[0]
                 + np.einsum("pfi,pfi->pf", third, first) * lengths[1])
        total[start:start + 64] = (2 * np.arctan2(turned, under)).sum(axis=1) / (4 * np.pi)
    return total


JOINED = {}  # {skin key: (each vertex's segment, each segment's joined faces mask)}, a rig's own


def joined_faces(rig):
    """(each skin vertex's segment, its heaviest skin weight; [segments x faces] whether a face's segment (its first
    corner's) is that segment, its parent or a child of it: the skin across one joint), once per rig."""
    faces = np.asarray(rig["faces"], dtype=np.int32)
    weights = np.asarray(rig["weights"], dtype=float)
    parents = np.asarray(rig["parents"], dtype=np.int64)
    key = hashlib.sha1(np.ascontiguousarray(weights).tobytes() + faces.tobytes() + parents.tobytes()).hexdigest()
    if key not in JOINED:
        joints = np.arange(len(parents))
        joined = ((joints[:, None] == joints[None, :]) | (parents[:, None] == joints[None, :])
                  | (joints[:, None] == parents[None, :]))
        owner = weights.argmax(axis=1)
        JOINED[key] = owner, joined[:, owner[faces[:, 0]]]
    return JOINED[key]


def folded(verts, rig):
    """The share of a posed skin's area (`verts`, the rig's own `faces`, no pockets closed) folding into itself: each
    vertex by its area, a vertex lying inside itself (`doubled`) whose inward point winds FOLD_OWN or more on the skin
    across its segment's joints (`joined_faces`: a knee's shin into its thigh); one pressing into a segment further off
    (an upper arm into the chest past its shoulder, a hand on the thigh) is contact, not a fold."""
    faces = np.asarray(rig["faces"], dtype=np.int32)
    inside, area, points = doubled(verts, faces)
    owner, joined = joined_faces(rig)
    verts = np.asarray(verts, dtype=np.float64)
    folds = np.zeros(len(verts), dtype=bool)
    for segment in np.unique(owner[inside]):
        mine = np.flatnonzero(inside & (owner == segment))
        folds[mine] = winding(points[mine], verts[faces[joined[segment]]]) >= FOLD_OWN
    return float(area[folds].sum() / area.sum())


def unfolded(verts, rig):
    """Whether a posed skin is kept in the skin rows and the share: it folds into itself over at most FOLD_MOST of its
    area (`folded`)."""
    return folded(verts, rig) <= FOLD_MOST


def bones_outside(model, path, worlds, segments, rig):
    """({"all": %, region: %} of the bone vertices outside SOMA's posed skin over every EVERY-th frame whose skin is
    `unfolded`, {"frames": the folded ones' count, "all": their own %, "no unfolded frame": whether every one folds})
    (`FOLD_MOST`). With no unfolded frame the shares are {}: no measure, never 0 or NaN."""
    data, meshes, shares, apart = mujoco.MjData(model), bone_meshes(model), {}, []
    for qpos, world in list(zip(path, worlds))[::EVERY]:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        points, bones = bone_points(model, data, meshes)
        labels = np.array([scored(region(bone)) for bone in bones])
        posed = skin(rig, segments, world)
        out = outside(points, posed, envelope(rig))
        if not unfolded(posed, rig):
            apart.append(float(out.mean()))
            continue
        shares.setdefault("all", []).append(float(out.mean()))
        for label in sorted(set(labels)):  # sorted: a set of strings orders by the process's hash seed
            shares.setdefault(label, []).append(float(out[labels == label].mean()))
    return ({label: round(100 * float(np.mean(found)), 2) for label, found in shares.items()},
            {"frames": len(apart), "all": round(100 * float(np.mean(apart)), 2) if apart else 0.0,
             "no unfolded frame": not shares})


class FrameSkins:
    """SOMA's skin posed on each frame of `worlds` (`skin`, the head's pockets closed: `envelope`), each a winding
    mesh built once, against which `outside` marks a path's kept bone vertices on the same frames, every frame at
    once; `labels` is each vertex's scored region. RRA's frame check (`rra.within_frames`) reads it: a move that puts
    bones out of the skin the fitted path kept them in is lowered on that frame (#70, 2026-11-05)."""

    def __init__(self, model, rig, segments, worlds):
        self.device = card_device()  # every frame's marks at once on the card where there is one (#70, 2026-11-05)
        faces = warp.array(envelope(rig).ravel(), dtype=int, device=self.device)
        self.model, self.data, self.bones = model, mujoco.MjData(model), bone_meshes(model)
        self.meshes = [warp.Mesh(points=warp.array(skin(rig, segments, world).astype(np.float32), dtype=warp.vec3,
                                                   device=self.device), indices=faces, support_winding_number=True)
                       for world in np.asarray(worlds, dtype=float)]
        self.labels = np.array([scored(region(name)) for _, kept, name in self.bones for _ in kept])

    def outside(self, path):
        """Whether each kept bone vertex lies outside its frame's skin on `path` (MuJoCo's qpos on `model`, one a
        frame of `worlds`): frames x vertices."""
        points = []
        for qpos in path:
            self.data.qpos[:] = qpos
            mujoco.mj_kinematics(self.model, self.data)
            points.append(bone_points(self.model, self.data, self.bones)[0])
        signs, depths = (warp.zeros((len(points), len(points[0])), dtype=float, device=self.device)
                         for _ in range(2))
        warp.launch(marks_kernel, dim=signs.shape, device=self.device,
                    inputs=[warp.array([mesh.id for mesh in self.meshes], dtype=warp.uint64, device=self.device),
                            warp.array(np.asarray(points, dtype=np.float32), dtype=warp.vec3, device=self.device),
                            signs, depths])
        return signs.numpy() >= 0


class PosedSkins:
    """SOMA's skin posed on each frame in MuJoCo's frame (`verts` frames x points x 3, float32), its `faces` and which
    frames are `kept` (`unfolded`; all when not given): the target level 3 holds the bones inside, a frame not
    kept giving no rows. Indexed by frames like an array; `joined` puts several end to end."""

    def __init__(self, verts, faces, kept=None):
        self.verts = np.asarray(verts, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.kept = np.ones(len(self.verts), dtype=bool) if kept is None else np.asarray(kept, dtype=bool)

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, frames):
        return PosedSkins(self.verts[frames], self.faces, self.kept[frames])

    @staticmethod
    def joined(parts):
        """`parts` end to end, refused when their faces differ (one person's mesh)."""
        parts = list(parts)
        if any(not np.array_equal(part.faces, parts[0].faces) for part in parts):
            raise ValueError("posed skins of different meshes")
        return PosedSkins(np.concatenate([part.verts for part in parts]), parts[0].faces,
                          np.concatenate([part.kept for part in parts]))


def posed_skins(rig, segments, worlds):
    """SOMA's skin on every frame of `worlds` (`skin`, linear blend) in MuJoCo's frame, each kept when `unfolded`: on
    the card when Warp sees one, every frame at once (`card_skins`, `card_folded`), else frame by frame on the CPU, the
    reference the card's equals within float64's rounding (#70, 2026-11-05: 19 to 23 s a clip on the CPU)."""
    device = card_device()
    if device == "cpu":
        own = np.stack([skin(rig, segments, world) for world in np.asarray(worlds, dtype=float)])
        return PosedSkins(own @ TO_OWN, envelope(rig), [unfolded(posed, rig) for posed in own])
    own = card_skins(rig, segments, worlds, device)
    return PosedSkins(own.numpy() @ TO_OWN, envelope(rig), card_folded(own, rig, device) <= FOLD_MOST)


@warp.kernel
def _blended(moves: warp.array2d(dtype=warp.mat44d), weights: warp.array2d(dtype=warp.float64),
             points: warp.array(dtype=warp.vec3d), posed: warp.array2d(dtype=warp.vec3d)):
    """Per (frame, vertex): `skin`'s linear blend, each segment's move of the vertex weighed by its weight."""
    frame, vert = warp.tid()
    total = warp.vec3d()
    for segment in range(weights.shape[1]):
        move = moves[frame, segment]
        point = points[vert]
        moved = warp.vec3d(move[0, 0] * point[0] + move[0, 1] * point[1] + move[0, 2] * point[2] + move[0, 3],
                           move[1, 0] * point[0] + move[1, 1] * point[1] + move[1, 2] * point[2] + move[1, 3],
                           move[2, 0] * point[0] + move[2, 1] * point[1] + move[2, 2] * point[2] + move[2, 3])
        total = total + weights[vert, segment] * moved
    posed[frame, vert] = total


def card_skins(rig, segments, worlds, device):
    """`skin` of every frame of `worlds` at once on `device` (frames x points, Warp's vec3d): float64, as the CPU's."""
    names = [str(joint) for joint in rig["names"]]
    joints = [names.index(segment) for segment in segments]
    moves = np.asarray(worlds, dtype=float) @ np.linalg.inv(np.asarray(rig["bind"], dtype=float)[joints])
    weights = np.ascontiguousarray(np.asarray(rig["weights"], dtype=float)[:, joints])
    points = np.asarray(rig["points"], dtype=float)
    posed = warp.zeros((len(moves), len(points)), dtype=warp.vec3d, device=device)
    warp.launch(_blended, dim=posed.shape, device=device,
                inputs=[warp.array(np.ascontiguousarray(moves), dtype=warp.mat44d, device=device),
                        warp.array(weights, dtype=warp.float64, device=device),
                        warp.array(points, dtype=warp.vec3d, device=device), posed])
    return posed


@warp.kernel
def _inward(posed: warp.array2d(dtype=warp.vec3d), faces: warp.array(dtype=warp.int32),
            ring_starts: warp.array(dtype=warp.int32), ring_faces: warp.array(dtype=warp.int32),
            areas: warp.array2d(dtype=warp.float64), inward: warp.array2d(dtype=warp.vec3d),
            inward32: warp.array2d(dtype=warp.vec3), posed32: warp.array2d(dtype=warp.vec3), fold_in: warp.float64):
    """Per (frame, vertex): `doubled`'s area (its faces' crossed sides summed, their length) and its point FOLD_IN in
    along its normal, float64 and float32, and the vertex in float32 for the frame's mesh."""
    frame, vert = warp.tid()
    summed = warp.vec3d()
    for slot in range(ring_starts[vert], ring_starts[vert + 1]):
        face = ring_faces[slot]
        first = posed[frame, faces[3 * face]]
        summed = summed + warp.cross(posed[frame, faces[3 * face + 1]] - first, posed[frame, faces[3 * face + 2]] - first)
    area = warp.length(summed)
    point = posed[frame, vert] - fold_in * (summed / warp.max(area, warp.float64(1.0e-30)))
    areas[frame, vert] = area
    inward[frame, vert] = point
    inward32[frame, vert] = warp.vec3(warp.float32(point[0]), warp.float32(point[1]), warp.float32(point[2]))
    here = posed[frame, vert]
    posed32[frame, vert] = warp.vec3(warp.float32(here[0]), warp.float32(here[1]), warp.float32(here[2]))


@warp.kernel
def _doubled_frames(meshes: warp.array(dtype=warp.uint64), points: warp.array2d(dtype=warp.vec3),
                    found: warp.array2d(dtype=warp.int32)):
    """`_doubled` on every frame's mesh at once: 1 where a point winds 1.5 or more."""
    frame, point = warp.tid()
    query = warp.mesh_query_point_sign_winding_number(meshes[frame], points[frame, point], 1.0e6, 2.0, 1.5)
    found[frame, point] = warp.where(query.sign < 0.0, 1, 0)


@warp.kernel
def _joined_winding(posed: warp.array2d(dtype=warp.vec3d), inward: warp.array2d(dtype=warp.vec3d),
                    faces: warp.array(dtype=warp.int32), frames: warp.array(dtype=warp.int32),
                    verts: warp.array(dtype=warp.int32), owners: warp.array(dtype=warp.int32),
                    starts: warp.array(dtype=warp.int32), joined: warp.array(dtype=warp.int32),
                    windings: warp.array(dtype=warp.float64)):
    """Per doubled (frame, vertex): `winding` of its inward point on its owner's `joined_faces`, exact in float64, times
    4 pi (the host divides, as `winding` does)."""
    pair = warp.tid()
    frame = frames[pair]
    point = inward[frame, verts[pair]]
    owner = owners[verts[pair]]
    total = warp.float64(0.0)
    for slot in range(starts[owner], starts[owner + 1]):
        face = joined[slot]
        first = posed[frame, faces[3 * face]] - point
        second = posed[frame, faces[3 * face + 1]] - point
        third = posed[frame, faces[3 * face + 2]] - point
        lengths = warp.vec3d(warp.length(first), warp.length(second), warp.length(third))
        turned = warp.dot(first, warp.cross(second, third))
        under = (lengths[0] * lengths[1] * lengths[2] + warp.dot(first, second) * lengths[2]
                 + warp.dot(second, third) * lengths[0] + warp.dot(third, first) * lengths[1])
        total = total + warp.float64(2.0) * warp.atan2(turned, under)
    windings[pair] = total


def card_folded(posed, rig, device):
    """`folded` of every frame of `posed` (`card_skins`) at once on `device`: each frame's posed skin as a mesh built
    as the CPU builds it (SAH, so the winding number's approximation reads the same tree), its doubled vertices, then
    their exact winding on their segment's joined faces. Returns each frame's share, float64."""
    faces = np.asarray(rig["faces"], dtype=np.int32)
    frames, count = posed.shape
    starts, ringed = rings(faces, count)
    flat = warp.array(faces.ravel(), dtype=warp.int32, device=device)
    areas = warp.zeros((frames, count), dtype=warp.float64, device=device)
    inward = warp.zeros((frames, count), dtype=warp.vec3d, device=device)
    inward32, posed32 = (warp.zeros((frames, count), dtype=warp.vec3, device=device) for _ in range(2))
    warp.launch(_inward, dim=(frames, count), device=device,
                inputs=[posed, flat, warp.array(starts, dtype=warp.int32, device=device),
                        warp.array(ringed, dtype=warp.int32, device=device), areas, inward, inward32, posed32,
                        FOLD_IN])
    indices = warp.array(faces.ravel(), dtype=int, device=device)
    meshes = [warp.Mesh(points=posed32[frame], indices=indices, support_winding_number=True, bvh_constructor="sah")
              for frame in range(frames)]
    found = warp.zeros((frames, count), dtype=warp.int32, device=device)
    warp.launch(_doubled_frames, dim=(frames, count), device=device,
                inputs=[warp.array(np.array([mesh.id for mesh in meshes], dtype=np.uint64), dtype=warp.uint64,
                                   device=device), inward32, found])
    pair_frames, pair_verts = np.nonzero(found.numpy())
    area = areas.numpy()
    folds = np.zeros((frames, count), dtype=bool)
    if len(pair_frames):
        owner, joined = joined_faces(rig)
        lists = [np.flatnonzero(mask).astype(np.int32) for mask in joined]
        windings = warp.zeros(len(pair_frames), dtype=warp.float64, device=device)
        warp.launch(_joined_winding, dim=len(pair_frames), device=device,
                    inputs=[posed, inward, flat, warp.array(pair_frames.astype(np.int32), dtype=warp.int32, device=device),
                            warp.array(pair_verts.astype(np.int32), dtype=warp.int32, device=device),
                            warp.array(owner.astype(np.int32), dtype=warp.int32, device=device),
                            warp.array(np.cumsum([0, *map(len, lists)]).astype(np.int32), dtype=warp.int32,
                                       device=device),
                            warp.array(np.concatenate(lists), dtype=warp.int32, device=device), windings])
        folds[pair_frames, pair_verts] = windings.numpy() / (4 * np.pi) >= FOLD_OWN
    return (area * folds).sum(axis=1) / area.sum(axis=1)


def bone_bodies(model):
    """Every kept bone vertex (`bone_meshes`' order) as (its body's id, its place in the body's frame): points x 3."""
    bodies, places = [], []
    for geom, kept, _ in bone_meshes(model):
        start = model.mesh_vertadr[model.geom_dataid[geom]]
        turn = np.zeros(9)
        mujoco.mju_quat2Mat(turn, model.geom_quat[geom])
        places.append(model.mesh_vert[start + kept].astype(float) @ turn.reshape(3, 3).T + model.geom_pos[geom])
        bodies += [int(model.geom_bodyid[geom])] * len(kept)
    return np.array(bodies, dtype=int), (np.concatenate(places) if places else np.zeros((0, 3)))


def skin_cost(count):
    """One bone vertex's residual weight: SKIN_GOAL over the `count` kept vertices, as a cost multiplies a depth."""
    return float(np.sqrt(SKIN_GOAL / max(count, 1)))


def card_device():
    """The card when Warp sees one, else the CPU."""
    return "cuda:0" if warp.is_cuda_available() else "cpu"


class SkinMeshes:
    """A `PosedSkins`' frames as Warp meshes (exact winding numbers) on `device`, with their float64 points (`verts`,
    each frame's from `first_verts`) for `apart`."""

    def __init__(self, skins, device):
        self.device = device
        faces = warp.array(skins.faces.ravel(), dtype=int, device=device)
        self.meshes = [warp.Mesh(points=warp.array(verts, dtype=warp.vec3, device=device), indices=faces,
                                 support_winding_number=True) for verts in skins.verts]
        self.ids = warp.array(np.array([mesh.id for mesh in self.meshes], dtype=np.uint64), dtype=warp.uint64,
                              device=device)
        count = skins.verts.shape[1] if len(skins) else 0
        self.first_verts = warp.array(np.arange(len(skins), dtype=np.int32) * count, dtype=warp.int32, device=device)
        self.verts = warp.array(skins.verts.reshape(-1, 3).astype(np.float64), dtype=warp.vec3d, device=device)
        self.faces = warp.array(skins.faces.ravel(), dtype=warp.int32, device=device)
        self.kept = warp.array(skins.kept.astype(np.int32), dtype=warp.int32, device=device)
        starts, ringed = rings(skins.faces, count)
        self.ring_starts = warp.array(starts, dtype=warp.int32, device=device)
        self.ring_faces = warp.array(ringed, dtype=warp.int32, device=device)

    def apart(self, points, frames):
        """Each point's offset off its frame's skin and that offset's Jacobian in the point (`apart_kernel`; points
        n x 3 in MuJoCo's frame, `frames` n: its mesh's index), float64: (n x 3, n x 3 x 3)."""
        found = warp.zeros(len(points), dtype=warp.vec3d, device=self.device)
        bends = warp.zeros(len(points), dtype=warp.mat33d, device=self.device)
        warp.launch(apart_kernel, dim=len(points), device=self.device,
                    inputs=[self.ids, self.first_verts, self.kept, self.verts, self.faces, self.ring_starts,
                            self.ring_faces,
                            warp.array(np.asarray(frames, dtype=np.int32), dtype=warp.int32, device=self.device),
                            warp.array(np.asarray(points, dtype=np.float64), dtype=warp.vec3d, device=self.device),
                            found, bends])
        return found.numpy().astype(float), bends.numpy().astype(float)
