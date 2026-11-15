"""SOMA's segment volumes and the masses, centres of mass and inertias they give (#70; pure numpy): the person's closed
mesh at its bind filled with cells (`filled`), each cell given to the bone group whose SOMA stretch is nearest
(`shares`), each group's volume, centre and spread (`measured`) and its girth for its length (`slenderness`), and its
mass, centre of mass and inertia's trace at Dempster's segment densities, all brought to the person's mass (`weighed`).

Motion cannot tell weight, so the masses come from the body's shape (the owner's design, #70). The densities are
assumed: Dempster 1955 (WADC TR 55-159, eight cadavers aged 52 to 83) as Winter 2009 tabulates them (Biomechanics and
Motor Control of Human Movement, Table 4.1), the same for every person; Chandler et al. 1975 measured lower ones (upper
arm 0.997 against 1.07 kg/l, forearm 1.043 against 1.13, hand 1.079 against 1.16, thigh 1.018 against 1.05), so a
segment's own may be some 7% under them (`DENSITY_RANGE`)."""
import numpy as np

CELL = 0.005  # metres: the cells' edge (a hand's volume in about 3000 cells)
# kg per litre by bone group (Winter 2009 Table 4.1, Dempster's): the trunk below the chest as abdomen and pelvis,
# the chest as thorax, the shoulder as Dempster's shoulder mass, the neck and head as head and neck.
DENSITY = {"Hips": 1.01, "Spine1": 1.01, "Spine2": 1.01, "Chest": 0.92, "Neck1": 1.11, "Neck2": 1.11, "Head": 1.11,
           "Shoulder": 1.04, "Arm": 1.07, "ForeArm": 1.13, "Hand": 1.16, "Leg": 1.05, "Shin": 1.09, "Foot": 1.10,
           "ToeBase": 1.10}
DENSITY_RANGE = (0.93, 1.0)  # a segment's own density against Dempster's: Chandler et al. 1975 to Dempster


def density(group):
    """A bone group's density (kg/l), its side dropped."""
    return DENSITY[group.removeprefix("Left").removeprefix("Right")]


def filled(points, faces, cell=CELL):
    """The centres of the cells inside a closed mesh (n x 3): each column along y between its crossings of the surface
    in pairs (a column through an edge is moved a hair off it)."""
    points = np.asarray(points, dtype=float)
    low, high = points.min(axis=0), points.max(axis=0)
    xs = np.arange(low[0] + cell / 2, high[0], cell) + 1.234e-7
    zs = np.arange(low[2] + cell / 2, high[2], cell) + 2.345e-7
    ys = np.arange(low[1] + cell / 2, high[1], cell)
    hits = {}
    for triangle in points[np.asarray(faces)]:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = triangle
        area = (bx - ax) * (cz - az) - (cx - ax) * (bz - az)
        if abs(area) < 1e-14:
            continue
        columns = np.flatnonzero((xs >= min(ax, bx, cx)) & (xs <= max(ax, bx, cx)))
        rows = np.flatnonzero((zs >= min(az, bz, cz)) & (zs <= max(az, bz, cz)))
        if not len(columns) or not len(rows):
            continue
        x, z = np.meshgrid(xs[columns], zs[rows], indexing="ij")
        one = ((bx - x) * (cz - z) - (cx - x) * (bz - z)) / area
        two = ((cx - x) * (az - z) - (ax - x) * (cz - z)) / area
        three = 1.0 - one - two
        inside = (one >= 0) & (two >= 0) & (three >= 0)
        for column, row, height in zip(np.broadcast_to(columns[:, None], x.shape)[inside],
                                       np.broadcast_to(rows[None, :], x.shape)[inside],
                                       (one * ay + two * by + three * cy)[inside]):
            hits.setdefault((column, row), []).append(height)
    found = []
    for (column, row), heights in hits.items():
        crossings = np.sort(heights)
        if len(crossings) % 2:
            continue
        within = np.searchsorted(crossings, ys) % 2 == 1
        found.append(np.column_stack([np.full(within.sum(), xs[column]), ys[within], np.full(within.sum(), zs[row])]))
    return np.concatenate(found) if found else np.zeros((0, 3))


def shares(cells, stretches):
    """Each cell's bone group (index into `stretches`): the one with the stretch (a segment from start to end, or
    several) nearest it. `stretches`: [[(start, end), ...] per group]."""
    lines = [(group, np.asarray(start, dtype=float), np.asarray(end, dtype=float))
             for group, pieces in enumerate(stretches) for start, end in pieces]
    best, owner = np.full(len(cells), np.inf), np.zeros(len(cells), dtype=int)
    for group, start, end in lines:
        along = end - start
        reach = np.clip((cells - start) @ along / max(along @ along, 1e-12), 0.0, 1.0)
        away = np.linalg.norm(cells - start - reach[:, None] * along, axis=1)
        nearer = away < best
        best[nearer], owner[nearer] = away[nearer], group
    return owner


def measured(cells, owner, count, cell=CELL):
    """Each group's (volume m3, centre (3), spread: the integral of the squared distance to the centre over its
    volume, m5, each cell a small cube)."""
    found = []
    for group in range(count):
        mine = cells[owner == group]
        if not len(mine):
            found.append((0.0, np.zeros(3), 0.0))
            continue
        centre = mine.mean(axis=0)
        volume = len(mine) * cell ** 3  # each cell's own spread too: a cube's is its volume times 3 edge2 / 12
        found.append((volume, centre, float(((mine - centre) ** 2).sum()) * cell ** 3 + volume * cell ** 2 / 4))
    return found


GIRTH_MIDDLE = (0.2, 0.8)  # the share of a stretch its girth is read over: clear of the cells its ends share


def slenderness(cells, owner, group, start, end, cell=CELL):
    """A group's girth for its length: the width of its mean cross-section (the square root of its cells' volume
    between GIRTH_MIDDLE of its stretch, start to end, over that span's length) over the stretch's length."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    along = (cells[owner == group] - start) @ (end - start) / length ** 2
    within = np.count_nonzero((along >= GIRTH_MIDDLE[0]) & (along <= GIRTH_MIDDLE[1]))
    return float(np.sqrt(within * cell ** 3 / ((GIRTH_MIDDLE[1] - GIRTH_MIDDLE[0]) * length))) / length

def weighed(groups, found, kg):
    """Each group's (kg, centre of mass, inertia's trace kg m2 about it): its volume at its density, all brought to
    `kg` together (the trace of a solid's inertia is twice its density times its spread)."""
    raw = [density(group) * 1000.0 * volume for group, (volume, _, _) in zip(groups, found)]
    share = kg / sum(raw)
    return {group: (mass * share, centre, 2.0 * density(group) * 1000.0 * share * spread)
            for group, mass, (_, centre, spread) in zip(groups, raw, found)}
