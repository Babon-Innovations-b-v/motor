"""MyoFullBody made movable above the lower back (#70, the owner: "MyoFullBody should be fully movable"), on the spec
of the pinned source, the download untouched: the neck's two joints the source leaves commented out (`NECK`), one
three-axis thoracic joint where the torso meets lumbar1 (`THORACIC`; the wraps only lumbar muscles use laid on
their vertebra, `LUMBAR_WRAPS`), and the shoulder girdle's couplings to the
arm's elevation dropped, its clavicle and scapula joints given their own two-sided ranges (`GIRDLE`) and each phantom
tied to the joint it undoes (`PHANTOMS`); its feet given the twist on the shin their hinges lack and the ankle's
and subtalar joint's human ranges (`FEET`, 2026-11-01); its fingers' knuckles let bend back as a person's do
(`KNUCKLES`, 2026-11-01); its shin's turn on the femur let free of the knee's flexion (`KNEE`, 2026-11-04); its upper
lumbar spine led of its own (`LUMBAR`), its lower neck let flex and bend sideways (`CERVICAL`) and its narrower single
ranges widened to the published ones (`RANGES`, 2026-11-04). A body or coupling the source does not have is left
alone (a toy wheel in the tests)."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# MyoHead's own lines (head/assets/myohead_rigid_chain.xml, commented out there), as written: {body: joint}.
NECK = {"neck": {"name": "neck_rotation", "pos": [-0.02, 0.4, 0.0], "axis": [0.2, 1.0, 0.0], "range": [-1.4, 1.4]},
        "head": {"name": "neck_flexion", "pos": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "range": [-0.87, 1.05]}}
# The thoracic spine's motion lumped at T12/L1 (landmarks' T12_L1 on the torso), the lumbar levels' axes and order
# (FE about z, flexion negative; LB about x; AR about y). Ranges: the thoracic spine's normal active range of motion,
# the upper end of Magee's (Orthopedic Physical Assessment, 6th ed. 2014, ch. 8: forward flexion 20-45 deg, extension
# 25-45 deg, side flexion 20-40 deg, rotation 35-50 deg), in line with White and Panjabi's summed T1-T12 segment
# ranges (Clinical Biomechanics of the Spine, 2nd ed. 1990, table 2-4). Names never end like bone_groups.TRUNK's.
THORACIC = ("torso", "myotorso", [-0.01368, 0.11893, 0.00045],
            [("thoracic_FE", [0.0, 0.0, 1.0], [-np.radians(45), np.radians(45)]),
             ("thoracic_LB", [1.0, 0.0, 0.0], [-np.radians(40), np.radians(40)]),
             ("thoracic_AR", [0.0, 1.0, 0.0], [-np.radians(50), np.radians(50)])])
# The torso's wrap geoms that only lumbar muscles wrap, and the vertebra each is laid on (#98, 2026-11-07). The source
# welds the torso to lumbar1 and ties the lumbar levels to one leader, so back_cylinder_l, on the torso, kept its place
# to its muscles, the iliocostalis lumborum IL_L2 to IL_L4 (sacrum to lumbar2-4, none across T12/L1), their sites
# 13 to 26 mm off its surface. With THORACIC's joint, the lumbar levels led of their own (LUMBAR), the torso's own
# scale and the fit's moves of lumbar1, the sites came within 3 mm and the cylinder's side site turned past them:
# IL_L2 wrapped the cylinder's far side, up to four times its length range's top, on most frames of a squat. On
# lumbar1 it still did on some frames; on lumbar2, the highest vertebra the muscles join, on none.
LUMBAR_WRAPS = {"back_cylinder_l": "lumbar2"}
# The girdle's couplings to shoulder_elv dropped (myoarm_bimanual_assets.xml: the clavicle's two turns, the scapula's
# three, and the phantom that undoes each), so the clavicle and the scapula lead, each on its own range; each phantom
# (unrotscap, unrothum: MoBL-ARMS' undo of the girdle's turn, Holzbaur et al. 2005) is tied at -1 to the joint it
# undoes, so the scapula turns on the thorax by the acromioclavicular joints alone and the humerus' joints stay
# thoracohumeral, as MoBL's rhythm had them (a phantom left free shares its joint's axis: the clavicle's own turn then
# has no target, and the card's level 3 swung the pair bound to bound, 2026-10-31). Each range by the sign the arm's
# lift drives the joint, the model's own end kept where it is wider (the published rhythm stays reachable):
# sternoclavicular r2 (lift negative: retraction) to the model's 43 deg (Ludewig et al. 2009, JBJS 91(2):378: 15-30
# deg in elevation), protraction 20 deg; r3 (lift positive: elevation) to 45 deg, depression 10 deg (Moore, Dalley and
# Agur, Clinically Oriented Anatomy, 7th ed. 2014, table 6.4). Acromioclavicular r3 (lift positive: upward rotation)
# to the model's 70 deg, downward rotation 10 deg; r1 (posterior tilt) to the model's 32 deg, anterior tilt 20 deg;
# r2 (lift negative: external rotation) 25 deg each way (the scapula on the thorax in elevation and at rest: Ludewig
# 2009; McClure et al. 2001, JSES 10(3):269).
GIRDLE = {"sternoclavicular_r2": [-0.75, np.radians(20)], "sternoclavicular_r3": [-np.radians(10), np.radians(45)],
          "acromioclavicular_r1": [-np.radians(20), 0.552], "acromioclavicular_r2": [-np.radians(25), np.radians(25)],
          "acromioclavicular_r3": [-np.radians(10), 1.228]}
PHANTOMS = {"unrotscap_r2": "sternoclavicular_r2", "unrotscap_r3": "sternoclavicular_r3",
            "unrothum_r1": "acromioclavicular_r1", "unrothum_r2": "acromioclavicular_r2",
            "unrothum_r3": "acromioclavicular_r3"}
# The foot's twist on the shin (#70, the owner, 2026-11-01: SOMA's foot turns about the shin's long axis, MyoFullBody's
# hinges cannot): one hinge `ankle_rotation_{side}` on the talus at the ankle's centre, about the tibia's long axis
# (knee to ankle centre), before the ankle's flexion, positive turning the toes in (adduction) on both sides. At the
# ankle: the ankle complex's transverse-plane axis is the tibia's long axis (Brockett and Chapman 2016, Orthop Trauma
# 30(3):232, doi:10.1016/j.mporth.2016.04.015), and under full body load the talus turns about the vertical axis in the
# talocrural joint as the leg turns, the talonavicular joint beyond it (Lundberg 1989, Acta Orthop Scand Suppl 233,
# p. 15: 5.0 and 12 degrees within a 20-degree internal to 10-degree external leg rotation, not end ranges), joints
# MyoFullBody's rigid foot lumps into this one. Range: the ankle joint complex's abduction 20 and adduction 25 degrees
# (Gunaratne and Tamura 2024, Sensors 24(24), PMC11679897, a review of active ankle orthoses beside Grimston et al.
# 1993, Foot Ankle 14:215, who measured the complex's abduction and adduction; the review's sentence names no source).
# Ranges widened to human ones, never to what a clip asks:
# plantar flexion to 55 degrees (the ankle complex's "10 to 20 of dorsiflexion through to 40-55 of plantarflexion",
# Brockett and Chapman 2016, after Grimston et al. 1993 and Stauffer et al. 1977), dorsiflexion to 33 (averages 13 to
# 33 across studies: Roaas and Andersson 1982, Boone and Azen 1979, Lundberg et al. 1989, Stauffer et al. 1977, as
# OrthopaedicsOne's foot-and-ankle biomechanics review gathers them), subtalar inversion to 23 (the frontal plane's
# "23 inversion, 12 eversion", Brockett and Chapman 2016 after Stauffer et al. 1977); eversion keeps the model's 20
# (wider than 12: the published model stays reachable).
FEET = {"twist": ("ankle_rotation", [-np.radians(20), np.radians(25)]),
        "ranges": {"ankle_angle": [-np.radians(55), np.radians(33)], "subtalar_angle": [-0.349066, np.radians(23)]}}
# The fingers' knuckles bend back (#70, 2026-11-01): MyoFullBody's metacarpophalangeal flexion of the index to the
# little finger (mcp2 to mcp5) stops at 0 (the finger straight on its metacarpal), so SOMA's flatter hand, which asks
# them 20 to 30 degrees back, leaves the index and little distal phalanges outside its skin. A person's finger MCP
# joint hyperextends: extension 0 to 45 degrees beside flexion 0 to 90 (the AAOS average ranges, Greene and Heckman,
# The Clinical Measurement of Joint Motion, AAOS 1994, as Norkin and White, Measurement of Joint Motion, 5th ed. 2016,
# reproduce them; Eaton's hand reference table: "(0-45H)/90"). Extension widened to 45 degrees; flexion keeps the
# model's 90; the thumb's MCP (mp_flexion) and every other joint keep theirs.
# The shin turns on the femur of its own (#70, 2026-11-04): MyoFullBody's knee sets the tibia's turn about its long axis
# (knee_angle_rotation3) from the flexion alone (polycoef 0, 0.3695, -0.1695, 0.0252: 0.26 rad at 90 degrees, its
# range end), so seated the fitted shin stood 6 to 10 degrees from SOMA's Shin turn and the lateral malleolus swung
# out of SOMA's skin. A person's flexed knee turns the tibia of its own: at 90 degrees of flexion 30 degrees medially
# and 40 laterally (Kapandji, The Physiology of the Joints, vol. 2, the lower limb: the knee's active axial rotation),
# none at full extension. The coupling is dropped and the joint leads on the published model's own range (15 degrees
# each way, inside the flexed knee's), never widened; the knee's other couplings stay.
KNEE = ("knee_angle_rotation3", "knee_angle")
KNUCKLES = {"stems": tuple(f"mcp{finger}_flexion" for finger in "2345"), "extension": np.radians(45)}
# The upper lumbar spine bends of its own (#70, 2026-11-04): MyoFullBody sets every lumbar level from one leader per
# axis on lumbar5 (flex_extension, lat_bending, axial_rotation) in fixed ratios, so its lumbar4 (SOMA's Spine1) and
# lumbar1 (SOMA's Spine2) can only bend in one proportion. A person's lumbar levels move each on its own range (White
# and Panjabi, Clinical Biomechanics of the Spine, 2nd ed. 1990, the lumbar table of segmental ranges: flexion and
# extension combined L1-L2 12, L2-L3 14, L3-L4 15, L4-L5 16, L5-S1 17 degrees, each with its own limits). The levels
# above L4 (L3/L4, L2/L3, L1/L2: SOMA's Spine2) lead from L3/L4's joints, the two above it tied to them in the model's
# own ratios; L3/L4 leads on the model's own range (never widened), L4/L5 stays tied to lumbar5's leaders.
LUMBAR = ("L3_L4", ("L2_L3", "L1_L2"), ("FE", "LB", "AR"))
# The lower neck bends (#70, 2026-11-04): MyoHead's neck turns only about its long axis at C7/T1 and flexes at C0/C1,
# so SOMA's Neck1, riding the neck, can neither flex nor bend sideways on the chest. The cervical spine below C1 flexes,
# extends and bends sideways at every level (White and Panjabi 1990, the cervical table: C1-C2 to C7-T1 lateral
# bending 0 to 11 degrees a level). Its flexion and sideways bend are lumped at C7/T1 (the neck's own place, as the
# thorax's are at T12/L1), on the neck's axes as the torso's (flexion about z, negative; side bending about x). Ranges:
# with C0/C1's 49.8 of flexion and 60.2 of extension, the whole neck to Magee's 90 of flexion and 70 of extension
# (Orthopedic Physical Assessment, 6th ed. 2014, ch. 3: flexion 80-90, extension 70, side flexion 20-45); side bending
# 45 each way (Magee's upper end, the AAOS's 45: Joint Motion, Method of Measuring and Recording, 1965).
CERVICAL = ("neck", "myohead", [("cervical_FE", [0.0, 0.0, 1.0], [-np.radians(40), np.radians(9.8)]),
                                ("cervical_LB", [1.0, 0.0, 0.0], [-np.radians(45), np.radians(45)])])
# Single ranges narrower than a person's, widened to the published one (#70, 2026-11-04), never past it, each side:
# - hip_rotation +-40 to 45 each way: the AAOS average hip internal and external rotation (Greene and Heckman, The
#   Clinical Measurement of Joint Motion, AAOS 1994; AAOS 1965: 45 and 45).
# - elbow_flex 0-130 to 0-145: Kapandji's active elbow flexion (The Physiology of the Joints, vol. 1, the upper limb:
#   145 active, 160 passive), inside the AAOS's 150.
# - wrist flexion +-45 to +-70: MoBL-ARMS' own (Holzbaur et al. 2005), inside the AAOS's 80 of flexion and 70 of
#   extension (Greene and Heckman 1994).
# - wrist deviation (+: radial, its index finger turns to the thumb's side) -10 to 30 ulnar, the AAOS's 30 (Greene
#   and Heckman 1994); the model's 25 radial kept (wider than the AAOS's 20: the published model stays reachable).
# - mtp_angle (+: flexion; one hinge for all five toes on the metatarsal break) +-30 to +-40: the AAOS's lesser toes'
#   metatarsophalangeal flexion 40 and extension 40 (AAOS 1965), inside the great toe's 45 and 70.
# Not widened: knee_angle (0-120, the AAOS's 135): its five couplings (the tibia's slides and sideways turn, the
# kneecap's) are the published model's curves measured to 120 degrees only, and reading them past it is no anatomy.
RANGES = {"hip_rotation": [-np.radians(45), np.radians(45)], "elbow_flex": [0.0, np.radians(145)],
          "flexion": [-np.radians(70), np.radians(70)], "deviation": [-np.radians(30), np.radians(25)],
          "mtp_angle": [-np.radians(40), np.radians(40)]}


def movable(spec):
    """The spec's neck, thorax (its lumbar wraps kept on the lumbar spine), shoulder girdle, feet, knuckles, knees,
    upper lumbar spine, lower neck and narrow ranges made movable in place (the module's ten changes)."""
    _neck(spec)
    _thoracic(spec)
    _lumbar_wraps(spec)
    _girdle(spec)
    _feet(spec)
    _knuckles(spec)
    _knee(spec)
    _lumbar(spec)
    _cervical(spec)
    _ranges(spec)


def _neck(spec):
    """MyoHead's two neck joints added as the source's commented lines give them (its class's defaults)."""
    for body, joint in NECK.items():
        found = spec.body(body)
        if found is not None:
            found.add_joint(default=spec.find_default("myohead"), limited=True, **joint)


def _thoracic(spec):
    """The thoracic joint's three hinges added on the torso at T12/L1."""
    body, kind, place, axes = THORACIC
    found = spec.body(body)
    if found is None:
        return
    for name, axis, span in axes:
        found.add_joint(default=spec.find_default(kind), name=name, pos=place, axis=axis, range=span, limited=True)


def _lumbar_wraps(spec):
    """`LUMBAR_WRAPS` and their side sites laid from the torso on their vertebrae (`_laid_on`)."""
    torso = spec.body(THORACIC[0])
    if torso is None:
        return
    for geom in [geom for geom in torso.geoms if geom.name in LUMBAR_WRAPS]:
        sides = [site for site in torso.sites
                 if site.name.startswith(f"{geom.name}_") and site.name.endswith("_sidesite")]
        for element in (geom, *sides):
            _laid_on(spec, torso, element, LUMBAR_WRAPS[geom.name])


# An element's fields `_laid_on` does not copy: its identity and place (set apart), and the methods.
NOT_COPIED = {"id", "parent", "name", "signature", "info", "frame", "compiler", "plugin", "classname", "alt", "pos",
              "quat", "fromto", "userdata"}


def _quaternion(turn):
    """A scipy rotation as MuJoCo's [w, x, y, z]."""
    return np.roll(turn.as_quat(), 1)


def _turn(quat):
    """MuJoCo's [w, x, y, z] as a scipy rotation."""
    return Rotation.from_quat(np.roll(np.asarray(quat, dtype=float), -1))


def _laid_on(spec, body, element, below):
    """A geom or site of `body` moved onto `below`, a body it hangs from, where it stands at the zero pose (its fields
    kept, its name the same, so a tendon's wrap reads it as before)."""
    place, turn, at = np.asarray(element.pos, dtype=float), _turn(element.quat), body
    while at.name != below:
        place, turn, at = np.asarray(at.pos) + _turn(at.quat).apply(place), _turn(at.quat) * turn, at.parent
    adder = at.add_geom if isinstance(element, mujoco.MjsGeom) else at.add_site
    made = adder(default=element.classname)
    for field in dir(element):
        if not field.startswith("_") and field not in NOT_COPIED and not callable(getattr(element, field)):
            setattr(made, field, getattr(element, field))
    made.pos, made.quat = place, _quaternion(turn)
    name = element.name
    spec.delete(element)
    made.name = name


def _girdle(spec):
    """The girdle's couplings to shoulder_elv deleted, each of its joints given GIRDLE's range; each phantom's tied at
    -1 to the joint it undoes, on that joint's range turned round."""
    for equality in list(spec.equalities):
        stem, side = equality.name1[:-2], equality.name1[-2:]  # the side's '_l'/'_r' off
        if not equality.name2.startswith("shoulder_elv"):
            continue
        if stem in GIRDLE:
            spec.delete(equality)
            spec.joint(equality.name1).range = GIRDLE[stem]
        elif stem in PHANTOMS:
            equality.name2 = PHANTOMS[stem] + side
            equality.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
            low, high = GIRDLE[PHANTOMS[stem]]
            spec.joint(equality.name1).range = [-high, -low]


def _feet(spec):
    """Each foot's twist hinge added on its talus before the ankle's flexion (`FEET`), the ankle's and subtalar
    joint's ranges widened."""
    name, span = FEET["twist"]
    for side in "lr":
        talus = spec.body(f"talus_{side}")
        if talus is None or spec.joint(f"ankle_angle_{side}") is None:
            continue
        shin_axis = -np.asarray(talus.pos, dtype=float) / np.linalg.norm(talus.pos)  # tibia origin (knee) from the ankle
        turned = Rotation.from_quat(np.roll(np.asarray(talus.quat, dtype=float), -1)).as_matrix()
        axis = turned.T @ shin_axis * (-1.0 if side == "l" else 1.0)  # +: toes in on both sides
        later = [_kept(joint) for joint in talus.joints]
        for joint in list(talus.joints):
            spec.delete(joint)
        talus.add_joint(default=spec.find_default("myolegs"), name=f"{name}_{side}", type=mujoco.mjtJoint.mjJNT_HINGE,
                        pos=[0.0, 0.0, 0.0], axis=axis, range=span, limited=True)
        for kept in later:
            talus.add_joint(**kept)
        for stem, wide in FEET["ranges"].items():
            spec.joint(f"{stem}_{side}").range = wide


def _knuckles(spec):
    """Each finger MCP flexion joint's (`KNUCKLES`) lower end moved to the knuckle's extension, its flexion kept."""
    for stem in KNUCKLES["stems"]:
        for side in "lr":
            joint = spec.joint(f"{stem}_{side}")
            if joint is not None:
                joint.range = [-KNUCKLES["extension"], float(joint.range[1])]


def _knee(spec):
    """Each knee's coupling of the shin's turn to its flexion (`KNEE`) deleted, the joint kept on its own range."""
    follower, leader = KNEE
    for equality in list(spec.equalities):
        if equality.name1[:-2] == follower and equality.name2[:-2] == leader:
            spec.delete(equality)


def _lumbar(spec):
    """Each axis' L3/L4 joint (`LUMBAR`) untied from lumbar5's leader, the levels above it tied to it in the model's
    ratios; a model without them is left alone, one whose ties are not plain ratios refused."""
    lead, above, axes = LUMBAR
    ties = {equality.name1: equality for equality in spec.equalities if equality.type == mujoco.mjtEq.mjEQ_JOINT}
    for axis in axes:
        own = ties.get(f"{lead}_{axis}")
        if own is None:
            continue
        followers = [ties[f"{level}_{axis}"] for level in above]
        for tie in (own, *followers):
            if tie.name2 != own.name2 or any(tie.data[at] for at in (0, 2, 3, 4)):
                raise ValueError(f"{tie.name1}: not a plain ratio of {own.name2} ({tie.name2}, {tie.data[:5]})")
        ratio = float(own.data[1])
        spec.delete(own)
        for tie in followers:
            tie.name2 = f"{lead}_{axis}"
            tie.data[:5] = [0.0, float(tie.data[1]) / ratio, 0.0, 0.0, 0.0]


def _cervical(spec):
    """The lower neck's flexion and side bending (`CERVICAL`) added on the neck at its turn's place, after it."""
    body, kind, axes = CERVICAL
    found = spec.body(body)
    turn = spec.joint(NECK[body]["name"])
    if found is None or turn is None:
        return
    for name, axis, span in axes:
        found.add_joint(default=spec.find_default(kind), name=name, pos=turn.pos, axis=axis, range=span,
                        limited=True)


def _ranges(spec):
    """Each side's joint of `RANGES` given its published range."""
    for stem, span in RANGES.items():
        for side in "lr":
            joint = spec.joint(f"{stem}_{side}")
            if joint is not None:
                joint.range = span


def _kept(joint):
    """A joint's settings, to add it again after another (MjSpec adds joints last only)."""
    return {field: getattr(joint, field) for field in
            ("name", "type", "pos", "axis", "range", "limited", "ref", "springref", "stiffness", "damping",
             "armature", "frictionloss", "margin", "group", "solref_limit", "solimp_limit", "solref_friction",
             "solimp_friction", "actfrclimited", "actfrcrange", "actgravcomp", "align", "springdamper")}
