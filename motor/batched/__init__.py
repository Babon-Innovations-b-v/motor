"""Every frame's MuJoCo state at once on the card (#70, 2026-11-05; the owner: "should prob just fix that now aswell,
so we know what were working with"): MuJoCo Warp, the model Newton's SolverMuJoCo puts on the card (`model.built` on a
CUDA card), one world a state: the position and velocity stages (kinematics, the tendons' and muscles' paths, lengths,
speeds and moment arms, the passive forces) and the inverse dynamics (`rne` with the accelerations), so the movement's
needed generalised forces (`needed`: rne + armature x qacc - passive, as `muscles.Physics.needed`) of every frame come
in one launch a stage. MuJoCo Warp is float32: what it gives is MuJoCo's C (float64) to its rounding, about 1e-6 of
each value (the stage test's tolerances measure it on the clips). `load` lays one state into a CPU `MjData`, so the
frame's own reads (the skin riding its bones, a point's Jacobian, the muscles' gains, their paths) read it as they read
MuJoCo's C. The model's masses, inertias and centres of mass are put on the card again whenever RRA's model step moves
them (`sync`). Collision and constraints are left out on the card (disabled on its model): nothing here reads them.
Run to run the card gives the same bits (#70, 2026-11-06: two runs of main on one L4 parted by 0.52 of an activation):
MuJoCo Warp's sums (a parent's share of its children, a tendon's length and Jacobian, the passive forces, the moment
rows' places) are atomic adds, whose order a card does not keep, so its kernels run in Warp's run-to-run mode
(`deterministic`: each add recorded, then summed in thread order), only the stages read here (not the inertia, its
tendon armature, the tendons' bias, nor the bias forces before `rne` with the accelerations), and as many worlds a
launch as its records fit in (`chunk_worlds`)."""
import contextlib

import numpy as np

from motor.batched._balls import within_balls_card
from motor.batched._least import least_all

CHUNK = 1024  # worlds a launch, at most
# Records a run-to-run launch may keep for one array (each an 8-byte key and its value, sorted after the kernel):
# 2^27, about 1.6 GB, a part of an L4's 24 GB the person's IK and least squares leave free.
RECORDS_MOST = 1 << 27
# Body chains MuJoCo Warp's `_spatial_geom_tendon` walks for one wrap at most (four of its six `_accumulate_jac_chain`
# calls, a wrapped side's two segments of two ends; an unwrapped side's two), each adding at most its tendon's Jacobian
# nonzeros once: the most records one thread leaves on the tendons' Jacobian. Too few, and Warp stops the launch.
CHAIN_WALKS = 4
COPIED = ("xpos", "xquat", "xipos", "subtree_com", "cdof", "actuator_length", "actuator_velocity", "moment_rownnz",
          "moment_rowadr", "moment_colind", "actuator_moment", "ten_wrapadr", "ten_wrapnum", "wrap_obj", "wrap_xpos",
          "qfrc_passive", "xmat", "ximat")


class Card:
    """MuJoCo Warp's model of `model` (`mjw_model`, Newton's SolverMuJoCo's), and its data a chunk of worlds."""

    def __init__(self, model, mjw_model):
        import mujoco_warp
        self.mjw, self.model, self.mjw_model = mujoco_warp, model, mjw_model
        self.mjw_model.opt.run_collision_detection = False
        bits = mujoco_warp.DisableBit
        flags = int(self.mjw_model.opt.disableflags) | int(bits.CONSTRAINT) | int(bits.CONTACT)
        self.mjw_model.opt.disableflags = flags
        self.data = {}
        self.synced = None
        self.records = records_bound(model)
        self.chunk = chunk_worlds(model, self.records)

    def sync(self):
        """The model's masses, inertias and centres of mass on the card as they stand on the CPU's model."""
        now = (self.model.body_mass.copy(), self.model.body_inertia.copy(), self.model.body_ipos.copy())
        if self.synced is not None and all(np.array_equal(a, b) for a, b in zip(now, self.synced)):
            return
        for name, values in zip(("body_mass", "body_inertia", "body_ipos"), now):
            target = getattr(self.mjw_model, name)
            target.assign(np.asarray(values, dtype=np.float32).reshape(target.numpy().shape))
        self.synced = now

    def _data(self, worlds):
        """A Data of `worlds` worlds, made once a size."""
        if worlds not in self.data:
            self.data[worlds] = self.mjw.make_data(self.model, nworld=worlds, nconmax=1, njmax=1)
        return self.data[worlds]

    def evaluate(self, qpos, qvel, qacc, names=COPIED):
        """Every state's (rows of `qpos`, `qvel`, `qacc`) needed generalised forces and what `load` lays (or only
        `names` of it): a dict of host arrays, one row a state, float64."""
        self.sync()
        qpos, qvel, qacc = (np.atleast_2d(np.asarray(values, dtype=float)) for values in (qpos, qvel, qacc))
        step = self.chunk
        with deterministic(self.records):
            parts = [self._chunk(qpos[start:start + step], qvel[start:start + step], qacc[start:start + step], names)
                     for start in range(0, len(qpos), step)]
        found = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        found["qpos"], found["qvel"], found["qacc"] = qpos, qvel, qacc
        return found

    def _chunk(self, qpos, qvel, qacc, names):
        """`evaluate` on at most `chunk` states: one Data of exactly that many worlds."""
        model = self.model
        data = self._data(len(qpos))
        data.qpos.assign(qpos.astype(np.float32))
        data.qvel.assign(qvel.astype(np.float32))
        data.qacc.assign(qacc.astype(np.float32))
        stages(self.mjw, self.mjw_model, data)
        found = {name: getattr(data, name).numpy() for name in set(names) | {"qfrc_passive"}}
        found = {name: values.astype(float) if values.dtype.kind == "f" else values for name, values in found.items()}
        found["needed"] = data.qfrc_bias.numpy().astype(float) + model.dof_armature * qacc - found["qfrc_passive"]
        return found

    def load(self, found, row, data):
        """State `row` of `evaluate`'s `found` laid into the CPU's `data` (MuJoCo's C reads it as its own)."""
        data.qpos[:], data.qvel[:], data.qacc[:] = found["qpos"][row], found["qvel"][row], found["qacc"][row]
        for name in COPIED:
            getattr(data, name).reshape(-1)[:] = found[name][row].reshape(-1)
        return found["needed"][row]


# How far a person's record (out.json's entry) on the card may lie from the same run on MuJoCo's C, the processor's
# (#70, 2026-11-05: "quality must not move"): the bones outside SOMA's skin (percentage points, every region), the joint
# error (mm) and turn error (degrees, both the median and the 95th percentile, free and at a limit), RRA's follows
# (degrees, before and after), its kept and least frame shares, and on every frame the ground's push and the pelvis'
# residual force (shares of the weight), the residual moment and the largest reserve (N m), each muscle's activation.
CARD_TOLERANCES = {"bones outside %": 0.1, "joints mm": 0.05, "turns deg": 0.01, "follows deg": 0.01, "share": 0.01,
                   "ground push / weight": 0.002, "residual force / weight": 0.002, "residual moment N m": 0.2,
                   "reserve N m": 0.5, "activation": 0.01}


def apart(base, card):
    """Where a person's record on the card (`card`, out.json's person entry) lies further than CARD_TOLERANCES from
    the processor's (`base`): ["<measure>: <base> against <card>", ...], none when within all."""
    misses = []

    def check(name, wanted, got, tolerance):
        wanted, got = np.asarray(wanted, dtype=float), np.asarray(got, dtype=float)
        if wanted.shape != got.shape:
            misses.append(f"{name}: shapes {wanted.shape} against {got.shape}")
            return
        off = float(np.abs(wanted - got).max(initial=0.0))
        if not off <= tolerance:
            misses.append(f"{name}: {off:.4g} apart (allowed {tolerance:g})")

    followed = [entry["model"]["fit"]["following, the path the physics takes"] for entry in (base, card)]
    regions = sorted(followed[0]["bones outside skin %"])
    check("bones outside %", [followed[0]["bones outside skin %"][region] for region in regions],
          [followed[1]["bones outside skin %"].get(region, np.nan) for region in regions],
          CARD_TOLERANCES["bones outside %"])
    for measure in ("joints mm", "turns deg"):
        for part in ("free", "at a limit"):
            check(f"{measure} {part}", followed[0][measure][part][:2], followed[1][measure][part][:2],
                  CARD_TOLERANCES[measure])
    moved = [entry["adjusted"] for entry in (base, card)]
    check("follows deg", moved[0]["follows deg"], moved[1]["follows deg"], CARD_TOLERANCES["follows deg"])
    check("shares", [moved[0]["kept share"], moved[0]["least frame share"]],
          [moved[1]["kept share"], moved[1]["least frame share"]], CARD_TOLERANCES["share"])
    weight = float(base["weight N"])
    frames = [entry["frames"] for entry in (base, card)]
    each = lambda index, field: np.array([frame[field] for frame in frames[index]], dtype=float)
    check("ground push / weight", each(0, "ground") / weight, each(1, "ground") / weight,
          CARD_TOLERANCES["ground push / weight"])
    check("residual force / weight", each(0, "residual")[:, :3] / weight, each(1, "residual")[:, :3] / weight,
          CARD_TOLERANCES["residual force / weight"])
    check("residual moment N m", each(0, "residual")[:, 3:], each(1, "residual")[:, 3:],
          CARD_TOLERANCES["residual moment N m"])
    check("reserve N m", each(0, "reserve"), each(1, "reserve"), CARD_TOLERANCES["reserve N m"])
    check("activation", each(0, "activations"), each(1, "activations"), CARD_TOLERANCES["activation"])
    return misses


def stages(mjw, mjw_model, data):
    """MuJoCo Warp's position and velocity stages and the inverse dynamics with the accelerations, only what `evaluate`
    reads: `fwd_position` without the inertia (`crb`, `tendon_armature`; nothing reads it, `factorize` off) and
    `fwd_velocity` without its own `rne` and the tendons' bias (`rne` with the accelerations writes the bias forces
    again, and the tendons' bias went with them), so no kernel whose records outgrow `records_bound` runs."""
    import warp as wp
    from mujoco_warp._src import forward
    mjw.fwd_kinematics(mjw_model, data)
    mjw.transmission(mjw_model, data)
    wp.launch(forward._actuator_velocity, dim=(data.nworld, mjw_model.nactuator),
              inputs=[data.qvel, data.moment_rownnz, data.moment_rowadr, data.moment_colind, data.actuator_moment],
              outputs=[data.actuator_velocity], block_dim=mjw_model.block_dim.actuator_velocity)
    wp.launch(forward._tendon_velocity, dim=(data.nworld, mjw_model.ntendon),
              inputs=[mjw_model.ten_J_rownnz, mjw_model.ten_J_rowadr, mjw_model.ten_J_colind, data.qvel, data.ten_J],
              outputs=[data.ten_velocity])
    mjw.com_vel(mjw_model, data)
    mjw.passive(mjw_model, data)
    mjw.rne(mjw_model, data, flg_acc=True)


def records_bound(model):
    """The most records one thread of `stages` leaves on one array in Warp's run-to-run mode: a tendon wrap's body
    chains (CHAIN_WALKS of them, each at most its tendon's Jacobian nonzeros); every other sum there is one record a
    thread, which Warp counts itself."""
    return max(1, CHAIN_WALKS * int(np.max(model.ten_J_rownnz, initial=0)))


def chunk_worlds(model, records):
    """Worlds a launch so its widest summing kernel (a world's bodies, wraps or actuators, `records` each) keeps at most
    RECORDS_MOST records an array; at most CHUNK, at least one."""
    return int(min(CHUNK, max(1, RECORDS_MOST // (records * max(model.nbody, model.nwrap, model.nu, 1)))))


@contextlib.contextmanager
def deterministic(records):
    """MuJoCo Warp's summing modules (`smooth`, `passive`) and the kernels it makes as it goes in Warp's run-to-run
    mode, `records` the most a thread keeps in `smooth` (`records_bound`). Its made kernels are made again the first
    time the modes change (their cache's key leaves the mode out), as Newton's SolverMuJoCo does; Warp's mode is put
    back after."""
    import warp as wp
    from mujoco_warp._src import passive, smooth, warp_util
    mode = wp.DeterministicMode.RUN_TO_RUN
    changed = False
    for module, most in ((smooth, records), (passive, 0)):
        wanted = {"deterministic": mode, "deterministic_max_records": most}
        held = wp.get_module_options(module=module)
        if any(held.get(name) != value for name, value in wanted.items()):
            wp.set_module_options(wanted, module=module)
            changed = True
    if changed:
        warp_util._KERNEL_CACHE.clear()
    before = wp.config.deterministic, wp.config.deterministic_max_records
    wp.config.deterministic, wp.config.deterministic_max_records = mode, records
    try:
        yield
    finally:
        wp.config.deterministic, wp.config.deterministic_max_records = before


def root_pose(model, found):
    """The root body's turn (states x 3 x 3) and place (states x 3) in `evaluate`'s `found`."""
    root = model.jnt_bodyid[0]
    return found["xmat"][:, root].reshape(-1, 3, 3), found["xpos"][:, root]


def same_layout(model, mjw_model):
    """Whether MuJoCo Warp's model holds `model`'s sizes (its bodies, dofs, tendons, wraps, actuators, nJmom)."""
    return all(getattr(model, name) == getattr(mjw_model, name) for name in ("nbody", "nv", "nq", "ntendon", "nwrap",
                                                                             "nu", "nJmom"))


__all__ = ["CARD_TOLERANCES", "Card", "apart", "chunk_worlds", "deterministic", "least_all", "records_bound",
           "root_pose", "same_layout", "stages", "within_balls_card"]
