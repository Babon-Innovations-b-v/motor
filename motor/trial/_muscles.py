"""MOTOR step 4 (#98), stage B's muscles: their paths at every node on MuJoCo's own model, MuJoCo's own force curves
on the card, and the activation dynamics between the nodes.

- `Paths`: every node's muscle-tendon lengths and moment arms on the leading joints (MuJoCo's C, float64: its
  tendons' wraps as the model has them, the couplings' map carrying a follower's arm onto its leader). The design's
  polynomials of the joints (OpenCap's) were tried first and refused: fitted to MuJoCo Warp at 4,000 poses about a
  squat's own, 61 of MyoFullBody's 416 muscles (the obliques and the rectus spanning nine spine joints, the
  supraspinatus, whose tendon jumps 62 mm to another side of a wrap within 5 mrad) missed by more than 1 mm of length
  or 2 mm of moment arm on poses held out (2026-11-08). The
  solve needs no gradient of MuJoCo: its Gauss-Newton model takes the muscles' lengths, speeds and arms as MuJoCo
  gives them (the arms' own change with the pose left out; the cost always exact).
- `Curves`: MuJoCo's muscle gain and bias (`mju_muscleGain`, `mju_muscleBias`: the force-length and force-velocity
  curves, the passive curve, the peak force by `scale / acc0` where the model asks) written in Warp, float64, and
  their derivatives on the length and the speed by Warp's own tape (one backward pass a curve).
- `excitations`: the excitation each muscle's activation needs between two nodes under MuJoCo's first-order activation
  dynamics (`mju_muscleDynamics`, its smoothing width 0 as MyoFullBody sets it: the activation's time constant scaled
  by 0.5 + 1.5 a, the deactivation's divided by it, Millard et al. 2013), at the span's midpoint."""
import mujoco
import numpy as np
import warp as wp

MINIMUM = 1e-15  # MuJoCo's mjMINVAL


@wp.func
def gain_length(length: wp.float64, lowest: wp.float64, highest: wp.float64) -> wp.float64:
    """MuJoCo's `mju_muscleGainLength`."""
    half = wp.float64(0.5)
    if length < lowest or length > highest:
        return wp.float64(0.0)
    low = half * (lowest + wp.float64(1.0))
    high = half * (wp.float64(1.0) + highest)
    if length <= low:
        x = (length - lowest) / wp.max(wp.float64(MINIMUM), low - lowest)
        return half * x * x
    if length <= wp.float64(1.0):
        x = (wp.float64(1.0) - length) / wp.max(wp.float64(MINIMUM), wp.float64(1.0) - low)
        return wp.float64(1.0) - half * x * x
    if length <= high:
        x = (length - wp.float64(1.0)) / wp.max(wp.float64(MINIMUM), high - wp.float64(1.0))
        return wp.float64(1.0) - half * x * x
    x = (highest - length) / wp.max(wp.float64(MINIMUM), highest - high)
    return half * x * x


@wp.kernel
def _curves(lengths: wp.array2d(dtype=wp.float64), speeds: wp.array2d(dtype=wp.float64),
            ranges: wp.array2d(dtype=wp.float64), accelerations: wp.array(dtype=wp.float64),
            gain_parameters: wp.array2d(dtype=wp.float64), bias_parameters: wp.array2d(dtype=wp.float64),
            gains: wp.array2d(dtype=wp.float64), biases: wp.array2d(dtype=wp.float64)):
    """MuJoCo's `mju_muscleGain` and `mju_muscleBias` of every node's every muscle."""
    node, muscle = wp.tid()
    one, half = wp.float64(1.0), wp.float64(0.5)
    low_length, high_length = ranges[muscle, 0], ranges[muscle, 1]
    force = gain_parameters[muscle, 2]
    if force < wp.float64(0.0):
        force = gain_parameters[muscle, 3] / wp.max(wp.float64(MINIMUM), accelerations[muscle])
    optimum = (high_length - low_length) / wp.max(wp.float64(MINIMUM), gain_parameters[muscle, 1]
                                                  - gain_parameters[muscle, 0])
    length = gain_parameters[muscle, 0] + (lengths[node, muscle] - low_length) / wp.max(wp.float64(MINIMUM), optimum)
    speed = speeds[node, muscle] / wp.max(wp.float64(MINIMUM), optimum * gain_parameters[muscle, 6])
    along = gain_length(length, gain_parameters[muscle, 4], gain_parameters[muscle, 5])
    fastest = gain_parameters[muscle, 8]
    room = fastest - one
    pace = fastest
    if speed <= -one:
        pace = wp.float64(0.0)
    elif speed <= wp.float64(0.0):
        pace = (speed + one) * (speed + one)
    elif speed <= room:
        pace = fastest - (room - speed) * (room - speed) / wp.max(wp.float64(MINIMUM), room)
    gains[node, muscle] = -force * along * pace
    passive_force = bias_parameters[muscle, 2]
    if passive_force < wp.float64(0.0):
        passive_force = bias_parameters[muscle, 3] / wp.max(wp.float64(MINIMUM), accelerations[muscle])
    passive_optimum = (high_length - low_length) / wp.max(wp.float64(MINIMUM), bias_parameters[muscle, 1]
                                                          - bias_parameters[muscle, 0])
    stretch = bias_parameters[muscle, 0] + (lengths[node, muscle] - low_length) / wp.max(wp.float64(MINIMUM),
                                                                                         passive_optimum)
    knee = half * (one + bias_parameters[muscle, 5])
    bias = wp.float64(0.0)
    if stretch > one:
        if stretch <= knee:
            x = (stretch - one) / wp.max(wp.float64(MINIMUM), knee - one)
            bias = -passive_force * bias_parameters[muscle, 7] * half * x * x
        else:
            x = (stretch - knee) / wp.max(wp.float64(MINIMUM), knee - one)
            bias = -passive_force * bias_parameters[muscle, 7] * (half + x)
    biases[node, muscle] = bias


class Curves:
    """MuJoCo's muscle curves of a model's `muscles` (actuator indices) on the card: `evaluated(lengths, speeds)`."""

    def __init__(self, model, muscles):
        self.count = len(muscles)
        self.parameters = [wp.array(np.ascontiguousarray(values, dtype=np.float64), dtype=wp.float64)
                           for values in (model.actuator_lengthrange[muscles], model.actuator_acc0[muscles],
                                          model.actuator_gainprm[muscles, :9], model.actuator_biasprm[muscles, :9])]

    def evaluated(self, lengths, speeds):
        """Every node's every muscle's (gain, bias, d gain / d length, d gain / d speed, d bias / d length), each
        [N, muscles]; the force of an activation a is gain a + bias (negative: a pull), as MuJoCo's."""
        shape = np.shape(lengths)
        arrays = [wp.array(np.ascontiguousarray(values, dtype=np.float64), dtype=wp.float64, requires_grad=True)
                  for values in (lengths, speeds)]
        outputs = [wp.zeros(shape, dtype=wp.float64, requires_grad=True) for _ in range(2)]
        tape = wp.Tape()
        with tape:
            wp.launch(_curves, dim=shape, inputs=[*arrays, *self.parameters], outputs=outputs)
        found = [output.numpy() for output in outputs]
        tape.backward(grads={outputs[0]: wp.ones(shape, dtype=wp.float64)})
        by_length, by_speed = (array.grad.numpy().copy() for array in arrays)
        tape.zero()
        tape.backward(grads={outputs[1]: wp.ones(shape, dtype=wp.float64)})
        bias_length = arrays[0].grad.numpy().copy()
        return found[0], found[1], by_length, by_speed, bias_length


class Paths:
    """A model's `muscles` (actuator indices) along every node on MuJoCo's C: `at(qpos)`."""

    def __init__(self, model, muscles, couplings):
        self.model, self.muscles, self.couplings = model, muscles, couplings
        self.data = mujoco.MjData(model)
        self.dense = np.zeros((model.nu, model.nv))

    def at(self, qpos):
        """Every node's (lengths [N, M], moment arms d length / d leading joint [N, M, J], the root's six left out)."""
        model, data, dense = self.model, self.data, self.dense
        lengths, arms = [], []
        maps = self.couplings.maps(qpos)
        for pose, mapped in zip(qpos, maps):
            data.qpos[:] = pose
            mujoco.mj_fwdPosition(model, data)
            mujoco.mju_sparse2dense(dense, data.actuator_moment, data.moment_rownnz, data.moment_rowadr,
                                    data.moment_colind)
            lengths.append(data.actuator_length[self.muscles].copy())
            arms.append((dense[self.muscles] @ mapped)[:, 6:])
        return np.array(lengths), np.array(arms)


def time_constants(activations, rising, dynamics):
    """MuJoCo's time constant (s) of each activation [..., M] (clamped to [0, 1]) rising or falling (`rising` bool),
    and its derivative on the activation: the activation's dynprm[0] (0.5 + 1.5 a), the deactivation's dynprm[1] /
    (0.5 + 1.5 a); `dynamics` the muscles' dynprm [M, 3]."""
    clamped = np.clip(activations, 0.0, 1.0)
    inside = (activations > 0.0) & (activations < 1.0)
    scale = 0.5 + 1.5 * clamped
    rise, fall = dynamics[:, 0] * scale, dynamics[:, 1] / scale
    rise_slope, fall_slope = dynamics[:, 0] * 1.5 * inside, -dynamics[:, 1] * 1.5 * inside / scale ** 2
    return np.where(rising, rise, fall), np.where(rising, rise_slope, fall_slope)


def excitations(activations, step, dynamics):
    """The excitation [N - 1, M] each span between two nodes needs under MuJoCo's activation dynamics (smoothing
    width 0): e = a_mid + tau(a_mid, rising) (a[k+1] - a[k]) / step, and its derivatives on the span's first and
    second node's activation."""
    middle = 0.5 * (activations[:-1] + activations[1:])
    pace = (activations[1:] - activations[:-1]) / step
    constant, slope = time_constants(middle, pace > 0.0, dynamics)
    found = middle + constant * pace
    by_middle = 1.0 + slope * pace
    return found, 0.5 * by_middle - constant / step, 0.5 * by_middle + constant / step
