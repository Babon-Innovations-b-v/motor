"""The 12-row step fixture (Jacobians, residuals, poses on their bounds, couplings, frozen dofs) and card/proto's
step and projection in float64 that the card's are checked against (test fixture,
test_the_card_step_and_projection_equal_the_cpu_prototype.py)."""

ROWS, RESIDUALS, DOFS = 12, 30, 20
FROZEN = (3, 7)
COUPLINGS = ((5, 2, (0.01, 0.6, -0.2, 0.05, 0.01)), (9, 2, (0.0, -0.4, 0.1, 0.0, 0.02)),
             (11, -1, (0.05, 0.0, 0.0, 0.0, 0.0)))  # follower, leader (-1: none), polyval coefficients
BOUND = 0.25  # float32-exact; dofs 0, 1, 4 and 6 range over +-BOUND
BOUNDED = (0, 1, 4, 6)
HOLD_PASSES = 3  # as _step.HOLD_PASSES: solves after the first, a held dof whose descent points inside released


def fixture():
    """The rows' Jacobians, residuals, poses (some on their bounds), rest pose and seeds, float32-exact."""
    import numpy as np
    generator = np.random.default_rng(7)
    jacobian = generator.normal(size=(ROWS, RESIDUALS, DOFS)).astype(np.float32).astype(float)
    residuals = generator.normal(scale=1.0, size=(ROWS, RESIDUALS)).astype(np.float32).astype(float)
    qpos = generator.uniform(-0.2, 0.2, size=(ROWS, DOFS))
    qpos[::2, BOUNDED] = BOUND
    qpos[1::2, BOUNDED] = -BOUND
    rest = generator.uniform(-0.05, 0.05, size=DOFS).astype(np.float32).astype(float)
    seeds = generator.uniform(-0.2, 0.2, size=(ROWS, DOFS)).astype(np.float32).astype(float)
    exit_rows(jacobian, residuals)
    lower = np.full(DOFS, -1e30)
    upper = np.full(DOFS, 1e30)
    lower[list(BOUNDED)], upper[list(BOUNDED)] = -BOUND, BOUND
    return jacobian, residuals, qpos.astype(np.float32).astype(float), rest, seeds, lower, upper


def exit_rows(jacobian, residuals):
    """Rows 0 (dofs 0 and 1 on their upper bound) and 1 (on their lower): both push out, and once dof 1 is held dof
    0's descent points back inside, so the active set must release it (holds only added leave its step 0)."""
    for row, sign in ((0, 1.0), (1, -1.0)):
        jacobian[row] = 0.0
        residuals[row] = 0.0
        jacobian[row, 0, 0], jacobian[row, 0, 1], jacobian[row, 1, 1] = 1.0, -2.0, 1.0
        residuals[row, 0], residuals[row, 1] = sign, -sign


def prototype_step(jacobian, residuals, qpos, rest, lower, upper):
    """card/proto/lm.py's step in float64 (hinge-only: coordinate = dof), its holds over HOLD_PASSES passes (a held dof
    released when its model gradient points its descent back inside), with its held rows and released count."""
    import numpy as np
    jac = jacobian.copy()
    spread = np.ones((ROWS, DOFS))
    for follower, leader, poly in COUPLINGS:
        if leader >= 0:
            lead = qpos[:, leader] - rest[leader]
            slope = sum(power * poly[power] * lead ** (power - 1) for power in range(1, 5))
            jac[:, :, leader] += jac[:, :, follower] * slope[:, None]
            spread[:, leader] += slope ** 2
    free = [dof for dof in range(DOFS) if dof not in FROZEN and dof not in {c[0] for c in COUPLINGS}]
    sub = jac[:, :, free]
    grad = np.einsum("frd,fr->fd", sub, residuals)
    hess = np.einsum("frd,fre->fde", sub, sub)
    hess[:, range(len(free)), range(len(free))] += (1e-2 * (residuals ** 2).sum(1) + 1e-9)[:, None] * spread[:, free]
    step = -np.linalg.solve(hess, grad[..., None])[..., 0]
    at = qpos[:, free]
    held = np.zeros_like(step, dtype=bool)
    on_low, on_high = at <= lower[free] + 1e-9, at >= upper[free] - 1e-9
    released = 0
    for _ in range(HOLD_PASSES):
        slope = np.einsum("frd,fr->fd", sub, residuals + np.einsum("frd,fd->fr", sub, step))
        inside = held & ((on_low & (slope < 0)) | (on_high & (slope > 0)))
        released += int(inside.sum())
        held = (held & ~inside) | (~held & ((on_low & (step < 0)) | (on_high & (step > 0))))
        keep = ~held
        hess_kept = hess * (keep[:, :, None] & keep[:, None, :]) + np.eye(len(free))[None] * held[:, :, None]
        step = -np.linalg.solve(hess_kept, (grad * keep)[..., None])[..., 0] * keep
    full = np.zeros((ROWS, DOFS))
    full[:, free] = step
    return full, np.linalg.norm(step, axis=1), int(held.any(1).sum()), released, free


def prototype_projected(qpos, rest, seeds, lower, upper):
    """card/proto/gpu.py's `projected`: ranges, couplings by polyval, frozen from the seeds."""
    import numpy as np
    found = np.clip(qpos, lower, upper)
    for follower, leader, poly in COUPLINGS:
        lead = found[:, leader] - rest[leader] if leader >= 0 else 0.0
        found[:, follower] = rest[follower] + sum(poly[power] * lead ** power for power in range(5))
    found[:, list(FROZEN)] = seeds[:, list(FROZEN)]
    return found
