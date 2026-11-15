"""The six-link hinge chain the objectives check runs on, and Newton's own objectives' rows on it (test fixture,
test_the_card_step_and_projection_equal_the_cpu_prototype.py)."""

LINKS = 6
OBJECTIVE_ROWS = 6


def chain_model():
    """A six-link hinge chain (axes x, y, z in turn, 0.3 m apart) on the card."""
    import newton
    import warp as wp
    builder = newton.ModelBuilder()
    parent, joints = -1, []
    for link in range(LINKS):
        body = builder.add_link(xform=wp.transform((0.0, 0.0, 0.3 * link), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.05)
        axis = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))[link % 3]
        joints.append(builder.add_joint_revolute(parent, body, axis=axis, parent_xform=wp.transform(
            (0.0, 0.0, 0.3 if parent >= 0 else 0.0), wp.quat_identity())))
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device="cuda:0")


def residuals_and_jacobian(model, poses, objectives):
    """Newton's IK residuals and analytic Jacobian at poses [rows, coords] for one objective list."""
    import warp as wp
    from newton import ik
    solver = ik.IKSolver(model, poses.shape[0], objectives, optimizer="lm",
                         jacobian_mode=ik.IKJacobianType.ANALYTIC)._impl
    context = solver._ctx_solver(wp.array(poses.astype("float32"), dtype=wp.float32, device="cuda:0"))
    solver._residuals_analytic(context)
    return context.residuals.numpy().copy(), solver._jacobian_at(context).numpy().copy()


def log_rate(turn):
    """SO(3)'s inverse left Jacobian at a log turn: the exact turn Jacobian is it times Newton's spin rows."""
    import numpy as np
    angle = np.linalg.norm(turn)
    across = np.array([[0.0, -turn[2], turn[1]], [turn[2], 0.0, -turn[0]], [-turn[1], turn[0], 0.0]])
    bend = 1.0 / angle ** 2 - (1.0 + np.cos(angle)) / (2.0 * angle * np.sin(angle))
    return np.eye(3) - 0.5 * across + bend * across @ across


def objective_rows(coords):
    """OBJECTIVE_ROWS rows' poses, offsets, unit turns, weights, places and unit wanted turns, seeded alike."""
    import numpy as np
    generator = np.random.default_rng(11)
    poses = generator.uniform(-1.0, 1.0, size=(OBJECTIVE_ROWS, coords))
    offsets = generator.normal(0.0, 0.05, size=(OBJECTIVE_ROWS, 3)).astype(np.float32)
    turns = generator.normal(size=(OBJECTIVE_ROWS, 4)).astype(np.float32)
    turns /= np.linalg.norm(turns, axis=1, keepdims=True)
    weights = generator.uniform(0.5, 2.0, OBJECTIVE_ROWS).astype(np.float32)
    places = generator.normal(0.0, 0.5, size=(OBJECTIVE_ROWS, 3)).astype(np.float32)
    wanted = generator.normal(size=(OBJECTIVE_ROWS, 4)).astype(np.float32)
    wanted /= np.linalg.norm(wanted, axis=1, keepdims=True)
    return poses, offsets, turns, weights, places, wanted
