"""`solvers.least` for many problems at once on the card (#70, 2026-11-05): every frame's static optimisation (one
bounded least squares a frame, `muscles.Physics.shared`) side by side, in float64, by the same projected Newton steps
(Bertsekas 1982): the gradient, the bounds that hold it back fixed, the Newton step on the rest by Woodbury's identity
(the inner matrix solved by elimination with partial pivoting, as numpy's), the step halved until the cost falls
(every halving's trial at once, the first that lowers the cost enough taken), a problem done when its projected
gradient is under SOLVE_TOLERANCE of its largest linear term and left as it stands from then on. Problems of different
widths are padded with columns of nothing bounded at 0, which never move. A round is one CUDA graph for every problem,
its stops read every CHECK_EVERY rounds."""
import functools

import numpy as np
import warp as wp

from motor.solvers import SOLVE_ROUNDS, SOLVE_TOLERANCE

DEVICE = "cuda:0"
HALVINGS = 35  # the line search's trial steps, 1 to 2^-34: `least` stops halving under a step of 1e-10, 34 halvings
TILE_WIDTH = 16  # the tiled products' columns a step (fp64 tiles: two of rows x 16 in shared memory)
TILE_THREADS = 64
ROW_BLOCK = 32  # a tiled product's rows a block
LANES = 64  # a block's threads over one sum's columns (each problem's row, pair of rows or halving a block)
CHECK_EVERY = 10  # rounds between reads of which problems still go (a round is one captured CUDA graph)


@wp.kernel(enable_backward=False)
def _times(system: wp.array3d[wp.float64], x: wp.array2d[wp.float64], out: wp.array2d[wp.float64],
           active: wp.array[wp.int32]):
    """out = system x, per problem and row: a block a row, its lanes over the columns."""
    problem, row, lane = wp.tid()
    if active[problem] == 0:
        return
    total = wp.float64(0.0)
    for column in range(lane, x.shape[1], LANES):
        total += system[problem, row, column] * x[problem, column]
    found = wp.tile_extract(wp.tile_sum(wp.tile(total)), 0)
    if lane == 0:
        out[problem, row] = found

@wp.kernel(enable_backward=False)
def _gradient(system: wp.array3d[wp.float64], product: wp.array2d[wp.float64], x: wp.array2d[wp.float64],
              squares: wp.array2d[wp.float64], linear: wp.array2d[wp.float64], gradient: wp.array2d[wp.float64],
              active: wp.array[wp.int32]):
    """system^T (system x) + squares x - linear."""
    problem, column = wp.tid()
    if active[problem] == 0:
        return
    total = wp.float64(0.0)
    for row in range(product.shape[1]):
        total += system[problem, row, column] * product[problem, row]
    gradient[problem, column] = total + squares[problem, column] * x[problem, column] - linear[problem, column]


@wp.kernel(enable_backward=False)
def _linear(system: wp.array3d[wp.float64], wanted: wp.array2d[wp.float64], linear: wp.array2d[wp.float64]):
    """system^T wanted."""
    problem, column = wp.tid()
    total = wp.float64(0.0)
    for row in range(wanted.shape[1]):
        total += system[problem, row, column] * wanted[problem, row]
    linear[problem, column] = total


@wp.kernel(enable_backward=False)
def _projected(x: wp.array2d[wp.float64], gradient: wp.array2d[wp.float64], highest: wp.array2d[wp.float64],
               enough: wp.array[wp.float64], active: wp.array[wp.int32], projected: wp.array[wp.float64]):
    """Each problem's projected gradient's largest entry; a problem under its `enough` is done."""
    problem, lane = wp.tid()
    if active[problem] == 0:
        return
    largest = wp.float64(0.0)
    for column in range(lane, x.shape[1], LANES):
        value = x[problem, column]
        clipped = wp.min(wp.max(value - gradient[problem, column], wp.float64(0.0)), highest[problem, column])
        largest = wp.max(largest, wp.abs(value - clipped))
    found = wp.tile_extract(wp.tile_max(wp.tile(largest)), 0)
    if lane == 0:
        projected[problem] = found
        if found <= enough[problem]:
            active[problem] = 0

@wp.kernel(enable_backward=False)
def _free(x: wp.array2d[wp.float64], gradient: wp.array2d[wp.float64], highest: wp.array2d[wp.float64],
          projected: wp.array[wp.float64], free: wp.array2d[wp.float64],
          active: wp.array[wp.int32]):
    """1 for a column its bound does not hold (Bertsekas' epsilon: a bound this near, pushed on, holds), else 0."""
    problem, column = wp.tid()
    if active[problem] == 0:
        return
    near = wp.min(projected[problem], wp.float64(1e-6))
    value, slope = x[problem, column], gradient[problem, column]
    held = (value <= near and slope > wp.float64(0.0)) or (value >= highest[problem, column] - near
                                                          and slope < wp.float64(0.0))
    free[problem, column] = wp.where(held, wp.float64(0.0), wp.float64(1.0))


@wp.kernel(enable_backward=False)
def _weighed(system: wp.array3d[wp.float64], squares: wp.array2d[wp.float64], free: wp.array2d[wp.float64],
             weighed: wp.array3d[wp.float64], active: wp.array[wp.int32]):
    """system_free squares_free^-1, the held columns 0."""
    problem, row, column = wp.tid()
    if active[problem] == 0:
        return
    weighed[problem, row, column] = free[problem, column] * system[problem, row, column] / squares[problem, column]


@wp.kernel(enable_backward=False)
def _trials(x: wp.array2d[wp.float64], step: wp.array2d[wp.float64], highest: wp.array2d[wp.float64],
            trials: wp.array3d[wp.float64], active: wp.array[wp.int32]):
    """Every halving's trial point, x + 2^-halving step clipped to the bounds (problems x columns x halvings)."""
    problem, column, halving = wp.tid()
    if active[problem] == 0:
        return
    size = wp.pow(wp.float64(0.5), wp.float64(halving))
    trials[problem, column, halving] = wp.min(wp.max(x[problem, column] + size * step[problem, column],
                                                     wp.float64(0.0)), highest[problem, column])


@functools.cache
def _products(rows, columns):
    """A tiled kernel, one block a problem and ROW_BLOCK of its rows: left [problem] (rows x width) times right
    [problem] (width x columns), or, `columns` None, times right [problem]^T (rows x rows): Warp's tile matrix
    products over the width TILE_WIDTH columns at a time (each problem's system read once a product, not once an
    entry; fp64 tiles of a block's rows fit the card's shared memory where all of MyoFullBody's 101 rows did not)."""
    block, tile = wp.constant(ROW_BLOCK), wp.constant(TILE_WIDTH)
    square = columns is None
    column_count = wp.constant(rows if square else columns)

    def kernel(left: wp.array3d[wp.float64], right: wp.array3d[wp.float64], active: wp.array[wp.int32],
               out: wp.array3d[wp.float64]):
        problem, part = wp.tid()
        if active[problem] == 0:
            return
        first = part * block
        total = wp.tile_zeros(shape=(block, column_count), dtype=wp.float64)
        for start in range(0, left.shape[2], tile):
            mine = wp.tile_load(left[problem], shape=(block, tile), offset=(first, start))
            if wp.static(square):
                theirs = wp.tile_transpose(wp.tile_load(right[problem], shape=(column_count, tile),
                                                        offset=(0, start)))
                wp.tile_matmul(mine, theirs, total)
            else:
                theirs = wp.tile_load(right[problem], shape=(tile, column_count), offset=(start, 0))
                wp.tile_matmul(mine, theirs, total)
        wp.tile_store(out[problem], total, offset=(first, 0))

    kernel.__name__ = kernel.__qualname__ = f"least_products_{rows}_{columns}"
    return wp.kernel(enable_backward=False, module="unique")(kernel)


@wp.kernel(enable_backward=False)
def _plus_identity(inner: wp.array3d[wp.float64], active: wp.array[wp.int32]):
    """inner + I: Woodbury's inner matrix."""
    problem, row = wp.tid()
    if active[problem] == 0:
        return
    inner[problem, row, row] = inner[problem, row, row] + wp.float64(1.0)


@wp.kernel(enable_backward=False)
def _trial_products(tiled: wp.array3d[wp.float64], products: wp.array3d[wp.float64], active: wp.array[wp.int32]):
    """The trials' products laid halving by halving (problems x halvings x rows) from the tiled product (problems x rows
    x halvings)."""
    problem, halving, row = wp.tid()
    if active[problem] == 0:
        return
    products[problem, halving, row] = tiled[problem, row, halving]


@wp.kernel(enable_backward=False)
def _trial_costs(products: wp.array3d[wp.float64], wanted: wp.array2d[wp.float64], x: wp.array2d[wp.float64],
                 trials: wp.array3d[wp.float64], squares: wp.array2d[wp.float64], gradient: wp.array2d[wp.float64],
                 costs: wp.array2d[wp.float64], along: wp.array2d[wp.float64], active: wp.array[wp.int32]):
    """Each halving's trial cost, 0.5 (|system moved - wanted|^2 + squares moved^2), and the gradient along its
    move."""
    problem, halving, lane = wp.tid()
    if active[problem] == 0:
        return
    total = wp.float64(0.0)
    for row in range(lane, wanted.shape[1], LANES):
        left = products[problem, halving, row] - wanted[problem, row]
        total += left * left
    slope = wp.float64(0.0)
    for column in range(lane, x.shape[1], LANES):
        moved = trials[problem, column, halving]
        total += squares[problem, column] * moved * moved
        slope += gradient[problem, column] * (moved - x[problem, column])
    found = wp.tile_extract(wp.tile_sum(wp.tile(total)), 0)
    sloped = wp.tile_extract(wp.tile_sum(wp.tile(slope)), 0)
    if lane == 0:
        costs[problem, halving] = wp.float64(0.5) * found
        along[problem, halving] = sloped


@wp.kernel(enable_backward=False)
def _pulled(system: wp.array3d[wp.float64], squares: wp.array2d[wp.float64], free: wp.array2d[wp.float64],
            gradient: wp.array2d[wp.float64], pulled: wp.array2d[wp.float64],
            active: wp.array[wp.int32]):
    """system_free squares_free^-1 gradient_free."""
    problem, row, lane = wp.tid()
    if active[problem] == 0:
        return
    total = wp.float64(0.0)
    for column in range(lane, squares.shape[1], LANES):
        total += free[problem, column] * system[problem, row, column] * gradient[problem, column] / squares[
            problem, column]
    found = wp.tile_extract(wp.tile_sum(wp.tile(total)), 0)
    if lane == 0:
        pulled[problem, row] = found

@wp.kernel(enable_backward=False)
def _solve(inner: wp.array3d[wp.float64], pulled: wp.array2d[wp.float64], active: wp.array[wp.int32],
           solved: wp.array2d[wp.float64]):
    """inner^-1 pulled by Gaussian elimination with partial pivoting, in place on `inner` and `solved`, as numpy's
    solve (LAPACK's getrf): the inner matrix holds the contacts' columns over their 1e-6 squared costs and the balance
    rows weighed 100, so its condition reaches 1e14, where a Cholesky factor's steps went astray (2026-11-05: the
    clips' muscles 65% more active than `least`'s)."""
    problem = wp.tid()
    if active[problem] == 0:
        return
    size = pulled.shape[1]
    for row in range(size):
        solved[problem, row] = pulled[problem, row]
    for column in range(size):
        pivot = column
        largest = wp.abs(inner[problem, column, column])
        for row in range(column + 1, size):
            value = wp.abs(inner[problem, row, column])
            if value > largest:
                largest = value
                pivot = row
        if pivot != column:
            for other in range(size):
                held = inner[problem, column, other]
                inner[problem, column, other] = inner[problem, pivot, other]
                inner[problem, pivot, other] = held
            held = solved[problem, column]
            solved[problem, column] = solved[problem, pivot]
            solved[problem, pivot] = held
        for row in range(column + 1, size):
            factor = inner[problem, row, column] / inner[problem, column, column]
            for other in range(column, size):
                inner[problem, row, other] = inner[problem, row, other] - factor * inner[problem, column, other]
            solved[problem, row] = solved[problem, row] - factor * solved[problem, column]
    for back in range(size):
        row = size - 1 - back
        value = solved[problem, row]
        for later in range(row + 1, size):
            value -= inner[problem, row, later] * solved[problem, later]
        solved[problem, row] = value / inner[problem, row, row]


@wp.kernel(enable_backward=False)
def _step(system: wp.array3d[wp.float64], squares: wp.array2d[wp.float64], free: wp.array2d[wp.float64],
          gradient: wp.array2d[wp.float64], solved: wp.array2d[wp.float64], step: wp.array2d[wp.float64],
          active: wp.array[wp.int32]):
    """The Newton step on the free columns by Woodbury: -(gradient - system^T solved) / squares, 0 where held."""
    problem, column = wp.tid()
    if active[problem] == 0:
        return
    back = wp.float64(0.0)
    for row in range(solved.shape[1]):
        back += system[problem, row, column] * solved[problem, row]
    step[problem, column] = -free[problem, column] * (gradient[problem, column] - back) / squares[problem, column]


@wp.kernel(enable_backward=False)
def _cost(product: wp.array2d[wp.float64], wanted: wp.array2d[wp.float64], x: wp.array2d[wp.float64],
          squares: wp.array2d[wp.float64], cost: wp.array[wp.float64],
          active: wp.array[wp.int32]):
    """0.5 (|system x - wanted|^2 + squares x^2), `product` system x."""
    problem, lane = wp.tid()
    if active[problem] == 0:
        return
    total = wp.float64(0.0)
    for row in range(lane, wanted.shape[1], LANES):
        left = product[problem, row] - wanted[problem, row]
        total += left * left
    for column in range(lane, x.shape[1], LANES):
        total += squares[problem, column] * x[problem, column] * x[problem, column]
    found = wp.tile_extract(wp.tile_sum(wp.tile(total)), 0)
    if lane == 0:
        cost[problem] = wp.float64(0.5) * found

@wp.kernel(enable_backward=False)
def _chosen(costs: wp.array2d[wp.float64], along: wp.array2d[wp.float64], before: wp.array[wp.float64],
            size: wp.array[wp.float64],
            active: wp.array[wp.int32]):
    """`least`'s line search: the first halving whose cost is under the cost before plus 1e-4 of the gradient along its
    move, or the last (`least` stops halving under a step of 1e-10: 2^-34)."""
    problem = wp.tid()
    if active[problem] == 0:
        return
    halvings = costs.shape[1]
    chosen = halvings - 1
    for halving in range(halvings):
        if chosen == halvings - 1 and costs[problem, halving] <= before[problem] + wp.float64(1e-4) * along[
                problem, halving]:
            chosen = halving
    size[problem] = wp.pow(wp.float64(0.5), wp.float64(chosen))


@wp.kernel(enable_backward=False)
def _taken(x: wp.array2d[wp.float64], step: wp.array2d[wp.float64], size: wp.array[wp.float64],
           highest: wp.array2d[wp.float64], active: wp.array[wp.int32]):
    """x moved by its chosen step, clipped to the bounds, for a problem still going."""
    problem, column = wp.tid()
    if active[problem] != 0:
        x[problem, column] = wp.min(wp.max(x[problem, column] + size[problem] * step[problem, column],
                                           wp.float64(0.0)), highest[problem, column])


def card(values, dtype=wp.float64):
    """A float64 card array of `values`."""
    return wp.array(np.ascontiguousarray(values, dtype=np.float64 if dtype == wp.float64 else np.int32), dtype=dtype,
                    device=DEVICE)


def least_all(systems, wanteds, costs, highests):
    """`solvers.least` of every problem (lists: system rows x columns_i, wanted rows, costs and highest columns_i;
    rows alike), on the card at once: each problem's x."""
    count, rows = len(systems), len(wanteds[0])
    widths = [len(values) for values in costs]
    width = max(widths)
    if width == 0:  # nothing to share: no contact touches on any frame
        return [np.zeros(0) for _ in range(count)]
    system, wanted = np.zeros((count, rows, width)), np.array(wanteds, dtype=float)
    squares, highest = np.ones((count, width)), np.zeros((count, width))
    for problem, (mine, cost, high) in enumerate(zip(systems, costs, highests)):
        system[problem, :, :widths[problem]] = mine
        squares[problem, :widths[problem]] = np.asarray(cost, dtype=float) ** 2
        highest[problem, :widths[problem]] = high
    system, wanted, squares, highest = card(system), card(wanted), card(squares), card(np.minimum(highest, 1e300))
    linear = wp.zeros((count, width), dtype=wp.float64, device=DEVICE)
    wp.launch(_linear, dim=(count, width), inputs=[system, wanted], outputs=[linear], device=DEVICE)
    enough = card(SOLVE_TOLERANCE * np.maximum(1.0, np.abs(linear.numpy()).max(axis=1, initial=0.0)))
    x, gradient, step, free = (wp.zeros((count, width), dtype=wp.float64, device=DEVICE) for _ in range(4))
    product = wp.zeros((count, rows), dtype=wp.float64, device=DEVICE)
    pulled, solved = (wp.zeros((count, rows), dtype=wp.float64, device=DEVICE) for _ in range(2))
    inner = wp.zeros((count, rows, rows), dtype=wp.float64, device=DEVICE)
    projected, before, size = (wp.zeros(count, dtype=wp.float64, device=DEVICE) for _ in range(3))
    products = wp.zeros((count, HALVINGS, rows), dtype=wp.float64, device=DEVICE)
    tiled = wp.zeros((count, rows, HALVINGS), dtype=wp.float64, device=DEVICE)
    trials = wp.zeros((count, width, HALVINGS), dtype=wp.float64, device=DEVICE)
    weighed = wp.zeros((count, rows, width), dtype=wp.float64, device=DEVICE)
    parts = -(-rows // ROW_BLOCK)
    costs, along = (wp.zeros((count, HALVINGS), dtype=wp.float64, device=DEVICE) for _ in range(2))
    active = card(np.ones(count), wp.int32)

    def one_round():
        """One projected Newton step for every problem still going (`least`'s loop body)."""
        wp.launch(_times, dim=(count, rows, LANES), inputs=[system, x], outputs=[product, active], block_dim=LANES,
                  device=DEVICE)
        wp.launch(_gradient, dim=(count, width), inputs=[system, product, x, squares, linear],
                  outputs=[gradient, active],
                  device=DEVICE)
        wp.launch(_projected, dim=(count, LANES), inputs=[x, gradient, highest, enough, active], outputs=[projected],
                  block_dim=LANES, device=DEVICE)
        wp.launch(_cost, dim=(count, LANES), inputs=[product, wanted, x, squares], outputs=[before, active],
                  block_dim=LANES,
                  device=DEVICE)
        wp.launch(_free, dim=(count, width), inputs=[x, gradient, highest, projected], outputs=[free, active],
                  device=DEVICE)
        wp.launch(_weighed, dim=(count, rows, width), inputs=[system, squares, free], outputs=[weighed, active],
                  device=DEVICE)
        wp.launch_tiled(_products(rows, None), dim=[count, parts], inputs=[weighed, system, active], outputs=[inner],
                        block_dim=TILE_THREADS, device=DEVICE)
        wp.launch(_plus_identity, dim=(count, rows), inputs=[inner, active], device=DEVICE)
        wp.launch(_pulled, dim=(count, rows, LANES), inputs=[system, squares, free, gradient], outputs=[pulled, active],
                  block_dim=LANES, device=DEVICE)
        wp.launch(_solve, dim=count, inputs=[inner, pulled, active], outputs=[solved], device=DEVICE)
        wp.launch(_step, dim=(count, width), inputs=[system, squares, free, gradient, solved], outputs=[step, active],
                  device=DEVICE)
        wp.launch(_trials, dim=(count, width, HALVINGS), inputs=[x, step, highest], outputs=[trials, active],
                  device=DEVICE)
        wp.launch_tiled(_products(rows, HALVINGS), dim=[count, parts], inputs=[system, trials, active], outputs=[tiled],
                        block_dim=TILE_THREADS, device=DEVICE)
        wp.launch(_trial_products, dim=(count, HALVINGS, rows), inputs=[tiled], outputs=[products, active],
                  device=DEVICE)
        wp.launch(_trial_costs, dim=(count, HALVINGS, LANES),
                  inputs=[products, wanted, x, trials, squares, gradient], outputs=[costs, along, active],
                  block_dim=LANES, device=DEVICE)
        wp.launch(_chosen, dim=count, inputs=[costs, along, before], outputs=[size, active], device=DEVICE)
        wp.launch(_taken, dim=(count, width), inputs=[x, step, size, highest, active], device=DEVICE)

    graph = None
    for done in range(SOLVE_ROUNDS):
        if done == 0:
            one_round()  # eagerly: every kernel loaded before the capture
        else:
            if graph is None:
                with wp.ScopedCapture(device=DEVICE, force_module_load=False) as capture:
                    one_round()
                graph = capture.graph
            wp.capture_launch(graph)
        if done % CHECK_EVERY == CHECK_EVERY - 1 and not active.numpy().any():
            break
    found = x.numpy()
    return [found[problem, :widths[problem]] for problem in range(count)]
