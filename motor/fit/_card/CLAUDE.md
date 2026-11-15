# motor/fit/_card/

Level 3 on the card (#70, the card build plan's step 1; owner 2026-10-29: "the card is why we go for MuJoCo and its
ecosystem"): mink's per-frame step batched over rows on the RTX 5080, in Warp 1.18.0 beside Newton 1.7.0. The mink
path in `../_frames.py` stays the reference and the one that runs until the card path equals it within the plan's
tolerances (cost ratio |x-1| <= 1e-5, angles <= 0.01 deg, sites <= 0.01 mm, every frame); nothing here replaces it
before that.

- `_step.py` (`Reduction`, `damped_step`): mink's damped Gauss-Newton step (1e-2 |r|^2 + 1e-9 on every full dof's
  diagonal, so S^T S in the reduced dofs: a leader's damping times 1 plus its followers' slopes squared; damped once,
  heavy_squat's row 148 swung elv_angle_r and its slope -1 follower 1 rad a step, never stopping, 2026-10-29) in the
  reduced dofs (frozen out, each follower's Jacobian column onto its leader's through polyval's slope, three
  active-set passes after the first solve holding dofs at a bound that push out (one pass let a toy row with two
  dofs on their bounds push a third out and back every step), a held one released when its model gradient
  J^T (r + J s) points its descent back inside (2026-10-29: holds only added kept heavy_kimodo's knee on 0 in the
  chain gate, mink's 4e-4 rad inside; test rows 0 and 1)), solved by Warp's tile Cholesky, Newton's own
  IKOptimizerLM pattern. Reduced size on MyoFullBody measured 37 (128 dofs, 40 frozen, 51 followers, no follower
  leads another, 2026-10-29), inside sm_120's shared memory (fits up to 79); the 37 kernel equals float64 to 1.6e-7
  rad on 497-residual rows, 0.43 ms a step at 136 rows, 10.4 ms at 4,000.
- `_project.py` (`Projection`, `project`): after each step, ranges clipped, followers set to polyval of their
  leader, frozen coordinates back to the row's seed, in MuJoCo's qpos layout (Newton's quaternion order converted
  by the caller).
- `_objective.py` (`PlacePerProblem`, `TurnPerProblem`): Newton's IKObjectivePosition and IKObjectiveRotation
  (canonicalised error; the turn's Jacobian the log's exact one, mink's: SO(3)'s inverse left Jacobian times
  Newton's spin rows, which alone are off by the log's bend 0.1 rad from the target) with offset and weight per problem (rows of several bodies or
  level-1 sets in one batch), analytic; equal to the built-in ones row by row (test file, six-link chain, 1e-6).
- `_forward.py` (`Forward`, `frames`): every row's link frames in float64 (free, revolute, prismatic, fixed; Newton's
  eval_fk maths), for the residuals; `_objective.wide_residuals` gives both objectives' residuals in float64 on them.
  Its joint places and axes are Newton's, which add_mjcf builds in float32: on mean's jump (2026-11-06) its float64
  frames of mink's pose stood up to 7.3 um off MuJoCo's at the left hand (0.25 um a body, jumps at the welded
  bimanual attachment's six-digit turn), so a fit on them minimises another model. `Exact` (`frames` takes either) lays
  the same link frames from MuJoCo's compiled model in float64 (`_level2_wide._kinematics`, mj_kinematics' steps, one
  set) and `Exact.drive(optimizer)` has Newton's analytic Jacobian read the same (its FK's link frames these in
  float32, each hinge's twist (anchor x axis, axis) and slide's from MuJoCo's anchors and axes; the free root's
  Newton's own, exact at its identity place): `_solve.solved(exact=)`, `_clip.Clip.angles(exact=True)`, which level 3
  (`_level3`) uses; level 1's `_sets`, `evaluation._card_batch` and `rra._footed_card` still read Newton's places.
- `_solve.py` (`solved`, `NotConverged`): the loop (Newton's analytic Jacobian in float32, the residuals and the stop
  rule's cost in float64 (`_forward`), the damped step, Newton's integrate, the projection; root quaternion order
  converted on the card), a row stopping when its reduced step is under 1e-6 or its cost moved under 1e-10 of itself
  over two steps (1e-9 until 8258bcab: mean_squat's chain stopped 0.047 deg short in elv_angle_r's flat valley), or (as mink's frames, `../_frames.solved`, 2026-10-29) a step moved its float64 residuals by under
  1e-8 (arms_raise's shoulder gimbal drift; its proto-clip and toy tests unchanged, its own break test and the
  arms_raise gate not yet run); NotConverged at 300. Since 2026-10-31 every step after the first is one CUDA graph (`wp.ScopedCapture` of
  the step, replayed; the step count on the card, `_counted`, read by `_stopped`): the first step runs eagerly so
  every kernel is loaded before the capture. The level 1 profile showed ~270k Python launches over 4 sets of tall's
  7 clips (~12 s of 72 under cProfile); the toy clip's angles and steps are bit for bit the same captured or not
  (4 tiled copies, night/gpu1/check/toy_solve.py), and the sampled frames' card phase went 7.3 to 3.5 s on 4 sets. Measured 2026-10-29 on the proto clip (136 frames, each row seeded as mink's
  frame was), float32 residuals: angles 0.020 deg and sites 0.029 mm max, all in elv_angle_l/r (the shoulder's
  elevation plane near the hanging arm, a flat direction); mink's answer is the float64 minimum there (scipy's least
  squares at 1e-15 from both answers lands on mink's to 4e-5 deg and 0.0002 mm), so the miss is the card's. With the
  residuals in float64 and the cost stop at 1e-7: angles 0.012 deg, sites 0.017 mm (2 frames over); at 1e-9 (the
  rule): EQUAL on every frame, angles 0.0025 deg, sites 0.0042 mm, cost ratio 2.2e-8, couplings 1.3e-7 rad, ranges
  held, 14 / 26 steps median / max (a step stop of 1e-8 never fires: 114 rows NotConverged at 300). Speed with float32 residuals (one batch, no chunking
  needed up to 12,784 rows): 1,925 rows/s at 1,088, 4,094 at 4,080, 6,889 at 12,784 (1.86 s, 2.7 GB on the card).
- The clip test (2026-10-29, in the test file): a toy MJCF (free root, three ranged hinges, a polyval follower in its
  own body since Newton merges two hinges in one body into a D6, a frozen hinge), 10 frames, mink set up as
  `_frames.Frames` (a place target turned as its site now; a target turned identity adds SE(3)'s V^-1 term and moved
  mink 0.8 deg off the plain place minimum): card equals mink to 0.0012 deg, costs equal to 5 digits, 5 to 28 steps.
- `solved_in_chunks`: `solved` over CHUNK_ROWS (8,000, the plan's) rows at a time, the caller's `optimizer_for(start,
  stop)` one IKSolver a chunk (its buffers are per row); equal to one batch on the toy (test).
- Gate measure (2026-10-29, exact turn Jacobian, three hold passes, one batch, proto clip tiled): EQUAL on every frame
  (angles 0.0017 deg, sites 0.0042 mm, cost ratio 2.0e-8, couplings 1.3e-7 rad, ranges held, 13 / 38 steps median /
  max); 136 rows 2.38 s (57 rows/s, the first launch's compile in it), 1,088 0.89 s (1,223/s, 244 MiB), 4,080 1.60 s
  (2,558/s, 916 MiB), 12,784 3.81 s (3,359/s, 2,740 MiB on the card). Nothing replaces mink yet.
- `_seeds.py` (`first`, `kept`, `newton_order`; plan step 2): iteration 0 starts each frame from mink's own angles
  for it (angles.npz, written by `evaluation._case` beside a clip's measures), later iterations from each frame's own
  last accepted solve; a cold seed (rest pose) is never used. Seeded from the last iteration's angles with the toy's
  places moved 0.5 or 3 mm, one row cycled for good (2,000 steps): the float32 pose's floor, not the step (the float64
  step equals the card's; its model gain 3.7e-12, the pose's rounding moves the cost 9e-9 either way). `_solve`'s cycle
  stop (cost back within 1e-2 of the change between, two steps apart, by a step under 1e-4 since 8258bcab, mean_squat's 2-cycle at mink's angle stepping 3e-5) ends it; a monotone fall
  never fires it. The seed check is in the test file (3 mm on: equals mink). On the proto clip (2026-10-29), each row
  seeded from mink's own answer for its frame (the cache) and mink re-solved from the same seed: EQUAL on every frame
  with the places as they were and moved 0.5 or 3 mm (angles 0.0018 deg, sites 0.0036 mm, cost ratio 2.1e-8 at most,
  steps median / max 3 / 12 to 9 / 24). Speed shares the card with env1: 4,080 rows took 1.5 to 1.7 s with the card
  idle and 3.7 s with env1 holding it at 99%; quote rows/s only with nvidia-smi's utilisation beside it.
- `_measure.py` (`measures`, `Skins`, `Posed`; plan step 3): the measures level 1 scores on ("joints mm", "per joint
  mm", "turns deg", "per group deg", "bones outside skin %", "bones outside depth mm", "bones outside all %"), keyed as
  `evaluation._measure`; the winding sign and the depth outside by `fit._skin.marks_kernel`, the CPU's own kernel (the
  depths within 4.6e-6 mm of the CPU's on the toy, float32 closest points, 2026-10-30);
  its other readings stay on the CPU. The skin posed every EVERY-th frame once per clip across every nudged set (`posed_skins`, keyed by `posed_key`: the rig, segments and worlds a set never changes; float32 points kept read-only on the host, `Skins` a Warp mesh each per case; 5.8 s of acr7's 4-set profile, 2026-10-30);
  the poses by MuJoCo's own float64 kinematics on the host (with MuJoCo Warp's float32 poses one hand vertex fell
  across the skin on heavy_kimodo, 2026-10-29); joint and turn errors float64 Warp kernels (the turn by
  arccos of the trace as the CPU: atan2 differed 0.001 deg on targets not quite orthonormal; the target is
  SOMA's turn as its nearest rotation, `evaluation._measure.segment_turn`, since 2026-11-01: the played worlds sit up to
  3.5e-6 off orthonormal and the arccos read a 0.06 degree shoulder turn as 0, the taped measure 0.06); the winding test on the
  card. Gate 2026-10-29 (heavy_kimodo, heavy_rigwalk from the seed cache): every measure equal to 4e-13, the skin
  shares exact, no rounded value differs; 0.05 / 0.09 s against the CPU's 2.8 / 2.5 s, the skins cached apart.
- `_level2.py` (`jacobian`; plan step 4, first piece): level 2's exact Jacobian of `../_constants.Problem.residuals`
  in its own sparse pattern, for `least_squares(jac=...)`: per frame the unknowns reach the dofs by `dof_map` (root
  place one for one, its rotation vector by SO(3)'s right Jacobian, MuJoCo's free joint turning in its body frame, a
  leader its dof, a follower by polyval's slope), place rows the site's place Jacobian, turn rows -J_l^-1(log)
  R_site^T times the site's turn Jacobian, a marker's move its body's turn, an inside joint's move its parent's turn
  on every site it carries, the priors diagonal. The frames' kinematics are MuJoCo's float64 on the host
  (`Kinematics`, the reference); `_level2_wide` gives the same on the card in float64 (below). Before 2026-10-31 the
  card's were MuJoCo Warp's float32 (`_level2_card.CardKinematics`, every sampled frame a world; retired with
  `fit.card_jacobian` once `_level2_wide` replaced it: 1e-5 of a column off, and the host still made level 2's
  residuals and polish). On fit's toy
  (test, `tests/card_level2.py`) the host's equals central differences to 1.8e-10 of a column.
  Gate on heavy (host kinematics, 2026-10-29, build/4/gate.py): 12,333 rows x 1,483 unknowns, 283,149 nonzeros; the
  exact Jacobian equals scipy's 2-point to 8.0e-7 of a column (plan 1 %); level 2 with jac=exact against today's:
  constants 0.0011 mm, coordinates 3.2e-4 deg, cost 6.5e-10 relative, both 24 evaluations, status 2; 6.4 s against
  36.7 s (a Jacobian 0.20 s on the host, 1.7 s by differences). With the card's kinematics: 5.2e-6 of a column;
  constants 0.019 mm, coordinates 5.8e-4 deg, cost 8.7e-9 relative, 6.2 s against 35.5 s. `frame_rows` gives COO
  triplets (the frame's own block's nonzeros, the move blocks vectorised, `moved_blocks` and the model's `couplings` made once per Jacobian, not
  per frame, 809dd8a2; each frame's qpos too, given by `jacobian`, which made it already for the kinematics: 1.67 s of
  frame_rows' 5.6 s on heavy x 4 sets under cProfile went to making it twice, 2026-10-30; the toy's follower is
  linear in its leader, so the test file cannot see a wrong frame's pose there; each frame's turn rows' -J_l^-1(log) R_site^T made for every frame and group in one pass, `turn_bends`, scipy's log and `left_jacobian_inverses` batched, 2026-10-30: under cProfile 0.98 s and 0.90 s per-frame loops on heavy x 4 sets; equal per row to 2e-15, test; a dense block over every
  unknown then csr took the Jacobian to 0.20 s): heavy's Jacobian 0.066 s with the card's kinematics, level 2 3.0 s
  against 32.3 s (constants 0.015 mm, coordinates 0.0020 deg, cost 6.4e-9 relative, 25 evaluations against 24). Wired as an
  option: `../_constants.staged(jacobian=...)` (2-point differences the default and the reference; `constants` does
  not pass it until the 10 tuning bodies' gate is equal); the toy solved both ways in the test file.
- `_clip.py` (`Clip`; plan step 5, first piece): the card loop's parts made from a fitted MuJoCo model as
  `../_frames.Frames` sets mink up (the turn and marker sites with `_frames.weights`' costs, the active joint
  couplings, the followers-free model's slide and hinge ranges, the frozen dofs) beside a Newton model of the same
  file; `angles(turns, points, seeds)` every frame from its own seed. The toy's fixture is set up by it (equal to the
  hand setup bit for bit, 2026-10-29); joint types compared as ints (MuJoCo's enums never equal the model's ints).
  `for_fitting(fitting, path)` builds it for a `Fitting` from the person's file (constants written there), refused
  when Newton's sites at rest miss MuJoCo's by over `MAPPED` (1e-5 m). Its ranges are `followers_free_ranges` (the joint
  arrays `coordinate_bounds` reads, the followers' limits off), not `Fitting.limited_model`'s deep copy of the whole
  model: 2.5 s of tall_kimodo's 14 s card process went to copying the meshes (2026-10-29, cProfile); angles bit for
  bit the same, the toy's setup checks the two bounds equal. On heavy_kimodo's gate dump (2026-10-29,
  build/5/clip_real.py, each frame seeded from mink's cached answer): sites 0.0073 mm apart at rest; 136 rows 1.55 s,
  3 / 32 steps; angles 0.00035 deg, sites 0.0013 mm off mink's.
- `_per_row.py` (`per_row_fk`; level 3 across sets, first piece): Newton's batched FK with each row's joint places
  (`joint_X_p`) read from a [sets, joints] array by the row's set, so one batch can hold every set's frames (rows = set
  x frame) though each set's inside-joint moves change its body places (Newton's IK has no worlds: one model, so the
  places go per row). Newton's own `eval_single_articulation_fk`, so equal to `eval_fk_batched` on each set alone bit
  for bit (2026-10-30, the chain, 3 sets x 4 frames). `rowwise(optimizer, set_places, row_set)` overrides an
  IKOptimizerLM instance's `_fk_two_pass` and `_compute_motion_subspace` with the same per-row read (Newton's own
  kernels, the place read by the row's set; an instance override, not a subclass, since Newton specialises the class
  per size): its analytic residuals and Jacobian equal each set's own model bit for bit (2026-10-30, the chain, a
  position objective). `_forward.Forward(model, set_places, row_set)` does the same in float64 (rows read their set's
  places; several sets with no row_set refused): equal to Forward on each set alone bit for bit (2026-10-30).
  `_solve.solved` / `solved_in_chunks(..., set_places=, row_set=)` (host arrays; chunks slice row_set) wire both: the
  toy's clip as 3 sets (inside joints moved ~2 mm, targets 1 mm apart) in one batch equals each set alone within
  0.01 deg / 0.01 mm (`tests/card_across.py`, 2026-10-30; at ~1 cm moves from mink's base seeds 3 rows did not
  converge in 300 steps, not looked into). `_sets.Sets(clips)` (one `Clip` per set, set 0 the base) builds the set x
  frame rows from several sets, each row its own set's marker offsets, weights and targets on set 0's model, the
  places per row; refused when a set's reduction or projection (frozen, limited, couplings) or its sites or bodies
  differ from set 0's. `Sets.angles(turns, points, seeds)` (one array per set) equals each set's `Clip.angles` alone
  within 0.01 deg / 0.01 mm on the toy (`tests/card_sets.py`: 3 sets, inside joints, offsets, weights and targets
  apart, 2026-10-30). The couplings' polys go per set too (`Reduction.poly` / `Projection.poly` [sets, couplings, 5],
  read by the row's set in `_step._slopes` and `_project._projected`; one set [1, ...] is shared by every row;
  `Sets` stacks each set's and `refusal` leaves `poly` out): a real body's nudged set's file rounds its polys (7e-7
  max, 1.5e-5 relative, 2026-10-30), so equal-poly sets alone would refuse it; on the toy with each set's follower
  slope 0.05 apart, equal to each Clip alone, reading set 0's for every row 4.2 / 10.2 deg off. Toy timing, 4 sets x 10 frames warm: one batch 186 ms against 384 ms alone (2.07x, equal
  exactly). Still to come: the body run calling `Sets`, and the 4-set timing on a real body's short clip.
- `newton_model_for` (2026-10-31) keeps Newton's model of a `Fitting`'s file (`BUILT`, weakly keyed by the Fitting):
  asked again, it moves the kept build's inside joints to the fitting's current moves (`set_places`), so level 1's
  set builds Newton once for level 2's sampled frames and once more never (the clips, constants laid on, reuse it).
  4 centre sets x tall's 7 clips (night/gpu2/prof f, g2): every row bit for bit equal, the clip builds 4.5 to 2.3 s;
  test `card_inside.a_kept_build_moves_to_the_fittings_moves` (red with the kept build returned unmoved).
- Newton's `add_mjcf` merges a MuJoCo body's several joints into one D6 joint into that body, placed at its first
  joint (checked 2026-10-30: hinge, hinge, slide on one body gave one joint `a1_a2_a3`), so `_clip.moved_inside`'s map
  of child body to joint has one entry a body and moving it moves the whole body (`tests/card_inside.py`; keyed by
  the joint's own label instead, KeyError). Myo joints on one body at different places are not checked here. `_clip.set_places(places, joints, held, moves)` (2026-10-30) gives a nudged set's joint places from one
  build (`inside_joints`: the Newton joint into each `Fitting.inside` body) by moving those joints by the set's moves
  less the ones the build holds, so `_sets.Sets` needs no Newton build per set: on the toy equal to a build per set
  (none, 1 and 3 x the move) to 1e-6 m; dropping `held` 10 mm off (test). Not yet called by the body run.
- One test file (`tests/test_the_card_step_and_projection_equal_the_cpu_prototype.py`): a 12-row fixture against the
  CPU prototype's float64 maths, the objectives on a six-link chain (`tests/card_chain.py`), the loop and the seeds on
  the toy clip (`tests/card_toy.py`: the MJCF, mink's answer set up as `_frames.Frames`, the card's setup, the
  errors), the measures on a toy body (`tests/card_body.py`, its check there too), level 2's two checks in `tests/card_level2.py`; it needs Newton's Python and a CUDA card, else prints `skipped`. About 20 s.
- `_skin_step.py` (`BoneSkin`, `SkinRows`, 2026-11-01): level 3's bones-in-skin rows (`../_skin_rows`' three offset
  rows a kept bone vertex, `../CLAUDE.md`) put into `_step.damped_step`'s normal equations as `extra` (J^T J, J^T r and
  their squared sum in the reduced dofs), never stored as rows: per row and body its points' M = c^2 sum A^T P A and
  m = c^2 sum A^T o (`_bodies`, one thread a row and body, its points in order, no atomics), each (body, reduced
  column) pair's world motion from Newton's joint_S_s and the couplings' slopes (`_columns`), then sum V^T M V and
  V^T m (`_normal`). A distinct row skin is one Warp mesh (~3.6 MB on the card: 600 measured 2.2 GB), so with skins a
  chunk is at most SKIN_CHUNK_ROWS rows; rows of several sets on one frame share it. `_solve` judges each step with
  the skin (`_judged`: a trial pose whose float64 cost is not under the kept pose's is turned back and the row's
  added damping raised, `_step`'s `lam`), the stop rule reading kept poses only, as `../_frames.judged_solve`.
  The toy (`tests/card_skin.py`): card equals mink's `SkinTask` to 0.005 deg and 0.003 mm on every frame at
  `TOY_GOAL` (at the real 1e3 on 8 points, 0.018 deg in a flat valley of the same cost).
- `_level2_wide.py` (`Wide`, 2026-10-31; level 1's card process sat at ~1.3 cores with the card near 0 %, level 2's
  per-frame host loops its largest share): a `../_constants.Problem` whose `residuals` and `exact` (its Jacobian)
  are float64 with every frame's kinematics on the card: each frame's pose made for all frames in one array pass on
  the host (`poses`: `Problem.qpos` and `Fitting.coupled`), one launch of MuJoCo's own mj_kinematics steps a thread
  a frame (`_kinematics`: body_pos, body_quat, a hinge about its anchor, a slide, the free joint from qpos; float64),
  one of the picked sites (`_sites`, mj_local2Global) and one of their Jacobian columns (`_jacobians`, mj_jacSite's:
  a hinge's world axis and axis x (point - anchor), a slide's axis, the free joint's world places and body-frame
  turns), then `_level2.frame_rows`' rows for every frame at once (`mappings` its `dof_map`s), in its order. The
  turns' targets enter as scipy reads R_site^T R_target (`polar`: a target off orthogonal at 1e-12 replaced by its
  nearest rotation, which R_site^T carries). Ball joints refused. `staged(wide=True)` uses it for least_squares and
  the polish; `told` stays on the host's Jacobian (with `Wide`'s, equal to 8e-16, the pivoted QR picked the thorax's x
  where the host picks the root's on tall: two equally told axes, the same residuals, 2.85 mm apart, the lumbar
  between them moved). Measured 2026-10-31: the toy's residuals and Jacobian equal the host's to 2.3e-16 and 7.8e-16
  (test, 1e-12, the sparse layout the host's too); tall's set s00000 (30 frames, 1908 told unknowns) residuals
  3.4e-16 / Jacobian 8.3e-16 off the host's (a few entries exact zeros on one side only, so not the host's
  sparse layout there), a residual 1.9 against 9.6 ms and a Jacobian 18 against 45 ms;
  `staged` with it against the host's exact Jacobian throughout: constants 3.6e-7 mm, angles 2.1e-7 deg, 47
  evaluations each, 3.5 s against 6.0 s.
- `_level2_lm.py` (`Set`, `solved`, `steps`, `Arrow`; the card plan's step 2, the owner's exception in `../CLAUDE.md`,
  2026-11-03): level 2's told unknowns by Levenberg-Marquardt (Marquardt's lambda times each unknown's own diagonal,
  as scipy's x_scale="jac"; Nielsen's update; a step clipped into the bounds and kept only when the cost falls; an
  unknown on a bound its descent pushes past held: held within `BOUND_HELD` of it, the first proof's six bodies, 2026-11-03,
  ended 1.1e-8 of their cost over scipy's, their constants and angles within 1e-5 mm and 5e-6 deg). Per round per set the float64 residuals and
  exact Jacobian (`Wide`), the arrow's blocks as sparse products on the host (each frame's W x W, its rows of Hfc, the
  constants' C x C); then every frame of every set in one launch: its block factored by Warp's tile Cholesky in
  float64 (`_frames_kernel`, COLUMNS right-hand columns a block) and solved for [-g_f | Hfc], the Schur sum per set
  (`_schur`); each set's constants by one dense Cholesky on the host (scipy), the frames' steps back from it. The host
  stops a set when its free gradient is under `POLISHED` ("settled"), lambda passes LAMBDA_MOST ("stalled") or ROUNDS;
  `_constants.polished` runs after it as after scipy's. Toy (test, `tests/card_level2.py`): equals `staged(wide=True)`
  within the proof tolerances, settled by itself, its step the dense damped solve to 1e-9 (two sets, a constant held),
  two sets at once each equal alone, and with the constants' bound at 1 mm, where scipy's trf stops short of the
  bound (its cost 3.2e-8 of itself over the card's: the proof judges the card's cost only above scipy's). A wrong
  Schur step still settles (the control hides it), so the step is checked itself. On the real bodies the card's own
  solve ends "stalled" (lambda past LAMBDA_MOST at the float64 floor, free gradient a few 1e-10) and the polish
  settles it. Proved on all 19 tuning bodies and level 1's gate columns, the default since 2026-11-03 (`../CLAUDE.md`).
- `_solve.solved` takes two more things for `rra`'s footing (2026-11-05, #70): `hold`, rows held off the Jacobian with
  no step control (its `normal(slopes, reduced, qpos)`, given the reduced Jacobian and the pose; the skin's `extra` now
  takes them too and reads only the slopes), and `bounds`, each row's own ranges in the step's active set (`_step._kept`)
  and the projection (`_project._projected`), in place of the reduction's and the projection's one set. Neither given,
  every kernel reads as before (level 3 unchanged: the card's tests pass as they did).
- `_level3.py` (`angles`, 2026-11-06, #70 speed): level 3 every frame at once behind `fit.LEVEL3_CARD` (off; mink
  stays the reference, `fit.card_angles` the card's way on its own). mink's frames are a recurrence (each frame's fit
  without the skin from the frame before's fit with it, then the skin's from that), so it is solved in sweeps: every
  frame not yet final one row, seeded from the frame before's last answer (the first from `Frames.start`), its fit
  without the skin then with it (`_clip.Clip.angles`); a frame is final once the frame before is final and moved
  under SETTLED_TURN (1e-4 rad) / SETTLED_PLACE (2e-6 m) between sweeps, and the first frame not final is final after
  the next sweep by construction, so at most one sweep a frame, the later ones over the frames not final only. The
  first sweep's seeds: a chain without the skin every STRIDE-th (5) frame, one row a solve. Toy (fit's test file,
  `level_3_on_the_card_is_minks_frames`, 17 frames in the shin's skin): 2 sweeps (17 then 16 rows), angles 1.5e-5
  deg, sites 0.0002 mm, bone depths 0.00014 mm off mink's. The toy has one minimum a frame, so the recurrence's
  history (elv_angle's flat valley, float32 Jacobians) is the real bodies' measure (fair-level3-card), not its.
  On mean's jump and env1's standing (fair-level3-card, L4, 2026-11-06) the sweeps on Newton's places stopped
  0.0059 to 0.0073 deg off mink in the shoulder girdle (sternoclavicular_r2/r3 with their unrotscap followers at -1:
  the arm keeps its turn, the clavicle moves only its place, a weak direction), sites 0.020 to 0.024 mm, 6 frames a
  clip over the gate, every frame's girdle biased ~0.002 deg; not the recurrence (mink from the card's frame before
  lands on its own answer to 1e-13 deg), not the stop rule (stops off, the card sat on the same cost for 100 steps),
  but the model: the card's own float64 cost was lower at its answer than at mink's (Newton's float32-built places).
  On MuJoCo's frames for the residuals alone the bad frames re-solved from mink's frame before land within 0.00015
  deg and 0.0005 mm of mink, but the whole jump on them still left frame 133 0.0249 mm off: seeded from the first
  sweep's frame 132 (under 1e-4 rad from its last, so final) the float32 Jacobian on Newton's places found no step
  that lowered the cost and stopped (LAM_MOST); with the Jacobian on MuJoCo's geometry too (`drive`) from the same
  seed 0.00016 mm. Whole clips on Warp's CPU device (`_clip.DEVICE` set to cpu by hand; the module's default is
  cuda:0 and `_solve` captures CUDA graphs only on a card), 2 sweeps each: jump (180 frames) joints 0.00043 deg,
  sites 0.00089 mm, bone depths 0.00018 mm, the same 22,467 points outside; standing (150) 0.00016 deg, 0.00055 mm,
  0.00019 mm, 19,221 outside: every frame within the gate (scratch l3fix).
  The test (`level_3_on_the_card_reads_minks_own_model_at_the_shoulder`, a toy girdle with MyoFullBody's own places)
  checks the frames equal mj_kinematics' to 1e-12, the hinge twists MuJoCo's to 1e-6, and that every level 3 solve
  reads both.
- **A clip's rows get CLIP_STEPS (3000), a footing's MOST_STEPS (300)** (#98, 2026-11-07): seven video squats
  and jumps failed level 3's chain on `NotConverged` at 300, each one row whose shoulder stood at zero elevation (its
  range's end), the plane of elevation seen only through the scapula's couplings: under mink's damping at the clip's
  cost (about 0.02) its steps crawled along elv_angle and its follower, the cost falling 1e-11 a step; given 3000, four
  stop by the unchanged stop rules (a fit then 455 s, against 50), and three still
  crawl past it and fail: the crawl itself (mink's damping in a direction the targets barely see) is not fixed.
  `solved(judged_from=...)` judges the steps without the skin too (`rra`'s card footing: the hold's cost in the
  judged cost, `Held.cost`), its damping lowered on every kept step and never put back to 0.
