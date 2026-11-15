# motor/fit/

The one solver (#70; SOMA is the truth): MyoFullBody fitted to the clip's SOMA body. `__init__.py` is the interface
(`followed`, `placed`, `targets`, `fitting`, `constants`, `angles`, `following`, `riding`, `constants_record`,
`METHOD`); `_model.py` the fitting model (`Fitting`: MuJoCo's own compile of the person's file, a calibration site per
bone group, a site per landmark marker, the constants laid on `body_pos`/`site_pos` in
place, the stretches' lengths), `_frames.py` mink per frame, `_constants.py` the constants' sparse least squares,
`_follow.py` the judgement and the segments' riding, `_skin.py` the bones outside SOMA's posed skin by region
(warp's winding number, every 10th frame; `marks_kernel` beside it each point's depth outside, its distance to the
skin's closest point, 0 inside, `depth_mm` a region's root mean square: level 1's smooth skin residual, the share
staying the judged measure; one kernel for the CPU and the card), `_frozen.py` main's calibration table frozen (227b29b2), `_card/` level 3's step and projection on the card (plan step 1, being proved equal to mink; mink stays the reference); level 2's exact Jacobian too, and level 2's residuals and Jacobian in float64 with the frames' kinematics on the card (`_card._level2_wide`), which `constants` has `staged` use (`wide=True`; scipy's 2-point differences stay `staged`'s default and the reference, `told` reads the host's).

- **Reuse, not handwritten maths** (owner, 2026-10-28): per frame mink 1.3.0 (Apache-2.0, pinned in
  `pyproject.toml` with qpsolvers 4.13.0 and daqp 0.10.3), its `FrameTask`s for the
  groups' turns (orientation only, on the calibration sites) and the places (position only, each target's turn its
  site's own so the error is the plain place error), `EqualityConstraintTask` as a hard constraint (the couplings,
  OpenSim's default), `ConfigurationLimit` on the leading joints only, `DofFreezingTask` for the fingers; the
  constants by scipy's `least_squares` with bounds. No handwritten iterative solver here; never bring one back.
  One exception, the owner's (2026-11-03, "yes make", #70 comment 6607949871, to the card plan's step 2 in comment
  6607775555): level 2's constants by our own Levenberg-Marquardt with the Schur complement on the card
  (`_card/_level2_lm`: each sampled frame's coordinates eliminated by the batched tile Cholesky level 3 uses, one
  dense Cholesky for the constants, many sets batched; the host keeps `told`, the float64 gradient check `POLISHED`
  and the stop). `_constants.ON_CARD` (on since 2026-11-03: every `staged(wide=True)`; `card=False` gives scipy's,
  `staged_across` several sets at once) after it equalled `staged(wide=True, card=False)` plus the polish on all 19
  tuning bodies (l1j's split, l1j's start point; scratch `/tmp/l2card/proof`): constants within 0.0057 mm, angles
  9e-5 deg, cost 4.4e-9 relative at most (every body at or below scipy's), free gradient 5.1e-11 to 9.9e-11; and
  level 1's 6 gate columns (g03's 15 clips, g10's 9, `/tmp/l2gate`) passed the 2026-10-30 rule (below): relative
  0.03 to 1.6 %, step cosine at least 0.99987, the start set's own card-against-host gap 9.3e-5. Speed per body
  (one set, six bodies sharing the card): the card's solve 1.2 to 2.6 s plus the polish 0.1 to 0.7 s, 1.84 s
  median, against scipy's plus polish 4.1 to 6.6 s, 5.4 s median (38 s against 100 s over the 19); `told` 0.3 to
  0.5 s either way. No other handwritten solver.
- **Weights are OpenSim's IK's** (InverseKinematicsSolver, Simbody's OrientationSensors and Markers goals): each goal
  normalised by its own count, here two goals (the turns, and the landmarks' markers, each weighed by its
  trust); the turns are the main term. Never tune a weight to pass a gate. The markers' goal's sum is each
  `Fitting`'s own `marker_goal` (taken from `_frames.MARKER_GOAL` when `fitting` builds it): it is a level 1 unknown,
  so sets solved across one another each keep their own, never the module's last value.
- **SOMA's group ends are a target** (#70, the owner's plan, 2026-10-29): every bone group's start that is a SOMA
  rig joint (`group_ends`: hips, knees, ankles, toes' bases, sternoclavicular, shoulders, elbows, wrists, the lumbar,
  thoracic and skull joints, the neck's middle too (the hips' middle left out: its two hip ends already hold both its rig joints, never counted twice): one rule for every end, a middle of two bodies' centres
  put on their nearest common holder at rest by `end_site`, the pelvis and the thorax, SOMA's side the mean of its two
  rig joints, `landmarks.soma_joints`) is a marker at MyoFullBody's centre there (`end_marker`; the head's moved by its shift read in
  SOMA's skin, `calibration.END_SHIFTS`, 2026-11-01; every end's shift a level 1 unknown since 2026-11-03, `tuning._ends`, laid into that table by level 1's worker), never moved by the solve, weighed END_WEIGHT times the landmarks' mean trust, the same for
  every end, in the markers' goal beside the landmarks (whose weights keep their own sum, `Fitting.goal`); joints inside
  a group come from the landmarks and couplings. Before (2026-10-29 upstream fit) SOMA's joints were only the measure,
  and before that a SOMA rule centre as target pulled the ankle 16 mm up the shank. Since the ends are a target, the
  joint measure partly measures what the solver pulls on: judge with the bones inside SOMA's skin and the turns on a
  frozen calibration table beside it. Level 1 tunes the turns on the table as it stands (`evaluation._measure.judged_turn`,
  `_card/_measure.turn_errors`' default: what the stage's follows targets ride), main's frozen one recorded beside it.
- **No size unknowns** (#70 level 2: lengths straight from SOMA): the lengths are the sizing's (`inside/sizing`),
  never solved here. The size-factor stage (14 mirrored size groups pulled to 1) never ran and was deleted
  (2026-10-29, the audit's item 9) with its record fields. The constants are each free marker's move off its link,
  pulled to it at its own weight, and the joints inside groups (below); the plantar pads and the skull's five (`landmarks.HELD`)
  never move, so a foot cannot sink through SOMA's sole nor the skull into SOMA's head (2026-11-04).
- **The joints inside groups are person constants** (#70 level 2, 2026-10-29): every body whose parent is another
  body of its own bone group (the lumbar levels, clavicle to scapula, the wrist bones, talus to calcaneus, the radius)
  and that a marker rides, it or a body below it (`Fitting.inside`), moves in its parent, solved in the markers' stage
  with the markers' moves and the sampled frames' angles, once per body, within `MARK_MOST`, pulled to its place at
  the markers' mean weight; never per frame. The moves are written in the person's file (`myo.inside_moved`), so
  Newton's model has them too; the record keeps each in mm (`inside joints moved mm`). No group or joint special-cased. Only the axes the data can tell are solved (`_constants.told`, 2026-10-29: level 2's optimum was
  flat, card and 2-point ending 1.1-1.55 mm apart): an inside joint axis whose column of level 2's exact Jacobian at
  the start (the host's double kinematics, never the card's floats) lies within `TOLD` of the span of every other
  column (the markers' moves, the frames' coordinates, root place included) and of the inside axes kept before it
  in the model's order (QR without pivoting; column-pivoted until 2026-10-31, when tall's c3, a landmark's trust,
  showed the pivot flipping between humphant_l's y and humphant1_l's at a 5 % trust change: level 2's constants
  jumped ~1 cm between level 1's nudged sets, their STEP and STEP/2 central differences 99.9 % apart; in the model's
  order the kept set is the same in every set and they are 0.1 % apart, night/gpu2/c3) stays at its file's place. On the tall walker 38 of 96 axes: the pelvis (its common shift with
  the sacrum is the root's place), the welded root, thorax and cervical spine (the bimanual attachment above takes
  them), the capitate and metacarpals (their markers' own moves take them), each humphant1's y; every other axis,
  the clavicle phantoms and scapulae too, is told (least: a scapula's y, 4e-3).
  The other columns' span is taken by their Gram matrix (`_constants.off_span`, sparse Cholesky) when their smallest
  singular value is over `WELL_TOLD` of their largest (then the pivoted QR keeps their whole span), else by that QR
  (`qr_off_span`); heavy_kimodo: 0.15 s against 2.5-5.6 s, the same axes kept.
- **Level 2 ends settled to float64** (#70, 2026-10-30): after scipy's `least_squares` (trf/lsmr, which stalled
  2.5 to 3 mm short of settled on heavy_kimodo) `_constants.polished` takes Gauss-Newton steps (the host-exact
  Jacobian, a sparse direct solve, halved until the cost falls) on the told unknowns more than `BOUND_HELD` inside
  their bounds until their gradient is under `POLISHED` (1e-10), else raises. When no halving lowers the cost, the
  whole step is judged by the free gradient instead (`steadied`: taken when it lowers it): on mean_kimodo the step's
  predicted fall at 1.59e-10 was 4.4e-17 on a cost of 0.867, under the residuals' own float64 noise (reading the fall
  off the residuals' differences instead of two summed costs did not help), so the cost test refused every halving. A halving is taken only on a strict fall (2026-10-30): with `<=` heavy_kimodo's polish took halvings 1e-13 long at an equal cost round after round and stood at 1.48e-10 for its last 10 of 30 rounds, `steadied` never asked; strict, b4/steps' run (13 sets, three bodies) ended with no polish refused, the 36 it logged under 1e-10 in 14 to 23 rounds. `Problem.path` is float64: the card's
  float32 angles rounded every qpos the residuals read (a 1e-6 step on root z moved it 9.5e-7), so the polish
  stalled at 6.5e-8 and the "exact Jacobian is off" reading was the residuals'. Toy: 2.0e-7 without the polish,
  under 1e-10 with it, 0.028 s; heavy_kimodo's polish 2.4 s. `least_squares` runs at scipy's own tolerances: the
  polish does the settling (at ftol=xtol 1e-12, gtol 1e-10 the whole real solve took 405 s; toy without the polish
  at the loose tolerances 9.8e-7; heavy_kimodo at them: 22 s, polish 5.0 s, free gradient 9.7e-11).
- **The calibration is the static table** (`calibration`), a site's quaternion R_cal^T on the group's frame body:
  never fitted, never per person.
- **The followers' own ranges are not held** (`Fitting.limited_model`): 35 of MyoFullBody's 51 couplings have a
  follower range narrower than its polynomial over its leader's range (the published model's); held, they stopped a
  leader 50 degrees short of a reachable pose (solver study, 2026-10-28). OpenSim's assembler and the old fit did not
  hold them either.
- **One segment drives one bone group** (`followed`, owner 2026-11-01: "the segments to bonegroups actually CAN be one
  to one"): a segment whose group's frame body an earlier group already rides drives none. Only Neck2 shares one
  (MyoFullBody's neck is one rigid body, Neck1's): before, Neck1 and Neck2 both pulled it and it stood at their
  compromise. Neck2 is still judged (its riding and turn measure). Where one to one still cannot be met by
  construction, measured at level 1's start (mean and tall, 14 clips, per group root mean square): Spine1 10.7 and
  Spine2 6.7 degrees (the lumbar levels follow one leader, `LACKS`), the feet 7.8 to 8.2 and toes 8.0 to 8.4 (three
  hinges and a toe hinge), the forearms 5.8 and arms 6.0 (a hinge elbow), the shins 3.8 to 4.6 (a hinge knee), the
  neck 4.7 and 3.7; and Hips 3.9 and the thighs 3.6 to 4.0, whose joints could reach SOMA's turn: a fault still to
  find (the neighbours' misses shared through the turns' least squares is the guess, not measured). Markers or group
  ends 100 times weaker moved none of them by over 0.3 degrees.
- **Every group SOMA has is a turn target** (`followed`): leaving out the groups with no joint of their own (thorax,
  neck, head) or only followers (lumbar above L5, shoulder girdle) measured worse on the Kimodo walker (joints 24 mm
  median against 17, 2026-10-28, before SOMA's joints were a goal).
- **The constants are one sparse problem** with the angles of `SAMPLE` frames (`_constants.Problem`): alternating
  mink and the constants gave the same answer on the walker (joints 16.9 against 16.7 mm median) but converged
  linearly and did not give back the toy's constants (test); the sparse solve takes about a minute on 30 frames.
- What stays (measured 2026-10-29 on both test walkers, median over the frames, SOMA's joints no goal): the joints
  16 mm off SOMA's, the shoulders 44 to 50, the head 20 to 25 (rigid on the thorax); the groups' turns 5.2 to 5.9
  degrees on the median of those not at a limit, the head 23 (rigid). A frame that does not converge in `MOST_STEPS` fails
  the run (`NotConverged`), never kept silently. A frame's fit ends on a step under `STEP_DONE`, its own norm or its change to the
  tasks' and couplings' weighted errors (2026-10-29: at zero shoulder elevation elv_angle and its follower cancel, a
  direction no task sees; mink drifted along it 1e-4 rad a step for ever on every body's arms_raise; with the rule
  mean, heavy and short converge on all 7 clips, the other clips' angles moved at most 7e-6 rad).
- **No card, the CPU's reference** (2026-11-03: a 16-core machine with no CUDA failed every sim-body row on
  "Invalid device identifier: cuda:0"): when Warp sees no CUDA device (`on_card`) the stage's level 2 poses its sampled
  frames by mink (`posed_paths` -> `mink_paths`, each recording from its own start) and solves its constants by
  scipy's `least_squares` on the host's exact Jacobian (`staged(..., Problem.exact, wide=False)`); level 3 is mink
  (`fit.LEVEL3_CARD` off; on, with a card, `_card/_level3`'s sweeps, every frame at once) and the skin measures take Warp's CPU device (`_skin.card_device`). With a card nothing changes. On env1's
  person_centre (CUDA hidden against the card through the card queue, both side by side on this PC, 20 and 16 min)
  every bones-outside share and every turn measure of both followings equal to the record's rounding, the markers'
  moves and inside joints too (0.0 mm apart); level 2's evaluations 22 against 21. Level 2's card solve
  (`_level2_lm`, `_level2_wide`) itself runs on Warp's default device, so `staged(wide=True)` also runs with no card.
- **Mink stops on its joint bounds where the card does not** (#70, 2026-10-30): on heavy_short_legged frame 16 mink
  ends at its ConfigurationLimit bounds (cost 0.307) while the card's solve ends at 0.0094 inside every range; that
  frame held two right-arm constants on their 0.03 m bound (level 2 cost 0.900 against 0.602 with card frames). So
  level 2's sampled frames are posed on the card, across sets (`card_paths`, `constants_across`: one launch a frame for every set; `evaluation/_card_sets`
  solves each body's constants that way); the bar is no worse than the CPU per body, better where mink is stuck. Why (measured 2026-10-30): not
  an early stop. From frame 15's seed mink's first full step is 2.0 rad (tangent norm) and lands in a second basin;
  its next 66 steps shrink geometrically, the cost flat at 0.30691 to 1e-15, and `moved < STEP_DONE` fires at a true
  local minimum. Seeded at the card's frame 16, mink stays there (cost 0.00929). So a toy for the card's edge needs a
  first step that overshoots into another basin, not a hinge driven past its range (that one mink slides along).
  Across sets against one set after another (measured 2026-10-30, card frames on both): 32 sets of heavy_arms_raise
  8.9 s a set against 17.8 s a nudged set alone; gate6a on heavy_kimodo + tall_kimodo, step cosine at least 0.99975
  on all 6 columns, relative under 1 % but ASIS_l trust 2.26 %, whose card-against-CPU gap (6.3e-5 a step) is the size
  of the start set's own gap (7.7e-5): float noise on a column that barely moves tall_kimodo, not a wiring error.
  The gate's rule (coordinator, 2026-10-30): a column passes at a relative gap of at most 2 %, or, above that,
  when its absolute gap is no larger than the start set's own card-against-CPU gap; on that rule all 6 pass.
  Card frames against mink frames on heavy_arms_raise (2026-10-30): the earlier gap (joints mm max 64.5 against
  59.6, shin bones outside 2.1 % against 0.65 %) came from fitting level 2 on that one recording against the body's
  seven, not from the card. Like with like (seven recordings, set s00000) card frames read joints mm max 59.4 against
  59.6, bones outside 13.9 % against 14.7 %, foot 17.6 against 19.2 %, hand 49.4 against 54.2 %; worse by a hair on
  shin (0.82 against 0.65 %, depth 0.19 against 0.17 mm) and thigh (18.0 against 17.7 %). That hair is the solvers'
  tolerance, not a fault: level 3's leg and spine angles differ at most 0.0005 rad, the root at most 0.9 mm, the leg
  constants are equal, so bone points within 0.2 mm of the skin flip side; only the shoulder chain moves (at most
  0.022 rad, its constants traded within their box), which is where the hand and forearm gain.
  On the body's seven clips (heavy, 4 real sets, measured 2026-10-30 with the card and CPU shared): across sets 70.8 s
  against 112.1 s one after another, measures within 0.0033; about 5.7 s against 22 s a nudged set after the first.
  Tall's seven clips the same way (4 sets): 73.5 s against 110.5 s, measures within 0.033 mm; about 4.7 s against
  23.2 s a nudged set. Every tuning body has seven clips, so a Jacobian's ~4.5 h would be ~0.9 to 1.15 h, one
  process a body, if a process holding 138 sets keeps a set's card memory (measured on 4 sets only).
- **The skin rows are the stage's, not level 1's** (#70, 2026-11-01, the coordinator's decision): level 3 with the
  skin rows is not a smooth function of level 1's unknowns (a few frames jump between minima under small moves; heavy
  frame 28 0.052 rad apart between two solves of the same route), so level 1's gradient through it cannot be checked
  (the gate's references 7 to 91 % apart among their steps). One switch decides: `sim_body.clip_angles` (and
  `fitted_person`) holds them on the stage's per-frame fit, the film's people; level 1's
  evaluation, sets and blocks run without them (`evaluation.SKIN_ROWS = False`: `_case.fitted_case`, which then
  poses no skin at all, `targets(..., skins=False)`,
  `_card_batch.batched_angles`, `tuning/_adjoint/_real`), its objective the bones outside the skin measured on that
  skinless fit, as in run l1g. The taped skin rows (`tuning/_adjoint/_taped`) stay for later, unused by level 1.
- **Level 3 holds the bones inside the skin; level 2 does not** (#70, 2026-11-01, the owner: "we just know bones need
  to be inside the skin"): `targets` gives SOMA's skin posed on every frame (`_skin.PosedSkins`, MuJoCo's frame) beside
  the turns and markers, and level 3 holds every kept bone vertex (`_skin.bone_bodies`, the measure's 80 a mesh) in it:
  `_skin.skin_cost` (sqrt(SKIN_GOAL / points), one goal over its own count as OpenSim weighs its goals) times its
  offset off the frame's skin, zero inside by the winding number, the closest point the nearest of the float32
  query's face and the faces touching its corners (`_skin.rings`: the query's face flips between near-tied
  neighbours) measured in float64 (`_skin.apart_kernel`; float32 depths left level 3's cost noisy at 2e-6 of itself,
  so no stop rule fired). The offset's Jacobian in the point is the projector off the closest feature (n n^T on a
  face, I - e e^T on an edge, I at a corner); level 3 steps on the offset's three rows a point (`_skin_rows.BonePoints`):
  a depth row's own Gauss-Newton square (n n^T everywhere) misses the skin's bend at the triangle mesh's edges and
  corners, and its steps overshot there (a toy frame crawled 300 steps, a real sampled frame cycled for good). Mink's
  `_frames.SkinTask` (its QP objective the offset rows, its error the depth rows) and the card's `_card/_skin_step`
  (the same rows put into the damped step's normal equations), both with each step judged (Levenberg-Marquardt's
  control, `_frames.judged_solve`, `_card/_solve._judged`): plain steps ran away from a cold sampled frame (cost 3e4
  to 5e8). Without skins every solve is as before, bit for bit. Level 2 (the person's constants, `_constants`) holds
  no skin rows (the coordinator's decision, 2026-11-01, on the sweep below): with them its polish did not settle (the
  depth's max(0, d) kink cycling one coordinate's rows) and it traded the person's constants for skin, inside joints
  moved 40 to 52 mm; its sampled frames are posed without the skin too (`card_paths`). Never bring them back without
  a polish that settles on them and constants that stay the person's.
- **A pose's skin is measured once** (#70 speed, 2026-11-05): mink's judged step asks `SkinTask`'s error twice and
  its QP objective once at one pose, and a turned-back step asks again at the kept pose (pickup: 5,438
  closest-point launches for about 1,660 skin steps, 35 s of level 3's 50 on an L4). `SkinTask._apart` keeps the
  last `SEEN` (2) poses' measures by the bone points' bytes, cleared by `set_skin`; the kernel is deterministic, so
  the angles are the same bit for bit (toy: 71 launches to 22, test). Frames cannot share one launch: each frame's
  fit without the skin starts from the frame before's fit with it, so lockstep frames would change the solve.
- **The skin goal, swept** (mean's 15 clips, l1g's point, the fold rule on, 2026-11-01; scratch
  `/tmp/skinl23/runs/s5_*`). Skin rows in levels 2 and 3: level 2's polish does not settle (goal 1e2: 8.4e-9 after 80
  rounds, the stall rule ending it at round 84 when given 160; 1e3: 3.2e-7; 1e4: 1.1e-3; env1 seated at 1e2: 1.5e-7
  after 160), the max(0, d) kink cycling one coordinate's rows; and at 1e2 it trades the person's constants for skin
  on the 31 sampled frames: inside joints moved 40 to 52 mm (scapulae, humeri, radii, sacrum), joints 12.1 -> 30.0 mm,
  forearm outside 2.7 -> 34.3 %, shoulder 1.7 -> 18.6 %, all 11.45 -> 13.53 %. Skin rows in level 3 only (level 2
  without): all outside 11.45 -> 9.57 / 6.95 / 4.88 % at 1e2 / 1e3 / 1e4 (thigh 19.5 -> 15.9 / 8.8 / 3.5, foot 23.9
  -> 20.9 / 15.3 / 10.6, hand 23.3 -> 15.6 / 9.4 / 5.4, shin 12.0 -> 10.0 / 6.6 / 5.2, head 19.8 -> 19.2 / 17.4 /
  15.0), turns 4.662 -> 4.61 / 4.603 / 4.735 deg (hands 0.15/0.26 -> 1.12/1.46 at 1e3, Spine1 10.83 -> 11.13), joints
  12.07 -> 12.49 / 14.14 / 15.51 mm, every level 3 frame stopped. The 81 of 281 measured frames whose skin folds are
  apart in every run (their bones 24.1 % outside). SKIN_GOAL is 1e3. On env1 (level 3 only, the first frame's fix, l1apply's point; scratch
  `/tmp/skinl23/s5/env1/final_g1e2`, `final_g1e3`), main -> 1e2 -> 1e3: seated all outside 20.96 -> 20.45 -> 13.82 %,
  turns free median 4.75 -> 4.74 -> 4.90 deg, residual after RRA 0 -> 0.21 N, 3.28 N m -> 1.36 N, 8.46 N m (frames
  within the residual limits 100 -> 98.7 -> 96.7 %); standing all 21.71 -> 20.48 -> 13.48 %, turns 4.25 -> 4.25 ->
  4.67 deg, residual 3.99 N, 5.77 N m -> 4.47, 9.45 -> 3.43, 5.15 (within limits 84.0 -> 78.7 -> 87.3 %). No goal
  leaves both env1 people's turns and residual no worse: 1e2 barely moves the bones and raises the standing residual,
  1e3 moves them most and raises the seated residual on a few frames.
- **Every frame takes its skin rows from its own fit without them** (`Frames.fitted`, the first frame's since
  2026-11-01, every frame's since 2026-11-04): from the frame before, the skin rows pulled a fast-swinging arm onto the
  hips', Spine2's or chest's skin and held it there for many frames, inside the step limit, so nothing raised (the
  mean body's running wrist 94 to 168 mm off SOMA's on 7 of its 11 scored frames, kimodo's 109 to 309 mm, rigwalk's
  right arm from frame 60 on); re-solving the stage's own frames so, running's wrist 17 to 21 mm and kimodo's 15 to
  18 mm, in about half the time (the skin rows start near their answer). The first frame's history: the skin
  pulls a bone point to the nearest skin, whichever part's, and counts it inside any part's. From the standing start
  env1's seated astronaut's hanging forearms stood in his thighs' skin: frame 0 ran 62 judged steps into a pose with
  the radii and scaphoids beside the thighs (their nearest skin LeftLeg/RightLeg), the arms, forearms and hands 23 to
  43 degrees off, and every later frame started from it (level 3 alone, goal 1e3, every 10th frame: forearm 9.8 ->
  63.9 %, shoulder 2.6 -> 36.7 %, head 5.5 -> 31.5 %). From its own fit without the skin: all 21.0 -> 13.8 %, forearm
  18.3, shoulder 4.1, head 5.4, hand 31.6 -> 19.2, thigh 25.1 -> 15.1 %, every group's turn within 0.5 degree of the
  fit without the skin. Standing (env1 centre) hid the same fault behind a better share: from the standing start the
  left arm stood 18.3 degrees off (shoulder_elv_l at its range end on every frame), all 21.7 -> 8.7 % but shoulder 3.5
  -> 12.4, head 5.9 -> 13.3; from its own fit all 13.5, shoulder 9.2, head 6.7 %, the left arm 4.9 degrees. Solving
  every frame so instead read the same (seated all 14.1 %) for 4.2 times the steps. The forearm's rise is the hand's
  trade: the fingers are frozen (`DofFreezingTask`), their second phalanges 13 mm outside SOMA's hand, and the wrist
  carries them in (the second phalanges 13 -> 2 mm deep) with the left radius 0.4 -> 2.7 mm. The mean body's sweep
  never met it: the card's level 3 starts every frame from mink's angles without the skin (`_card/_seeds.first`), so
  its 15 clips read the same after the change (all 6.95 %, turns 4.603 deg, joints 14.14 mm). The whole env1 run
  (level 2 without skin rows, l1apply's point, goal 1e3; scratch `/tmp/skinl23/s5/env1/fix_l3_g1e3`), main's fit ->
  the standing start -> the fix: seated all 20.96 -> 20.10 -> 13.82 %, forearm 9.8 -> 63.9 -> 18.35, shoulder 2.6 ->
  36.6 -> 4.0, head 5.5 -> 31.5 -> 5.4, turns at a limit 600 -> 1744 -> 600 of 3450, residual before RRA 0 -> 200 ->
  1.5 N (after RRA 0 -> 215 -> 1.4 N, 8.5 N m on its worst frame); standing all 21.71 -> 8.65 -> 13.48 %, forearm
  15.5 -> 23.4 -> 15.4, shoulder 3.6 -> 12.4 -> 9.1, head 6.0 -> 13.3 -> 6.7, leading joints at a range end on over
  5 % of frames 5 -> 8 -> 4, after RRA 4.0 -> 4.8 -> 3.4 N.
- **The skin's side is the exact winding number's** (2026-10-31): every skin mesh is built with Warp's
  `support_winding_number=True` (`_skin.marks`, `_card/_measure.Skins`, `tuning/_adjoint/_measures`). Without it
  Warp's sign query misjudged points of a posed skin that crosses itself: a forearm point 5.7 mm off mean_running's
  skin read outside, then inside after a 0.003 mm move (exact winding number 1 both times), and the jump of its depth
  made the trust and end weight columns' differences scale as 1/step on running, squat and sit_stand (night/gpu2/c3);
  on frame 12's skin 1 of 3,000 random points within a centimetre sat on the wrong side, none with it; the test's
  four overlapping spheres 169 of 4,000.
- **The head's pockets are closed for the inside test** (2026-11-01): SOMA's mesh is closed but holds inward
  pockets (the mouth's slit 46 mm wide and 13 mm deep, the eye sockets, nostrils, ears), and the teeth and skull read
  outside in them. `_skin.envelope` adds each `pockets` (faces whose outward normal meets the mesh again within
  POCKET_REACH, mostly riding Head) as its convex hull, a shell of its own after the skin's faces, which keep their
  indices. Every inside test uses it (`bones_outside`, `_card/_measure.Skins`, `evaluation/_measure`,
  `tuning/_adjoint/_real.skin_parts`); sizing uses the skin as it is. On mean_running's re-fit (frames 0, 75, 82) the
  jaw outside went 28.2/42.5/45.5 % to 24.8/38.8/42.8 %, the head region 17.4/31.2/35.5 % to 15.9/29.7/34.4 %, "all"
  under 0.1 point: level 1's score changes with it.
- **A frame whose skin folds into itself has no skin rows** (the owner, 2026-11-01: a deep squat's skin folds into
  itself, which no skeleton can follow): `_skin.folded` is the share of a posed skin's area folding into itself (a
  vertex, by a third of its faces' area, whose point `FOLD_IN` 1 mm in along its normal winds 1.5 or more on the
  skin's own faces, no pockets closed, and `FOLD_OWN` 1.5 or more on the faces across its own segment's joints alone,
  `_skin.joined_faces`: its segment, parent and children). Skin pressing into a segment further off is contact, not a
  fold (#70, 2026-11-01: l1h scored NaN, its five heavy standing clips, g19_sidestep, short_heavy_sidestep and
  _circle, heavy_short_legged_rigwalk and _sidestep, 0.5 to 2.8 % inside themselves by the whole skin on every frame,
  99 of 99 frames folded; the overlap was the upper arms into the chest past the shoulder and a hand on the thigh, not
  the rest pose, which is 0.007 % at most; by the joined skin 0.00 to 0.17 %, 0 of 99 folded. Mean's 15 clips: 81 ->
  52 of 281, every squat and kneel frame over 0.7 % still folded, arms_raise 8 -> 0 and reach 7 -> 0; scratch
  `/tmp/foldfix`). Over `FOLD_MOST` (0.5 %) the frame is left out of the skin rows of levels 1 (the taped
  `tuning/_adjoint/_real.skin_parts`) and 3 (`PosedSkins.kept`, zero offsets in `apart_kernel` and the card's
  `_skin_step._points`; the taped level 3 rows drop its copies) and out of the bones-outside share and depth
  (`bones_outside`, `evaluation/_measure`, `_card/_measure`), each of which reports them apart: `skin folded frames`
  {frames, all: their bone vertices' share outside, and in the stage's record `no unfolded frame`}. A clip
  whose every skin frame folds has no share at all ("bones outside all %" None, no region's share or depth, never 0 or
  NaN): `evaluation.case_score` leaves it out of the score, `reported` and level 1's held_out.json name it ("clips
  with no unfolded frame"), `tuning.whole` gives it no rows, its exact block has none. Its basis (mean's 15 clips, every 10th frame, main's fit before the
  skin rows, 2026-11-01, by the whole-skin rule then): 182 of 281 frames fold over 0, holding 86 % of the bones' squared depth outside; 81 over
  0.5 %, 79 % of it, their bones 24.8 % outside against 16.0 % on the rest; 31 over 2 %, 72 %, 36.8 % against 16.2 %;
  the worst mean_kneel's frames 130 to 140 and mean_squat. The fold does not itself read a bone outside (the union
  counts as inside): it marks poses past what the model's ranges reach. A frame's share costs 35 to 110 ms on the CPU (measured on the 20 clips above).
- **The record states the honest measures first** (#70's audit item 2, 2026-10-29): `following` gives the bones
  outside SOMA's skin by region and the turns on main's frozen table before the joint mm (partly the ends' own pull)
  and the tree's-table turns; the frozen table is never edited to follow the tree.
- `following` and `riding` read the bone-group and landmark tables of MyoFullBody itself, so the toy test does not
  reach them; they are checked on the walkers' runs (#70's numbers).
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
- **The kneecap is not held in the skin** (`_skin.KNEECAP`, #98, 2026-11-07): it rides the quadriceps' tendon where
  MyoFullBody's couplings put it (sizing keeps them), and SOMA's linear-blend skin caves at a bent knee; held there,
  the fit turned the knee to keep it in.
