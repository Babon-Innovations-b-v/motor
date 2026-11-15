# motor/rra/

OpenSim's residual reduction (RRA, the parent overlay): `reweighed` (RRATool's mass and torso centre of mass step),
`reduced` (the pelvis' and trunk's capped, low-passed move), `within_follows`, `within_frames`, and the measures the
`adjusted` record reads (`turned_off`, `off_clip`); `_linear.py` holds what a state asks of the body and its
linearisation, `_cap.py` the caps, `_footed.py` `Footing` (the feet kept where they stood by the legs: mink per frame,
set up as the fit's and through its `fit.solved`, a frame not converged fails the run; the handwritten damped least
squares it replaced, 40 fixed steps with ranges by clipping, went 2026-10-29: one solver in sim_body).

- Moved unchanged in #70's split; changed since by #70 only where its review asked (2026-10-29): the mass step bounded
  (`MASS_MOST`) and `Footing` on mink; the torso's centre of mass moved only across its own mean up (`level`,
  2026-11-09, #98), as RRATool moves it in its body's x and z: fitted in all three axes, the vertical one read off the
  frames' small tilts alone and nearly every video run took it to the 10 cm cap.
- `Footing` holds each moving leg joint in its range, or where the frame it puts the feet back to (`before`, the
  fitted path) already stands past a stop, within where it stands (2026-11-03, #70): camp's walking astronaut_1, its
  right knee carried 0.38 degrees past its 0 stop by the path's low-pass on 27 of 150 frames, had that knee pulled
  in at no move, the hip turned 18 degrees to keep the foot, so `within_frames` found those frames 3.55 times the cap
  with every share at 0 and failed the run. The cap and the follows limit are unchanged.
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
- On a CUDA card (2026-11-05, #70) `Footing` given the person's file solves every frame at once there
  (`_footed_card.CardFooting`: the fit's level-3 card loop, `fit._card._solve.solved`, with Footing's own tasks, each
  row's own ranges and `Held`, the card's FreeHold: Gram-Schmidt on the tasks' reduced Jacobian); its float32 legs'
  joints come back with the frame's own frozen coordinates and the followers set again in float64. mink stays the
  reference and the solve without a card; the test's toy holds the two within 1e-4. RRA's residuals, the trunk's
  moments and the linearisation's sensitivities (`sensitivities`, `CARD_STEPS`) take every frame's state at once on
  the card (`batched`).
- `Footing` holds each leg's free directions where the frame stands them (`FreeHold`, 2026-11-04, #70): the null
  space of the feet's and the pelvis' height's weighted Jacobian over the moving leads, so it moves no foot. A leg has
  more joints than its foot's six directions (the ankle's twist; the knee's shin turn freed from the flexion), and at
  the fit's damping a step along a direction no foot sees was unbounded: env1's seated astronaut's left leg, its
  subtalar on its stop, cycled for good and failed the run (`fit.NotConverged`).
  The directions are the frame's own pose's, fixed for its solve, and its steps judged (`fit.solved(...,
  judged_from=FOOTED_LAM_FIRST)`, Levenberg-Marquardt's control from 1e-6) (2026-11-07, #98): at a straight leg the
  bend left the feet's rank and came back a hair off it, so a hold found again each step held it on one step and let it
  go on the next, and the undamped step at a straight knee bent it 0.7 rad; video drop jumps and walks (knees
  straight at take-off, landing and heel strike, the estimator's a hair past their stop) cycled 300 steps.
  Its judged damping is lowered ten times on every kept step and never put back to 0 (`decaying`, Levenberg-Marquardt's
  own rule), a kept step at no more than its first damping that lowers the cost by under FOOTED_GAIN of it ends the
  frame (a foot no pose reaches), and a frame whose feet already stand where they stood (FOOTED_SETTLED, a micrometre)
  is its own answer, on the card too (2026-11-07, #98): put back to 0, every other step was the undamped one turned
  back and five drop jumps' near-straight legs, their hip's rotation, shin's turn and ankle's twist all but one axis,
  crept 300 steps at 1e-12 to 8e-7 of cost; and RRA's frame check, lowering a move to nothing, got the card's float32
  legs back a rounding off the fitted path. On the card the same: `Held` finds its directions at the solve's first
  pose and keeps them, and the steps are judged with the hold's cost (`fit._card._solve.solved(judged_from=...)`);
  before, a straight leg's frames never stopped there and went to mink. `CardFooting`'s Newton-against-MuJoCo check is
  0.1 mm (`FK_MATCHED`): Newton's float32 forward kinematics stands the right file's bodies 0.003 to 0.011 mm off.
- `within_frames` keeps the move's bones inside SOMA's skin as the fitted path holds them, frame by frame and region by
  region (`fit.FrameSkins`, `skin_excess`, `SKIN_SLACK`), its shares smooth (`held_share`) and the move off the fitted
  path low-passed with the feet kept again (`smoothed_off`) (2026-11-05, #70): the freed joints' pickup kept 0.75 of a
  move that turned the arms and head with the trunk, its hands 37.7% outside the skin, its shins 30 times the clip's
  jerk.
- `cap_rows` moves, lands and poses the path moved by `total` and each column's nudge of it as one stacked path
  (`moved` takes copies x frames, each frame's couplings' map once; the run's `land` takes whole copies of the clip,
  each against the fitted path), so on the card every nudge's legs are one solve, not one a column; `sensitivities`
  keeps a frame's couplings' map for a nudged state whose leaders stand as the frame's (2026-11-06, #70: the two
  together 19 s in 3 calls of the jump's 135 s). Bit for bit the column-by-column answer: each row's solve is its own.
- `within_follows` keeps RRA's whole move for a fitted path already further off its clip than the `follows` limit
  (#98, 2026-11-07): no move meets a limit the fit itself is past (most video fits 5 to 9 degrees off, every
  share 0; held at the fit's own measure, still 0), so the
  per-frame cap (`within_frames`) holds it and the `follows` gate fails the fit's own excess.
