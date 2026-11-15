# motor/trial/

MOTOR (#98): the whole trial as one solve on the card, built step by step from the design on #98 (comment
6721156999): 0 feasibility, 1 whole-trial movement, 2 inverse dynamics, contact and the pelvis residual,
3 tuning, 4 muscles (here now), 5 batching, 6 into the stage. `__init__.py` is the interface (`solved`, `solved_on`, `METHOD`);
`_spline.py` the uniform cubic B-spline and its banded normal equations (pure numpy and scipy); `_rows.py` every
frame's Gauss-Newton block of the fit's own cost from level 3's card pieces (`fit._card`), all frames as rows at once;
`_dynamics.py` step 2's rows (the nodes' states, the card's inverse dynamics, the smooth contact, the torques and their
Jacobians); `_stage_a.py` step 2's solve (`dynamics_solved`); `_muscles.py` step 4's muscle paths (MuJoCo's C), MuJoCo's muscle
curves in Warp and the activation dynamics; `_stage_b.py` step 4's solve (`muscles_solved`).

- **Step 1 is the fit's cost over the whole trial, nothing new in it**: the groups' turns, the landmarks' markers and
  the bones inside SOMA's posed skin, exactly as level 3 weighs them, plus the spline's jerk; the leading joints
  inside their ranges by box bounds on their control points. Its jerk weight is not tuned: each coordinate's is the
  one that puts the solve's smoothing at the chain's own CUTOFF (`jerk_weights`). No weight or offset is ever fitted
  against reference angles (the owner's patented method; tuning is step 3's, against force plates and EMG).
- **The root**: its place a spline like a joint's; its turn the fit's turn low-passed (`smooth_turns`, the chain's
  filter) times exp of a body-frame rotation-vector spline. Newton's root columns are carried onto these unknowns
  through Newton's own world twists (`Layout.carried`: S^-1 U J_r), checked against central differences.
- **On in a job by `"motor": true`** (`sim_body.fitted_person`): the path is MOTOR's in place of the fit's low-passed
  one (`run_person` skips `filtered`); RRA and the rest of the chain run on it as before. Off, the run is the chain's.
- **Step 2, stage A** (`_stage_a`, job `"motor": "dynamics"`, `sim_body.run_person` after the skin's stance credit):
  the same spline's unknowns take every node's inverse dynamics (MuJoCo Warp on the card, `batched.Card`), a smooth
  Hunt-Crossley contact with smoothed Coulomb friction of every skin point within `BAND` of its floor (measured from
  the seen floor: the stance credit), its force the motion's alone, and tau = G^T (needed - J_c^T f_c) on the leading
  coordinates: the root's six rows (the pelvis' residual over the weight and weight x the centre's height) by an
  augmented Lagrangian to the stage's limits, the joints' as an effort. Levenberg-Marquardt on the banded normal
  equations (the optimiser the design left to this step: a kept step's Jacobians only; each node's rows touch its three
  control points, so the band is step 1's). Jacobians: the needed forces' on the place by one-sided card differences,
  on the acceleration MuJoCo's mass matrix (exact), on the speed the joints' damping (the velocity product terms left
  out, a Gauss-Newton model; the cost is always exact); the contact's through each point's Jacobian. The contact's
  settings are physical (step 3's tuning kept them); the penalty and the
  effort start as scales of the fit's cost at the warm start. The chain (RRA, the static optimisation) then runs on
  stage A's path and measures it as any path; stage A's own residual, contact force a side and rounds are its record
  under "motor"/"dynamics".
- **Step 3, tuning** (2026-11-08): the contact's settings and the effort share are a job's `"motor settings"`
  (`_dynamics.Settings.of`, refusing a name it does not know; the record keeps them), tuned in SCORE (`tools/labval/motor_tune.py`, on
  training data only, against force plates with stage A's own contact). The landmarks' weights are B3D's EM step (`motor_tune weights`: 1 / each one's
  pooled mean squared miss over the training solves, over the median, inside 0.2-5), a job's `"motor landmarks"`
  (`motor.retrusted`: the fit's trusts times them, the markers' goal kept); every solve records its landmarks' misses
  (`landmark errors mm`). Before stage A's first round the contact is settled (`Contact.settle`): a body whose lowest
  skin point sits under its floor at the warm start has its floor lowered by that depth on that frame (a seat laid at a known height under a skin the video sat lower would otherwise push many times the body's weight). The couplings and the
  place Jacobian's nudge run on all rows at once (`Coupled`, `integrated`: the same numbers as `model.Couplings` and
  `mj_integratePos` row by row).
- **Step 4, stage B** (`_stage_b`, job `"motor": "muscles"`, after stage A in `sim_body.run_person`): stage A's
  spline and every node's 416 activations solved together; each joint's reserve (stage A's torque less the muscles'
  moment arms x MuJoCo's own force, gain a + bias) squared at a weight raised until the actuated ones are within 1
  %BW*ht, the pelvis' rows by stage A's augmented Lagrangian carried on, the activations' effort cubed, MuJoCo's
  activation dynamics as a hinge on each span's excitation. Ligament-held turns, unspanned joints (the neck) and a
  joint pushed back in near an end of its range (STOP_REACH, smooth) carry their torque at STOP_SHARE. Each step
  eliminates the activations node by node (Woodbury) onto the movement's banded system. The muscles' lengths and
  moment arms are MuJoCo's own at every node (`_muscles.Paths`): the design's polynomials of the joints were refused
  on MyoFullBody (61 of 416 muscles, the trunk's and the supraspinatus, past 1 mm / 2 mm on held-out poses); the Gauss-Newton model leaves the arms' own change with the pose out. The cost is evaluated on
  MuJoCo's C (`_dynamics.Exact`: the card's float32 hid the steps), and the speed Jacobian takes every velocity
  product by card differences (stage A's damping-only model left the pelvis' rows 1.3 off along stage B's step and no
  step was kept). The frames' activations are stage B's; the chain still runs on its path and measures it.
- Card only: `solved` raises where Warp sees no CUDA card. The banded solve runs on the host (scipy's
  `solveh_banded`, 5,000-25,000 unknowns, three control points either side).
- Test: `test_one_spline_solve_lowers_the_trials_cost_and_jerk_within_its_ranges.py` (the card's toy, 40 frames;
  step 2 on the same toy built as a person is, a skin grid under its root on a flat floor; step 4 on it with five
  muscles).
