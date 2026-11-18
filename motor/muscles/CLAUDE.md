# motor/muscles/

The static optimisation, frame by frame (`Physics`): `needed` (inverse dynamics), `contacts` (`contact.pushes` on the
person's skin, `contact.Skin`), `stops` and `shared` (the static optimisation in OpenSim's two stages, `staged`, each a
bounded least squares by `solvers.least`: the ground reaction force first from the pelvis' six rows, RRA's balance
(`balance_asked`) shared over the touching points (`share_asked`, SHARE_COST, SHARE_ROUNDS), then the joints' stops and
the muscles' activations with it fixed (`joints_asked`); owner, 2026-11-05, #70: one least squares of all three was
ill-posed, a 1e-6 nudge of the contacts moving an activation by up to 0.79), its `touches` each touching part's push, centre of pressure (weighed by each push along its contact's
normal) and touching ground points, a foot one part (`parts_of`: its calcaneus and toes under its Foot segment's name;
2026-11-05, the heel's and the toes' pushes apart drew an arrow up the shin and one lying in the floor). What they cannot give is the pelvis' residual and each joint's reserve: measured, never applied (the
parent overlay).

- `shared` is `problem` (the frame's static optimisation, unsolved, from the state laid in `data`), `solved_all` (its
  stages, each every frame's at once: `least_each`, on the card `batched.least_all`, else `solvers.least` one by one)
  and `share_of` (the record); `each`
  gives every frame's needed forces with its state laid in `data`, on the card at once when the model is there
  (`batched.Card`), else frame by frame on MuJoCo's C (2026-11-05, #70).
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
- A knee's free turn (`LIGAMENTS`: knee_angle_rotation3, freed of the flexion by myo._movable's KNEE) is carried by
  its ligaments and condyles, two-sided columns at CONTACT_COST beside the stops (`ligaments`; #98, 2026-11-07):
  MyoFullBody's vasti reach it 0.3 to 0.5 cm off its axis, and its few newton metres, left to the muscles, set the
  vasti's split (at a deep squat one vastus fully active, another at 0).
- **Each foot carries its contact's share** (#1, 2026-11-18, the owner: "its foot conctact"; `split_row`, SPLIT):
  with both feet down the share step also asks the feet's vertical split to be their contacts' (each sole's push
  lengths squared, summed: each touching point's `contact.closeness` times its area), so the split is the contact's, not the video body's frontal
  moment frame to frame; what that leaves of the balance is the pelvis' residual.
