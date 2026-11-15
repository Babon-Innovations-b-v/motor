# motor/calibration/

The static calibration table (#70; pure numpy, its test runs in the score venv): `TABLE`, per bone group, the fixed
turn R_cal between SOMA's segment frame and MyoFullBody's group frame (R_body = R_segment @ R_cal), read once from
the two models' rest poses (SOMA's template bind against MyoFullBody's default pose, couplings applied) by aligning
each group's anatomical axes (`aligned`, `frame`, `REFERENCES`); `turn(group)` gives it as a matrix.

- `FINGERS` (2026-11-01, read again 2026-11-06): the held fingers' rest turns, one per joint for both hands, so MyoFullBody's hand at rest
  is SOMA's template hand's shape (SOMA's hand is one rigid shape; MyoFullBody's published zero has the fingers straight
  and the thumb opposed, 67.6 % of the finger bones' vertices outside the template's skin, the thumb 25 to 45 mm;
  24.7 % with them, the thumb inside). `_measure.fingers_measured` reads them on the table as it stands (paste TABLE
  first): each finger bone vertex as deep in SOMA's skin as the joints' ranges let it, both hands one solve, each hand
  laid at SOMA's Hand joint, where the fit holds it (2026-11-01 to 11-06 each hand was moved as one body to where its
  palm stood least outside: the turns fit a hand off the joint, the thumb, index and little finger stood out at it, the
  wrist's end shift read 18 mm up the forearm to bury them and every fingertip bone stood 20 to 22 mm short of SOMA's;
  read again 2026-11-06 on the mean body's rig, the wrist's shift zero, the fingertips 6.5 to 13.4 mm off SOMA's in
  the fit, the little's 2.5 to 3.5, numbers beside FINGERS); laid in the person's file by `myo._rested`.
  The wrists' end shift stays zero: read with the new fingers it is 6.6 mm, 1.1 % more of the hand inside for 3 mm
  less reach on every fingertip. The finger knuckles bend back 45 degrees (`myo._movable.KNUCKLES`, read again 2026-11-01), so the index
  can reach SOMA's, which stands dorsal of where MyoFullBody's metacarpal puts it. Static like the table: never per person or per clip.
- Never fit it, never per person or per clip: it would absorb a person's posture (a curved spine) and erase it
  (owner, 2026-10-28). It is the geometry every retargeting tool uses, not fitted to reference data.
- `_measure.py` measures it again in Newton's own Python and prints `TABLE`; run it after a model or template change
  and paste the rows. Measured 2026-10-28: the same as the tables study's within 0.002 (quaternion parts).
- A right group's turn is its left one's times diag(-1, -1, 1); a middle group turns no body sideways (the test).
- **The toes' hinge's two groups share its line, the lumbar pair the whole lumbar's** (2026-11-04, #70): `HINGED`
  (foot and toes about the MT1 to MT5 heads, the same landmarks on both models) keep that reference exact and square
  the long axis to it (`hinged`); `LONG_LINES` gives Spine1 and Spine2 the line from L5/S1 to T12/L1 (SOMA's Spine1
  to Chest joints). The `follows` gate's error was almost all a fixed offset per segment, the same in every frame,
  clip and person: the toes' pitch against the foot and the lumbar levels' rest curvature differed between the two
  models by a turn no MyoFullBody joint makes (the oblique mtp hinge, one lumbar joint spread over five levels). A rest
  solve of every leading joint against SOMA's bind (scratch, not kept) left Spine1 13.4, Spine2 23.0 and each toe 15.8
  degrees no joint reaches with the old rows; with these 0.5, 0.9 and 2.6. The feet left `INSIDE_GROUPS`. The elbow
  stays on its line: hinged too, its forearm's 12.2 degrees went to 0.5 at rest, but the bones then lie along
  MyoFullBody's carrying angle, 12 degrees off SOMA's forearm (env1 standing: forearm bones outside 1.9 -> 12.6 %).
- **The feet are levelled** (`LEVELLED`, `levelled`; 2026-11-06, #98: "a foot the floor carries lies flat"): after
  their hinge's rows, each foot's row is turned about its hinge line (exact, so the toes still share it) until SOMA's
  bind foot, which stands flat on its floor, is MyoFullBody's calcaneus standing flat at its default pose (its long
  axis rising as the default's). On the landmark lines alone it was the calcaneus 3.0 degrees toes-down (the two
  feet's shapes differ: the ankle to mtp line's rise over the sole is each model's own; the sole's own line, heel pad
  to MT pad, moved it 0.3), so the sim body's calcaneus stood toes-down where SOMA's sole lay flat. Static, read at rest: never per person or per clip.
- `INSIDE_GROUPS` (2026-11-01, #70; the owner and the coordinator: the Head's row by its skull and jaw in SOMA's
  head, then every group where it helps): their rows are read by `_measure.inside_measured`, the group's bone vertices
  laid at rest with their start centre (`myo_parts`, scaled) at SOMA's start joint, then turned and shifted rigidly to
  their least depth outside SOMA's skin (the head's pockets closed); a middle group turns about its body's z alone, a
  side's pair shares one mirrored turn (`turned`); the shift is dropped. Level 1's re-read keeps them (`remeasured`).
  Anchored at the frame body's origin instead (before), the neck's bones stood 48 cm above SOMA's neck (its body's
  origin is at the torso's) and the shoulders' and chest's far off too, so the solve fell into 61.7, 32.5 and 7.2
  degree turns. Kept where the bones outside fell at rest and no group's turn error rose on the mean body's 15 clips:
  Head (the decided one: 6.6 degrees, 30.0 to 3.1 % outside at rest; head 25.1 to 19.5 % on the clips, its turn
  error 4.81 to 5.01 degrees), Shins (5.5 degrees, 12 to 2 %; shin 20.0 to 17.4 %, thigh 28.4 to 24.3 %, their
  error 5.35/6.35 to 4.23/5.05), Feet (1.5, 10 to 2 %; 8.43/8.34 to 7.93/7.76), Hips (0.2) and ForeArms (0.1). Left
  on their lines: Toes (7.0 degrees, 48 to 16 % at rest, but foot 28.3 to 32.3 % on the clips and their error up),
  Hands (2.1 degrees, 14.7 to 13.8 %, the left's error 0.15 to 0.17), Legs (0.3 degrees, their error up 0.02), Chest
  and Neck1 (more outside at rest). All together: bones outside 15.12 to 13.84 %, turns 4.99 to 4.66 degrees, joints
  10.78 to 12.07 mm (the shins' turn moves the knee and ankle centres).
- **The head's row is laid on its landmarks** (`LAID_GROUPS`, `_measure.laid_measured`; 2026-11-06, #70, the owner:
  "nose isnt at nose, chin isnt at chin"): its turn about the body's z and C0_C1's shift in the middle plane solved so
  the skull's seven landmarks (vertex, forehead, opisthocranion, both sides, the face's nasion and menton) meet SOMA's
  template skin; their links then written as SOMA's skin off the laid bone points. The skin read (above) had turned
  it 9.8 degrees chin up and held it 6 mm up and 8 mm forward (`END_SHIFTS`): SOMA's oral cavity stood open (the
  pockets reached only the lips' slit, `fit._skin.POCKET_REACH`), so the teeth inside it read outside. At rest: nasion
  31 -> 0, menton 30 -> 0 mm off their links; skull and jaw outside 0.0 % with the cavity closed. On the fit (CPU, a
  frame in 15): nasion 20.5 -> 3.9, menton 23.4 -> 3.1, vertex 19.4 -> 2.2 mm standing; pickup and jump 21 to 24 ->
  5 to 8; the head outside 5.6 -> 0.6 % standing. C0_C1's shift is now the lay's (0.4 mm up, 0.5 forward).
- A stretch's ends are always MyoFullBody's joint centres (`myo_parts`), a reference's points the landmarks' bone
  points: until 2026-10-29 the landmark `sternoclavicular_{side}` (7 mm off the clavicle's origin) shadowed the
  centre of that name, turning the shoulders' rows about 2 degrees; measured again, they are main's. Level 1's
  worker (`../evaluation/_case.py`) reads the table again at rest from each evaluation's group-end centres.
- `END_SHIFTS` (2026-11-01, #70): a group end's shift read in SOMA's skin (`_measure.end_shift_measured`: the group's
  bones laid at its row on the template, every shift on a 2 mm grid within 24 mm, the least share outside; the share
  is a step, so no least squares). Only the head's (C0_C1, 6 mm up and 8 mm forward): at SOMA's Head joint 20.6 % of
  the skull and jaw's vertices stand outside the template's skin (the dropped shift of its row's read), 2.5 % shifted.
  `fit.end_marker` moves that end's marker by it. Mean's 15 clips: head 17.4 -> 13.0 %, all 6.95 -> 6.39 %, turns
  within 0.01 degree; env1 head 6.7 -> 3.0 % standing, 5.4 -> 2.7 % seated. Static: never per person or per clip.
- `END_READS` (2026-11-03, #70): every group start end's shift read the same way, level 1's starts for its segment
  end unknowns (`tuning._ends`; the owner: "segment ends ... is better" than joint centre to joint centre); the stage
  never reads it. Level 1's worker lays an evaluation's shifts into `END_SHIFTS` in place (`evaluation._case.load_shifts`).
