# motor/contact/

The body on the scene's ground: `Skin` (the person's drawn surface, SOMA's mesh, each point the blend of the
segments its weights name, each segment riding its bone: the contact surface, a seat's as a floor's), `lowests`
(the whole skin's lowest point over its ground and each part's, placed once) and `pushes` (the skin points the ground pushes: within `GAP` of the drawn triangle under them or into it, a friction
pyramid under `SLIDE_MOST`, else sliding against the slip, each acting at the ground under it: the record's `touches`, each part's push and centre
of pressure, `muscles.touches`), each point's push directions `closeness` long (1 on or into the ground, 0 at `GAP`),
so a point's push fades in as it comes within `GAP` and the solve never hands a point crossing it its whole share at
once (2026-11-05: a foot's centre of pressure moved 3 to 6 cm a frame standing, base pickup); `surface` (each collider's points in 26 directions) is
the evaluation's measure only. MyoFullBody's colliders are not the contact (2026-11-01): a seated astronaut's stood 16
to 27 mm inside his skin, which sat 10 mm into his floor, so the floor carried 0 N of his 121 N weight.

- #70: SOMA is the truth: nothing here moves the body. Ground contact is where the fit to SOMA puts the feet; the old
  fit's holds (`stances`, `on_ground`, `unslipped`, `landed`) and `grounded` (the path moved so its lowest point
  stood on the ground) went with it on 2026-10-28.
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
- **The skin is laid on SOMA's** (#98, 2026-11-07, `Skin.registered`): each segment's riding moved by the one rigid
  turn and shift (Kabsch over its points and REGISTERED_FRAMES frames of the fitted path) that best lays its carried
  skin on SOMA's. Carried from MyoFullBody's ankle, a foot's skin stood 5 to 11 mm over SOMA's and turned 4 to 6
  degrees on video bodies; laid, within about 1 mm. Nothing moves the body: only where its skin rides its bones.
- **A foot the video shows standing is pushed from its sole** (#98, `Skin.credited`, the job's targets' `stance`,
  frames x 2: a video estimator's static contact or the caller's own stance): on those frames its points are measured
  from its lowest point (at most STANCE_HIGH, 5 cm, tuned against force plates) instead of the ground, on `pushes`, `pushes_all` and
  `touching_all` alike (a state's credit is its frame's, states being whole copies of the clip). A clip without a
  stance is pushed as before.
