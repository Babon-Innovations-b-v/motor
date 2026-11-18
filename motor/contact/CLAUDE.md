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
- **Each foot's contact is MOTOR's own** (#1, 2026-11-17, `feet_touching`, `Skin.credited`): from its lowest skin
  point's height over a floor fitted per trial and foot (the median of its heights on its slow frames) and the slower
  of its bodies' speeds, with hysteresis (in under TOUCH_HIGH and TOUCH_FAST while not rising past RISING, or at any
  speed DEEP under the floor or under LANDED having fallen faster than FALLING within FALLEN frames, a drop jump's
  landing; out over LEAVE_HIGH, or over LEAVE_FAST unless DEEP, or under LANDED within FALLEN frames of touching or
  rising slower than RISING, so a walking foot lifting off from under its floor leaves on its speed; tuned against
  the plates' loaded frames on subjects 2 to 5). GEM-X's static flag, used
  before, came 3 to 4 frames late at heel strike. A touching foot is pushed from its lowest point, its sole laid on its ground there whether over
  it or into it (toes 17 to 29 mm into the floor at push-off pushed hundreds of points); a foot not touching is not
  pushed at all (credit AWAY), on `pushes`, `pushes_all` and `touching_all` alike (a state's credit is its frame's,
  states being whole copies of the clip).
- **A point pushes by its area** (`point_areas`, `Skin.areas`): its directions are `closeness` times the square root of
  its share of the skin's area, so the least squares that shares the load gives each point its area's share, not
  each point an equal one (SOMA's low mesh: 127 toe points of 0.6 cm2 under a sole against 17 of 7 cm2 under the heel
  and midfoot).
- **The feet's split follows the contacts' timeline** (#1, 2026-11-18, `feet_shares`, `Skin.shares`, set by
  `credited`): while both feet touch, the left foot's share of their load, a step's linear handover from the foot
  lifting off to the one coming down, otherwise each foot ramping in or out over RAMP and equal between;
  `muscles.split_row` asks it of the share step.
