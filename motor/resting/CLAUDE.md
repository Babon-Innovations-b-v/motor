# motor/resting/

Each muscle's tendon slack length calibrated at MyoSuite's `stand` (#70, the owner's option 3 of 2026-10-29): a muscle
standing past `REST_MOST` (1.1) of its optimal fibre length has its length range moved so it stands at `REST_MOST`;
its optimal fibre length, peak force and curves are MyoSuite's. `myo.person_file` applies it after the path scaling
and records it as the model record's `resting`. Chosen over keeping MyoSuite's and over a Thelen 2003 curve on 2026-10-29: the comparison is on #70.

- MuJoCo has no slack length of its own: L0 and LT are read off `lengthrange` and `range`; moving the range's two ends
  together moves LT only. MyoFullBody's LT is below zero for 32 of the 74 muscles it moves, so keeping LT + L0 is not
  possible there.
- Its test runs in MOTOR's own Python (Newton and MuJoCo, pinned in pyproject.toml).
