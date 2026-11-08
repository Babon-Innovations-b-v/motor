# motor/

The MOTOR package: the solver (docs/bible.md `architecture/solver`, `architecture/calibration`). Empty until the
solver moves in from SCORE (SCORE #101).

- **A module is a folder here** with its interface in `__init__.py`, its private files `_prefixed`, its one test file
  beside it (or in its `tests/` folder) and its overlay once it has rules of its own.
- **It imports nothing of SCORE**: no store, rows, spend or paths of SCORE's home. What a caller needs comes in as
  arguments and goes out as return values or files the caller names.
- **Never the owner's patented method** (root `CLAUDE.md`): the angles are MyoFullBody's own coordinates after the fit.
