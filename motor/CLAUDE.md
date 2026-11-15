# motor/

The MOTOR package: the solver (docs/bible.md `architecture/solver`, `architecture/calibration`), moved in from SCORE's
`score/stages/sim_body/inside` at SCORE ae0f57ed6 (SCORE #101): `run` (SCORE's `sim_body.py`), one folder a step,
`trial` (SCORE's inner `motor`, renamed so it does not shadow the package) and `following` (SCORE's
`evaluation/_measure.py`, the one piece of the tuning harness the solver's tests read). Each module's own overlay
came with it; their `#N` are SCORE's issues.

- **A module is a folder here** with its interface in `__init__.py`, its private files `_prefixed`, its one test file
  beside it (or in its `tests/` folder) and its overlay once it has rules of its own.
- **Imports are package imports** (`from motor import fit`, `from motor.fit import _skin`), never a bare name on
  `sys.path` as SCORE's run folder had them; the tests' own helpers (`fit/_card/tests/card_*.py`) are the one
  exception, put on `sys.path` by the test that reads them.
- **It imports nothing of SCORE**: no store, rows, spend or paths of SCORE's home. What a caller needs comes in as
  arguments and goes out as return values or files the caller names.
- **Never the owner's patented method** (root `CLAUDE.md`): the angles are MyoFullBody's own coordinates after the fit.
