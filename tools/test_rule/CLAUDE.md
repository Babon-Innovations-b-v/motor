# tools/test_rule/

The guard on the test rule (`docs/bible.md` `workflow/module-standard`), SCORE's own guard kept the same here.
`check.py` is the way in; `make tests` and the push gate run it first and fail on any fault.

- **The time budget is the Makefile's, not the guard's** (owner, 2026-11-07): `make tests` kills a file at
  `TEST_SECONDS` (60) and prints it as over budget; `SUITE_SECONDS` (600) is the whole suite's limit
  (`make test-budget`).
- **What it refuses**: a module with more than one test file, a test file not named `test_<invariant>.py`, a test file
  whose docstring names no break that turns it red, a second `test_*` function in a file (unless its docstring has a
  line `second test: <why>`), checks that run only under `__main__`, a SyntaxWarning.
- **Its lists only shrink** (`ALLOWLIST`, `MAIN_ONLY`, `MAY_SKIP`, `FUNCTION_DEBT`). They start empty here; the code
  moved in from SCORE brings its entries as SCORE held them on the move's day, at their new paths. Never add an entry
  to dodge a fault.
- **A run that printed skipped did not run** (`check.py --ran`, asked by `make tests`): it fails unless `MAY_SKIP`
  names the file with the reason (a GPU test: "needs a GPU").
- **A change to the guard's logic is made in SCORE first** and copied here, so the two repositories judge tests the
  same way.
