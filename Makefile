# The project's commands. No CI runs these: the push gate (.claude/hooks/local/pre-push) runs the test rule and every
# touched module's test file on a push to main.

.PHONY: env tests test-budget

# MOTOR's Python environment (pyproject.toml, uv.lock) in .venv.
PYTHON := $(CURDIR)/.venv/bin/python
# Every test file, or only the ones named: make tests TESTS="<module>/test_<invariant>.py ...". Each is a plain script,
# or, when it has test_ functions and no __main__ block, a pytest file; each runs under a memory cap so a heavy one can
# never take the machine down. First the test rule's guard (tools/test_rule/check.py: one test file a module, named
# test_<invariant>.py, its docstring naming its break), always over every module, TESTS given or not. A file whose
# output says skipped did not run, and fails, unless the guard's MAY_SKIP names it with the reason.
TESTS ?= $(shell git ls-files 'motor/*test_*.py' 'tools/*/test_*.py' 'paper/*/test_*.py' '.claude/*/tests/test_*.py')
MEMORY_CAP ?= $(shell systemd-run --user --scope -q true >/dev/null 2>&1 && echo systemd-run --user --scope -q -p MemoryMax=16G)
# The files run side by side, at most JOBS at once (JOBS=1: one after another).
JOBS ?= 6
# The time budgets (docs/bible.md workflow/module-standard), the one place every gate reads them from: one test file at
# most TEST_SECONDS (owner, 2026-11-07: "we have a one test max of 60 sec"), the whole suite at most SUITE_SECONDS
# (owner, 2026-11-07: "a max of 10 min for the full suit"). `make test-budget` prints both.
TEST_SECONDS ?= 60
SUITE_SECONDS = 600

# One file (its path the first argument), in the shell xargs starts: its line, and its path added to $$FAILED.
define ONE_TEST
test=$$1; log=$$(mktemp); runner="$(PYTHON)";
if grep -q '^def test_' $$test && ! grep -q '^if __name__' $$test; then runner="$(PYTHON) -m pytest -q -p no:cacheprovider"; fi;
start=$$(date +%s);
if $(MEMORY_CAP) timeout $(TEST_SECONDS) $$runner $$test > $$log 2>&1 && { ! grep -qi skipped $$log || $(PYTHON) $(CURDIR)/tools/test_rule/check.py --ran $$test $$log > $$log.ran || { cat $$log.ran >> $$log; false; }; }; then echo "ok    $$test ($$(( $$(date +%s) - start )) s)";
else code=$$?; { if [ $$code = 124 ]; then echo "FAIL  $$test (over $(TEST_SECONDS) s: the module standard's time budget)"; else echo "FAIL  $$test ($$(( $$(date +%s) - start )) s)"; fi; tail -n 15 $$log; } > $$log.fail; cat $$log.fail; rm -f $$log.fail; echo $$test >> $$FAILED; fi;
rm -f $$log $$log.ran
endef
export ONE_TEST

env:
	uv sync -q --frozen

tests: env
	@FAILED=$$(mktemp); export FAILED; \
	$(PYTHON) $(CURDIR)/tools/test_rule/check.py || echo tools/test_rule/check.py >> $$FAILED; \
	printf '%s\n' $(TESTS) | xargs -r -P $(JOBS) -n 1 sh -c "$$ONE_TEST" one; \
	failed=$$(paste -sd ' ' $$FAILED); rm -f $$FAILED; \
	if [ -n "$$failed" ]; then echo "failed: $$failed"; exit 1; fi

test-budget:
	@echo "file $(TEST_SECONDS)"; echo "suite $(SUITE_SECONDS)"
