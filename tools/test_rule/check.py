"""The guard on the test rule (docs/bible.md workflow/module-standard; owner, 2026-10-26: "only 1 but one actual good
meaning full test file per module ... i feel like that rule kinda drifted by now"): it lists every way the repository's
test files break the rule, so the rule cannot drift again. `make tests` runs it and fails on any fault.

    python3 tools/test_rule/check.py                 # every module in the repository
    python3 tools/test_rule/check.py <test file> ... # only the modules those files belong to

A test file is a file named `test_*.py` or `*_test.py`, tracked or new (not ignored), outside `vendor/`. Its faults:

- **named for its module, not its invariant**: every test file is `test_<invariant>.py`.
- **no break**: its module docstring has no line naming the break that turns it red ("break" and "red" in one statement,
  at most 200 characters apart, as "Breaks (each applied once, seen red, reverted), in ...:").
- **more than one test function in a file** (owner, 2026-11-01): a test file holds ONE `test_*` function (a method of
  a `Test*` class counts); parametrized cases of one check are one function. A script-style file's checks count the
  same (2026-11-07): each entry of its top-level `CHECKS` and each top-level `def check_*`, a name counted once. A
  docstring line starting `second test:` with a reason excuses it, but only up to the count the file has on
  origin/main (one for a file not there): a note never grows a file. `FUNCTION_DEBT` holds the files over it that day
  with their count: one may not grow, one that shrank must lower its entry, one down to one test (or gone) must lose it.
- **tests that run only under `__main__`**: a test file with an `if __name__ == ...` block and no `test_*` function
  holds checks pytest never finds. `MAIN_ONLY` lists the files that did so on 2026-11-07; it only shrinks.
- **a SyntaxWarning** (an invalid escape sequence, say) is a fault of the file that raises it.
- **more than one test file in a module**: a module is the directory its test file sits in, a `tests/` folder counting
  as its parent's (`.claude/hooks/tests` is `.claude/hooks`). In a flat folder of independent tools (`FLAT_TOOL_FOLDERS`)
  a top-level test file belongs to the tool script its docstring names first (as `tools/<folder>/<tool>.py`, else as
  `<tool>.py`, else as the bare name), and one that names none is a fault of its own.

`ALLOWLIST` holds the modules other agents' sweeps are still reworking, dated, each pattern with the test files and
test functions it covered on 2026-11-07; like `FUNCTION_DEBT` it only shrinks (a pattern covering more is a fault, one
covering less must lower its entry, one covering none must go), and nothing else goes in it.

    python3 tools/test_rule/check.py --ran <test file> <its output file>

is the run's own fault (`make tests` asks it for each file whose output says skipped): a file that printed a line
starting `skipped`, or whose pytest summary is skips only, did not run, unless `MAY_SKIP` names it with the reason.
The output is one summary line, then one line a fault.
"""
import ast
import fnmatch
import functools
import pathlib
import re
import subprocess
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]

# Folders of independent command-line tools: each top-level tool script is a module of its own.
FLAT_TOOL_FOLDERS = ()

# The lists below only ever shrink. They start empty here (2026-11-08): the code moved in from SCORE brings its own
# entries as SCORE held them on the move's day, at their new paths, and nothing else is ever added.
# Modules still being swept, each pattern with the (test files, test functions) it covered when counted.
ALLOWLIST_DATE = "2026-11-08"
ALLOWLIST_COUNT_DATE = "2026-11-08"
ALLOWLIST = {}

# Test files whose checks run only under `__main__` (pytest finds none of them).
MAIN_ONLY_DATE = "2026-11-08"
MAIN_ONLY = ()

# Test files that may print `skipped` (a check not run on this box), each with the reason.
MAY_SKIP = {}

# Test files holding more than one test function, each with its count on the day it was listed.
FUNCTION_DEBT_DATE = "2026-11-08"
FUNCTION_DEBT = {}

# The docstring line that excuses a second test function, with its reason after the colon.
SECOND_TEST = re.compile(r"^\s*second test:\s*\S", re.IGNORECASE | re.MULTILINE)

# The pytest summary of a file that ran nothing but skips ("3 skipped in 0.1s").
ONLY_SKIPS = re.compile(r"^\s*\d+ skipped\b", re.MULTILINE)

# "break" and "red" in one statement, which may wrap over the docstring's lines.
BREAK_LINE = re.compile(r"\bbreaks?\b.{0,200}?\bred\b|\bred\b.{0,200}?\bbreaks?\b", re.IGNORECASE)


def is_test_file(path):
    """Whether a repository path names a test file under the rule (vendor/ is third-party, kept as released)."""
    name = path.name
    return (not path.parts or path.parts[0] != "vendor") and name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py"))


def test_files(root):
    """The repository's test files, tracked or new and not ignored, that exist on disk, as paths from the root."""
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            check=True, capture_output=True, text=True).stdout
    paths = {pathlib.PurePosixPath(entry) for entry in listed.split("\0") if entry}
    return sorted(path for path in paths if is_test_file(path) and (root / path).is_file())


def allowlisted(path, allowlist):
    """Whether a test file sits in a module the allowlist still exempts."""
    return any(fnmatch.fnmatch(str(path), pattern) for pattern in allowlist)


@functools.lru_cache(maxsize=None)
def parse(source):
    """A source's tree (None when it does not parse) and the lines of the SyntaxWarnings parsing it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
    raised = tuple(warning.lineno for warning in caught if issubclass(warning.category, SyntaxWarning))
    return tree, raised


def parsed(root, path):
    """A test file's tree and SyntaxWarnings (see `parse`)."""
    return parse((root / path).read_text(encoding="utf-8"))


def docstring(root, path):
    """A test file's module docstring ("" when it has none), or None when the file does not parse."""
    tree, _ = parsed(root, path)
    return None if tree is None else ast.get_docstring(tree) or ""


def named_tool(root, path, text):
    """The tool script of a flat folder that a top-level test file's docstring names first, or None."""
    folder = path.parent
    tools = {sibling.stem for sibling in (root / folder).glob("*.py")
             if not is_test_file(pathlib.PurePosixPath(sibling.name)) and sibling.name != "__init__.py"}
    for pattern in (re.escape(str(folder)) + r"/(\w+)\.py\b", r"\b(\w+)\.py\b", r"\b(\w+)\b"):
        for found in re.finditer(pattern, text):
            if found.group(1) in tools:
                return found.group(1)
    return None


def module_of(root, path, text):
    """The module a test file belongs to, as a path from the root, or None for a flat-folder test naming no tool."""
    folder = path.parent
    if folder.name == "tests":
        folder = folder.parent
    if str(path.parent) in FLAT_TOOL_FOLDERS:
        tool = named_tool(root, path, text or "")
        return folder / tool if tool else None
    return folder


def pytest_tests(tree):
    """The names of the test functions pytest finds in a tree (a `Test*` class's methods as `Class.method`)."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.update(f"{node.name}.{item.name}" for item in node.body
                         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"))
    return names


def script_checks(tree):
    """The names of a script-style file's checks: its top-level `CHECKS` entries and `def check_*` functions."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("check_"):
            names.add(node.name)
        elif (isinstance(node, ast.Assign) and isinstance(node.value, (ast.Tuple, ast.List))
              and any(isinstance(target, ast.Name) and target.id == "CHECKS" for target in node.targets)):
            names.update(entry.id if isinstance(entry, ast.Name) else f"CHECKS[{index}]"
                         for index, entry in enumerate(node.value.elts))
    return names


def tree_count(tree):
    """How many test functions a tree defines, pytest's and the script's checks, each name once; 0 for None."""
    return 0 if tree is None else len(pytest_tests(tree) | script_checks(tree))


def test_count(root, path):
    """How many test functions a test file defines (see `tree_count`), 0 when it does not parse."""
    return tree_count(parsed(root, path)[0])


def runs_only_under_main(root, path):
    """Whether a test file has an `if __name__ == ...` block and no test function pytest finds."""
    tree, _ = parsed(root, path)
    if tree is None or pytest_tests(tree):
        return False
    return any(isinstance(node, ast.If) and "__name__" in {name.id for name in ast.walk(node.test)
                                                           if isinstance(name, ast.Name)} for node in tree.body)


def base_count(root, path, base):
    """A test file's test-function count at the git ref `base`, or None when the file (or the ref) is not there."""
    shown = subprocess.run(["git", "-C", str(root), "show", f"{base}:{path}"], capture_output=True, text=True)
    return tree_count(parse(shown.stdout)[0]) if shown.returncode == 0 else None


def count_faults(path, count, text, debt, based):
    """One test file's fault under the one-function rule, against its debt entry if it has one.

    `based` gives the file's count on origin/main, asked only when a `second test:` note must be weighed.
    """
    owed = debt.get(str(path))
    if owed is None:
        if count <= 1:
            return []
        if not SECOND_TEST.search(text or ""):
            return [f"{path}: {count} test functions, one a file (merge them into one test of the answer that matters)"]
        before = based() or 1
        if count > before:
            return [f"{path}: {count} test functions, its `second test:` note covers the {before} it has on origin/main"
                    " (a note never grows a file)"]
        return []
    if count > owed:
        return [f"{path}: {count} test functions, its FUNCTION_DEBT entry allows {owed} (a debt file may not grow)"]
    if count <= 1:
        return [f"{path}: {count} test function now, delete its FUNCTION_DEBT entry"]
    if count < owed:
        return [f"{path}: {count} test functions now, lower its FUNCTION_DEBT entry from {owed} to {count}"]
    return []


def file_faults(path, text, raised=()):
    """One test file's own faults: its name, its docstring's break line and the SyntaxWarnings it raises."""
    faults = [f"{path}: raises a SyntaxWarning on line {line} (python3 -m py_compile shows it)" for line in raised]
    if not path.name.startswith("test_"):
        faults.append(f"{path}: named for its module, not its invariant (rename to test_<invariant>.py)")
    if text is None:
        faults.append(f"{path}: does not parse")
    elif not BREAK_LINE.search(" ".join(text.split())):
        faults.append(f"{path}: its docstring has no line naming the break that turns it red")
    return faults


def allowlist_faults(root, listed, allowlist):
    """The allowlist's own faults: a pattern covering more test files or functions than its entry, less, or none."""
    found = []
    for pattern, owed in sorted(allowlist.items()):
        covered = [path for path in listed if fnmatch.fnmatch(str(path), pattern)]
        count = (len(covered), sum(test_count(root, path) for path in covered))
        if not covered:
            found.append(f"ALLOWLIST {pattern}: covers no test file, delete its entry")
        elif count[0] > owed[0] or count[1] > owed[1]:
            found.append(f"ALLOWLIST {pattern}: covers {count[0]} test files, {count[1]} test functions, its entry"
                         f" allows {owed[0]}, {owed[1]} (an allowlisted module may not grow)")
        elif count != tuple(owed):
            found.append(f"ALLOWLIST {pattern}: covers {count[0]} test files, {count[1]} test functions now, lower its"
                         f" entry from {tuple(owed)} to {count}")
    return found


def stale_faults(listed, debt, main_only):
    """The FUNCTION_DEBT and MAIN_ONLY entries whose file is not in the tree."""
    present = {str(path) for path in listed}
    return [f"{test}: on {name} but not in the tree, delete its entry"
            for name, entries in (("FUNCTION_DEBT", debt), ("MAIN_ONLY", main_only))
            for test in entries if test not in present]


def main_faults(root, path, main_only):
    """A test file's fault under the pytest-finds-it rule, against its MAIN_ONLY entry if it has one."""
    only = runs_only_under_main(root, path)
    if only and str(path) not in main_only:
        return [f"{path}: its tests run only under __main__, pytest finds none (make them one test_ function)"]
    if not only and str(path) in main_only:
        return [f"{path}: pytest finds its test now, delete its MAIN_ONLY entry"]
    return []


def faults(root, named=(), debt=None, allowlist=None, main_only=None, base="origin/main"):
    """Every fault of the rule in the repository at `root`, or only in the modules of the `named` test files.

    `debt`, `allowlist` and `main_only` are `FUNCTION_DEBT`, `ALLOWLIST` and `MAIN_ONLY` when None, and `base` the
    ref a `second test:` note is weighed against; a test passes its own.
    """
    root = pathlib.Path(root)
    debt = FUNCTION_DEBT if debt is None else debt
    allowlist = ALLOWLIST if allowlist is None else allowlist
    main_only = MAIN_ONLY if main_only is None else main_only
    modules = {}
    found = []
    listed = test_files(root)
    if not named:
        found.extend(stale_faults(listed, debt, main_only))
        found.extend(allowlist_faults(root, listed, allowlist))
    for path in listed:
        if allowlisted(path, allowlist):
            continue
        text = docstring(root, path)
        module = module_of(root, path, text)
        if module is None:
            found.append(f"{path}: its docstring names none of {path.parent}'s tool scripts")
            module = path
        modules.setdefault(module, []).append((path, text))
    wanted = {pathlib.PurePosixPath(name) for name in named}
    for module, files in sorted(modules.items()):
        if wanted and not wanted & {path for path, _ in files}:
            continue
        if len(files) > 1:
            found.append(f"{module}: {len(files)} test files, one a module: {', '.join(str(path) for path, _ in files)}")
        for path, text in files:
            found.extend(file_faults(path, text, parsed(root, path)[1]))
            found.extend(count_faults(path, test_count(root, path), text, debt,
                                      functools.partial(base_count, root, path, base)))
            found.extend(main_faults(root, path, main_only))
    parse.cache_clear()  # the trees held to the interpreter's exit cost it two seconds to free
    return sorted(set(found))


def ran_faults(path, output, may_skip=None):
    """A test file's run's fault: it printed `skipped` (or pytest skipped all of it) and `may_skip` names it not."""
    may_skip = MAY_SKIP if may_skip is None else may_skip
    skipped = any(line.lstrip().lower().startswith("skipped") for line in output.splitlines())
    if not (skipped or ONLY_SKIPS.search(output)) or any(fnmatch.fnmatch(str(path), key) for key in may_skip):
        return []
    return [f"{path}: printed skipped, so it did not run (run it where it can, or name it in MAY_SKIP with the reason)"]


def main(argv):
    """Print the summary line and each fault; exit 1 when there is any."""
    if argv[:1] == ["--ran"]:
        found = ran_faults(argv[1], pathlib.Path(argv[2]).read_text(encoding="utf-8", errors="replace"))
        for fault in found:
            print(f"  {fault}")
        return 1 if found else 0
    found = faults(REPO, argv)
    print(f"test rule: {len(found)} fault(s) (allowlist of {ALLOWLIST_DATE}: {len(ALLOWLIST)} entries)")
    for fault in found:
        print(f"  {fault}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
