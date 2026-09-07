"""Licence rails (master plan C55/C56, and the Part 10 risk register).

Three gates live here, and all three are things a judge can check by reading a
file:

1. **No excluded dataset or checkpoint reaches a training manifest.** LEVIR-CD,
   LEVIR-MCI, SECOND, QAG-360K and the TinyCD / ChangeFormer checkpoints are
   excluded outright; VRSBench is evaluation-only and must never appear in
   training; DynamicEarthNet is not staged until its licence clears.
2. **Every shipped weight maps to a CREDITS entry marked CLEAR.** A ``.pt`` or
   ``.safetensors`` in the tree with no cleared provenance is exactly the failure
   the risk register calls "a restricted checkpoint reaches the deliverable".
3. **AROSICS is pinned ``>=1.0.0``**, which is a licence control, not a
   preference: pre-1.0 releases were GPL-3.0 (C58).

The earlier version of this file scanned for bare substrings, which meant the
ordinary English words "second" and "ground" tripped it — 6 false positives the
day real prose was written, and a test that cries wolf gets deleted rather than
fixed. Matching is now on **word-boundary identifiers**, and ``CREDITS.md`` is
exempt because recording an exclusion by name is the whole point of that file.
"""

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

#: Identifiers that must not appear in any manifest, config or data-prep code.
#: Written as regexes with word boundaries, so "SECOND" the dataset is caught and
#: "the second band" is not.
BLOCKLIST: dict[str, str] = {
    "tinycd": r"\btiny[-_ ]?cd\b",
    "changeformer": r"\bchange[-_ ]?former\b",
    "levir-cd": r"\blevir[-_ ]?cd\b",
    "levir-mci": r"\blevir[-_ ]?mci\b",
    "levir": r"\blevir\b",
    "second-dataset": r"\bsecond[-_ ](?:dataset|cc|cd)\b",
    "qag-360k": r"\bqag[-_ ]?360k\b",
    "dfc2023": r"\bdfc[-_ ]?2023\b",
    "sardet-100k": r"\bsardet[-_ ]?100k\b",
    "xview3": r"\bxview[-_ ]?3\b",
}

#: Eval-only sources. Legal to evaluate on, never legal to train on (C20).
EVAL_ONLY: dict[str, str] = {
    "vrsbench": r"\bvrsbench\b",
    "dota": r"\bdota\b",
}

#: Not staged until its licence clears (C61, C40).
#:
#: LS-SSDD sits here rather than in the blocklist because it is genuinely
#: half cleared: its annotation repository carries a verified Apache-2.0
#: LICENSE, but that repository ships no imagery. The 15 Sentinel-1 scenes
#: live behind a CAS portal whose terms were never read, and CREDITS carried
#: the dataset as fully CLEARED on the strength of the annotation badge
#: alone. Correcting the document is not enough -- the same badge would
#: justify staging it again next week.
NOT_STAGED: dict[str, str] = {
    "dynamicearthnet": r"\bdynamic[-_ ]?earth[-_ ]?net\b",
    "ls-ssdd": r"\bls[-_ ]?ssdd\b",
}

NEWLINE = chr(10)

SCANNED_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".cfg", ".toml")

#: CREDITS.md exists to name excluded sources and why. The plan's own judge-facing
#: answer quotes several of them by name.
EXEMPT_FILES = {"CREDITS.md", "TEAM_CONTEXT.md", "test_license_blocklist.py"}

SCANNED_DIRS = ("configs", "training", "satquery", "scripts")


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCANNED_DIRS:
        root = PROJECT_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SCANNED_SUFFIXES:
                continue
            if path.name in EXEMPT_FILES or {"__pycache__", "node_modules"} & set(path.parts):
                continue
            files.append(path)
    return files


def _strip_prose(path, content: str) -> str:
    """Python source with comments and docstrings blanked out.

    A blocklist name in a comment is usually the opposite of a violation:
    ``satquery/cd/model.py`` explains at length that TinyCD is
    non-commercial and that Open-CD's published weights are trained on
    LEVIR-CD -- which is precisely why neither is used. Matching raw text
    made this guard fire hardest on the file documenting compliance best,
    and the only way to silence it was to delete the reasoning.

    Docstrings and comments only -- deliberately NOT every string literal.
    A real violation usually IS a string (``ROOT = "/data/levir-cd"``),
    so blanking all of them would let the thing this guard exists to catch
    walk straight through. Verified by probe: that assignment still fails
    the test, while the prose explaining an exclusion no longer does.

    Blanking rather than deleting keeps every reported line number pointing
    at the same code. A file that will not parse falls back to its raw text:
    an unparsable file is no reason to stop checking it, and a missed
    reference is the expensive direction of this error.
    """
    if path.suffix != ".py":
        return content
    import ast
    import io
    import tokenize

    out = content.splitlines()

    def blank(r0, c0, r1, c1):
        for row in range(r0 - 1, min(r1, len(out))):
            line = out[row]
            first = c0 if row == r0 - 1 else 0
            last = min(c1 if row == r1 - 1 else len(line), len(line))
            if last > first:
                out[row] = line[:first] + " " * (last - first) + line[last:]

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return content
    for token in tokens:
        if token.type == tokenize.COMMENT:
            blank(token.start[0], token.start[1], token.end[0], token.end[1])

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return NEWLINE.join(out)
    for node in ast.walk(tree):
        # A docstring is a bare string expression-statement. Anything bound
        # to a name, passed as an argument or built into a path is not one,
        # and stays visible to the scan.
        if not isinstance(node, ast.Expr):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            blank(
                value.lineno,
                value.col_offset,
                value.end_lineno or value.lineno,
                value.end_col_offset or 0,
            )
    return NEWLINE.join(out)


def _hits(patterns: dict[str, str]) -> list[str]:
    violations: list[str] = []
    for path in _scanned_files():
        try:
            content = _strip_prose(path, path.read_text(encoding="utf-8")).lower()
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                line = content.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}:{line}: excluded source '{name}'"
                )
    return violations


def test_no_excluded_source_in_configs_or_training_code():
    violations = _hits(BLOCKLIST)
    assert not violations, (
        "Excluded datasets or checkpoints are referenced outside CREDITS.md "
        "(master plan C55/C56):\n" + "\n".join(violations)
    )


def test_eval_only_sources_absent_from_training_manifests():
    manifest_dir = PROJECT_ROOT / "training" / "data"
    violations: list[str] = []
    if manifest_dir.exists():
        for path in manifest_dir.rglob("*.jsonl"):
            content = path.read_text(encoding="utf-8").lower()
            for name, pattern in {**EVAL_ONLY, **NOT_STAGED}.items():
                if re.search(pattern, content):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: '{name}'")
    assert not violations, (
        "VRSBench/DOTA are evaluation-only (C20) and DynamicEarthNet is not staged "
        "until its licence clears (C61); neither may enter a training manifest:\n"
        + "\n".join(violations)
    )


def _git_ignored(path) -> bool:
    """True when git will never ship this file.

    The gate this test guards is "a restricted checkpoint reaches the
    deliverable". A file matched by ``.gitignore`` cannot: the ignore file
    already excludes ``*.pt``, ``*.safetensors`` and ``checkpoints/`` under the
    note that checkpoints go to the private HF repo, not here. Scanning the raw
    filesystem instead flagged working copies pulled down to run the model
    locally -- weights that live on a Modal Volume, are ignored by git, and are
    never packaged. Ask git rather than re-implement its matching rules; when
    git is unavailable, fall back to scanning everything, because a missed
    weight is the expensive direction of this error.
    """
    import subprocess

    try:
        done = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            # Explicit, and load-bearing on Windows: pytest replaces stdin with
            # an object that has no valid file handle, so an inherited stdin
            # makes CreateProcess fail with OSError(9, 'The handle is
            # invalid'). That was caught by the except below and reported as
            # "not ignored", which silently opened the guard -- the test passed
            # in a full run and failed file-only, purely on capture state.
            stdin=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    Path("/tmp/gi.log").open("a").write(f"{path.name} rc={done.returncode}" + chr(10))
    return done.returncode == 0


def test_every_shipped_weight_maps_to_a_cleared_credits_entry():
    """The risk register's 'restricted checkpoint reaches the deliverable' gate."""
    credits = (PROJECT_ROOT / "CREDITS.md").read_text(encoding="utf-8")
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".ruff_cache"}
    weights = [
        path
        for pattern in ("*.pt", "*.pth", "*.safetensors", "*.ckpt", "*.bin")
        for path in PROJECT_ROOT.rglob(pattern)
        # is_file() matters: node_modules/.bin is a *directory* whose name ends
        # in .bin, and globbing it in reported five phantom unlicensed weights.
        if path.is_file() and not skip_dirs & set(path.parts)
    ]
    weights = [w for w in weights if not _git_ignored(w)]
    unclear: list[str] = []
    for weight in weights:
        stem = weight.stem.lower()
        for name, pattern in BLOCKLIST.items():
            if re.search(pattern, stem):
                unclear.append(f"{weight.name}: matches blocklisted '{name}'")
                break
        else:
            block = _credits_block_for(credits, stem)
            if block is None or "clear" not in block.lower():
                unclear.append(f"{weight.name}: no CREDITS entry marked CLEAR")
    assert not unclear, (
        "Every shipped weight must map to a CREDITS entry marked CLEAR:\n"
        + "\n".join(unclear)
    )


def _credits_block_for(credits: str, stem: str) -> str | None:
    for line in credits.splitlines():
        if stem and stem in line.lower():
            return line
    return None


@pytest.mark.parametrize("path", [PROJECT_ROOT / "pyproject.toml"])
def test_arosics_is_pinned_above_1_0(path: Path):
    """Pre-1.0 AROSICS was GPL-3.0 (C58). The pin is a licence control."""
    content = path.read_text(encoding="utf-8")
    assert "arosics" in content.lower(), "AROSICS must stay declared so its pin is visible"
    match = re.search(r"arosics\s*>=\s*([0-9]+)\.", content, flags=re.IGNORECASE)
    assert match, "AROSICS must be pinned with an explicit >= constraint"
    assert int(match.group(1)) >= 1, "AROSICS must be pinned >=1.0.0; pre-1.0 was GPL-3.0"


def test_snap_is_never_imported():
    """SNAP is GPL-3.0 and runs as an external process only (section 4.2)."""
    forbidden = re.compile(r"^\s*(?:import|from)\s+(?:snappy|esa_snappy|pyroSAR)\b", re.MULTILINE)
    violations = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "satquery").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert not violations, (
        "SNAP must never be imported or linked, only invoked as a separate process "
        "(TEAM_CONTEXT rule 10):\n" + "\n".join(violations)
    )
