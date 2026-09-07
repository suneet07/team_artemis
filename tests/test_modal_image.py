"""The Modal image installs everything `pyproject.toml` says satquery needs.

Two deploys crash-looped on this in one session: `scikit-image` was absent
while `scikit-learn` was present (which reads as installed to a skim), and
`jsonschema` only surfaced when `/meta/health` imported the trace validator.
Both cost a three-minute deploy, a crash-loop email and a log dig to find.

The package declares its own runtime dependencies. The image that runs it must
install them, so comparing the two lists turns that class of bug into a local
test failure.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
MODAL_SCRIPT = PROJECT_ROOT / "scripts" / "modal_phase0.py"


def _requirement_names(text: str) -> set[str]:
    """Distribution names from requirement strings, normalised for comparison.

    PyPI treats ``-``, ``_`` and ``.`` as equivalent and is case-insensitive,
    so `scikit-image` and `scikit_image` are one package; comparing raw strings
    would report a false difference.
    """
    names = set()
    for raw in re.findall(r'"([^"]+)"', text):
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:[<>=!~\[].*)?$", raw)
        if match:
            names.add(re.sub(r"[-_.]+", "-", match.group(1)).lower())
    return names


def _declared_dependencies() -> set[str]:
    block = re.search(
        r"^dependencies = \[(.*?)^\]",
        PYPROJECT.read_text(encoding="utf-8"),
        re.S | re.M,
    )
    assert block, "no [project] dependencies block in pyproject.toml"
    return _requirement_names(block.group(1))


def _base_image_packages() -> set[str]:
    source = MODAL_SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"^base_image = \((.*?)^\)", source, re.S | re.M)
    assert block, "no base_image definition in modal_phase0.py"
    return _requirement_names(block.group(1))


def test_the_modal_base_image_installs_every_declared_dependency():
    declared = _declared_dependencies()
    installed = _base_image_packages()
    missing = sorted(declared - installed)
    assert not missing, (
        "scripts/modal_phase0.py's base_image is missing runtime "
        f"dependencies that pyproject.toml declares: {missing}. Any container "
        "importing satquery will raise ModuleNotFoundError on the first "
        "request that reaches them -- which surfaces as a crash-looping "
        "deploy, not as a local failure."
    )


def test_scikit_image_and_scikit_learn_are_not_confused_for_each_other():
    """They are different packages; having one says nothing about the other."""
    installed = _base_image_packages()
    assert "scikit-image" in installed, (
        "satquery.coreg, satquery.tools.catalog, tools.cv_utils, "
        "tools.thresholds, cd.dataset and api.server all import skimage"
    )
