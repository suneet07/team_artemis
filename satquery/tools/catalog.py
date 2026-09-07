"""Manifest name -> executable implementation (master plan section 4.6).

Kept separate from :mod:`satquery.tools.registry` so the registry stays what the
parameter gate needs — manifests and validation, importable with nothing but
``jsonschema`` and ``pyyaml`` — while the executables that pull in ``scipy`` and
``skimage`` live here. Anything wanting to *validate* a plan does not have to
import anything that can run one.

``lulc_classifier`` is the exception among the learned tools: its weights are
TU Berlin's published reBEN reference classifiers (MIT), not an adapter we
train, so they are available now and it registers below rather than later. It
still carries a learned confidence and is gated by the same manifest.

The remaining learned tools (``rs_vqa``, ``rs_ground_caption``, ``change_vqa``,
``optsar_fusion``, ``change_map``) are absent by design:
they need adapter weights, and the plan's scheduling rule is that training is
never on the demo's critical path. They register themselves through
:func:`register_implementation` when their weights land, and every routing,
validation and trace path below already accounts for them being missing.
"""

from satquery.tools.base import Tool
from satquery.tools.deterministic import (
    CentroidPriorTool,
    ChangeStatsTool,
    ObjectBoxFallbackTool,
    SarBackscatterTool,
    SpectralIndexTool,
    TextureSegTool,
)
from satquery.tools.lulc import LulcClassifierTool
from satquery.tools.wrappers import CoregCheckTool, TileScorerTool

__all__ = [
    "DETERMINISTIC_TOOLS",
    "available_tools",
    "implementation",
    "register_implementation",
]

_IMPLEMENTATIONS: dict[str, Tool] = {}


def register_implementation(tool: Tool) -> None:
    """Register (or replace) the executable behind a manifest name."""
    _IMPLEMENTATIONS[tool.name] = tool


for _tool in (
    SpectralIndexTool(),
    CentroidPriorTool(),
    SarBackscatterTool(),
    TextureSegTool(),
    ObjectBoxFallbackTool(),
    ChangeStatsTool(),
    CoregCheckTool(),
    TileScorerTool(),
):
    register_implementation(_tool)

#: The tools that run with no learned weights at all — the CPU-only headless
#: path (section 4.11) is built entirely from these. Captured *before*
#: ``lulc_classifier`` registers: it needs torch and downloaded weights, so a
#: headless CPU run that believed it was deterministic would fail at execution
#: rather than at planning.
DETERMINISTIC_TOOLS = tuple(sorted(_IMPLEMENTATIONS))

register_implementation(LulcClassifierTool())


def implementation(name: str) -> Tool | None:
    """The executable for ``name``, or None when only its manifest exists."""
    return _IMPLEMENTATIONS.get(name)


def available_tools() -> tuple[str, ...]:
    return tuple(sorted(_IMPLEMENTATIONS))
