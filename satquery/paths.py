import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(os.environ.get("SATQUERY_CONFIG_DIR", _REPO_ROOT / "configs"))
PREPROCESSING_CONFIG_PATH = CONFIG_DIR / "preprocessing.yaml"
TRACE_SCHEMA_PATH = CONFIG_DIR / "trace_schema.json"
TOOL_MANIFEST_SCHEMA_PATH = CONFIG_DIR / "tool_manifest_schema.json"
TOOLS_DIR = CONFIG_DIR / "tools"
