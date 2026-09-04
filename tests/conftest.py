from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_test_traces_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Configures test environment: SATQUERY_TRACES_DIR to tmp_path and stub serving enabled."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SATQUERY_TRACES_DIR", str(traces_dir))
    monkeypatch.setenv("SATQUERY_STUB_SERVING", "1")
