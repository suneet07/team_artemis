import os
import re
from pathlib import Path

# C55 & C56: Explicitly blocklisted datasets and model checkpoints, due to
# non-commercial / academic-only licences.
BLOCKLIST = {
    "tinycd",
    "changeformer",
    "levir-cd",
    "levir-mci",
    "second",
    "qag-360k",
}


def test_no_blocklisted_terms_in_manifests():
    """
    Ensures that no restricted datasets or model checkpoints are referenced
    in any training manifests, configs, or requirements.
    """
    project_root = Path(__file__).parent.parent

    # Directories to check
    dirs_to_check = [
        project_root / "configs",
        project_root / "training",
        project_root / "satquery",
    ]

    violations = []

    for d in dirs_to_check:
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith((".py", ".yaml", ".json", ".jsonl", ".txt")):
                    filepath = Path(root) / file
                    try:
                        with open(filepath, encoding="utf-8") as f:
                            content = f.read().lower()
                            for term in BLOCKLIST:
                                if term == "second":
                                    # Distinguish SECOND dataset from English ordinal
                                    pat = (
                                        r"\b(dataset|benchmark|model|data).*second\b|"
                                        r"\bsecond[-_]?(dataset|benchmark|cd)\b"
                                    )
                                    if re.search(pat, content):
                                        rel = filepath.relative_to(project_root)
                                        violations.append(
                                            f"Blocklisted term '{term}' found in {rel}"
                                        )
                                elif term in content:
                                    rel = filepath.relative_to(project_root)
                                    violations.append(
                                        f"Blocklisted term '{term}' found in {rel}"
                                    )
                    except Exception as e:
                        print(f"Skipping {filepath} due to read error: {e}")

    assert len(violations) == 0, "Found restricted terms:\n" + "\n".join(violations)
