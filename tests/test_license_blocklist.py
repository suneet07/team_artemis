import os
from pathlib import Path

# C55 & C56: Explicitly blocklisted datasets and model checkpoints, due to
# non-commercial / academic-only licences.
BLOCKLIST = {
    "tinycd",
    "changeformer",
    "levir-cd",
    "levir-mci",
    "second",
    "qag-360k"
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
                                if term in content:
                                    rel = filepath.relative_to(project_root)
                                    violations.append(
                                        f"Blocklisted term '{term}' found in {rel}"
                                    )
                    except Exception as e:
                        print(f"Skipping {filepath} due to read error: {e}")
                        
    assert len(violations) == 0, "Found restricted terms in the codebase:\n" + "\n".join(violations)
