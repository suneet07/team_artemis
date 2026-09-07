import json
from pathlib import Path

from satquery.agent.task_enum import RouterPath, Task


def test_routing_300_dataset_validity():
    dataset_path = Path(__file__).parent.parent / "training" / "eval" / "routing_300.jsonl"
    assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 300, f"Expected 300 records, got {len(records)}"

    tasks_found = set()
    for record in records:
        assert "id" in record
        assert "query_text" in record
        assert "input_context" in record
        assert "ground_truth" in record

        ctx = record["input_context"]
        assert "modalities" in ctx
        assert "bands" in ctx
        assert "crs" in ctx

        gt = record["ground_truth"]
        assert "expected_task" in gt
        assert "expected_tools" in gt
        assert "expected_router_path" in gt
        assert "expected_category" in gt

        # Verify valid values
        assert gt["expected_task"] in set(t.value for t in Task)
        assert gt["expected_router_path"] in set(p.value for p in RouterPath)

        tasks_found.add(gt["expected_task"])

    assert len(tasks_found) == len(Task), "Not all tasks are covered in the dataset"
