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


def test_routing_300_accuracy_rules_only():
    from satquery.agent.router import route_query

    dataset_path = Path(__file__).parent.parent / "training" / "eval" / "routing_300.jsonl"
    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    correct = 0
    for record in records:
        ctx = record["input_context"]
        modalities = ctx.get("modalities", [])
        task, router_path, notes = route_query(
            question=record["query_text"],
            modalities=modalities,
            image_count=len(modalities),
        )
        if task.value == record["ground_truth"]["expected_task"]:
            correct += 1

    accuracy = correct / len(records)
    print(f"Rules-only accuracy: {accuracy * 100:.1f}% ({correct}/{len(records)})")
    assert accuracy >= 0.80, f"Expected rules accuracy >= 80%, got {accuracy * 100:.1f}%"


def test_routing_heldout_dataset_validity():
    dataset_path = Path(__file__).parent.parent / "training" / "eval" / "routing_heldout.jsonl"
    assert dataset_path.exists(), f"Held-out dataset not found at {dataset_path}"

    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) >= 40, f"Expected >= 40 records, got {len(records)}"

    tasks_found = set()
    for record in records:
        assert "id" in record
        assert "query_text" in record
        assert "input_context" in record
        assert "ground_truth" in record

        gt = record["ground_truth"]
        assert gt["expected_task"] in set(t.value for t in Task)
        assert gt["expected_router_path"] in set(p.value for p in RouterPath)
        tasks_found.add(gt["expected_task"])

    assert len(tasks_found) == len(Task), "Held-out set must cover all 8 tasks"


def test_routing_heldout_accuracy_rules_only():
    from satquery.agent.router import route_query

    dataset_path = Path(__file__).parent.parent / "training" / "eval" / "routing_heldout.jsonl"
    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    correct = 0
    for record in records:
        ctx = record["input_context"]
        modalities = ctx.get("modalities", [])
        task, router_path, notes = route_query(
            question=record["query_text"],
            modalities=modalities,
            image_count=len(modalities),
        )
        if task.value == record["ground_truth"]["expected_task"]:
            correct += 1

    accuracy = correct / len(records)
    print(f"Held-out rules accuracy: {accuracy * 100:.1f}% ({correct}/{len(records)})")
    assert accuracy >= 0.85, f"Expected held-out accuracy >= 85%, got {accuracy * 100:.1f}%"


def test_routing_stage_2_llm_tiebreak():
    from satquery.agent.router import route_query

    # Mock client returns constrained JSON
    def mock_llm(q: str):
        return {
            "task": "change_map",
            "reason": "User asks for spatial distribution of differences across years",
        }

    task, path, notes = route_query(
        question="Show the evolution of this region",
        modalities=["optical", "optical"],
        image_count=2,
        allow_llm=True,
        llm_client=mock_llm,
    )
    assert task == Task.CHANGE_MAP
    assert path == RouterPath.LLM
    assert any("Stage 2 LLM tie-break" in n for n in notes)
