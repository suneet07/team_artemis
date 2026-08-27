import json
import random
from pathlib import Path

from satquery.agent.task_enum import RouterPath, Task

# Standard queries taken from the exact formats of existing benchmark datasets.
BENCHMARK_QUERIES = {
    Task.SINGLE_VQA: [
        # RSVQA style
        "Are there buildings in this image?",
        "What is the land cover here?",
        "Is there a body of water?",
        "Are there roads present?",
        "Is the area mostly forested?",
        # General VQA
        "How many ships are docked in the harbour?",
        "What is the predominant land use?",
    ],
    Task.SINGLE_CAPTION: [
        # BEN.txt style
        "Provide a description of the land cover.",
        "Describe the contents of this satellite image.",
        "Write a caption detailing the visible features.",
        "What does this region look like?",
    ],
    Task.SINGLE_GROUNDING: [
        # BEN.txt referring style
        "Highlight the water body referred to in the query.",
        "Find the forest area.",
        "Provide a bounding box for the urban region.",
        # Object level (RarePlanes, SpaceNet)
        "Where is the aircraft in this image?",
        "Locate all the buildings.",
    ],
    Task.CHANGE_DESCRIPTION: [
        # CDVQA style
        "Describe the changes between the first and latter image.",
        "What differences are visible over time?",
        "Identify the areas that have been modified.",
        "How has the land cover changed?",
    ],
    Task.CHANGE_VQA: [
        # CDVQA style
        "Have new buildings been constructed?",
        "Is there evidence of deforestation?",
        "Did the water level change?",
    ],
    Task.CHANGE_MAP: [
        # LEVIR / SpaceNet 7 style
        "Generate a change mask for this area.",
        "Provide a map of new construction.",
        "Produce a raster highlighting the differences.",
    ],
    Task.CROSSMODAL_EXTRACTION: [
        # SpaceNet 6 / OpenEarthMap-SAR style
        "Extract building footprints using both optical and SAR.",
        "Identify urban features across modalities.",
        "Find structures combining radar and optical data.",
    ],
    Task.CROSSMODAL_VQA: [
        # OSCD / multimodal style
        "Is the bridge visible in both optical and SAR?",
        "What features appear in the SAR but not optical?",
        "Are the ships moving between the two captures?",
    ]
}

def generate_dataset(output_path: Path, num_queries: int = 300):
    dataset = []
    
    tasks = list(Task)
    queries_per_task = num_queries // len(tasks)
    extra = num_queries % len(tasks)
    
    current_id = 1
    
    for i, task in enumerate(tasks):
        count = queries_per_task + (1 if i < extra else 0)
        
        for _ in range(count):
            # Mix valid (happy path) and invalid (refusal) queries
            is_refusal = random.random() < 0.2
            
            query = random.choice(BENCHMARK_QUERIES[task])
            
            if is_refusal:
                # Deliberately mismatch modalities or bands
                optical_task = "optical" in task.value or "caption" in task.value
                modalities = ["sar"] if optical_task else ["optical"]
                bands = ["VV"]
                expected_category = "parameter_gate"
                router_path = RouterPath.RULES
            else:
                modalities = ["optical", "sar"] if "crossmodal" in task.value else (
                    ["optical"] if random.random() > 0.3 else ["sar"]
                )
                bands = ["B02", "B03", "B04", "B08"] if "optical" in modalities else ["VV", "VH"]
                expected_category = "valid"
                # If valid, rules mostly route it unless ambiguous
                router_path = RouterPath.RULES if random.random() > 0.1 else RouterPath.LLM
            
            record = {
                "id": f"q_{current_id:04d}",
                "query_text": query,
                "input_context": {
                    "modalities": modalities,
                    "bands": bands,
                    "crs": "EPSG:4326"
                },
                "ground_truth": {
                    "expected_task": task.value,
                    "expected_tools": ["dummy_tool"] if not is_refusal else [],
                    "expected_router_path": router_path.value,
                    "expected_category": expected_category
                }
            }
            dataset.append(record)
            current_id += 1
            
    random.shuffle(dataset)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")
            
    print(f"Generated {len(dataset)} routing evaluation queries at {output_path}")

if __name__ == "__main__":
    out_dir = Path(r"C:\random projs webdev\Sat Query\training\eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset(out_dir / "routing_300.jsonl", 300)
