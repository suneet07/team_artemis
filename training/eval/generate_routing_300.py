import json
import random
from pathlib import Path

from satquery.agent.task_enum import RouterPath, Task

# Diverse benchmark queries with paraphrased surface forms across all tasks
BENCHMARK_QUERIES = {
    Task.SINGLE_VQA: [
        # RSVQA style
        "Are there buildings in this image?",
        "What is the land cover here?",
        "Is there a body of water?",
        "Are there roads present?",
        "Is the area mostly forested?",
        # General VQA & Paraphrases
        "How many ships are docked in the harbour?",
        "What is the predominant land use?",
        "Are agricultural fields visible in this scene?",
        "What is the primary surface type observed?",
        "Can you identify any industrial installations?",
        "Are residential zones visible?",
        "Is there an airport or landing strip?",
    ],
    Task.SINGLE_CAPTION: [
        # BEN.txt style & Paraphrases
        "Provide a description of the land cover.",
        "Describe the contents of this satellite image.",
        "Write a caption detailing the visible features.",
        "What does this region look like?",
        "Give an overall summary of the scene.",
        "Summarise the terrain and structures present.",
        "Summarize the visible environment.",
        "Provide a natural language description of this capture.",
    ],
    Task.SINGLE_GROUNDING: [
        # BEN.txt referring style & object-level targets
        "Highlight the water body referred to in the query.",
        "Find the forest area.",
        "Provide a bounding box for the urban region.",
        "Where is the aircraft in this image?",
        "Locate all the buildings.",
        "Find the storage tanks in this scene.",
        "Where are the runways located?",
        "Locate the bridge crossing the channel.",
        "Highlight the harbour facility.",
        "Where is the vehicle parking area?",
    ],
    Task.CHANGE_DESCRIPTION: [
        # CDVQA style & Paraphrases
        "Describe the changes between the first and latter image.",
        "What differences are visible over time?",
        "Identify the areas that have been modified.",
        "How has the land cover changed?",
        "Describe what differences emerged across these dates.",
        "Detail the structural changes observed over time.",
        "Explain how the urban boundary expanded between passes.",
    ],
    Task.CHANGE_VQA: [
        # CDVQA style & Paraphrases
        "Have new buildings been constructed?",
        "Is there evidence of deforestation?",
        "Did the water level change?",
        "Were any structures demolished between the dates?",
        "Has vegetated terrain decreased since the previous capture?",
        "Did new road construction occur between the observations?",
        "Has the river overflowed its banks since last year?",
    ],
    Task.CHANGE_MAP: [
        # LEVIR / SpaceNet 7 style & Paraphrases
        "Generate a change mask for this area.",
        "Provide a map of new construction.",
        "Produce a raster highlighting the differences.",
        "Create a binary change mask between the two dates.",
        "Map the areas of deforestation between observations.",
        "How much did the water area change between the two scenes?",
        "Produce a map of built-up expansion.",
    ],
    Task.CROSSMODAL_EXTRACTION: [
        # SpaceNet 6 / OpenEarthMap-SAR style & Paraphrases
        "Extract building footprints using both optical and SAR.",
        "Identify urban features across modalities.",
        "Find structures combining radar and optical data.",
        "Extract water body boundaries combining optical and SAR.",
        "Measure built-up extent using both SAR and multispectral imagery.",
        "Identify commercial structures by combining optical and radar.",
    ],
    Task.CROSSMODAL_VQA: [
        # OSCD / multimodal style & Paraphrases
        "Is the bridge visible in both optical and SAR?",
        "What features appear in the SAR but not optical?",
        "Are the ships moving between the two captures?",
        "Does the radar backscatter confirm the optical water boundary?",
        "Can you detect differences between the optical and SAR observations?",
        "Are the buildings identifiable in both optical and radar captures?",
    ],
}


def generate_dataset(output_path: Path, num_queries: int = 300, seed: int = 42):
    random.seed(seed)
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
                modalities = (
                    ["optical", "sar"]
                    if "crossmodal" in task.value
                    else (["optical"] if random.random() > 0.3 else ["sar"])
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
                    "crs": "EPSG:4326",
                },
                "ground_truth": {
                    "expected_task": task.value,
                    "expected_tools": ["dummy_tool"] if not is_refusal else [],
                    "expected_router_path": router_path.value,
                    "expected_category": expected_category,
                },
            }
            dataset.append(record)
            current_id += 1

    random.shuffle(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    print(f"Generated {len(dataset)} routing evaluation queries at {output_path}")


if __name__ == "__main__":
    out_file = Path(__file__).parent / "routing_300.jsonl"
    generate_dataset(out_file, 300)

