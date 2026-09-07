import json
import os

from satquery.qgen.gen_bbox import gen_bbox_qa
from satquery.qgen.loaders.rareplanes import RarePlanesLoader


def process_real_rareplanes():
    out_dir = r"c:\random projs webdev\Sat Query\sample_dataset_real"
    
    loader = RarePlanesLoader(data_dir=out_dir)
    
    canonical_records = []
    hf_records = []
    
    # Process up to 20 images
    for primitive in loader.emit(limit=20):
        # Pass through the generator pipeline
        qa_records = list(gen_bbox_qa(primitive))
        
        # Use the refer_box output if available
        refer_records = [r for r in qa_records if r["task"] == "refer_box"]
        if not refer_records:
            continue
            
        refer_record = refer_records[0]
        canonical_records.append(refer_record)
        
        hf_records.append({
            "id": refer_record["sample_id"],
            "image": refer_record["images"][0],
            "conversations": [
                {"from": "user", "value": refer_record["question"]},
                {"from": "assistant", "value": refer_record["answer"]}
            ]
        })
        
    jsonl_path = os.path.join(out_dir, "canonical_manifest_projected.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in canonical_records:
            f.write(json.dumps(rec) + "\n")
            
    hf_path = os.path.join(out_dir, "hf_training_data_projected.json")
    with open(hf_path, "w") as f:
        json.dump(hf_records, f, indent=2)
        
    print(f"Successfully projected and processed {len(canonical_records)} real image samples.")
    print(f"Projected data saved to: {out_dir}")

if __name__ == "__main__":
    process_real_rareplanes()
