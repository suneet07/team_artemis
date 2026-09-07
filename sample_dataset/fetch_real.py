import json
import os
import random
import urllib.request

from satquery.qgen.gen_bbox import gen_bbox_qa
from satquery.qgen.primitives import P2BoundingBox


def fetch_real_rareplanes():
    bucket_url = "https://rareplanes-public.s3.amazonaws.com"
    s3_keys = [
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_319.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_333.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_347.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_375.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_418.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_432.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_456.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_457.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_470.png",
        "real/train/PS-RGB_tiled/100_1040010029990A00_tile_471.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_219.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_220.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_235.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_236.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_268.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_269.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_274.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_460.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_476.png",
        "real/train/PS-RGB_tiled/100_1040010039437200_tile_509.png"
    ]

    out_dir = r"c:\random projs webdev\Sat Query\sample_dataset_real"
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    canonical_records = []
    hf_records = []
    
    for idx, key in enumerate(s3_keys):
        filename = os.path.basename(key)
        local_path = os.path.join(img_dir, filename)
        
        # Download real image
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            url = f"{bucket_url}/{key}"
            urllib.request.urlretrieve(url, local_path)
            
        sample_id = f"rareplanes_{idx:04d}"
        
        # Dummy bounding boxes for the visual output since projection requires GDAL
        ymin, xmin = random.randint(100, 300), random.randint(100, 300)
        ymax, xmax = ymin + random.randint(20, 100), xmin + random.randint(20, 100)
        
        primitive = P2BoundingBox(
            sample_id=sample_id,
            image_paths=[os.path.join("images", filename)],
            effective_gsd_m=[0.3],
            source="rareplanes",
            source_ann_id=f"ann_{idx}",
            licence="CC-BY-SA-4.0",
            provenance_chain=["rareplanes", "maxar_wv3"],
            split="train",
            bboxes=[{"class": "aircraft", "bbox": [ymin, xmin, ymax, xmax], "attributes": {}}]
        )
        
        # Pass through the generator pipeline
        qa_records = list(gen_bbox_qa(primitive))
        
        # Use the refer_box output
        refer_record = [r for r in qa_records if r["task"] == "refer_box"][0]
        canonical_records.append(refer_record)
        
        hf_records.append({
            "id": refer_record["sample_id"],
            "image": refer_record["images"][0],
            "conversations": [
                {"from": "user", "value": refer_record["question"]},
                {"from": "assistant", "value": refer_record["answer"]}
            ]
        })
        
    jsonl_path = os.path.join(out_dir, "canonical_manifest.jsonl")
    with open(jsonl_path, "w") as f:
        for rec in canonical_records:
            f.write(json.dumps(rec) + "\n")
            
    hf_path = os.path.join(out_dir, "hf_training_data.json")
    with open(hf_path, "w") as f:
        json.dump(hf_records, f, indent=2)
        
    print(f"Successfully processed {len(s3_keys)} real image samples.")
    print(f"Images downloaded to: {img_dir}")
    print(f"Data saved to: {out_dir}")

if __name__ == "__main__":
    fetch_real_rareplanes()
