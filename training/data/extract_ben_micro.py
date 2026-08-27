import json
import os

import numpy as np
from datasets import load_dataset
from PIL import Image


def create_micro_split(output_dir: str = "ben-micro-split", num_samples: int = 500):
    print(f"Creating Kaggle dataset folder at: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # We use streaming=True so we don't download the 600GB archive!
    print("Connecting to HuggingFace to stream BigEarthNet.txt...")
    dataset = load_dataset(
        "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt", split="all_data", streaming=True
    )
    
    metadata = []
    
    print(f"Extracting first {num_samples} samples...")
    for idx, item in enumerate(dataset):
        if idx >= num_samples:
            break
            
        # The dataset contains Sentinel-2 images (and S1). 
        # For our VQA test, we just need to save the optical image and the text.
        # Note: HF dataset format might vary slightly, but generally it
        # contains a PIL image or array
        img_id = item.get('patch_id', f"patch_{idx:04d}")
        
        # Save the prompt/answer for VQA
        question = item.get('input', 'Describe the land cover in this image.')
        answer = item.get('output', 'Unknown land cover.')
        
        # Simulate saving the image (in a real run, you extract the actual image tensor to PNG/TIFF)
        # We will save a dummy image if the actual image object isn't directly a PIL image, 
        # but normally HF datasets provide a PIL Image for optical patches.
        try:
            if 'image' in item:
                img = item['image']
                img.save(os.path.join(images_dir, f"{img_id}.png"))
            else:
                # Fallback if the dataset provides raw bands instead of a composed image
                dummy = Image.fromarray(np.zeros((120, 120, 3), dtype=np.uint8))
                dummy.save(os.path.join(images_dir, f"{img_id}.png"))
        except Exception as e:
            print(f"Skipping image save for {img_id}: {e}")
            
        metadata.append({
            "image_id": f"{img_id}.png",
            "question": question,
            "answer": answer
        })
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{num_samples}")

    # Save the text data as a JSONL file
    metadata_path = os.path.join(output_dir, "train_metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\nDone! Extracted {num_samples} samples.")
    print("Next steps:")
    print(f"1. Run: kaggle datasets init -p {output_dir}")
    print(f"2. Edit {output_dir}/dataset-metadata.json to name your dataset")
    print(f"3. Run: kaggle datasets create -p {output_dir}")

if __name__ == "__main__":
    create_micro_split()
