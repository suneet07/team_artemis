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
    skipped: list[str] = []

    print(f"Extracting first {num_samples} samples...")
    for idx, item in enumerate(dataset):
        if idx >= num_samples:
            break

        # The dataset contains Sentinel-2 images (and S1).
        # For our VQA test, we just need to save the optical image and the text.
        # Note: HF dataset format might vary slightly, but generally it
        # contains a PIL image or array
        img_id = item.get("patch_id", f"patch_{idx:04d}")

        # Save the prompt/answer for VQA
        question = item.get("input", "Describe the land cover in this image.")
        answer = item.get("output", "Unknown land cover.")

        # TEAM_CONTEXT mistake 2: this branch used to write a 120x120 array of
        # zeros whenever the record carried no composed image, and it appended a
        # metadata row alongside it. Twenty-one constant-black PNGs entered
        # ben-micro-split that way, a loss curve was read off them, and the run
        # looked like it was learning. A silent black-image fallback is the most
        # expensive line in this file's history.
        #
        # It now skips the record and says so. A missing sample is recoverable;
        # a fabricated one is not.
        image = item.get("image")
        if image is None:
            print(f"Skipping {img_id}: record carries no composed image")
            skipped.append(str(img_id))
            continue
        try:
            array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
        except Exception as error:  # noqa: BLE001 - one bad record must not stop the run
            print(f"Skipping {img_id}: image could not be read ({error})")
            skipped.append(str(img_id))
            continue

        if array.size == 0 or array.min() == array.max():
            # Constant image: black, white, or a single fill value. Never
            # training data, whatever the pipeline upstream believes.
            print(f"Skipping {img_id}: image is constant ({array.min()}), not real imagery")
            skipped.append(str(img_id))
            continue

        Image.fromarray(array.astype(np.uint8)).save(os.path.join(images_dir, f"{img_id}.png"))
        metadata.append({"image_id": f"{img_id}.png", "question": question, "answer": answer})

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{num_samples}")

    # Save the text data as a JSONL file
    metadata_path = os.path.join(output_dir, "train_metadata.jsonl")
    with open(metadata_path, "w") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! Extracted {len(metadata)} usable samples of {num_samples} requested.")
    if skipped:
        print(
            f"Skipped {len(skipped)} record(s) with no usable image: "
            f"{', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}"
        )
    if not metadata:
        raise SystemExit(
            "No usable images were extracted. Do NOT upload: an empty or black-image "
            "split is exactly the failure that produced a believable loss curve on "
            "nothing at all (TEAM_CONTEXT mistake 2)."
        )
    print(
        "\nLICENCE: the annotation layer is CDLA-Permissive 1.0 and derived "
        "annotations inherit it. dataset-metadata.json must NOT declare CC0-1.0 "
        "(TEAM_CONTEXT mistake 6)."
    )
    print("Next steps:")
    print(f"1. Run: kaggle datasets init -p {output_dir}")
    print(f"2. Edit {output_dir}/dataset-metadata.json to name your dataset")
    print(f"3. Run: kaggle datasets create -p {output_dir}")


if __name__ == "__main__":
    create_micro_split()
