# Data Preparation Assignment: RarePlanes

## 1. The Target Adapter: `rs_ground_caption`
You are assigned to prepare training data for the **`rs_ground_caption`** adapter. 
This is a LoRA adapter for our Vision-Language Model (Qwen3-VL-4B-Instruct). Its specific purpose is **object grounding**—meaning it must learn to look at an image and reply with exact bounding box coordinates for objects of interest (e.g., aircraft, ships), or take coordinates as input and describe the object inside them.

## 2. The Dataset: RarePlanes
You will be processing the **RarePlanes** dataset. 
- **The Raw Format:** RarePlanes provides high-resolution Maxar satellite images alongside **GeoJSON / COCO JSON** files containing bounding boxes and attributes for aircraft.
- **Why you are manipulating it:** Vision-Language Models cannot read raw GeoJSON files. They train on conversational text (Questions and Answers). You must build the bridge that turns those raw coordinates into conversational text.

## 3. The Workflow Strategy
You must follow the project's **"Manifest First"** data pipeline. Do not attempt to download the entire 50GB+ dataset to your local machine.

### Step A: Download Metadata Only
Connect to the RarePlanes AWS Open Data bucket and download **only** the annotation JSON files. 

### Step B: Sub-sample and Create the Manifest
We do not need the entire dataset for this adapter. 
1. Write a Python script to analyze the annotations.
2. Select a balanced subset of images (e.g., 5,000 images) that provide a good mix of single-aircraft scenes, crowded tarmac scenes, and empty backgrounds (to prevent false positives).
3. Save the IDs of these selected images into a text file called `manifest.txt`.

### Step C: Template Generation & Coordinate Normalization
For every image in your `manifest.txt`, your Python script must read its original JSON bounding boxes and construct text-based Q&A pairs.

> [!IMPORTANT]  
> **Coordinate Normalization:** VLM models usually expect coordinates scaled from `0` to `1000` relative to the image dimensions (e.g., `[ymin, xmin, ymax, xmax]`). Your script must calculate this conversion from the raw pixel coordinates.

**Required JSON Output Format (Hugging Face Native):**
Our pipeline trains native Hugging Face models using `transformers`, `peft` (LoRA), and `bitsandbytes`. Therefore, your output must be structured so that Hugging Face's `apply_chat_template()` function can parse it natively. You must output a standard conversation array exactly like this:

```json
[
  {
    "id": "rareplanes_001",
    "image": "images/wv3_scene_104.png",
    "conversations": [
      {
        "from": "user",
        "value": "Find the aircraft in this image and provide its bounding box."
      },
      {
        "from": "assistant",
        "value": "There is an aircraft located at [450, 200, 500, 250]."
      }
    ]
  },
  {
    "id": "rareplanes_002",
    "image": "images/wv3_scene_105.png",
    "conversations": [
      {
        "from": "user",
        "value": "What object is located at [100, 300, 150, 350]?"
      },
      {
        "from": "assistant",
        "value": "That bounding box contains a small aircraft."
      }
    ]
  }
]
```

### Step D: Linguistic Augmentation
If every user query in your JSON is exactly *"Find the aircraft"*, the model will overfit to that exact phrase. 
- You must create a bank of 10 to 15 text templates (e.g., *"Locate the plane"*, *"Give me the coordinates for the jet"*, *"Where are the aircraft located?"*).
- Have your script randomly sample from these templates when constructing the `conversations` array so the model learns natural language variations.

### Step E: Targeted Download and Staging
1. Write a final Python script that reads your `manifest.txt` and downloads **only** those specific images from AWS.
2. Put the images in a folder alongside your final generated `conversations.json`.
3. Upload this finalized package as a **Private Kaggle Dataset**. The ML engineers will attach this directly to their native Hugging Face training notebooks.
