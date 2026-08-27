# SatQuery AI — Exhaustive Dataset Technical Specifications

This document serves as an exhaustive technical reference for all datasets integrated into the SatQuery AI project. It details processing levels, exact band configurations, image formats, bit-depths, metadata schemas, and specific "jump scares" derived from deep technical review of the source papers.

## 1. BigEarthNet / BigEarthNet.txt (reBEN / BEN-MM)
**Primary Role:** VQA, Captioning, Region Grounding, Classifier Pretraining.
### Specifications:
- **Version:** BigEarthNet v2.0 (reBEN), which upgraded atmospheric correction and updated the CLC map to reduce label noise. Contains 549,488 image patch pairs.
- **Optical (Sentinel-2):**
  - **Processing Level:** Level-2A (Atmospherically corrected using Sen2Cor v2.11).
  - **Bands & Resolution:** 12 spectral bands provided at three native Ground Sampling Distances (GSD):
    - **10m (120x120 px):** B02 (Blue), B03 (Green), B04 (Red), B08 (NIR).
    - **20m (60x60 px):** B05, B06, B07 (Vegetation Red Edge), B8A (Narrow NIR), B11, B12 (SWIR).
    - **60m (20x20 px):** B01 (Coastal Aerosol), B09 (Water Vapor).
- **SAR (Sentinel-1):**
  - **Processing Level:** Level-1 Ground Range Detected (GRD).
  - **Bands:** Dual-polarization (VV and VH), C-band. Co-registered with S2.
- **Labels:** 19 CORINE land-cover classes (mapped down from the original 43). Multi-label (sigmoid), not single-label (softmax).

> [!WARNING]
> **Engineering Jump Scares:**
> - **Variable Patch Sizes:** The S2 patches physically differ in array dimensions based on GSD. You must implement interpolation (typically upsampling 20m and 60m bands to 120x120) before stacking them into the 3-composite inputs.
> - **No Float32 Out-of-the-Box:** Raw values require proper normalization based on Sentinel-2 10,000 scale factors.

## 2. SpaceNet 7 / MUDS (Multi-Temporal Urban Development)
**Primary Role:** Expert-labelled change detection (`change_map`, `change_vqa`).
### Specifications:
- **Imagery Source:** Planet Labs monthly satellite imagery mosaics.
- **Scale:** 24 monthly mosaics per geography across ~100 global geographies. Total >40,000 km².
- **Resolution:** 4.0 meters GSD.
- **Format:** 8-bit, 4-band (RGBA: Red, Green, Blue, Alpha) electro-optical (EO).
- **Metadata Structure:** Follows SpatioTemporal Asset Catalog (STAC) standards.
- **Labels:** >11 million individual building footprints. Provided as **GeoJSON** polygons with Well-Known Text (WKT) geometries.
- **Metric:** SpaceNet Change and Object Tracking (SCOT).

> [!WARNING]
> **Engineering Jump Scares:**
> - **Alpha Channel:** The imagery is RGBA, not RGB. The alpha mask dictates valid imagery vs. nodata.
> - **Persistent IDs:** Every building footprint is given a unique identifier (address) to track construction/demolition over time. Your pipeline must parse these IDs to calculate accurate counting and change metrics (D4).
> - **Empty Geometries:** The dataset explicitly uses `POLYGON EMPTY` in GeoJSON to represent scenes with no buildings. Handle this without throwing coordinate parsing errors.

## 3. SpaceNet 6: MSAW (Multi-Sensor All-Weather Mapping)
**Primary Role:** D1 validation, X-band/single-pol stress testing, cross-modal `optsar_fusion`.
### Specifications:
- **Optical (Maxar WorldView-2):** 0.5m GSD. Panchromatic, pan-sharpened RGB, and RGBNIR (4-band).
- **SAR (Capella Space):**
  - **Sensor Type:** X-band synthetic aperture radar (aerial-mounted to mimic space-borne).
  - **Resolution:** 0.5m x 0.25m (Range x Azimuth).
  - **Processing Levels Provided:** 
    - **SLC (Single Look Complex):** Raw phase and complex information.
    - **MAG-POL (Georeferenced Magnitude and Polarimetry):** Georeferenced, 6-band imagery (4 channels of quad-pol intensity: HH, HV, VH, VV + 2 Pauli decomposition channels). Backscatter intensity is in **decibel (dB)** units.
- **Labels:** ~48,000 building footprints (plus height statistics derived from the 3DBAG dataset).

> [!WARNING]
> **Engineering Jump Scares:**
> - **SAR Band Count:** The MAG-POL imagery contains 6 bands. Your optical-SAR fusion pipeline is built around Sentinel-1's dual-pol. You MUST extract the VV or HH bands to synthesize the single-pol RISAT fallbacks correctly.
> - **Test Set Imbalance:** The official testing/scoring data is *SAR-only* to simulate cloudy scenarios.

## 4. OpenEarthMap-SAR
**Primary Role:** Sub-meter cross-modal fusion, single-pol SAR validation.
### Specifications:
- **Imagery Source:** Satellite SAR imagery (Umbra Spotlight).
- **Geographic Scale:** 5,033 images (1024x1024 pixels) across 35 regions.
- **Resolution:** Very high resolution ranging from 0.15m to 0.5m GSD.
- **Polarization:** Single-pol (VV or HH).
- **Preprocessing:** Radiometrically normalized backscatter intensity standardized to an **8-bit** format (to match the original optical OpenEarthMap).
- **Labels:** 8 land cover classes (bareland, rangeland, developed space, road, tree, water, agricultural land, building).

> [!WARNING]
> **Engineering Jump Scares:**
> - **8-bit SAR:** The data has already been stretched and compressed to 8-bit. Standard raw float32 backscatter thresholds (e.g., `-18 dB` for water) will **not** work directly on this dataset without reverse-calibration.
> - **Noisy Pseudo-labels:** Only ~20 images per region are manually annotated. The rest are generated by optical models. The "Bareland" class has an IoU of ~0.02. You must implement label confidence filtering (omitting low-confidence pixels) during training.

## 5. RarePlanes
**Primary Role:** Object-level grounding for aircraft (`rs_ground_caption`).
### Specifications:
- **Imagery Source:** Maxar WorldView-3 (WV3).
- **Image Formats Provided:** 
  - 8-bit RGB
  - 16-bit 9-channel Multi-Spectral (MS)
  - 16-bit Panchromatic (PAN)
- **Resolution:** 0.3m to 1.25m GSD. Off-nadir collection angles range from 3.2 to 29.6 degrees.
- **Tiling:** Native imagery is pre-tiled into 512x512 pixel chips with 20% overlap.
- **Annotations:** COCO JSON and GeoJSON. Contains ~14,700 real and ~630,000 synthetic aircraft.

> [!WARNING]
> **Engineering Jump Scares:**
> - **Dense Attribute Metadata:** Annotations include 10 fine-grained attributes (length, wingspan, wing-shape, propulsion, vertical stabilizers, canards, etc). If your model only needs bounding boxes, strip this metadata out early to save RAM and parsing time.

## 6. LS-SSDD-v1.0 (Large-Scale SAR Ship Detection)
**Primary Role:** Object-level grounding for ships.
### Specifications:
- **Imagery Source:** Sentinel-1 Interferometric Wide (IW) swath mode. Both VV and VH polarizations.
- **Dimensions:** 15 ultra-large images unified to 24,000 × 16,000 pixels, directly cropped into 9,000 sub-images.
- **Format:** 3-channel, 24-bit grayscale **.JPG**.
- **Labels:** Expert-labeled XML formats bounding boxes (`Xmin`, `Xmax`, `Ymin`, `Ymax`), verified by AIS (Automatic Identification System).

> [!WARNING]
> **Engineering Jump Scares:**
> - **Format Compression:** Because the images are stored as 3-channel JPGs, standard SAR radiometric calibration (e.g., SNAP tools converting digital numbers to sigma nought) cannot be applied. It is purely a computer-vision dataset at this stage.
> - **Pure Background Hybrid Training (PBHT):** The dataset deliberately leaves "pure background" land images (without ships) in the dataset to act as negative samples. DO NOT filter these out; they are vital to suppressing false alarms on land surfaces.

## 7. HRSCD (High-Resolution Semantic Change Detection)
**Primary Role:** Sub-meter semantic change detection.
### Specifications:
- **Imagery Source:** Aerial RGB orthophotos (BD ORTHO database by IGN France).
- **Scale & Coverage:** 291 coregistered image pairs (2006 and 2012) at 10,000 × 10,000 pixels.
- **Resolution:** 0.5m GSD.
- **Labels:** Pixel-level binary change masks and semantic land cover masks grouped into 5 hierarchical classes (derived from Copernicus Urban Atlas).

> [!WARNING]
> **Engineering Jump Scares:**
> - **Aerial vs Satellite Geometry:** Orthophotos suffer from building lean (relief displacement) that differs fundamentally from space-borne satellite nadir perspectives. 
> - **Label Alignment:** Use the community-refined "HRSCD-Clean" subset. The raw labels are extremely coarse. 
> - **Licensing Constraint:** The 2006 imagery falls under a non-redistributable caveat. Ensure it is excluded from any public demo or shipped artifacts.

## 8. OSCD + S1 Extension (Onera Satellite Change Detection)
**Primary Role:** Optical-SAR multimodal change detection.
### Specifications:
- **Optical (Sentinel-2):** Multispectral bi-temporal pairs.
- **SAR (Sentinel-1):** Community-provided extensions (e.g., from DS-UNet implementations or GEE pulls). 
- **Preprocessing:** Georeferenced, radiometrically calibrated, and co-registered to the S2 optical grid.
- **Format:** Dual-polarization (VV and VH). Measurements are in sigma nought ($\sigma^0$) backscatter values in the dB scale.

> [!WARNING]
> **Engineering Jump Scares:**
> - **Value Ranges:** Because the SAR data is already in dB scale (logarithmic), negative float values (e.g., -15.4) are the standard representation. Ensure your normalization functions expect this domain rather than linear digital numbers.

## 9. SARLANG-1M
**Primary Role:** SAR QA, Captioning, Modality Dropout.
### Specifications:
- **Scale:** 118,331 SAR images paired with 1,126,277 text annotations. Contains 1,012 distinct question types covering 1,696 object types.
- **Sourcing:** An aggregation from 4 distinct sub-datasets collected by >12 different satellites.
- **Resolution Variation:** Hierarchical resolutions ranging from **0.1m to 25m**.

> [!WARNING]
> **Engineering Jump Scares:**
> - **Extreme Scale Variance:** Feeding a 0.1m image and a 25m image into the same network without context will destroy performance. The agent *must* inject the Ground Sampling Distance (GSD) into the prompt string (GSD-conditioned prompting) and utilize multi-scale augmentation.

## 10. RSVQA-HR
**Primary Role:** High-resolution single-image VQA.
### Specifications:
- **Imagery Source:** USGS High-Resolution Orthophoto (HRO).
- **Resolution:** 15cm (0.15m) GSD.
- **Scale:** ~10,659 images (512x512 px) yielding >1.06 million Question-Answer pairs.
- **Generation:** Triplets generated automatically by querying OpenStreetMap (OSM) vector data.

> [!WARNING]
> **Engineering Jump Scares:**
> - **Linguistic Exploitation (Answer-Centric):** Because questions are procedurally generated from OSM templates, they contain rigid syntactic structures. Standard VQA models often bypass the visual encoder entirely and "guess" based on language priors. You must monitor test-set splits heavily to detect if the model is ignoring the image.
