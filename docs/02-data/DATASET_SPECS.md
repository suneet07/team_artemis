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

---

# Part B — Access & parsing log

Part A above is derived from the source papers: what each dataset *says* it is.
This part is derived from actually downloading and parsing them: what each
dataset *turned out to be*, what the access path costs, and what went wrong.

The two disagree often enough to be worth separating. Where a paper-derived
claim in Part A is contradicted by a hands-on finding here, **this part wins**,
and the contradiction is written down rather than quietly corrected — a spec
that was wrong once will be believed again by the next reader.

**Convention for new entries.** One section per dataset, added the first time
anyone tries to parse it, extended every time. Record:

1. **Source of truth** — the exact record/URL the files came from.
2. **Files and real sizes** — measured, not quoted from a landing page.
3. **Actual schema** — column names and row counts as loaded, not as documented.
4. **Access path** — the sequence that worked, including any trick that avoided
   a large download.
5. **Gotchas** — each with the symptom it produces, so it is recognisable next
   time rather than merely listed.
6. **Open problems** — unresolved, with what was already ruled out.

## B1. BigEarthNet v2.0 / reBEN — parsed 2026-08-29

**Source of truth:** Zenodo record [10891137](https://zenodo.org/records/10891137).
Licence CDLA-Permissive 1.0.

### Files and real sizes (measured by HTTP HEAD, not quoted)

| File | Size | Holds |
|---|---|---|
| `metadata.parquet` | 3.6 MB | labels, split, country — **no geometry** |
| `metadata_for_patches_with_snow_cloud_or_shadow.parquet` | small | the excluded patches |
| `Reference_Maps.tar.zst` | **282 MB** | per-patch georeferenced GeoTIFFs |
| `BigEarthNet-S2.tar.zst` | **59 GiB** | the S2 patch imagery |
| `BigEarthNet-S1.tar.zst` | ~51 GiB | the S1 patch imagery |

### `metadata.parquet` actual schema

480,038 rows. Columns, exactly:

```
patch_id, labels, split, country, s1_name, s2v1_name,
contains_seasonal_snow, contains_cloud_or_shadow
```

**There is no geometry column of any kind** — no bounds, no CRS, no transform,
no centroid. This contradicted the assumption written into
`satquery/ingest/copernicus/patch_grid.py`, which named this file as the source
of per-patch georeferencing for the grid-convention verify. It is not, and
following that pointer costs an afternoon before the absence becomes obvious.

Note also 480,038 rows against the 549,488 patches Part A quotes: the ~69k
difference is the snow/cloud/shadow patches, which live in the *other* parquet.
Anyone computing corpus size from this file alone will be short and will not
be told.

### Where the georeferencing actually lives

`Reference_Maps.tar.zst`. Its members are per-patch GeoTIFFs carrying CRS and
transform, which is authoritative dataset-supplied geometry.

**Access trick — do not download 282 MB for one patch.** A single patch settles
the grid convention, so `scripts/fetch_reben_reference_bounds.py` streams the
archive with `tarfile` in `"r|"` mode over `compression.zstd` (Python 3.14+;
falls back to the `zstandard` package) and stops at the first usable member.
Costs a few MB.

**The sampled patch must be off-diagonal.** Where the two trailing indices are
equal, `row_col` and `col_row` produce identical bounds, so a diagonal patch is
consistent with both conventions and proves nothing. The script skips them.

**Verified sample:**

```
patch:  S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
CRS:    EPSG:32633
size:   120 x 120 px @ 10 m
bounds: 331200.0  5330400.0  332400.0  5331600.0
```

Kept at `data/reben/reference_sample.tif` as evidence.

### Gotchas

- **`N9999` is a placeholder baseline.** reBEN overwrites the real processing
  baseline in every patch id. A CDSE catalogue query filtering on the id
  verbatim returns an empty list, which reads like a withdrawn product rather
  than a malformed query. Match on mission, level, sensing time, orbit and tile
  and **exclude the baseline field**. The patch above resolves to the live
  product `S2A_MSIL2A_20170613T101031_N0500_R022_T33UUP_20231008T194656.SAFE`.
- **Split disagreement.** `metadata.parquet` puts `..._26_57` in the **test**
  split, but `ben-micro-split/train_metadata.jsonl` uses it as training data.
  The micro-split's provenance needs checking before any number is reported off
  it — this is the shape of a train/test leak.
- **The repo's BEN chips are fabricated.** All 21 PNGs in
  `ben-micro-split/images/` are pure black (`min 0 max 0 std 0.0`). They came
  from a missing-image fallback and a loss curve was once read off them. Never
  point a training or timing run at that directory.

### CDSE `/vsis3/` access — what works

- Two separate credential sets, not interchangeable: OIDC username/password for
  the OData catalogue, and S3 access key/secret generated in the CDSE S3 keys
  manager. The account password does not authenticate S3.
- **`rasterio.Env` rejects AWS credentials as kwargs** — `EnvError: GDAL's AWS
  config options can not be directly set. AWS credentials are handled
  exclusively by boto3.` Passing them there fails at open time.
- **`rasterio.session.AWSSession` needs boto3**, which this project does not
  otherwise depend on and which would follow into the Modal image for one
  endpoint. Symptom without it: `AttributeError: 'NoneType' object has no
  attribute 'Session'`.
- **What works instead:** set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `AWS_S3_ENDPOINT` in `os.environ` — GDAL's `/vsis3/` driver reads them there —
  and pass only the non-credential options to `rasterio.Env`. Restore the
  previous values afterwards so credentials do not outlive the read.
- Endpoint is `eodata.dataspace.copernicus.eu`, **path-style**
  (`AWS_VIRTUAL_HOSTING=FALSE`). Virtual hosting resolves `eodata.<endpoint>`
  and 404s every band.

### Grid convention — SETTLED 2026-08-29

`col_row_from_northwest`. **Column first, row second**, counted from the tile's
north-west corner.

The module shipped with `row_col_from_northwest` as its placeholder, described
in the code as "standard raster order, which is the most likely". **It was
wrong.** Had the guard not blocked manifest emission, every patch would have
been read at its transpose and carried another patch's CORINE labels, with loss,
accuracy and the D1 agreement rate all staying plausible.

Evidence, re-checkable:

```
patch:  S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
bounds: 331200.0 5330400.0 332400.0 5331600.0   (EPSG:32633, from its reference map)
tile:   origin (300000, 5400000) @ 10 m         (from the granule geotransform)

(331200  - 300000)  / 10 = 3120 px = 26 x 120  -> first index is the COLUMN
(5400000 - 5331600) / 10 = 6840 px = 57 x 120  -> second index is the ROW
```

Both exact, no tolerance slack, and the patch is off-diagonal so the two
readings genuinely disagreed.

A unit test (`test_window_is_a_whole_patch_on_the_ten_metre_grid`) had encoded
the *guess* and was passing. A green test suite was not evidence here.

### `/vsis3/` band paths — the wildcard trap

`_band_url` used to return a glob:

```
/vsis3/eodata/.../GRANULE/*/IMG_DATA/R10m/*_B04_*.jp2
```

on the assumption that GDAL expands it. **GDAL does not.** `/vsis3/` treats the
path as a literal object key, asks S3 for a key containing an asterisk, and gets
nothing; with `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` it cannot list to resolve
either, so it retries rather than erroring. Symptom: an open that hangs with no
error message, indefinitely.

The names cannot be constructed either — the granule subdirectory is
`L2A_T33UUP_A010315_20170613T101608`, carrying the absolute orbit and datastrip
sensing time, neither of which appears in the product name. It must be listed.

**The OData Nodes API lists the SAFE tree and needs no authentication:**

```
GET {catalogue}/Products({id})/Nodes({product_name})/Nodes(GRANULE)/Nodes({granule_dir})/Nodes(IMG_DATA)/Nodes(R10m)/Nodes
```

`resolve_band_paths()` walks this and caches per product, so a granule group
lists once. Measured cost: catalogue resolve ~18 s, three Nodes calls ~4 s,
`rasterio.open` over `/vsis3/` **0.7 s** on a 10980x10980 JP2.

**Band resolution must be chosen per band, never by listing order.** B02, B03
and B04 are published at *both* 10 m and 20 m. A flat merge of the two directory
listings lets whichever was walked last win — which silently resolved the
visible bands to 20 m and would have halved the GSD of every true-colour
composite with no error anywhere.

### Debugging note

Three separate "hangs" here were an artefact of piping the command through
`| tail -N`: `tail` buffers the whole stream until EOF, so a working 23-second
run and a genuinely stuck one both print nothing. Run these unbuffered
(`python -u`, no pipe) when timing anything against CDSE.

### Mixed native resolutions in one patch

Each band is read as a window of *its own* native raster, so a 1200 m patch
comes back as 120x120 for the 10 m bands and **60x60 for the 20 m bands**.
`write_patch` stacked them directly and died with
`ValueError: all input arrays must have the same shape` — for every patch, so
this blocks the whole fetch rather than corrupting a few.

A GeoTIFF holds one grid and one transform, so the 20 m bands are put on the
10 m grid with nearest-neighbour: it replicates measured values instead of
interpolating SWIR into numbers no sensor produced.

**The upsampling must be recorded, not just performed.** After it, the raster's
own pixel size says 10 m for all six bands, and GSD-conditioned prompting would
tell the model a resolution B11 and B12 never had. The written tags therefore
carry `grid_gsd_m`, `upsampled_bands`, `upsampled_from_gsd_m`, `resampling`, and
a per-band `native_gsd_m`.

### First real corpus — fetched 2026-08-29

21/21 patches, 0 failures, one granule, **500 canonical rows**, 63 composites.
Independent confirmation that the geometry is right: the written GeoTIFF's
transform origin is `(331200.0, 5331600.0)`, which is exactly the bounds the
reference map gave — the convention derived through the granule geotransform
lands on the same ground as the dataset's own georeferencing.

Sanity of the pixels, checked rather than assumed:

```
band B02 native=10m  min=1109 max=2792 mean=1366
band B04 native=10m  min=1100 max=3326 mean=1475
band B08 native=10m  min=2176 max=7829 mean=4554
band B11 native=20m  min=1721 max=4262 mean=2929
```

Values sit in the L2A 10,000-scale range, and NIR (4554) far exceeds red (1475),
which is the vegetation signature expected over Austria in June. Every composite
has std > 49 — compare the fabricated `ben-micro-split/images/` PNGs at std 0.0.

### Open problems

- **Split disagreement — RESOLVED 2026-08-29, and it was real.** All **21** of
  the micro-split's patches are in reBEN's **test** split, while
  `ben-micro-split/train_metadata.jsonl` presents them as training data. The
  first fetched corpus inherited that and labelled 500 canonical rows `train`.
  Nothing was damaged — the only run against them was a timing smoke that
  reports no accuracy — but training on them and then quoting a BEN.txt number
  would have inflated it silently.

  Fixed in two places: `build_ben_manifest.py` now checks the requested split
  against `metadata.parquet` and **hard-stops** on disagreement (override is
  explicit and must be justified), and `data/chips/canonical.jsonl` is
  relabelled `test` with `split_source` recording why.

  The general lesson for every other dataset in this file: **a split label that
  ships with a derived subset is not evidence.** Check it against the original
  release.

## B2. RSVQA-LR — parsed 2026-08-29

**Source of truth:** Zenodo record [6344334](https://zenodo.org/records/6344334).
Licence **CC-BY-4.0**, read from the record's own metadata rather than the paper.

### Files and real sizes

| File | Size | Holds |
|---|---|---|
| `Images_LR.zip` | 90.6 MB | 772 tif images |
| `all_questions.json` | 14.6 MB | **the question text, all splits together** |
| `all_answers.json` | 8.7 MB | **the answer text, all splits together** |
| `LR_split_<split>_questions.json` | 2.6–11.4 MB | membership flags only |

Whole dataset is ~140 MB — it is the cheapest real benchmark in the project.

### The trap: the split files contain no text

`LR_split_test_questions.json` holds 33,212 rows shaped
`{"id": N, "active": true|false}`. That is a **membership flag and nothing
else**. The question and answer strings are only in `all_questions.json` and
`all_answers.json`, which span every split. Reading the split file alone yields
33,212 rows of nothing that still look like data.

Correct join: take ids where `active` is true in the split file, then look the
text up by `id` in `all_questions.json` and by `question_id` in
`all_answers.json`.

### Imagery

256x256, **3-band uint8 RGB**, EPSG:3857, Sentinel-2 at 10 m (so 2,560 m a
side). PIL opens them directly, so no conversion step is needed. Staged as
shipped — eval preprocessing must match how the published numbers were made.

### Question types are load-bearing

Rows are typed `rural_urban`, `presence`, `count`, `comp`, and the reported
metric is **average accuracy — the mean of per-type accuracies**, not overall
accuracy. The types are severely unbalanced; on the staged test split:

```
comp          4002
presence      2955
count         2947
rural_urban    100
```

A row that loses its `question_type` moves the headline number silently, and
`rural_urban` at 100 rows carries the same weight in AA as `comp` at 4,002.

### One image means one view

RSVQA-LR ships a single RGB tif per sample, not a three-composite stack.
`RealChipDataset` **repeats a view** when a caller pins a composite count
(`dataset.py:213`), so running the C22 two-vs-three ablation here would compare
a picture against a copy of itself and report a real-looking delta. That
ablation belongs on the BEN chips, which have three genuinely different
composites.

### Staged 2026-08-29

`scripts/stage_rsvqa_lr.py --split test` -> **10,004 rows over 100 images**,
verified end to end through `load_canonical_manifest` and `RealChipDataset`,
with GSD-conditioned prompting applied
(`[ground sample distance: 10 m] Is it a rural or an urban area`).
