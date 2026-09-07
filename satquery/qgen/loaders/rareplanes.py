import json
import os
from collections.abc import Iterator

import rasterio

from satquery.qgen.primitives import Loader, P2BoundingBox


class RarePlanesLoader(Loader):
    """
    Loads RarePlanes annotations and fetches 512x512 image chips.
    Project geographic coordinates to pixel coordinates using rasterio.
    """
    source = "rareplanes"
    licence = "CC-BY-SA-4.0"
    provenance_chain = ["rareplanes", "maxar_wv3"]
    
    def __init__(self, data_dir: str = r"c:\random projs webdev\Sat Query\sample_dataset_real"):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "images")
        self.ann_dir = os.path.join(data_dir, "annotations")
        
    def emit(self, limit: int = 20) -> Iterator[P2BoundingBox]:
        """
        Emits P2BoundingBox primitives parsed from local GeoJSONs and PNGs.
        """
        if not os.path.exists(self.ann_dir):
            return
            
        count = 0
        for ann_file in os.listdir(self.ann_dir):
            if not ann_file.endswith(".geojson"):
                continue
                
            if count >= limit:
                break
                
            sample_id = ann_file.replace(".geojson", "")
            img_file = f"{sample_id}.png"
            img_path = os.path.join(self.img_dir, img_file)
            ann_path = os.path.join(self.ann_dir, ann_file)
            
            if not os.path.exists(img_path):
                continue
                
            with open(ann_path) as f:
                geojson = json.load(f)
                
            bboxes = []
            
            # Open the image to read its Geotransform matrix
            try:
                with rasterio.open(img_path) as src:
                    for feature in geojson.get("features", []):
                        geom = feature.get("geometry", {})
                        if geom.get("type") == "Polygon":
                            coords = geom.get("coordinates", [[]])[0]
                            
                            pixel_rows = []
                            pixel_cols = []
                            for lon, lat in coords:
                                row, col = src.index(lon, lat)
                                pixel_rows.append(row)
                                pixel_cols.append(col)
                                
                            ymin, ymax = min(pixel_rows), max(pixel_rows)
                            xmin, xmax = min(pixel_cols), max(pixel_cols)
                            
                            props = feature.get("properties", {})
                            bboxes.append({
                                "class": "aircraft",
                                "bbox": [ymin, xmin, ymax, xmax],
                                "attributes": {
                                    "wingspan": props.get("wingspan"),
                                    "num_engines": props.get("num_engines")
                                }
                            })
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
                
            yield P2BoundingBox(
                sample_id=sample_id,
                image_paths=[os.path.join("images", img_file)],
                effective_gsd_m=[0.3],
                source=self.source,
                source_ann_id=sample_id,
                licence=self.licence,
                provenance_chain=self.provenance_chain,
                split="train",
                bboxes=bboxes
            )
            count += 1
