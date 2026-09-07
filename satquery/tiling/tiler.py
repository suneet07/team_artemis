from collections.abc import Generator
from typing import Any

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None

from satquery.config import preprocessing_config


class GeoTiler:
    """
    P4: GeoPixel-style full scene tiling.
    Chops massive satellite scenes into smaller, overlapping patches
    that models can process within VRAM limits.
    """

    def __init__(self, tile_size: int | None = None, overlap: int | None = None):
        """Defaults come from `configs/preprocessing.yaml`, not from literals.

        The previous 1024/128 defaults were a second, silent tiling contract:
        1024x1024 is 1,048,576 px, four times the frozen `tiling.max_pixels`, so
        every tile this produced would have been downsampled inside the
        processor -- coarsening ground sample distance with no record of it,
        which is the exact failure `check_within_cap` exists to catch.
        """
        cfg = preprocessing_config().tiling
        self.tile_size = int(tile_size if tile_size is not None else cfg.tile_side_px)
        self.overlap = int(
            overlap
            if overlap is not None
            else round(self.tile_size * cfg.tile_overlap_fraction)
        )
        if self.overlap >= self.tile_size:
            raise ValueError("tile overlap must be smaller than the tile size")
        cfg.check_within_cap(self.tile_size, self.tile_size, where="tile")

    def get_tiles(self, dataset: Any) -> Generator[dict[str, Any], None, None]:
        """
        Yields overlapping tiles from a rasterio dataset.
        Returns a dict with the tile window, the image array, and the transform.
        """
        if rasterio is None:
            raise ImportError("rasterio is required for tiling.")

        width = dataset.width
        height = dataset.height
        stride = self.tile_size - self.overlap
        max_tiles = preprocessing_config().tiling.max_tiles
        emitted = 0

        for row_off in range(0, height, stride):
            for col_off in range(0, width, stride):
                if emitted >= max_tiles:
                    # Hard cap so scene size cannot blow the query SLA
                    # (section 4.4 step 5). Silently iterating 40,000 tiles off a
                    # Cartosat scene is how a 20 s p95 becomes an hour.
                    return
                emitted += 1
                # Clamp to dataset edges
                actual_height = min(self.tile_size, height - row_off)
                actual_width = min(self.tile_size, width - col_off)

                window = Window(
                    col_off=col_off, row_off=row_off, width=actual_width, height=actual_height
                )

                # Read the data for this window (all bands)
                data = dataset.read(window=window)

                # Get the window's affine transform
                window_transform = dataset.window_transform(window)

                yield {
                    "data": data,
                    "window": window,
                    "transform": window_transform,
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": actual_width,
                    "height": actual_height,
                }

    def assemble_tiles(self, tiles: list[dict[str, Any]], out_shape: tuple[int, int]) -> Any:
        """
        Takes a list of processed tile dicts (must contain 'mask' and 'window')
        and reconstructs the full-scene mask, handling overlaps.
        """
        import numpy as np

        # We use a simple max-pooling for overlapping boolean masks
        full_mask = np.zeros(out_shape, dtype=bool)

        for tile in tiles:
            mask = tile["mask"]
            w = tile["window"]
            # Logical OR across overlaps
            full_mask[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width] |= mask

        return full_mask
