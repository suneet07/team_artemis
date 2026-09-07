"""CLIP-backed tile relevance — the upgrade path for section 4.4 step 4.

The plan names "CLIP-style similarity between the query text and each tile".
:func:`satquery.tiling.content_scorer` is the deterministic default, because the
headless path (section 4.11) must run CPU-only inside one container with no
network and no model download. This module is the other half of that contract:
the same interface, backed by a real CLIP model, used when weights and a GPU are
available.

It plugs in through the existing ``TileScorer`` signature —
``(query: str, tile: np.ndarray) -> float`` — so nothing downstream changes:

    from satquery.tiling import plan_tiles, score_tiles
    from satquery.tiling.clip_scorer import ClipTileScorer

    scorer = ClipTileScorer()          # loads once
    ranked = score_tiles(query, tiles, scorer=scorer)

Two things it does that a naive wrapper does not. It scores a **batch** in one
forward pass, because one pass per tile over 256 tiles is the difference between
comfortably inside the 20 s query SLA and nowhere near it. And it takes tiles as
**arrays**, not as a path plus bounds, because by the time tiling has run the
pixels are already in memory and re-opening and re-cropping the scene per tile
reads it from disk once per tile.
"""

import importlib.util
from collections.abc import Sequence

import numpy as np

__all__ = ["ClipTileScorer", "clip_available"]

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def clip_available() -> bool:
    """Whether torch and transformers are installed.

    Uses ``find_spec`` rather than importing them: importing torch costs tens of
    seconds, and a capability *probe* that takes half a minute gets called once
    on a hot path and then blamed on something else.
    """
    return all(
        importlib.util.find_spec(name) is not None for name in ("torch", "transformers")
    )


def _to_rgb_uint8(tile: np.ndarray) -> np.ndarray:
    """Percentile-stretch an arbitrary tile to the 3-channel uint8 CLIP expects.

    CLIP was trained on 8-bit photographs. Handing it raw 12-bit reflectance or
    negative sigma-nought dB gives it a distribution it has never seen, and the
    similarity it returns is then not measuring what the caller thinks. The
    stretch is the same robust 2-98% used everywhere else in the pipeline.
    """
    array = np.asarray(np.ma.getdata(tile), dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    array = array[:3] if array.shape[0] >= 3 else np.repeat(array[:1], 3, axis=0)

    out = np.zeros(array.shape, dtype=np.uint8)
    for index, plane in enumerate(array):
        valid = np.isfinite(plane)
        if not valid.any():
            continue
        low, high = np.percentile(plane[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        scaled[~valid] = 0.0
        out[index] = (scaled * 255).astype(np.uint8)
    return np.transpose(out, (1, 2, 0))


class ClipTileScorer:
    """A ``TileScorer``: callable as ``scorer(query, tile) -> float`` in [0, 1].

    Also exposes :meth:`score_batch` for the case the caller has all the tiles,
    which is the one the tiling path is actually in.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str | None = None):
        if not clip_available():
            raise ImportError(
                "ClipTileScorer needs torch and transformers. The deterministic "
                "content_scorer is the CPU-only default and needs neither; install "
                'the extras with `pip install -e ".[train]"` to use this one.'
            )
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model_id = model_id
        # Named so `score_tiles` records what produced the ranking, rather than
        # attributing a CLIP ranking to "content_statistics" in the trace.
        self.__name__ = f"clip:{model_id}"

    def score_batch(self, query: str, tiles: Sequence[np.ndarray]) -> list[float]:
        """Cosine similarity of every tile against the query, in one pass."""
        import torch

        if not tiles:
            return []
        images = [_to_rgb_uint8(tile) for tile in tiles]
        inputs = self.processor(
            text=[query], images=images, return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # (n_images, n_texts) -> (n_images,). One text, so column 0.
            logits = outputs.logits_per_image[:, 0].float().cpu().numpy()

        # Map to [0, 1] by rank-free min-max over this scene's tiles. CLIP logits
        # are not calibrated and their absolute scale shifts with the prompt, so
        # an absolute threshold on them would mean nothing; the caller only ever
        # needs the ordering and a bounded score.
        if len(logits) == 1:
            return [1.0]
        low, high = float(logits.min()), float(logits.max())
        if high <= low:
            return [1.0] * len(logits)
        return [float((value - low) / (high - low)) for value in logits]

    def __call__(self, query: str, tile: np.ndarray) -> float:
        """Single-tile scoring, for interface compatibility.

        Prefer :meth:`score_batch`: called per tile this runs one forward pass
        each, and a single tile has nothing to normalise against, so it returns
        the squashed raw similarity instead.
        """
        import torch

        inputs = self.processor(
            text=[query], images=[_to_rgb_uint8(tile)], return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logit = float(self.model(**inputs).logits_per_image[0, 0])
        return float(1.0 / (1.0 + np.exp(-logit / 10.0)))
