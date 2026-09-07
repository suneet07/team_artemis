"""The three optical composites of C22, built from raw Sentinel-2 reflectance.

Section 4.2 feeds a multispectral source to the VLM as three RGB composites,
because a VLM takes three channels and Sentinel-2 has thirteen:

============= ====================== =====================================
composite     bands (R, G, B)        what it is there for
============= ====================== =====================================
true_colour   B04, B03, B02          what a human would recognise
false_colour  B08, B04, B03          vegetation vigour; NIR in the red slot
short_wave    B12, B11, B08          moisture, burn scars, bare soil
============= ====================== =====================================

``rs_ground_caption`` uses only the first two (C22): its sub-metre sources are
RGB and the deployment sensor, Cartosat-2S, has no SWIR at all. Feeding a
short-wave composite at training time that can never exist at inference teaches
the model to depend on a channel it will not get.

STRETCHING
----------
Sentinel-2 L2A is 16-bit reflectance scaled by 10000; a naive cast to uint8
clips almost everything into the bottom of the range. The stretch here is a
**percentile** stretch computed per composite, and the percentiles used are
recorded on the result. A stretch is a lossy, irreversible decision about
contrast, and one that is not written down cannot be reproduced or compared
across runs -- which is how two composite sets end up looking different for
reasons nobody can name.

The 20 m bands (B11, B12) are upsampled to the 10 m grid so all three composites
share one geometry. Upsampling does not create information; it makes the short-
wave composite spatially registerable with the other two, which is the only
thing it needs to do.
"""

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COMPOSITES",
    "Composite",
    "REQUIRED_BANDS",
    "build_composites",
    "composite_names_for",
    "stretch",
]

#: Band triples per composite, in (red, green, blue) slot order.
COMPOSITES: dict[str, tuple[str, str, str]] = {
    "true_colour": ("B04", "B03", "B02"),
    "false_colour": ("B08", "B04", "B03"),
    "short_wave": ("B12", "B11", "B08"),
}

#: Every band any composite needs. What the fetcher must be asked for.
REQUIRED_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B11", "B12")

#: Default percentiles. 2/98 rather than min/max: a single hot pixel or a cloud
#: edge would otherwise set the whole scale and flatten the image.
DEFAULT_PERCENTILES = (2.0, 98.0)


@dataclass
class Composite:
    """One three-channel image plus how it was made."""

    name: str
    bands: tuple[str, str, str]
    array: Any
    percentiles: tuple[float, float]
    #: Per-channel (low, high) reflectance values the stretch mapped to 0 and 255.
    cut_values: list[tuple[float, float]] = field(default_factory=list)

    def to_image(self):
        from PIL import Image

        return Image.fromarray(self.array)


def composite_names_for(adapter: str) -> tuple[str, ...]:
    """Composites an adapter actually trains on (C22).

    Two for ``rs_ground_caption``, three for everything else. Kept next to the
    composite definitions rather than derived from
    :func:`satquery.training.config.adapter_composites`, because a count is not
    a choice of *which*: dropping the third composite must drop the short-wave
    one specifically, not whichever happens to be last.
    """
    from satquery.training.config import adapter_composites

    count = adapter_composites(adapter)
    return tuple(list(COMPOSITES)[:count])


def stretch(
    channel, percentiles: tuple[float, float] = DEFAULT_PERCENTILES
) -> tuple[Any, tuple[float, float]]:
    """Percentile-stretch one band to uint8. Returns the array and its cut values.

    A degenerate band (all one value, which happens on water in SWIR and on
    nodata margins) would divide by zero; it maps to mid-grey instead, so the
    composite stays valid and the cut values recorded on it show what happened.
    """
    import numpy as np

    data = np.asarray(channel, dtype="float32")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.full(data.shape, 128, dtype="uint8"), (0.0, 0.0)
    low, high = (float(v) for v in np.percentile(finite, percentiles))
    if high <= low:
        return np.full(data.shape, 128, dtype="uint8"), (low, high)
    scaled = (np.clip(data, low, high) - low) / (high - low)
    return (scaled * 255.0).astype("uint8"), (low, high)


def _resample_to(array, shape):
    """Nearest-neighbour upsample a 20 m band onto the 10 m grid.

    Nearest and not bilinear: the 20 m bands are being aligned, not enhanced,
    and an interpolated SWIR value is a number no sensor measured.
    """
    import numpy as np

    data = np.asarray(array)
    if data.shape == tuple(shape):
        return data
    rows = (np.arange(shape[0]) * data.shape[0] // shape[0]).clip(0, data.shape[0] - 1)
    cols = (np.arange(shape[1]) * data.shape[1] // shape[1]).clip(0, data.shape[1] - 1)
    return data[np.ix_(rows, cols)]


def build_composites(
    bands: dict[str, Any],
    *,
    names: tuple[str, ...] | None = None,
    percentiles: tuple[float, float] = DEFAULT_PERCENTILES,
) -> list[Composite]:
    """Build the requested composites from a band dict.

    Raises on a missing band rather than substituting a neighbour. A composite
    silently built from the wrong band produces a plausible image that means
    something else -- the exact failure mode this whole package is arranged
    around.
    """
    import numpy as np

    names = names or tuple(COMPOSITES)
    target = None
    for band in ("B04", "B03", "B02", "B08"):
        if band in bands:
            target = np.asarray(bands[band]).shape
            break
    if target is None:
        raise ValueError(
            f"no 10 m band among {sorted(bands)}; the composite grid is defined by "
            "the 10 m bands and cannot be inferred from SWIR alone"
        )

    built: list[Composite] = []
    for name in names:
        triple = COMPOSITES[name]
        missing = [band for band in triple if band not in bands]
        if missing:
            raise ValueError(
                f"composite {name!r} needs {list(triple)} and is missing {missing}. "
                f"Fetch with bands={REQUIRED_BANDS!r}."
            )
        channels, cuts = [], []
        for band in triple:
            stretched, cut = stretch(_resample_to(bands[band], target), percentiles)
            channels.append(stretched)
            cuts.append(cut)
        built.append(
            Composite(
                name=name,
                bands=triple,
                array=np.stack(channels, axis=-1),
                percentiles=percentiles,
                cut_values=cuts,
            )
        )
    return built
