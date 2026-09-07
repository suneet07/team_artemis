"""The views served to an adapter must be the views it was trained on.

This is the failure the project's own notes call load-bearing: a train/serve
mismatch does not raise, it just scores the model on an input shape it never
saw. It cost two confidently wrong answers on ``_29_57`` -- a chip that is 100%
coniferous forest answering "yes" to both water and residential -- because the
served view was blue/green/red in the red/green/blue slots, repeated three
times, so NIR and SWIR never reached the model at all.

Pixel equality against the corpus PNGs, not a shape or a channel-count check:
the stretch is per channel and the 20 m bands are upsampled, and either of
those drifting would leave the shapes intact while changing what the model sees.
"""

from pathlib import Path

import numpy as np
import pytest

CHIP = "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_29_57"
GEOTIFF = Path("data/chips/geotiff") / f"{CHIP}.tif"
COMPOSITES = Path("data/chips/composites")
ROLES = ("true_colour", "false_colour", "short_wave")

pytestmark = pytest.mark.skipif(
    not GEOTIFF.exists() or not all((COMPOSITES / f"{CHIP}_{r}.png").exists() for r in ROLES),
    reason="sample chip and its corpus composites are not staged here",
)


def test_served_views_match_the_corpus_composites():
    from PIL import Image

    from satquery.ingest.scene import load_scene
    from satquery.tools.learned import _views

    views = _views([load_scene(GEOTIFF).scene])
    assert len(views) == 3

    for view, role in zip(views, ROLES, strict=False):
        reference = np.asarray(Image.open(COMPOSITES / f"{CHIP}_{role}.png").convert("RGB"))
        served = np.asarray(view.convert("RGB"))
        assert served.shape == reference.shape, role
        assert np.array_equal(served, reference), (
            f"{role} served view differs from the corpus composite by up to "
            f"{np.abs(served.astype(int) - reference.astype(int)).max()} levels"
        )


def test_the_three_views_are_distinct():
    """A duplicate view is the specific regression: one image filling all three
    slots is what hid NIR and SWIR from the adapter."""
    from satquery.ingest.scene import load_scene
    from satquery.tools.learned import _views

    served = [np.asarray(view.convert("RGB")) for view in _views([load_scene(GEOTIFF).scene])]
    for first in range(len(served)):
        for second in range(first + 1, len(served)):
            assert not np.array_equal(served[first], served[second]), (
                f"views {first} and {second} are identical"
            )


def test_view_count_matches_the_adapter_budget():
    """Each adapter gets the view count its corpus used, not a global 3.

    ``_VIEWS`` was one hardcoded 3 for every adapter. RSVQA rows (one image
    repeated to three) were right by coincidence; the two-image tasks were not.
    CDVQA carries two real views and was scored on two, and VRSBench captioning
    trains on two composites because Cartosat-2S has no SWIR -- both were being
    served a third, duplicated view they were never measured with.
    """
    from satquery.ingest.scene import load_scene
    from satquery.tools.learned import _views

    scene = load_scene(GEOTIFF).scene
    assert len(_views([scene], "rs_vqa")) == 3
    assert len(_views([scene], "rs_ground_caption")) == 2
    # Two real views pass through as two -- never padded up to the budget of 6,
    # and never truncated, which would hand the model a single date.
    assert len(_views([scene, scene], "change_vqa")) == 2


def test_a_single_view_is_repeated_to_the_budget():
    """The one safe expansion, and the one the corpus itself performs: an RSVQA
    row carries one image and the loader repeats it to three."""
    import numpy as np
    from PIL import Image

    from satquery.tools.base import Scene
    from satquery.tools.learned import _views

    plain = Scene(
        name="rgb.png",
        modality="optical",
        bands={"red": np.zeros((8, 8)), "green": np.zeros((8, 8)), "blue": np.zeros((8, 8))},
        inventory=load_inventory(),
    )
    views = _views([plain], "rs_vqa")
    assert len(views) == 3
    assert all(isinstance(v, Image.Image) for v in views)


def load_inventory():
    from satquery.ingest.band_inventory import BandInventory

    return BandInventory(bands={"red": 1, "green": 2, "blue": 3})
