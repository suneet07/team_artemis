import re

from satquery.ingest.band_inventory import BandInventory

BAND_NAMES = ("blue", "green", "red", "nir", "swir", "vv", "vh", "hh", "hv")
ORDER_PHRASES = {
    ("blue", "green", "red"): "RGB",
    ("blue", "green", "red", "nir"): "B,G,R,NIR",
}


def _match_descriptions(descriptions: tuple[str | None, ...]) -> dict[str, int]:
    matched: dict[str, int] = {}
    for index, desc in enumerate(descriptions):
        if not desc:
            continue
        tokens = set(re.findall(r"[a-z]+", desc.lower()))
        for name in BAND_NAMES:
            if name in tokens and name not in matched:
                matched[name] = index + 1
    return matched


def _order_phrase(order: tuple[str, ...]) -> str:
    return ORDER_PHRASES.get(order, ",".join(order))


def _derive_indices(bands: dict[str, int], has_swir: bool) -> list[str]:
    present = set(bands)
    if has_swir:
        present.add("swir")
    indices: list[str] = []
    for name, needed in (
        ("NDVI", ("nir", "red")),
        ("NDWI", ("green", "nir")),
        ("MNDWI", ("green", "swir")),
        ("NDBI", ("swir", "nir")),
    ):
        if all(band in present for band in needed):
            indices.append(name)
    return indices


def _sar_band_from_tags(tags: dict[str, str]) -> str | None:
    raw = tags.get("SAR_BAND") or tags.get("sar_band") or tags.get("BAND")
    if raw and raw.upper() in ("C", "X", "L", "S", "P"):
        return raw.upper()
    return None


def _complete_from_assumed_order(
    matched: dict[str, int], order: tuple[str, ...]
) -> tuple[dict[str, int] | None, list[str]]:
    notes: list[str] = []
    for name, index in matched.items():
        if index > len(order) or order[index - 1] != name:
            return None, [
                f"band descriptions contradict the assumed {', '.join(order)} order; "
                "assumption disabled so indices cannot silently corrupt"
            ]
    bands = dict(matched)
    for slot, name in enumerate(order, start=1):
        if name not in bands:
            bands[name] = slot
    notes.append(
        "sparse band descriptions matched a band description; completed from the "
        f"assumed {_order_phrase(order)} order"
    )
    return bands, notes


def build_band_inventory(
    meta,
    modality: str,
    bands_config=None,
) -> tuple[BandInventory, list[str]]:
    notes: list[str] = []
    bands = _match_descriptions(meta.descriptions)

    assumed = None
    if bands_config is not None:
        assumed = bands_config.assumed_orders.get(meta.count)

    if len(bands) < 2 and modality == "optical" and assumed:
        if bands:
            completed, complete_notes = _complete_from_assumed_order(bands, assumed)
            if completed is None:
                bands = {}
                notes.extend(complete_notes)
            else:
                bands = completed
                notes.extend(complete_notes)
        else:
            bands = {name: i + 1 for i, name in enumerate(assumed)}
            notes.append(
                f"no usable band descriptions; assumed {_order_phrase(assumed)} order"
            )

    polarisations = [name for name in ("vv", "vh", "hh", "hv") if name in bands]
    has_swir = "swir" in bands
    inventory = BandInventory(
        bands=bands,
        has_swir=has_swir,
        has_nir="nir" in bands,
        is_pan_only=modality == "optical" and meta.count == 1,
        polarisations=polarisations,
        sar_band=_sar_band_from_tags(meta.tags),
        sensor_hint=meta.sensor_hint,
    )
    inventory.computable_indices = _derive_indices(bands, has_swir)
    return inventory, notes
