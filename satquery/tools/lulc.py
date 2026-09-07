"""``lulc_classifier`` — the reBEN reference classifiers, behind the manifest.

:mod:`satquery.tools.catalog` lists ``lulc_classifier`` among the learned tools
"absent by design ... they register themselves through
:func:`register_implementation` when their weights land". These are those
weights: TU Berlin's BIFOLD models, published alongside the reBEN paper, MIT
licensed, 19 CORINE classes.

**Scoped to radar, and the scoping is measured.** Published average precision
is optical 0.714, radar 0.628, and the twelve-channel *fused* model 0.711 --
below optical alone, so early fusion buys nothing and D1 keeps fusing at the
decision level as section 4.7 argued.

On **our** rows the ordering inverts, and that decided the scope. With both
models answering the same 4,000 held-out questions, radar scored **0.7525** and
optical **0.4968** -- chance -- and radar won **87%** of their disagreements.
Optical land cover is already served twice over, by ``spectral_index`` and by
the VLM, both measured and both working. So this tool covers the one modality
neither of those can read.

**The band order is asserted, never inferred.** Getting it wrong does not raise:
it returns confident probabilities at exactly the chance rate. Measured, with
the S1 model on 2,000 rows: ``[VH, VV]`` scores 81.7% and ``[VV, VH]`` scores
49.9%. A sweep found that, but a sweep is not a safety mechanism -- it happily
reports the best of several wrong answers, which is how the twelve-channel path
went unnoticed at chance for three runs. So the caller states which band sits in
which position and this refuses input that does not match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from satquery.tools.base import Tool, ToolContext, ToolResult

__all__ = [
    "LulcClassifierTool",
    "CLASSES",
    "FOREST",
    "OPTICAL_MODEL",
    "RADAR_MODEL",
    "answer_for",
    "names_a_known_class",
]

#: Alphabetical, matching ``configilm``'s ``valid_labels_classification``. The
#: checkpoints' own ``class_names`` are "0".."18" and carry no meaning, so this
#: ordering is the only thing that maps a logit to a land-cover class.
CLASSES = (
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
)

#: Kept as a name only. Nothing in the serving path loads it: the optical arm
#: was measured at 0.4968 on 4,000 rows -- chance -- against radar's 0.7525, and
#: radar wins 87% of their disagreements. It stays declared because
#: `eval_bifold.py duet` still runs both arms to reproduce that comparison, and
#: a decision is easier to keep when the rejected option is named beside the
#: number that rejected it.
OPTICAL_MODEL = "BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0"
RADAR_MODEL = "BIFOLD-BigEarthNetv2-0/resnet50-s1-v0.2.0"

#: reBEN's own per-band statistics, from ``bigearthnet_common.constants``. S1 is
#: calibrated sigma-nought in decibels; S2 is Level-2A reflectance scaled by
#: 10,000. Feeding 8-bit stretched pixels to either is the failure
#: :func:`satquery.tools.deterministic.assert_decibels` guards against on the
#: deterministic path, and it is silent here too.
_S1_STATS = {
    "VV": (-12.619993741972035, 5.115911777546365),
    "VH": (-19.29044597721542, 5.464428464912864),
}
_S2_STATS = {
    "B02": (429.9430203, 572.41639287), "B03": (614.21682446, 582.87945694),
    "B04": (590.23569706, 675.88746967), "B05": (950.68368468, 729.89827633),
    "B06": (1792.46290469, 1096.01480586), "B07": (2075.46795189, 1273.45393088),
    "B08": (2218.94553375, 1365.45589904), "B8A": (2266.46036911, 1356.13789355),
    "B11": (1594.42694882, 1079.19066363), "B12": (1009.32729131, 818.86747235),
}

#: What the checkpoints were trained on, from
#: ``configilm.extra.BEN_lmdb_utils.BAND_COMBINATION_PREDEFINTIONS``. Note the
#: S2 order is **not** sorted -- B08 sits fourth -- and S1 puts **VH first**,
#: which is neither the filename order nor the Sentinel-1 convention.
_S1_ORDER = ("VH", "VV")
_S2_ORDER = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A")

_INPUT_PX = 120


@dataclass
class _Loaded:
    model: Any
    device: Any


#: The grouped question the corpus asks. "Is there forest" is answered by the
#: strongest of the three forest classes rather than by any single one.
FOREST = ("Broad-leaved forest", "Coniferous forest", "Mixed forest")

#: Threshold. 0.5 is the operating point 74.95% was measured at, and the sweep
#: in `eval_bifold.py` reports it as "the honest out-of-the-box number" -- a
#: better threshold exists but was chosen on the same split it is scored on.
ANSWER_THRESHOLD = 0.5


def names_a_known_class(question: str) -> bool:
    """Whether the question asks about a class this model actually predicts.

    The planner's half of :func:`answer_for`: same matching, without needing the
    probabilities. It is what lets a radar question skip the VLM entirely rather
    than calling it and discarding its answer.
    """
    text = question.lower()
    if "is there forest" in text:
        return True
    return any(name.lower() in text for name in CLASSES)


def answer_for(question: str, probs, threshold: float = ANSWER_THRESHOLD) -> str | None:
    """Turn 19 class scores into the yes/no the corpus asked for.

    This is the function `eval_bifold.py` scored 74.95% with, moved here so
    both it and the served path call one implementation. Two copies of a
    scoring rule is how a benchmark number and a product answer quietly stop
    meaning the same thing -- the same argument the answer formatter already
    makes, applied to the one component that had its rule living in a script.

    ``None`` when the question names no class this model knows, which is the
    signal to let something else answer rather than to guess.
    """
    by_lower = {name.lower(): index for index, name in enumerate(CLASSES)}
    text = question.lower()
    if "is there forest" in text:
        score = max(probs[CLASSES.index(name)] for name in FOREST)
    else:
        target = None
        for name, index in by_lower.items():
            if name in text:
                target = index
                break
        if target is None:
            return None
        score = probs[target]
    return "yes" if score >= threshold else "no"


class LulcClassifierTool:
    """Multi-label land cover, scoped to SAR -- the modality the VLM cannot read."""

    name = "lulc_classifier"

    def __init__(self) -> None:
        self._cache: dict[str, _Loaded] = {}

    # -- weights -----------------------------------------------------------
    def _load(self, repo: str, channels: int) -> _Loaded:
        """Load once per process. Imported lazily so ``catalog`` stays cheap.

        The catalogue is imported by the parameter gate and the headless CPU
        path, neither of which should pull in torch to validate a plan.
        """
        if repo in self._cache:
            return self._cache[repo]

        import timm
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        state = load_file(hf_hub_download(repo, "model.safetensors"))
        # Every tensor is a plain timm resnet50 under a wrapper prefix, so timm
        # loads them directly -- no configilm, no clone from TU Berlin's GitLab
        # at container start.
        state = {k.removeprefix("model.vision_encoder."): v for k, v in state.items()}
        # Built under an explicit float32 default, because this tool shares a
        # process with the VLM. Loading Qwen3-VL with dtype=bfloat16 leaves
        # torch's *global* default dtype at bfloat16, so `timm.create_model`
        # here produced a bf16 ResNet-50, `load_state_dict` cast BIFOLD's
        # float32 weights down into it, and the forward pass then died with
        #     Expected weight to have type Float but got BFloat16
        # -- but only inside the deployed API. Standalone, where nothing had
        # loaded a VLM first, the same code scored 74.95%. Setting the dtype
        # here rather than fixing the VLM keeps the fix next to the model that
        # needs float32, and holds however the runner is configured later.
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            model = timm.create_model(
                "resnet50", in_chans=channels, num_classes=len(CLASSES)
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # A silently randomised layer scores like a weak model rather than a
            # broken one, which is indistinguishable from a bad dataset.
            raise RuntimeError(
                f"{repo}: state dict mismatch, missing={sorted(missing)[:4]} "
                f"unexpected={sorted(unexpected)[:4]}"
            )
        # Belt and braces: whatever the state dict carried, the module runs in
        # float32 so it cannot inherit a caller's precision.
        model = model.float()
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = _Loaded(model=model.to(device), device=device)
        self._cache[repo] = loaded
        return loaded

    # -- input -------------------------------------------------------------
    @staticmethod
    def _stack(scene, order: tuple[str, ...], stats: dict[str, tuple[float, float]]):
        """Normalised channel stack in the order the checkpoint was trained on.

        Bands are addressed by name. A scene that cannot supply one of them is
        refused rather than padded: a zero channel where a band should be is
        indistinguishable, downstream, from a band that happens to read zero.
        """
        planes = []
        for band in order:
            try:
                plane = np.asarray(scene.band(band), dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - scene backends vary
                raise ValueError(
                    f"lulc_classifier needs band {band!r}; this scene does not "
                    f"provide it ({exc}). Expected {list(order)}."
                ) from exc
            mean, std = stats[band]
            planes.append((plane - mean) / std)
        return np.stack(planes)

    @staticmethod
    def _resize(array: np.ndarray) -> np.ndarray:
        """To the 120 px the checkpoints were trained at, bicubic as configilm does."""
        import torch
        import torch.nn.functional as functional

        tensor = torch.from_numpy(array).unsqueeze(0)
        if tensor.shape[-2:] != (_INPUT_PX, _INPUT_PX):
            tensor = functional.interpolate(
                tensor, size=(_INPUT_PX, _INPUT_PX), mode="bicubic", align_corners=False
            )
        return tensor

    # -- run ---------------------------------------------------------------
    def run(self, context: ToolContext) -> ToolResult:
        import torch  # noqa: F401  (kept adjacent to the tensor work below)

        top_k = int(context.params.get("top_k", 5))
        warnings: list[str] = []

        # Radar first, deliberately. On 4,000 held-out rows with both models
        # answering the same questions, radar scored 0.7525 and optical 0.4968
        # -- chance -- and radar won 87% of their disagreements. Optical land
        # cover is served by ``spectral_index`` and the VLM instead, so this
        # tool exists for the modality neither of those can read.
        # Radar only. There is no optical fallback, by decision.
        #
        # This used to fall through to `resnet50-s2` when no SAR view was
        # present, behind a warning. That path was unreachable -- the router
        # requires `has_sar` before planning this tool, and the parameter gate
        # refuses it against `required_modalities: [sar]` -- but unreachable is
        # not the same as safe: relaxing either guard would have silently served
        # a classifier measured at **0.4968 on 4,000 rows**, which is chance,
        # behind a warning most readers never open.
        #
        # Optical land cover is served by `spectral_index` and the VLM, both of
        # which are measured. Refusing here is the honest answer for a scene
        # this tool cannot speak about.
        radar = context.scene_of("sar")
        if radar is None:
            raise ValueError(
                "lulc_classifier answers from radar only; this bundle provides "
                f"{[s.modality for s in context.scenes]}. The optical classifier "
                "scores 0.4968 -- chance -- on the same rows, so there is no "
                "optical path here; optical land cover belongs to spectral_index."
            )
        scene, repo, order, stats = radar, RADAR_MODEL, _S1_ORDER, _S1_STATS

        batch = self._resize(self._stack(scene, order, stats))
        loaded = self._load(repo, len(order))
        with torch.no_grad():
            pixels = batch.to(loaded.device, dtype=torch.float32)
            probs = torch.sigmoid(loaded.model(pixels))[0].float().cpu().numpy()

        scores = probs.tolist()
        ranked = sorted(zip(CLASSES, scores, strict=False), key=lambda kv: -kv[1])[:top_k]
        # The classifier answers the question directly when the question names a
        # class it knows. On radar that is the right voice to answer in: this
        # model scores 0.7525 where the VLM scores 0.4968, and leaving the reply
        # to the VLM meant the component holding the measured number was
        # computing labels nobody read.
        verdict = answer_for(context.query_text, scores)
        labels = [{"class": name, "score": round(float(score), 4)} for name, score in ranked]

        return ToolResult(
            outputs=(
                {"labels": labels}
                if verdict is None
                else {"labels": labels, "answer": verdict}
            ),
            # The strongest class's own probability. Multi-label sigmoid, so
            # this is that class's confidence, not a distribution over classes.
            confidence=round(float(ranked[0][1]), 4) if ranked else None,
            confidence_basis="learned_logprob",
            warnings=warnings,
            param_provenance={
                "model": repo,
                "band_order": list(order),
                "input_px": _INPUT_PX,
                "modality_used": scene.modality,
            },
        )


def _self_check() -> None:
    """Fail loudly if the class list and the checkpoints ever disagree."""
    if len(CLASSES) != 19:
        raise AssertionError(f"reBEN defines 19 classes; found {len(CLASSES)}")
    if len(_S2_ORDER) != 10 or len(_S1_ORDER) != 2:
        raise AssertionError("band orders must be 10 optical and 2 radar channels")
    if set(_S2_ORDER) - set(_S2_STATS) or set(_S1_ORDER) - set(_S1_STATS):
        raise AssertionError("a band in the order has no published statistics")


_self_check()

_TOOL: Tool = LulcClassifierTool()
