"""Training samples: the canonical JSONL corpus, and shape-only synthetic data.

Two datasets live here and the difference between them is the point.

:class:`RealChipDataset` reads the canonical manifest -- actual imagery, actual
questions, actual answer strings in each benchmark's own vocabulary so the
formatter (section 6.4) is normalising rather than translating.

:class:`SyntheticShapeDataset` generates noise tiles for **timing only**. Step
cost depends on tensor shapes, not pixel values, so this is legitimate for
pricing a config -- and illegitimate for anything else. The first timing run
used constant black images and reported a falling loss curve as evidence that
training worked; it was evidence of nothing. The class is named for what it is
and its ``__getitem__`` answers come from a pool, so a LoRA cannot memorise one
string and drive the loss to zero while the labels are wired wrong.

GSD-CONDITIONED PROMPTING
-------------------------
Section 5.4 requires the real ground sample distance in the text context at
train *and* inference, so the model treats scale as a variable rather than a
hidden constant. SARLANG-1M alone spans 0.1 m to 25 m; without conditioning,
one distribution is being averaged over a 250x scale range.

The resolution policy adds the part that is easy to get wrong: inject the
**effective** GSD, not the native one. RSVQA-HR is 0.15 m natively and enters
training downsampled to 0.30 m. Conditioning on 0.15 m would teach the model a
scale its pixels do not have.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "CanonicalSample",
    "RealChipDataset",
    "SyntheticShapeDataset",
    "gsd_prompt_prefix",
    "load_canonical_manifest",
    "resolve_sample_images",
]


#: The most views one sample may carry. Not a modelling limit -- a budgeting
#: one: image tokens scale with view count, and the measured batch sizes for
#: every adapter assume a sample stays under this. Six is the largest any
#: current source produces (three composites at each of two dates).
MAX_VIEWS = 8


@dataclass
class CanonicalSample:
    """One row of the canonical training JSONL."""

    sample_id: str
    adapter: str
    task: str
    images: list[str]
    question: str
    answer: str
    answer_type: str = "text"
    image_roles: list[str] = field(default_factory=list)
    modality: list[str] = field(default_factory=list)
    effective_gsd_m: list[float] = field(default_factory=list)
    split: str = "train"
    source: str = ""
    licence: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CanonicalSample":
        known = {
            "sample_id",
            "adapter",
            "task",
            "images",
            "question",
            "answer",
            "answer_type",
            "image_roles",
            "modality",
            "effective_gsd_m",
            "split",
            "source",
            "licence",
        }
        # `adapter` and `task` are required alongside the obvious four. They
        # carry no default in the dataclass, so a row without them used to die
        # with a TypeError from the constructor instead of this message.
        missing = {
            "sample_id",
            "adapter",
            "task",
            "images",
            "question",
            "answer",
        } - set(row)
        if missing:
            raise ValueError(
                f"canonical row is missing required field(s) {sorted(missing)}: "
                f"{row.get('sample_id', row)!r}"
            )
        return cls(**{k: v for k, v in row.items() if k in known}, raw=dict(row))


def load_canonical_manifest(
    path: Path | str,
    *,
    adapter: str | None = None,
    split: str | None = None,
) -> list[CanonicalSample]:
    """Read a canonical JSONL manifest, optionally filtered.

    A malformed line is an error, not a skip. Silently dropping rows shrinks a
    training corpus by an amount nobody measures, and the run still completes.
    """
    path = Path(path)
    samples: list[CanonicalSample] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
            sample = CanonicalSample.from_row(row)
            if adapter is not None and sample.adapter != adapter:
                continue
            if split is not None and sample.split != split:
                continue
            samples.append(sample)
    if not samples:
        raise ValueError(
            f"{path} yielded no samples for adapter={adapter!r} split={split!r}. "
            "An empty corpus trains cleanly and teaches nothing, so this is an "
            "error rather than an empty loader."
        )
    return samples


def resolve_sample_images(sample: CanonicalSample, root: Path | str) -> list[Path]:
    """Absolute paths for a sample's images, checked for existence.

    Manifest paths are stored as written by whichever machine generated them,
    which on Windows means backslashes. Normalising here keeps the corpus
    portable instead of making every reader handle it.
    """
    root = Path(root)
    paths = []
    for entry in sample.images:
        relative = Path(str(entry).replace("\\", "/"))
        candidate = relative if relative.is_absolute() else root / relative
        if not candidate.exists():
            raise FileNotFoundError(
                f"{sample.sample_id}: image {entry!r} resolves to {candidate}, which "
                "does not exist. The manifest and the staged imagery have drifted."
            )
        paths.append(candidate)
    return paths


#: Questions that turn on RSVQA's size categories. Word-bounded so "largest
#: connected region" -- a superlative, not a category -- does not match.
_SIZE_WORDS = re.compile(r"\b(small|medium|large)\b", re.I)

#: RSVQA's size thresholds, from the paper's Table I. They are hard cut-offs on
#: OSM surface area -- an arbitrary convention the model cannot see -- and the
#: two scales define the same words **30x apart**. Left unstated, "is a small
#: building present?" carries two incompatible meanings across one corpus and
#: the model has to reverse-engineer both from examples. Stated, each row says
#: which rule applies and the contradiction disappears.
#:
#: Keyed by source rather than by GSD, because BigEarthNet.txt is 10 m like
#: RSVQA-LR but uses no size categories at all: its questions read "How large is
#: the area covered by X? a) 20 to 60%, b) ..." and carry their own ranges
#: inline. Asserting RSVQA's thresholds over a BEN row would state a rule that
#: is simply false for it.
_SIZE_RULES = {
    "RSVQA-HR": "small <100 m2, medium <500 m2, large >=500 m2",
    "RSVQA-LR": "small <3000 m2, medium <10000 m2, large >=10000 m2",
}


def _size_rule_for(source: str) -> str:
    for key, rule in _SIZE_RULES.items():
        if source.startswith(key):
            return rule
    return ""


def scale_prefix(
    gsd_values, question: str, source: str = "", images=None
) -> str:
    """Scale conditioning from primitives: pixel size, extent, size thresholds.

    **This is the one implementation.** Training, evaluation and the demo
    endpoint all reach the model through here, because a prompt the model was
    never trained on is indistinguishable from a weak model -- the demo would
    quietly measure the wrong thing and the eval would disagree with the
    serving path for reasons no score could explain.

    Three facts, each because the model cannot otherwise have it.

    **Ground sample distance** -- section 5.4.

    **Scene extent**, computed from the *loaded* image dimensions rather than
    the manifest, so it cannot drift from what the model is actually shown. The
    corpus spans a 1,000x range in scene area -- an RSVQA-HR tile is 6,088 m2
    and an RSVQA-LR tile is 6,553,600 m2 -- and GSD alone does not distinguish
    them. Without the extent, "what area is covered by X" is unanswerable except
    by memorising each source's tile size.

    **Size thresholds**, when the question uses a size word and the row's source
    defines one. RSVQA's two scales define "small" 30x apart -- under 3,000 m2
    at 10 m, under 100 m2 sub-metre -- so each row is told which rule it is
    under. BEN.txt gets none: it uses no size categories, stating its ranges
    inline in the question instead.

    Returns an empty string when there is no GSD rather than inventing one -- a
    wrong scale in the prompt is worse than no scale, because the model learns
    to trust the field.
    """
    values = [v for v in (gsd_values or []) if v]
    if not values:
        return ""

    parts = []
    if len({round(v, 4) for v in values}) == 1:
        parts.append(f"ground sample distance: {values[0]:g} m")
    else:
        parts.append(
            "ground sample distance per view: "
            + ", ".join(f"{v:g} m" for v in values)
        )

    if images:
        try:
            width, height = images[0].size
        except (AttributeError, TypeError, ValueError):
            width = height = 0
        if width and height:
            across, down = width * values[0], height * values[0]
            parts.append(f"scene {across:g} x {down:g} m, {across * down:,.0f} m2")

    if _SIZE_WORDS.search(question):
        rule = _size_rule_for(source or "")
        if rule:
            parts.append(rule)

    return "[" + "; ".join(parts) + "] "


def gsd_prompt_prefix(sample: CanonicalSample, images=None) -> str:
    """The prefix for one canonical row. Thin wrapper over :func:`scale_prefix`."""
    return scale_prefix(
        sample.effective_gsd_m, sample.question, sample.source or "", images
    )


class RealChipDataset:
    """Canonical manifest samples as ``{images, question, answer}`` dicts.

    Deliberately not a ``torch.utils.data.Dataset`` subclass: torch is an extra,
    and this class is exercised by CPU tests that must not import it. It
    satisfies the map-style protocol (``__len__`` / ``__getitem__``), which is
    all a ``DataLoader`` requires.
    """

    def __init__(
        self,
        samples: Sequence[CanonicalSample],
        root: Path | str,
        *,
        composites: int | None = None,
        gsd_conditioning: bool = True,
        resize_views: bool = False,
    ):
        if not samples:
            raise ValueError("RealChipDataset needs at least one sample")
        self.samples = list(samples)
        self.root = Path(root)
        self.composites = composites
        self.resize_views = resize_views
        self.gsd_conditioning = gsd_conditioning

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image

        sample = self.samples[index]
        paths = resolve_sample_images(sample, self.root)
        images = [Image.open(path).convert("RGB") for path in paths]

        # A manifest row carries as many views as its source has, and by default
        # a mismatch with the adapter's expected count is an error rather than
        # something to quietly correct.
        #
        # Truncating is the dangerous direction. ``change_vqa`` rows carry six
        # views (three composites at each of two dates); silently keeping the
        # first three would hand the model a single date, train without error,
        # converge on the answer prior, and surface only as a mediocre score
        # after a 15.7 h run. Repetition from a *single* view is the one safe
        # case: the corpus already does it for single-image sources, so it is
        # in-distribution rather than a guess.
        if self.composites and len(images) != self.composites:
            if len(images) == 1:
                images = images * self.composites
            elif self.resize_views:
                # Explicit, and only from callers measuring sequence length: the
                # timing sweep and the two-vs-three composite ablation both vary
                # the count on purpose.
                if len(images) < self.composites:
                    images = [
                        images[i % len(images)] for i in range(self.composites)
                    ]
                else:
                    images = images[: self.composites]
            elif len(images) <= MAX_VIEWS:
                # A row carrying several *real* views is not a mismatch to
                # correct -- it is what its source produced. change_vqa draws
                # from two: SpaceNet 7 is RGB-only Planet, so one composite per
                # date and two views a sample; the self-generated Sentinel
                # tiles carry true-colour, false-colour and short-wave at each
                # of two dates, so six. Forcing both to one number would either
                # truncate Sentinel down to a single date or pad SpaceNet 7
                # with duplicates. The adapter's `composites` stays the
                # sequence-length budget it was always used as, and these rows
                # pass through with the views they actually have.
                pass
            else:
                raise ValueError(
                    f"{sample.sample_id!r} carries {len(images)} view(s), beyond "
                    f"the {MAX_VIEWS}-view ceiling. Past this a single sample's "
                    "image tokens dominate the sequence and the batch size that "
                    "was measured for this adapter no longer holds."
                )

        prefix = (
            gsd_prompt_prefix(sample, images) if self.gsd_conditioning else ""
        )
        return {
            "images": images,
            "question": f"{prefix}{sample.question}",
            "answer": sample.answer,
            "sample_id": sample.sample_id,
        }

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]


class SyntheticShapeDataset:
    """Noise tiles at a chosen source size. **Timing only.**

    Legitimate use: pricing a config, where step cost is a function of tensor
    shapes and pixel values are irrelevant. Illegitimate use: anything that
    reads the loss as evidence. The answers come from a pool of four so a LoRA
    cannot memorise one and collapse the loss while labels are masked wrong --
    the failure that made the first run's curve meaningless.
    """

    QUESTION = (
        "Would you say that any arable land lies next to pastures in the image? "
        "Answer with reference to the visible land cover."
    )

    #: Near-identical token lengths, so sequence length stays controlled.
    ANSWERS = (
        "Yes. Arable parcels occupy the north-eastern quadrant and share a "
        "boundary with pasture along the drainage line.",
        "No. Coniferous stands dominate the southern slope and meet open "
        "heathland well short of the cultivated margin.",
        "Yes. Irrigated terraces follow the valley floor and abut grazing "
        "land immediately west of the settlement edge.",
        "No. Broad-leaved canopy covers the eastern third and borders only "
        "water, with no cultivated parcel adjacent.",
    )

    def __init__(
        self,
        *,
        source_size: int,
        composites: int,
        length: int = 4096,
        seed: int = 0,
        pool_size: int = 4,
    ):
        import numpy as np
        from PIL import Image

        self.length = length
        self.composites = composites
        self.source_size = source_size
        rng = np.random.default_rng(seed)
        self._pool = [
            Image.fromarray(
                rng.integers(0, 256, (source_size, source_size, 3), dtype=np.uint8)
            )
            for _ in range(pool_size)
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        images = [
            self._pool[(index + k) % len(self._pool)] for k in range(self.composites)
        ]
        return {
            "images": images,
            "question": self.QUESTION,
            "answer": self.ANSWERS[index % len(self.ANSWERS)],
            "sample_id": f"synthetic_{index}",
        }
