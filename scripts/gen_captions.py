"""Turn staged annotations into captions in the register VRSBench actually uses.

    python scripts/gen_captions.py build \
        --rareplanes /data/eval/rareplanes --sn2 /data/eval/sn2 \
        --sn6 /data/eval/sn6 --rsvqa-hr /data/eval/rsvqa_hr \
        --ben /data/manifests/ben_txt_train.jsonl \
        --out /data/manifests/rs_ground_caption_captions.jsonl

**Why generate rather than adopt a corpus.** No public caption set writes in
this register. SkyScript and RS5M are OSM-tag alt-text built for CLIP alignment;
ChatEarthNet is 10 m Sentinel-2 with no licence; RSICD and friends are Google
Earth-derived (C32) and one sentence long. VRSBench's 47-word object inventory
exists because GPT-4V generated it over DOTA imagery, and nobody imitates it. We
hold annotations that contain exactly the facts such a caption states -- class,
count, size, position -- so the caption is rendered from them.

**The register is measured, not invented.** Every constant below came from all
9,350 VRSBench references: median 47 words over 3 sentences; "small" in 53.8%
and "large" in 15.8%; counts spelled as words (one 25.8%, two 25.9%); positional
phrasing led by "in the" 37.8%, "on the" 27.1%, "at the" 24.7%, "towards the"
19.1%; and 86% opening with a sensor phrase naming GoogleEarth.

**The sensor phrase is a slot, never a constant.** VRSBench says GoogleEarth
because its imagery is GoogleEarth. The ISRO evaluation set is Cartosat-2S, and
a model taught to assert GoogleEarth would state a false provenance on every
Indian scene -- in a system whose whole claim is auditable, evidence-grounded
output. Each source fills ``{sensor}`` with what it actually is.

**Two length modes, as a request rather than a contract.** ~47 words matches
VRSBench; ~100 serves a detailed mode and hedges against an ISRO register we
cannot see. A third longer mode was considered and rejected: our sources are
aircraft and buildings, and 150 words about three aircraft can only be filled by
repetition or invention, which is precisely what this system must not learn. A
sparse scene gets a short caption whatever length was asked for.

**Register is conditioned on the instruction, not blended.** BEN.txt keeps its
own land-cover register under its own prompt instead of being averaged into the
object-inventory one -- with ~20k BEN rows against ~30k object rows, blending
would drift toward the majority and lose the register this file exists to teach.
The model learns that register follows the instruction, which is also what makes
it steerable at an unseen benchmark.

Licences, all verified in CREDITS: RarePlanes / SpaceNet 2 / SpaceNet 6
CC BY-SA 4.0; RSVQA-HR imagery USGS public domain with CC BY 4.0 annotations;
BEN.txt CDLA-Permissive 1.0.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gen_ground import (  # noqa: E402
    ROLE_PHRASES,
    _quadrant_key,
    _touches_edge,
)

#: Counts are spelled in 60%+ of references; digits are rare.
NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

#: Screen-relative, matching the reference vocabulary and its frequencies.
CELL_PHRASE = {
    ("top", "left"): ["top-left", "upper left", "top left corner"],
    ("top", "right"): ["top-right", "upper right", "top right corner"],
    ("bottom", "left"): ["bottom-left", "lower left", "bottom left corner"],
    ("bottom", "right"): ["bottom-right", "lower right", "bottom right corner"],
}
SIDE_PHRASE = {
    "top": ["top", "upper part", "top edge"],
    "bottom": ["bottom", "lower part", "bottom edge"],
    "left": ["left side", "left edge", "left"],
    "right": ["right side", "right edge", "right"],
}
#: "in the" 37.8%, "on the" 27.1%, "at the" 24.7%, "towards the" 19.1%.
LEAD_IN = ["in the", "on the", "at the", "towards the", "near the"]

#: VRSBench's most common four-word opening covers 17.3% of its captions, so a
#: generator with three openers is more repetitive than the thing it imitates.
OPENERS = [
    "The image, sourced from {sensor},",
    "The high-resolution image from {sensor}",
    "The image from {sensor}",
    "The aerial image from {sensor}",
    "This high-resolution image from {sensor}",
    "This aerial image from {sensor}",
    "The satellite image, sourced from {sensor},",
    "This image, captured by {sensor},",
    "The overhead image from {sensor}",
    "An aerial view from {sensor}",
]
VERBS = ["shows", "features", "captures"]

#: Per-source, so no caption asserts a provenance it does not have.
SENSOR = {
    "RarePlanes (real split)": "WorldView-3",
    "SpaceNet 2 (Vegas/Paris/Shanghai/Khartoum)": "WorldView-3",
    "SpaceNet 6 / MSAW (Rotterdam)": "Maxar",
    "RSVQA-HR": "USGS aerial imagery",
}

CAPTION_PROMPTS = {
    "concise": [
        "Describe the image in detail.",
        "Provide a detailed description of this satellite image.",
        "Write a caption for this aerial image.",
    ],
    "detailed": [
        "Describe this satellite image in detail, in about 100 words.",
        "Give an extended description of this aerial image.",
    ],
    "landcover": [
        "Describe the land cover shown in this satellite image.",
        "Explain the land cover classes visible in this image.",
    ],
}


def _words(n: int) -> str:
    return NUMBER_WORDS.get(n, "several" if n <= 12 else "numerous")


def _place(key, rng: random.Random) -> str:
    """A position phrase, sometimes as a corner and sometimes as a side."""
    vertical, horizontal = key
    if rng.random() < 0.45:
        return f"{rng.choice(LEAD_IN)} {rng.choice(CELL_PHRASE[key])}"
    axis = vertical if rng.random() < 0.5 else horizontal
    return f"{rng.choice(LEAD_IN)} {rng.choice(SIDE_PHRASE[axis])}"


#: Role names that already state a size. RarePlanes calls its classes "Small
#: Civil Transport/Utility" and "Large Civil Transport/Utility", so prefixing a
#: computed size produced "one small small civil aircraft" and "one large medium
#: civil aircraft" -- broken English, in a corpus meant to teach fluent prose.
_SIZE_IN_PHRASE = re.compile(r"\b(small|medium|large|tiny|huge)\b", re.I)


def _describe(count: int, size: str, noun: str, rng: random.Random) -> str:
    """A noun phrase, with the size word suppressed when the noun has one."""
    parts = []
    if count > 1:
        parts.append(_words(count))
    else:
        parts.append(rng.choice(["a", "one"]))
    if not _SIZE_IN_PHRASE.search(noun):
        parts.append(size)
    parts.append(noun)
    return " ".join(parts)


def _agree(items: list[str]) -> str:
    """Verb agreement for the surroundings clause.

    "Nearby are airport apron" is the kind of sentence a template writes and a
    reader notices immediately.
    """
    return "are" if len(items) > 1 else "is"


def _size_word(fraction: float) -> str:
    """"small" is in 53.8% of references and "large" in 15.8%.

    Keyed on the share of the frame the object covers, not on absolute area.
    RSVQA defines "small" thirty times apart between its 10 m and sub-metre
    splits (paper Table I), so any absolute threshold is wrong for some source;
    a fraction of the tile is scale-free and matches how a reader sees it.
    """
    return "large" if fraction >= 0.02 else "small"


def _sentence(parts: list[str]) -> str:
    text = " ".join(p.strip() for p in parts if p and p.strip())
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text[0].upper() + text[1:] + "." if text else ""


def render(objects, surroundings, source, rng, mode="concise") -> str:
    """One caption from an object inventory.

    Objects are ``(phrase, count, size, cell)``. The shape is the reference
    shape: an opener naming the sensor and the dominant content, then object
    placements, then surroundings.
    """
    if not objects:
        return ""
    sensor = SENSOR.get(source, "satellite imagery")
    opener = rng.choice(OPENERS).format(sensor=sensor)
    if not opener.endswith(","):
        opener += " " + rng.choice(VERBS)
    else:
        opener += " " + rng.choice(VERBS)

    head, *rest = objects
    phrase, count, size, cell, *_extra = head
    lead = f"{_words(count)} {size} {phrase}" if count > 1 else f"a {size} {phrase}"
    sentences = [_sentence([opener, lead, _place(cell, rng)])]

    limit = 2 if mode == "concise" else 4
    for phrase, count, size, cell, *_extra in rest[:limit]:
        subject = (
            f"{_words(count)} {size} {phrase}" if count > 1 else f"a {size} {phrase}"
        )
        verb = rng.choice(["are", "sit", "are visible"]) if count > 1 else rng.choice(
            ["is", "sits", "is visible"]
        )
        sentences.append(_sentence([subject, verb, _place(cell, rng)]))

    if surroundings:
        joined = ", ".join(surroundings[: 3 if mode == "concise" else 5])
        sentences.append(
            _sentence([rng.choice(
                ["The surrounding area includes", "Nearby are", "The scene also contains"
                 ]), joined])
        )
    return " ".join(s for s in sentences if s)


def _inventory(items, width, height, rng):
    """Group boxes into ``(phrase, count, size, cell, cells)``.

    ``cell`` is the group's dominant quadrant and ``cells`` the full count per
    quadrant. The fifth element used to be thrown away, which is why every mode
    rendered the same length: once the invented surroundings were removed, a
    group could only ever contribute one clause no matter how far its members
    were spread. That spread is real, annotated, and free.
    """
    grouped = defaultdict(list)
    for phrase, bbox in items:
        grouped[phrase].append(bbox)
    out = []
    for phrase, boxes in grouped.items():
        cells = Counter(_quadrant_key(b, width, height) for b in boxes)
        cell = cells.most_common(1)[0][0]
        share = sum(b[2] * b[3] for b in boxes) / max(1.0, width * height)
        out.append(
            (phrase, len(boxes), _size_word(share / max(1, len(boxes))), cell, cells)
        )
    out.sort(key=lambda t: -t[1])
    return out


# --------------------------------------------------------------------------
# Structural augmentation
#
# **A template teaches a template.** Five openers over one fixed skeleton is
# surface variation: the model can learn the sentence pattern instead of the
# image, and BLEU/ROUGE would reward it, because they score shared words. The
# blind floor is what detects that -- a memorising model raises its score and
# not its gap over describing the wrong image -- but detection after a training
# run is expensive. Breaking the skeleton first is cheaper.
#
# The plan (line 1131) specifies "template captions ... then LLM linguistic
# augmentation". An LLM pass over ~50k captions is roughly 28 GPU-hours, which
# does not earn its place, so this varies *structure* rather than wording:
# object order, how many facts share a sentence, whether the caption opens on
# the scene or the surroundings, and which of several constructions states a
# count. ``caption_diversity`` then measures whether it worked instead of
# assuming it.
# --------------------------------------------------------------------------

#: Constructions for one object group. Each states the same three facts --
#: count, size, position -- in a different grammatical shape.
OBJECT_FORMS = [
    "there {verb} {subject} {place}",
    "{place} {verb} {subject}",
    "{subject} {verb} {place}",
]

SCENE_LEADS = [
    "{opener} {verb_o} {summary}",
    "{opener} {verb_o} {summary}",
    "{opener} {verb_o} {setting} with {summary}",
]


def _augmented(objects, surroundings, source, rng, mode="concise"):
    """Render one caption with the structure itself randomised."""
    if not objects:
        return ""
    sensor = SENSOR.get(source, "satellite imagery")
    opener = rng.choice(OPENERS).format(sensor=sensor)
    verb_o = rng.choice(VERBS)

    groups = list(objects)
    # Order is a real degree of freedom: nothing about the scene says the
    # largest group must be described first, and always leading with it is a
    # pattern the model would learn instead of looking.
    if rng.random() < 0.5:
        rng.shuffle(groups)

    limit = 3 if mode == "concise" else 5
    groups = groups[:limit]

    summary = _describe(groups[0][1], groups[0][2], groups[0][0], rng)
    setting = rng.choice(
        ["a scene", "an area", "a stretch of ground", "a site"]
    )
    lead = rng.choice(SCENE_LEADS).format(
        opener=opener, verb_o=verb_o, summary=summary, setting=setting
    )
    sentences = [_sentence([lead, _place(groups[0][3], rng)])]

    # Sometimes two groups share a sentence, sometimes each gets its own. That
    # changes the sentence count between 2 and 4, matching the references'
    # spread rather than emitting exactly three every time.
    rest = groups[1:]
    index = 0
    while index < len(rest):
        take = 2 if (rng.random() < 0.35 and index + 1 < len(rest)) else 1
        clauses = []
        for noun, count, size, cell, *_extra in rest[index : index + take]:
            form = rng.choice(OBJECT_FORMS)
            if form.startswith("there "):
                verb = "are" if count > 1 else "is"
            elif form.startswith("{place}"):
                # Fronting the place inverts the clause, and "Towards the left
                # edge is visible a bomber" is not English. The predicative
                # forms are dropped only for this construction.
                verb = rng.choice(["are", "sit"] if count > 1 else ["is", "sits"])
            else:
                verb = (
                    rng.choice(["are", "sit", "are visible", "appear"])
                    if count > 1
                    else rng.choice(["is", "sits", "is visible", "appears"])
                )
            clauses.append(
                form.format(
                    subject=_describe(count, size, noun, rng),
                    verb=verb,
                    place=_place(cell, rng),
                )
            )
        sentences.append(_sentence([" and ".join(clauses)]))
        index += take

    # The honest length lever. A group whose members sit in more than one
    # quadrant genuinely occupies more of the frame than one clause says, and
    # naming where they fall is reading the annotation, not padding from a
    # word list. Concise mode skips it: ~50 words is the reference length and
    # the target register.
    if mode != "concise":
        for noun, count, _size, _cell, *extra in groups:
            cells = extra[0] if extra else None
            if not cells or count < 2 or len(cells) < 2:
                continue
            where = [
                rng.choice(CELL_PHRASE[key])
                for key, _n in cells.most_common(3)
            ]
            joined = (
                ", ".join(f"the {w}" for w in where[:-1])
                + f" and the {where[-1]}"
            )
            sentences.append(
                _sentence([
                    rng.choice([
                        f"The {noun} are spread across {joined} of the frame",
                        f"They fall across {joined}",
                        f"The group spans {joined}",
                    ])
                ])
            )
            break

    if surroundings:
        picked = list(surroundings)
        rng.shuffle(picked)
        picked = picked[: 3 if mode == "concise" else 5]
        joined = ", ".join(picked)
        singular = _agree(picked)
        lead = rng.choice(
            [
                f"The surrounding area includes {joined}",
                f"Nearby {singular} {joined}",
                f"The scene also contains {joined}",
                f"Around them {'lie' if len(picked) > 1 else 'lies'} {joined}",
                f"Bordering the area {singular} {joined}",
                f"{joined} {singular} visible around the site",
            ]
        )
        sentences.append(_sentence([lead]))
    return " ".join(x for x in sentences if x)


def caption_diversity(captions: list[str]) -> dict:
    """Is this a corpus or one sentence wearing hats?

    Reported at generation time so a templated corpus is caught before a GPU
    run, not diagnosed after one. The reference set is the yardstick: measured
    over VRSBench's own 9,350 captions, its most common four-word opening
    accounts for 17.3% and its distinct-4-gram ratio is high. A generated corpus
    whose top opening is 60% has not been augmented, whatever the code claims.
    """
    from collections import Counter

    if not captions:
        return {}
    openings = Counter(" ".join(c.split()[:4]) for c in captions)
    grams = Counter()
    for c in captions:
        tokens = c.lower().split()
        for i in range(len(tokens) - 3):
            grams[tuple(tokens[i : i + 4])] += 1
    total = sum(grams.values()) or 1
    lengths = [len(c.split()) for c in captions]
    sentences = [c.count(".") for c in captions]
    return {
        "captions": len(captions),
        "top_opening_share": round(openings.most_common(1)[0][1] / len(captions), 4),
        "distinct_openings": len(openings),
        "distinct_4gram_ratio": round(len(grams) / total, 4),
        "median_words": sorted(lengths)[len(lengths) // 2],
        "median_sentences": sorted(sentences)[len(sentences) // 2],
    }


def rareplanes_items(tile):
    """The annotated aircraft in one tile as ``(phrase, bbox)``, with frame size.

    Separated from the caption so the attribute rows can read the same set the
    caption is built from. Two code paths deriving the same list would drift,
    and the attribute supervision is only worth anything if it agrees with the
    caption it is meant to support.
    """
    width = tile.get("width") or 512
    height = tile.get("height") or 512
    boxes = [
        b for b in tile["boxes"]
        if not b.get("truncated") and not _touches_edge(b["bbox"], width, height)
    ]
    items = [
        (ROLE_PHRASES[b["role"]], b["bbox"])
        for b in boxes
        if b.get("role") in ROLE_PHRASES
    ]
    return items, width, height


#: Asked of the attribute rows. Phrased as a question rather than an
#: instruction because the benchmark asks questions, and an auxiliary task in a
#: different voice is one more thing for the adapter to disentangle.
ATTRIBUTE_PROMPTS = [
    "Which object types are visible in this image?",
    "List the kinds of objects present in this satellite image.",
    "Name the object categories visible in this aerial image.",
]


def attribute_answer(items) -> str:
    """The multilabel attribute set, as a comma-separated list.

    Types only, no counts and no positions: this is the multilabel signal, and
    folding counts in would make it a second captioning task rather than the
    prior the captioning task is supposed to lean on.
    """
    kinds = sorted({phrase for phrase, _bbox in items})
    return ", ".join(kinds)


def rareplanes_caption(tile, rng, mode):
    items, width, height = rareplanes_items(tile)
    if not items:
        return ""
    # No surroundings. RarePlanes annotates aircraft and nothing else, so a
    # runway, an apron or a terminal building in the caption would be asserted
    # from the source name rather than read from the frame -- and most of these
    # tiles are not airfield centres, so a good share of those assertions would
    # simply be false. The earlier version shuffled seven airport nouns into
    # every caption to buy length against the brevity penalty; that trades a
    # metric gain for training the model to hallucinate context, which is the
    # one failure mode a captioning adapter must not learn.
    #
    # The slot stays empty rather than removed: the OpenStreetMap join fills it
    # with tags read at the tile's own coordinates.
    return _augmented(
        _inventory(items, width, height, rng),
        [],
        "RarePlanes (real split)", rng, mode,
    )


def _polygon_caption(image_path, label_path, source, gsd, rng, mode):
    """Buildings from a SpaceNet GeoJSON, placed using the tile's transform.

    The label file alone cannot say where in the frame a building sits, so the
    GeoTIFF header (not its pixels) supplies the geotransform. Cheap, and
    unavoidable.
    """
    import rasterio
    from rasterio.transform import rowcol

    try:
        document = json.loads(Path(label_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    features = [f for f in document.get("features", []) if f.get("geometry")]
    if not features:
        return ""
    with rasterio.open(image_path) as handle:
        transform, width, height = handle.transform, handle.width, handle.height

    items = []
    for feature in features:
        coords = feature["geometry"].get("coordinates") or []
        while coords and isinstance(coords[0], list) and coords[0] and isinstance(
            coords[0][0], list
        ):
            coords = coords[0]
        points = [c for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        if len(points) < 3:
            continue
        try:
            rows_, cols_ = zip(*(rowcol(transform, x, y) for x, y, *_ in points), strict=False)
        except Exception:  # noqa: BLE001 - one unmappable polygon is not fatal
            continue
        x0, x1 = max(0, min(cols_)), min(width, max(cols_))
        y0, y1 = max(0, min(rows_)), min(height, max(rows_))
        if x1 <= x0 or y1 <= y0:
            continue
        items.append(("building", [x0, y0, x1 - x0, y1 - y0]))
    if not items:
        return ""
    # Same rule as RarePlanes: a building footprint file records buildings. It
    # does not record roads, vegetation, open ground, parking or water, so
    # sampling those five at random described whatever the generator felt like
    # rather than whatever the tile held.
    return _augmented(
        _inventory(items, width, height, rng),
        [],
        source, rng, mode,
    )


#: RSVQA-HR asks about these, and they are the vocabulary our aircraft-and-
#: buildings sources cannot teach -- roads 21%, water 17%, fields 19% of what
#: VRSBench captions mention.
_RSVQA_NOUNS = [
    "road", "roads", "water area", "water areas", "building", "buildings",
    "commercial building", "residential building", "nature reserve",
    "grass area", "forest", "parking",
]


def rsvqa_hr_facts(rows):
    """Fold RSVQA-HR's question set into one inventory per image.

    Only ``presence`` and ``count`` are read. ``area`` is excluded on evidence
    recorded in CREDITS: 62% of its answers are 0m2 where OpenStreetMap mapped
    nothing, and 275 rows claim more building area than the tile physically
    contains -- the largest being 21,264 m2 inside a 6,088 m2 tile. A caption
    stating those areas would be confidently wrong.

    Positions are not taken either. RSVQA phrases them relative to another
    object ("on the right of a water area"), which is not a frame position and
    cannot be rendered as one.
    """
    per_image = defaultdict(lambda: {"present": set(), "counts": {}})
    for row in rows:
        images = row.get("images") or []
        if not images:
            continue
        key = images[0]
        kind = row.get("question_type")
        question = str(row.get("question", "")).lower()
        answer = str(row.get("answer", "")).strip().lower()
        noun = next((n for n in _RSVQA_NOUNS if n in question), None)
        if not noun:
            continue
        if kind == "presence" and answer == "yes":
            per_image[key]["present"].add(noun.rstrip("s"))
        elif kind == "count" and answer.isdigit() and int(answer) > 0:
            singular = noun.rstrip("s")
            per_image[key]["counts"][singular] = max(
                per_image[key]["counts"].get(singular, 0), int(answer)
            )
    return per_image


def rsvqa_caption(facts, rng, mode):
    """A caption from presence and counts only -- no positions, no areas.

    Without frame positions this cannot use the object-inventory construction,
    so it states what is present and how much. That is honest about what the
    source knows, and still teaches the vocabulary the other sources lack.
    """
    counted = sorted(facts["counts"].items(), key=lambda kv: -kv[1])
    present = sorted(facts["present"] - set(facts["counts"]))
    if not counted and not present:
        return ""
    sensor = SENSOR["RSVQA-HR"]
    opener = rng.choice(OPENERS).format(sensor=sensor)
    parts = []
    for noun, count in counted[: 3 if mode == "concise" else 5]:
        parts.append(f"{_words(count)} {noun}{'s' if count > 1 else ''}")
    sentences = [
        _sentence([opener, rng.choice(VERBS), ", ".join(parts) or "a built-up area"])
    ]
    if present:
        sentences.append(
            _sentence([
                rng.choice(["Also visible are", "The scene includes",
                            "Present in the frame are"]),
                ", ".join(present[: 3 if mode == "concise" else 5]),
            ])
        )
    return " ".join(x for x in sentences if x)


def cmd_build(args) -> int:
    rng = random.Random(args.seed)
    rows: list[dict] = []

    def emit(image, caption, source, gsd, mode, extra=None, prompt=None,
             task="single_caption"):
        if not caption:
            return
        prompt = prompt or rng.choice(CAPTION_PROMPTS[mode])
        digest = sha1(f"{image}{prompt}{caption}".encode()).hexdigest()[:10]
        rows.append({
            "sample_id": f"cap_{digest}",
            "adapter": "rs_ground_caption",
            "task": task,
            "images": [image],
            "image_roles": ["scene"],
            "modality": ["optical"],
            "effective_gsd_m": [gsd],
            "question": prompt,
            "answer": caption,
            "answer_type": "free_text",
            "length_mode": mode,
            "split": "train",
            "source": source,
            **(extra or {}),
        })

    if args.rareplanes:
        root = Path(args.rareplanes)
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        for tile in plan["tiles"]:
            image = str(root / "tiles" / tile["file_name"])
            for mode in ("concise", "detailed"):
                emit(image, rareplanes_caption(tile, rng, mode),
                     "RarePlanes (real split)", 0.3, mode)
            # Attribute supervision on a share of tiles, not all of them. Each
            # tile already contributes two caption rows; emitting an attribute
            # row for every one would make a third of the RarePlanes corpus a
            # task the benchmark never asks about. At the default share it is
            # roughly a fifth, which is auxiliary rather than competing.
            items, _w, _h = rareplanes_items(tile)
            answer = attribute_answer(items)
            # A single-class tile teaches nothing here: "civil aircraft" is
            # what the source name already implies. The signal is which of
            # several types are present.
            if answer and answer.count(",") >= 1 and rng.random() < args.attr_share:
                emit(image, answer, "RarePlanes (real split)", 0.3, "concise",
                     prompt=rng.choice(ATTRIBUTE_PROMPTS), task="attribute_list")
        print(f"RarePlanes -> {len(rows)} row(s)", flush=True)

    for label, root_arg, gsd, source in (
        ("SpaceNet 2", args.sn2, 0.3,
         "SpaceNet 2 (Vegas/Paris/Shanghai/Khartoum)"),
        ("SpaceNet 6", args.sn6, 0.5, "SpaceNet 6 / MSAW (Rotterdam)"),
    ):
        if not root_arg:
            continue
        before = len(rows)
        root = Path(root_arg)
        plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
        # Progress inside the loop, not just at the end. This leg opens a
        # GeoTIFF header per tile off a network volume and is by far the
        # slowest part of the build; without a heartbeat a cancelled or hung
        # run is indistinguishable from a slow one, which cost us a full run.
        seen = 0
        total = len(plan["tiles"])
        for tile, modalities in plan["tiles"].items():
            seen += 1
            if seen % 250 == 0:
                print(f"  {label}: {seen}/{total} tiles, {len(rows)} row(s)",
                      flush=True)
            optical = modalities.get("PS-RGB")
            labels = modalities.get("geojson_buildings")
            if not optical or not labels:
                continue
            aoi = tile.split("/")[0] if "/" in tile else ""
            base = root / aoi if aoi else root
            image = base / "PS-RGB" / Path(optical).name
            label = base / "geojson_buildings" / Path(labels).name
            if not image.exists() or not label.exists():
                continue
            for mode in ("concise", "detailed"):
                emit(str(image),
                     _polygon_caption(image, label, source, gsd, rng, mode),
                     source, gsd, mode)
        print(f"{label} -> {len(rows) - before} row(s)", flush=True)

    if args.rsvqa_hr:
        before = len(rows)
        manifest = Path(args.rsvqa_hr)
        path = manifest if manifest.is_file() else manifest / "rsvqa_hr_train.jsonl"
        source_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for image, facts in rsvqa_hr_facts(source_rows).items():
            kinds = sorted(set(facts["present"]) | set(facts["counts"]))
            if len(kinds) >= 2 and rng.random() < args.attr_share:
                emit(image, ", ".join(kinds), "RSVQA-HR", 0.3, "concise",
                     prompt=rng.choice(ATTRIBUTE_PROMPTS), task="attribute_list")
            for mode in ("concise", "detailed"):
                emit(image, rsvqa_caption(facts, rng, mode), "RSVQA-HR", 0.3, mode)
        print(f"RSVQA-HR -> {len(rows) - before} row(s)", flush=True)

    if args.ben:
        before = len(rows)
        for line in Path(args.ben).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for item in record.get("qa", []):
                if item.get("task") != "single_caption":
                    continue
                # BEN keeps its own register under its own prompt. Rewriting its
                # climate-and-season prose into the object-inventory style would
                # invent positions a 120 px patch at 10 m cannot support.
                rows.append({
                    "sample_id": f"cap_ben_{record['patch_id'][:24]}",
                    "adapter": "rs_ground_caption",
                    "task": "single_caption",
                    "images": [f"chips/composites/{record['patch_id']}_true_colour.png"],
                    "image_roles": ["scene"],
                    "modality": ["optical"],
                    "effective_gsd_m": [10.0],
                    "question": item.get("question")
                    or rng.choice(CAPTION_PROMPTS["landcover"]),
                    "answer": item.get("answer", ""),
                    "answer_type": "free_text",
                    "length_mode": "landcover",
                    "split": "train",
                    "source": "BigEarthNet.txt",
                })
        print(f"BEN.txt -> {len(rows) - before} row(s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source = Counter(r["source"] for r in rows)
    by_mode = Counter(r["length_mode"] for r in rows)
    lengths = [len(r["answer"].split()) for r in rows if r["length_mode"] != "landcover"]
    summary = {
        "rows": len(rows),
        "by_source": dict(by_source),
        "by_mode": dict(by_mode),
        "median_words_generated": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "target_words": 47,
        "diversity": caption_diversity(
            [r["answer"] for r in rows if r["length_mode"] != "landcover"]
        ),
        "diversity_reference_vrsbench": {
            "top_opening_share": 0.173,
            "median_words": 47,
            "median_sentences": 3,
        },
    }
    Path(args.summary).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--rareplanes", default="")
    build.add_argument("--sn2", default="")
    build.add_argument("--sn6", default="")
    build.add_argument("--rsvqa-hr", default="")
    build.add_argument("--ben", default="")
    #: SpaceNet is deliberately excluded from attribute rows: a building
    #: footprint file yields exactly one type, so the row would read
    #: "building" and teach nothing.
    build.add_argument(
        "--attr-share",
        type=float,
        default=0.5,
        help="Fraction of multi-type tiles that also emit an attribute row. "
             "0 disables the auxiliary task entirely.",
    )
    build.add_argument("--out", default="/data/manifests/rs_ground_caption_captions.jsonl")
    build.add_argument("--summary", default="logs/caption_corpus.json")
    build.add_argument("--seed", type=int, default=0)
    build.set_defaults(func=cmd_build)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
