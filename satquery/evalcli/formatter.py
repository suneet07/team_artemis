"""Per-benchmark answer formatter (master plan section 6.4).

*"The water body covers approximately 3.4 km²" scores zero where the scorer
wants "yes". Format compliance is worth more points than a better model, and it
costs two days.*

This module is that layer. It post-normalises a free-form answer onto each
benchmark's closed answer vocabulary: yes/no, a single word, a canonical class
name, a canonical number format. It is deliberately **not** a metric: official
scorer scripts own the metrics, our harness only wraps them, and any gap between
our number and the official script's is a P0 bug (section 6.4).

Two properties matter more than cleverness here:

* it never invents an answer — if nothing in the text maps onto the vocabulary
  it returns the fallback and says so, because a wrong confident answer and an
  abstention score the same but only one of them is honest;
* it is pure and deterministic, so the unit tests in ``tests/`` can pin it
  against each benchmark's published examples, which is what section 6.4 asks
  for.
"""

import re
from dataclasses import dataclass

__all__ = [
    "AnswerFormat",
    "BENCHMARK_FORMATS",
    "FormattedAnswer",
    "contract_for",
    "format_answer",
    "format_box",
    "format_count",
    "format_count_class",
    "format_mcq",
    "format_ratio",
    "format_yes_no",
]

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")

#: Box coordinates only, and two things differ from ``_NUMBER``.
#:
#: A comma separates coordinates here and never marks a decimal, which is the
#: opposite of ``_NUMBER``'s assumption -- Qwen's "(560,190),(650,290)" is four
#: numbers, not two.
#:
#: And a digit welded to a letter or underscore is part of an identifier, not a
#: coordinate. Qwen's native grounding reply is
#: ``[{"bbox_2d": [287, 97, 333, 148]}]``, where a naive scan takes the ``2``
#: from ``bbox_2d`` as x1 and shifts every coordinate one place left -- turning
#: a correct box into a wrong one silently, which would read as a model failure
#: rather than a parsing bug. The same guard covers ``x1``/``y2`` key names.
_BOX_NUMBER = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z_])")

#: Phrases that assert nothing is there. A count question answered this way has
#: been answered -- with zero -- and refusing to read it penalises fluent
#: prose over a bare digit.
_ABSENT = re.compile(
    r"\b(?:no|none|zero|not any|no new|nothing|neither)\b|"
    r"\bno (?:evidence|buildings|change|new)\b|"
    r"\b(?:did|were|was|have|has|are|is) not\b",
    re.I,
)
#: A refusal is not an answer. "It is not possible to determine how many
#: buildings appeared" contains "not", so the absence pattern above matches it
#: -- but the model has declined, not counted zero, and scoring it as zero
#: would credit an abstention. Checked FIRST, and it wins.
_UNSURE = re.compile(
    "cannot|can" + chr(39) + "t|unable|not possible|impossible|"
    "unclear|uncertain|difficult to|hard to|"
    "insufficient|no way to|cannot be determined|not determinable",
    re.I,
)

_YES = ("yes", "true", "correct", "present", "affirmative", "there is", "there are", "it does")
_NO = (
    "no",
    "false",
    "incorrect",
    "absent",
    "negative",
    "there is no",
    "there are no",
    "it does not",
)

#: CDVQA / RSVQA style bucketed magnitude vocabulary. Ordered low to high.
_MAGNITUDE_BUCKETS = (
    (0.0, "no"),
    (0.02, "very small"),
    (0.1, "small"),
    (0.3, "medium"),
    (0.6, "large"),
    (1.01, "very large"),
)


#: CDVQA's six land-cover answers, verbatim as the annotations spell them --
#: underscores and the ``NVG_surface`` abbreviation included. A model answering
#: "low vegetation" has answered correctly and the normaliser maps it back; a
#: *scorer* that expected the spaced form would have failed the exact tokens the
#: benchmark ships.
CDVQA_CLASSES = (
    "NVG_surface",
    "buildings",
    "low_vegetation",
    "trees",
    "water",
    "playgrounds",
)

#: Ten-point bands, plus a bare "0" that means no change rather than the first
#: band. Read off the released annotations: 11 distinct values across
#: change_ratio and 9 across change_ratio_types, the latter simply never
#: reaching the top bands.
CDVQA_RATIO_BUCKETS = (
    "0",
    "0_to_10", "10_to_20", "20_to_30", "30_to_40", "40_to_50",
    "50_to_60", "60_to_70", "70_to_80", "80_to_90", "90_to_100",
)


@dataclass(frozen=True)
class AnswerFormat:
    """One benchmark's answer contract."""

    name: str
    kind: str  # "binary" | "vocabulary" | "count" | "free_text" | "bbox"
    vocabulary: tuple[str, ...] = ()
    lowercase: bool = True
    strip_punctuation: bool = True
    #: Divisor for ``bbox`` coordinates. 1.0 for our own 0-1 floats, 100 for
    #: VRSBench. Declared per contract because guessing it from magnitude
    #: cannot separate 0-100 from 0-1000 -- a 60 is either 0.60 or 0.06.
    box_scale: float = 1.0


#: The benchmarks the plan nominates. Vocabularies are the published closed sets;
#: anything outside them is a formatting bug on our side, not a model error.
BENCHMARK_FORMATS: dict[str, AnswerFormat] = {
    # change_vqa. Its answers are generated by us rather than published by a
    # benchmark, so the vocabularies below are the ones gen_change.py writes --
    # they are the contract, and a mismatch here scores a correct answer wrong.
    #
    # Without these, contract_for("change_vqa", ...) returns None, the raw
    # generation is compared verbatim, and "Yes, several new buildings
    # appeared" is marked wrong against a reference of "yes". That is a
    # formatting bug being counted as a model error, which is the exact failure
    # the answer-format harness exists to prevent.
    "change_presence": AnswerFormat("change_presence", "binary"),
    "change_direction": AnswerFormat(
        "change_direction", "vocabulary", ("increased", "decreased", "unchanged")
    ),
    "change_compare": AnswerFormat(
        "change_compare", "vocabulary", ("first", "second")
    ),
    "change_magnitude": AnswerFormat(
        "change_magnitude", "vocabulary", ("none", "a few", "dozens", "many")
    ),
    "change_where": AnswerFormat(
        "change_where", "vocabulary", ("north", "south", "east", "west")
    ),
    # Exact integers, capped at 20 by the generator -- so "count" and not
    # "count_class": there is no quantisation to undo here.
    "change_count": AnswerFormat("change_count", "count"),
    "rsvqa_lr_presence": AnswerFormat("rsvqa_lr_presence", "binary"),
    "rsvqa_lr_comparison": AnswerFormat("rsvqa_lr_comparison", "binary"),
    # LR counts are quantised by the dataset itself: "numerical answers are
    # quantized into the following categories: '0'; 'between 1 and 10'; ...
    # 'more than 1000'" (Lobry et al. 2020), because a 6.55 km2 tile can contain
    # thousands of objects and "it would be in most cases impossible to
    # distinguish 17139 objects on an image of 65536 pixels".
    #
    # RSVQA-HR counts are NOT quantised -- the same paragraph keeps them exact,
    # the maximum there being 89 -- which is why rsvqa_hr_count stays "count".
    # Quantising those would score us against a task the benchmark never set.
    "rsvqa_lr_count": AnswerFormat(
        "rsvqa_lr_count",
        "count_class",
        vocabulary=(
            "0",
            "between 1 and 10",
            "between 11 and 100",
            "between 101 and 1000",
            "more than 1000",
        ),
    ),
    "rsvqa_lr_rural_urban": AnswerFormat(
        "rsvqa_lr_rural_urban", "vocabulary", vocabulary=("rural", "urban")
    ),
    "rsvqa_hr_presence": AnswerFormat("rsvqa_hr_presence", "binary"),
    "rsvqa_hr_count": AnswerFormat("rsvqa_hr_count", "count"),
    "rsvqa_hr_area": AnswerFormat(
        "rsvqa_hr_area",
        "vocabulary",
        # Lobry et al. 2020's own class strings. The boundaries are 1/11/101,
        # not 0/10/100: the intervals do not overlap, and "0m2" is its own
        # class rather than the floor of the first one. Three of these five
        # were previously off by one, so a correct area answer scored zero on
        # the majority of the vocabulary.
        vocabulary=("0m2", "between 1m2 and 10m2", "between 11m2 and 100m2",
                    "between 101m2 and 1000m2", "more than 1000m2"),
    ),
    # --------------------------------------------------------------------
    # CDVQA, the organiser-nominated change benchmark. Eight question types.
    #
    # These vocabularies are read off the released annotations, not off the
    # paper's prose. The previous three entries were written from the prose and
    # were wrong in a way no test would catch: `change_ratio` was scored against
    # ("no", "very small", "small", ...) while the data answers "0_to_10" and
    # "80_to_90", and `largest_change` against ("no change", "low vegetation",
    # ...) while the data answers "NVG_surface" and "low_vegetation". Every
    # correct answer would have been marked wrong, and the model would have
    # looked incapable rather than mis-scored.
    #
    # Ratio buckets are ten-point bands plus a bare "0". "0" is not the same as
    # "0_to_10" -- it means no change at all -- so it stays a distinct token
    # rather than being folded into the first band.
    # --------------------------------------------------------------------
    "cdvqa_change_or_not": AnswerFormat("cdvqa_change_or_not", "binary"),
    "cdvqa_increase_or_not": AnswerFormat("cdvqa_increase_or_not", "binary"),
    "cdvqa_decrease_or_not": AnswerFormat("cdvqa_decrease_or_not", "binary"),
    "cdvqa_change_ratio": AnswerFormat(
        "cdvqa_change_ratio",
        "vocabulary",
        vocabulary=CDVQA_RATIO_BUCKETS,
    ),
    "cdvqa_change_ratio_types": AnswerFormat(
        "cdvqa_change_ratio_types",
        "vocabulary",
        vocabulary=CDVQA_RATIO_BUCKETS,
    ),
    "cdvqa_change_to_what": AnswerFormat(
        "cdvqa_change_to_what", "vocabulary", vocabulary=CDVQA_CLASSES
    ),
    "cdvqa_largest_change": AnswerFormat(
        "cdvqa_largest_change", "vocabulary", vocabulary=CDVQA_CLASSES
    ),
    "cdvqa_smallest_change": AnswerFormat(
        "cdvqa_smallest_change", "vocabulary", vocabulary=CDVQA_CLASSES
    ),
    "rsvqa_hr_comparison": AnswerFormat("rsvqa_hr_comparison", "binary"),
    "ben_binary_vqa": AnswerFormat("ben_binary_vqa", "binary"),
    # BEN.txt's MCQ answers are bare option letters. A model that names the
    # option instead of lettering it has still chosen correctly, so the letter
    # is recovered from the question's option list rather than scored as wrong.
    "ben_mcq": AnswerFormat("ben_mcq", "mcq", vocabulary=("a", "b", "c", "d")),
    "ben_caption": AnswerFormat("ben_caption", "free_text", lowercase=False),
    "ben_ref_detection": AnswerFormat("ben_ref_detection", "bbox", lowercase=False),
    # rs_ground_caption. Like change_vqa these are our own generator's contracts
    # rather than a published benchmark's, so gen_ground.py's answer_type field
    # is the authority: 'box' -> bbox, 'yesno' -> binary.
    "ground_reference": AnswerFormat("ground_reference", "bbox", lowercase=False),
    "ground_point": AnswerFormat("ground_point", "bbox", lowercase=False),
    "ground_presence": AnswerFormat("ground_presence", "binary"),
    # VRSBench -- PS-nominated for grounding. References read
    # "{<25><40><33><60>}" on a 0-100 integer scale, verified across all 64,636
    # coordinates in VRSBench_EVAL_referring.json (min 0, max 100). Scoring is
    # IoU, so the syntax need not match ours; the *scale* must, and a missing
    # box_scale here shrinks every reference 10x toward the origin.
    "vrsbench_referring": AnswerFormat(
        "vrsbench_referring", "bbox", lowercase=False, box_scale=100.0
    ),
}


#: ``(benchmark, question_type prefix) -> contract``. One table, so the harness
#: that reports a score and the headless mode a judge runs cannot drift apart:
#: an evaluation number is only a prediction of the graded result if both format
#: the answer the same way.
_CONTRACTS: tuple[tuple[str, str, str], ...] = (
    ("ben", "binary", "ben_binary_vqa"),
    ("ben", "mcq", "ben_mcq"),
    ("ben", "captioning", "ben_caption"),
    ("ben", "bounding_box", "ben_ref_detection"),
    ("rsvqa_lr", "presence", "rsvqa_lr_presence"),
    ("rsvqa_lr", "comp", "rsvqa_lr_comparison"),
    ("rsvqa_lr", "count", "rsvqa_lr_count"),
    ("rsvqa_lr", "rural_urban", "rsvqa_lr_rural_urban"),
    ("rsvqa_hr", "presence", "rsvqa_hr_presence"),
    ("rsvqa_hr", "comp", "rsvqa_hr_comparison"),
    ("rsvqa_hr", "count", "rsvqa_hr_count"),
    ("rsvqa_hr", "area", "rsvqa_hr_area"),
    # The generator's task names are the question types, so the prefix match
    # is exact rather than a family.
    ("change_vqa", "change_presence", "change_presence"),
    ("change_vqa", "change_direction", "change_direction"),
    ("change_vqa", "change_compare", "change_compare"),
    ("change_vqa", "change_magnitude", "change_magnitude"),
    ("change_vqa", "change_where", "change_where"),
    ("change_vqa", "change_count", "change_count"),
    # CDVQA, matched on its own question-type names.
    #
    # Order matters and is not alphabetical: this table returns the *first*
    # prefix match, and "change_ratio" is a prefix of "change_ratio_types".
    # Listed the other way round, every ratio-by-type row would be scored
    # against the plain ratio contract -- same vocabulary here, so it would
    # have looked fine, and would break silently the day the two diverge.
    ("cdvqa", "change_ratio_types", "cdvqa_change_ratio_types"),
    ("cdvqa", "change_ratio", "cdvqa_change_ratio"),
    ("cdvqa", "change_or_not", "cdvqa_change_or_not"),
    ("cdvqa", "change_to_what", "cdvqa_change_to_what"),
    ("cdvqa", "increase_or_not", "cdvqa_increase_or_not"),
    ("cdvqa", "decrease_or_not", "cdvqa_decrease_or_not"),
    ("cdvqa", "largest_change", "cdvqa_largest_change"),
    ("cdvqa", "smallest_change", "cdvqa_smallest_change"),
    ("rs_ground_caption", "ground_reference", "ground_reference"),
    ("rs_ground_caption", "ground_point", "ground_point"),
    ("rs_ground_caption", "ground_presence", "ground_presence"),
    ("vrsbench", "ref", "vrsbench_referring"),
)


def contract_for(benchmark: str, question_type: str) -> str | None:
    """The answer contract for one row, or None when the type has none."""
    for source, prefix, name in _CONTRACTS:
        if benchmark == source and str(question_type or "").startswith(prefix):
            return name
    return None


@dataclass(frozen=True)
class FormattedAnswer:
    text: str
    matched: bool
    basis: str

    def __str__(self) -> str:  # so it drops straight into a trace field
        return self.text


def _clean(text: str, fmt: AnswerFormat) -> str:
    cleaned = text.strip()
    if fmt.lowercase:
        cleaned = cleaned.lower()
    if fmt.strip_punctuation:
        cleaned = cleaned.strip(" .!\"'`\n\t")
    return re.sub(r"\s+", " ", cleaned)


def format_yes_no(text: str) -> FormattedAnswer:
    """Map free text onto ``yes``/``no``.

    Negation is checked before affirmation: "there are no buildings" contains
    "there are", and a naive first-match scan gets it exactly backwards.
    """
    lowered = f" {text.strip().lower()} "
    for token in _NO:
        if f" {token} " in lowered or lowered.strip().startswith(token):
            return FormattedAnswer("no", True, "negation match")
    for token in _YES:
        if f" {token} " in lowered or lowered.strip().startswith(token):
            return FormattedAnswer("yes", True, "affirmation match")
    return FormattedAnswer("no", False, "no polarity found; abstaining to the negative class")


def format_count(text: str, value: float | int | None = None) -> FormattedAnswer:
    """Canonical integer count. A count is never "approximately"."""
    if value is not None:
        return FormattedAnswer(str(int(round(float(value)))), True, "numeric input")
    match = _NUMBER.search(text)
    if match is None:
        # "No new buildings appeared" is an answer, and the answer is zero.
        # Measured on 240 dumped zero-shot predictions: 53 of 69 unrecoverable
        # rows were exactly this -- a stated absence with no digit in it. They
        # were being scored as unparsable, which counts a correct answer as a
        # format failure and, worse, does so ONLY for the untrained model. The
        # adapter answers with digits, so the gap it appeared to open over the
        # baseline was partly the harness reading one of them more generously.
        if _UNSURE.search(text):
            return FormattedAnswer("0", False, "model declined to answer")
        if _ABSENT.search(text):
            return FormattedAnswer("0", True, "stated absence read as zero")
        return FormattedAnswer("0", False, "no number in the answer text")
    return FormattedAnswer(
        str(int(round(float(match.group().replace(",", "."))))), True, "number parsed from text"
    )


def _count_class(value: int) -> str:
    if value == 0:
        return "0"
    if value <= 10:
        return "between 1 and 10"
    if value <= 100:
        return "between 11 and 100"
    if value <= 1000:
        return "between 101 and 1000"
    return "more than 1000"


def format_count_class(text: str, value: float | int | None = None) -> FormattedAnswer:
    """A count mapped onto RSVQA-LR's five classes.

    Accepts either the class wording or a bare number: a model that answers
    "257" has given a magnitude, and the benchmark's own answer space turns
    that into a class rather than discarding it.
    """
    cleaned = str(text).strip().lower()
    for entry in ANSWER_CLASSES_LR_COUNT:
        if cleaned == entry:
            return FormattedAnswer(entry, True, "exact class match")
    if value is not None:
        return FormattedAnswer(
            _count_class(int(round(float(value)))), True, "numeric input bucketed"
        )
    match = _NUMBER.search(cleaned)
    if match:
        number = int(round(float(match.group().replace(",", "."))))
        return FormattedAnswer(_count_class(number), True, "number parsed and bucketed")
    for entry in sorted(ANSWER_CLASSES_LR_COUNT, key=len, reverse=True):
        if entry in cleaned:
            return FormattedAnswer(entry, True, "class substring match")
    return FormattedAnswer(
        "0", False, f"answer '{str(text)[:60]}' carries no count; abstained to '0'"
    )


def format_ratio(ratio: float) -> FormattedAnswer:
    """Map a change ratio in [0, 1] onto the bucketed vocabulary.

    This is the D4 path: ``change_stats`` computes the ratio exactly, and this
    turns the exact number into the string the official scorer compares against.
    Every published model sits at 32-37% on these question types because it is
    guessing the bucket; we compute it.
    """
    ratio = float(max(0.0, min(1.0, ratio)))
    for upper, label in _MAGNITUDE_BUCKETS:
        if ratio <= upper:
            return FormattedAnswer(label, True, f"exact ratio {ratio:.4f} bucketed")
    return FormattedAnswer("very large", True, f"exact ratio {ratio:.4f} bucketed")


def _vocab_key(value: str) -> str:
    """Compare vocabulary entries and model output on equal terms.

    Case and word separators are presentation, not content. CDVQA spells its
    classes ``NVG_surface`` and ``low_vegetation``; a model answering "low
    vegetation" or "NVG surface" has answered correctly, and matching the raw
    strings marked both wrong. Worse, the answer was then *abstained* to the
    first vocabulary entry, so "low vegetation" scored as ``NVG_surface`` -- a
    wrong answer manufactured by the scorer.
    """
    return re.sub(r"[\s_\-]+", " ", value.strip().lower())


def _match_vocabulary(text: str, vocabulary: tuple[str, ...]) -> FormattedAnswer:
    cleaned = _vocab_key(text)
    keys = {entry: _vocab_key(entry) for entry in vocabulary}

    for entry, key in keys.items():
        if cleaned == key:
            return FormattedAnswer(entry, True, "exact vocabulary match")
    # Longest candidate first, so "no change" wins over "no" -- and, for
    # CDVQA's ratio bands, so "0 to 10" wins over the bare "0" that is also a
    # valid answer meaning no change at all. Ordering by the *normalised* key
    # is what makes that work: compared raw, "0_to_10" could not match "0 to
    # 10" at all and the bare "0" won by default.
    for entry in sorted(vocabulary, key=lambda e: len(keys[e]), reverse=True):
        if keys[entry] in cleaned:
            return FormattedAnswer(entry, True, "vocabulary substring match")
    return FormattedAnswer(
        vocabulary[0],
        False,
        f"answer '{text[:60]}' is outside the closed vocabulary; abstained to '{vocabulary[0]}'",
    )


#: A leading option letter: "b", "b)", "(b)", "b." -- but never the article in
#: "a large building", which is why a delimiter or end-of-string is required.
_MCQ_LEAD = re.compile(r"^\s*\(?([a-d])\)?\s*(?:[.):\-]|$)", re.I)
_OPTIONS = re.compile(r"\b([a-d])\)\s*(.+?)(?=\s+\b[a-d]\)|$)", re.I | re.S)
_WORDS = re.compile(r"[a-z0-9]+")
_AREA_VALUE = re.compile(r"(\d[\d,]*)\s*m2", re.I)

#: RSVQA-LR's five counting classes, in order.
ANSWER_CLASSES_LR_COUNT = (
    "0",
    "between 1 and 10",
    "between 11 and 100",
    "between 101 and 1000",
    "more than 1000",
)


def _tokens(text: str) -> list[str]:
    return _WORDS.findall(str(text).lower())


def format_mcq(text: str, question: str = "") -> FormattedAnswer:
    """The chosen option letter, recovered from a letter or from option text.

    Abstains to ``a`` when nothing is recoverable, matching every other
    contract here: the headless mode must always emit something a judge can
    score, and ``matched=False`` is what records that it was a guess.
    """
    lead = _MCQ_LEAD.match(str(text))
    if lead:
        return FormattedAnswer(lead.group(1).lower(), True, "option letter")

    answered = _tokens(text)
    for letter, option in _OPTIONS.findall(str(question)):
        candidate = _tokens(option)
        # The whole option phrase must appear, so one word shared between two
        # options cannot decide which was meant.
        if candidate and any(
            answered[i : i + len(candidate)] == candidate
            for i in range(len(answered) - len(candidate) + 1)
        ):
            return FormattedAnswer(letter.lower(), True, "option text matched")

    return FormattedAnswer(
        "a", False, f"answer '{str(text)[:60]}' names no option; abstained to 'a'"
    )


def _area_class(value: int) -> str:
    if value == 0:
        return "0m2"
    if value <= 10:
        return "between 1m2 and 10m2"
    if value <= 100:
        return "between 11m2 and 100m2"
    if value <= 1000:
        return "between 101m2 and 1000m2"
    return "more than 1000m2"


def format_box(
    text: str,
    image_size: tuple[int, int] | None = None,
    scale: float | None = None,
) -> FormattedAnswer:
    """Recover a normalised ``[x1 y1, x2 y2]`` box from whatever the model wrote.

    Grounding is 76% of ``rs_ground_caption``'s corpus and, until this existed,
    ``bbox`` fell through ``format_answer`` to "free text passed through" -- so a
    perfectly placed box scored zero unless its punctuation happened to match
    the reference string. That is the section 6.4 failure this module exists to
    prevent, and it was open on the largest task we have.

    **Say the scale; do not let it be guessed.** Coordinate conventions in this
    project's own inputs already span three decades of magnitude: our generator
    writes 0-1 floats, VRSBench -- the PS-nominated grounding benchmark --
    writes ``{<25><40><33><60>}`` on a **0-100** scale, and the Qwen-VL family
    has emitted 0-1000 integers and, later, absolute pixels against the
    *resized* input. ``scale`` divides the parsed numbers and is the only
    reliable answer; ``image_size`` handles pixels.

    The fallback when neither is given -- rescale anything above 1.5 by 1000 --
    is a guess kept only so a bare call still does something sane on Qwen
    output. It is **wrong by 10x on VRSBench**, which is why that benchmark's
    contract carries an explicit ``box_scale`` and the caller must pass it
    through. Every rescale records its basis so a wrong one is visible in the
    trace rather than silently shrinking every box toward the origin.

    A refusal is not a box. ``_UNSURE`` is checked first and returns
    ``matched=False`` rather than a plausible-looking rectangle, because an
    invented box and a missing one score the same but only one is honest.
    """
    raw = str(text)
    if _UNSURE.search(raw):
        return FormattedAnswer("", False, "model declined to locate the object")

    # Deliberately NOT _NUMBER: that pattern treats a comma as a decimal
    # separator, so Qwen's "(560,190),(650,290)" reads as two numbers -- 560.190
    # and 650.290 -- instead of four, and the box is silently rejected. Inside a
    # box a comma is always a separator.
    numbers = [float(n) for n in _BOX_NUMBER.findall(raw)]
    if len(numbers) < 4:
        return FormattedAnswer(
            "", False, f"answer '{raw[:60]}' carries {len(numbers)} number(s), need 4"
        )
    # The first four win. A model that narrates ("the box [0.1 0.2, 0.3 0.4] is
    # near the second at ...") has still answered with its first box, and taking
    # the last four would silently score the wrong object.
    x1, y1, x2, y2 = numbers[:4]

    basis = "normalised 0-1"
    if scale:
        x1, y1, x2, y2 = (v / scale for v in (x1, y1, x2, y2))
        basis = f"declared scale 0-{scale:g}"
    elif max(abs(v) for v in (x1, y1, x2, y2)) > 1.5:
        if image_size:
            width, height = image_size
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
            basis = f"pixel coords rescaled by {width}x{height}"
        else:
            x1, y1, x2, y2 = (v / 1000.0 for v in (x1, y1, x2, y2))
            basis = "assumed 0-1000 scale; pass image_size if pixels"

    # Order the corners rather than trusting them. A transposed box is a
    # plausible low-IoU rectangle, and box_iou cannot tell it from a genuine
    # miss -- see the same warning in training/metrics.py.
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:
        return FormattedAnswer("", False, f"degenerate box from '{raw[:60]}'")

    return FormattedAnswer(f"[{x1:.2f} {y1:.2f}, {x2:.2f} {y2:.2f}]", True, basis)


def format_answer(
    text: str,
    benchmark: str,
    *,
    numeric_value: float | None = None,
    ratio: float | None = None,
    question: str = "",
    image_size: tuple[int, int] | None = None,
) -> FormattedAnswer:
    """Normalise ``text`` onto ``benchmark``'s answer contract.

    ``numeric_value`` and ``ratio`` are the exact values a deterministic tool
    computed. When present they win over anything parsed out of prose — the
    whole point of D4 is that the number is not a guess.
    """
    fmt = BENCHMARK_FORMATS.get(benchmark)
    if fmt is None:
        raise KeyError(
            f"unknown benchmark '{benchmark}'; add its answer contract to "
            f"BENCHMARK_FORMATS rather than formatting by hand at the call site"
        )
    if fmt.kind == "binary":
        return format_yes_no(text)
    if fmt.kind == "bbox":
        return format_box(text, image_size, scale=fmt.box_scale)
    if fmt.kind == "count":
        return format_count(text, numeric_value)
    if fmt.kind == "count_class":
        return format_count_class(text, numeric_value)
    if fmt.kind == "mcq":
        return format_mcq(text, question)
    if fmt.kind == "vocabulary":
        if ratio is not None and "ratio" in fmt.name:
            return format_ratio(ratio)
        matched = _match_vocabulary(_clean(text, fmt), fmt.vocabulary)
        if not matched.matched and fmt.name.endswith("_area"):
            # A raw magnitude is an answer, just not in class wording. Bucket it
            # the way the references were bucketed rather than abstaining, so
            # "10631m2" is scored on whether the magnitude was right.
            raw = _AREA_VALUE.search(str(text)) or _NUMBER.search(str(text))
            if raw:
                digits = raw.group(1) if raw.re is _AREA_VALUE else raw.group()
                return FormattedAnswer(
                    _area_class(int(float(digits.replace(",", "")))),
                    True,
                    "raw area value bucketed",
                )
        return matched
    return FormattedAnswer(_clean(text, fmt), True, "free text passed through")
