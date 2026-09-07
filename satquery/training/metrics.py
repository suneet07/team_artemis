"""Scoring for the Phase 0 bake-off and the ablations.

**These are not the official scorers.** Section 6.4 (C8) is explicit that
reimplemented metrics routinely differ from official ones by large margins, and
that integrating the official scripts is a Phase 1 deliverable. What is here is
for *comparing our own runs to each other* -- base model A against base model B,
prior against no prior -- where a consistent metric is enough and an official
one is not available offline.

Every report these produce carries that caveat, because a bake-off number quoted
in a deck as if it were an RSVQA score is a scoring-integrity problem.

The one metric that is genuinely canonical here is **average accuracy (AA)**:
the mean of per-question-type accuracies rather than the mean over samples. It
is what RSVQA and CDVQA report and it is not the same number as overall
accuracy whenever the type distribution is skewed, which it always is.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

__all__ = [
    "OFFICIAL_SCORER_CAVEAT",
    "ScoreReport",
    "average_accuracy",
    "bleu",
    "box_iou",
    "caption_scores",
    "cider_d",
    "grounding_accuracy",
    "normalise_answer",
    "rouge_l",
    "score_predictions",
]

OFFICIAL_SCORER_CAVEAT = (
    "Scored with the in-repo comparator, NOT the official benchmark scorer. "
    "Valid for run-to-run comparison; not quotable as a benchmark result "
    "(master plan section 6.4, C8)."
)

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s.]")
#: A period is punctuation everywhere except between two digits. Keeping every
#: period to protect "12.5" left "Yes." scoring as a miss against "yes", which
#: is a formatter difference being counted as an accuracy difference.
_SENTENCE_STOP = re.compile(r"(?<!\d)\.|\.(?!\d)")
_SPACE = re.compile(r"\s+")


def normalise_answer(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace.

    Deliberately conservative: it does not map synonyms or round numbers. A
    normaliser that is too clever inflates every score it touches and hides the
    formatter bugs section 6.4 exists to catch.
    """
    text = _PUNCT.sub(" ", str(text).strip().lower())
    text = _SENTENCE_STOP.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


@dataclass
class ScoreReport:
    """Per-type accuracies plus the two headline numbers."""

    total: int = 0
    correct: int = 0
    per_type: dict[str, tuple[int, int]] = field(default_factory=dict)
    caveat: str = OFFICIAL_SCORER_CAVEAT

    @property
    def overall_accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def average_accuracy(self) -> float:
        """Mean of per-type accuracies -- what RSVQA and CDVQA report as AA."""
        if not self.per_type:
            return 0.0
        return sum(c / t for c, t in self.per_type.values() if t) / len(self.per_type)

    def as_dict(self) -> dict:
        return {
            "samples": self.total,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "average_accuracy": round(self.average_accuracy, 4),
            "per_type": {
                name: {"n": total, "accuracy": round(correct / total, 4) if total else 0.0}
                for name, (correct, total) in sorted(self.per_type.items())
            },
            "caveat": self.caveat,
        }


def score_predictions(
    predictions: list[str],
    references: list[str],
    question_types: list[str] | None = None,
) -> ScoreReport:
    """Exact-match accuracy after normalisation, broken down by question type."""
    if len(predictions) != len(references):
        raise ValueError(
            f"{len(predictions)} predictions against {len(references)} references; "
            "a length mismatch means the two lists are not aligned and any score "
            "computed from them is meaningless"
        )
    types = question_types or ["all"] * len(predictions)
    if len(types) != len(predictions):
        raise ValueError("question_types must align with predictions")

    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for prediction, reference, kind in zip(predictions, references, types, strict=True):
        hit = int(normalise_answer(prediction) == normalise_answer(reference))
        correct += hit
        buckets[kind][0] += hit
        buckets[kind][1] += 1

    return ScoreReport(
        total=len(predictions),
        correct=correct,
        per_type={name: (hit, total) for name, (hit, total) in buckets.items()},
    )


def average_accuracy(report: ScoreReport) -> float:
    return report.average_accuracy


def box_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """IoU of two ``(x1, y1, x2, y2)`` boxes in one shared coordinate space.

    Both boxes must already be in the same convention. This function cannot
    detect a transposed box -- it returns a plausible low IoU instead, which is
    exactly why ``satquery.qgen.boxes`` guards the convention rather than
    trusting a score to reveal the error.
    """
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


def grounding_accuracy(
    predicted: list[tuple[float, float, float, float] | None],
    reference: list[tuple[float, float, float, float]],
    thresholds: tuple[float, ...] = (0.5, 0.7),
) -> dict[str, float]:
    """acc@k for referring grounding -- what section 6.1 reports for G3.

    A prediction of ``None`` (the model refused or emitted no parsable box)
    counts as a miss rather than being dropped. Dropping unparsable outputs is
    how a grounding number ends up describing only the samples the model found
    easy.
    """
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference box lists must align")
    ious = [
        box_iou(p, r) if p is not None else 0.0
        for p, r in zip(predicted, reference, strict=True)
    ]
    result = {
        f"acc@{threshold}": sum(i >= threshold for i in ious) / len(ious)
        for threshold in thresholds
    }
    result["mean_iou"] = sum(ious) / len(ious) if ious else 0.0
    result["unparsable"] = sum(p is None for p in predicted) / len(predicted)
    return result


# ---------------------------------------------------------------------------
# Caption metrics
#
# VRSBench reports BLEU, METEOR, ROUGE-L and CIDEr for captioning. These are
# reimplementations, and the OFFICIAL_SCORER_CAVEAT above applies with extra
# force: caption metrics are more sensitive to tokenisation than accuracy is,
# so two correct implementations disagree more than you would expect. Use them
# to compare our own runs; never quote them beside a published number.
#
# METEOR is deliberately absent. It needs WordNet synonymy and a stemmer to mean
# anything, and a version without those is not METEOR -- it is a worse BLEU-1
# wearing its name, which is more misleading than omitting it.
# ---------------------------------------------------------------------------

_CAPTION_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens. Shared by every caption metric so they agree."""
    return _CAPTION_TOKEN.findall(str(text).lower())


def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i : i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def bleu(
    predictions: list[str], references: list[str], max_n: int = 4
) -> dict[str, float]:
    """Corpus BLEU with the standard brevity penalty.

    Corpus-level, not the mean of per-sentence scores: sentence BLEU on a short
    caption is dominated by whether one 4-gram happened to match, and averaging
    those is a different and much noisier statistic than what is published.
    """
    import math

    if len(predictions) != len(references):
        raise ValueError("predictions and references must align")
    clipped = [0] * max_n
    totals = [0] * max_n
    pred_len = ref_len = 0

    for prediction, reference in zip(predictions, references, strict=True):
        p_tokens, r_tokens = _tokenise(prediction), _tokenise(reference)
        pred_len += len(p_tokens)
        ref_len += len(r_tokens)
        for order in range(1, max_n + 1):
            p_grams = _ngrams(p_tokens, order)
            r_grams = _ngrams(r_tokens, order)
            for gram, count in p_grams.items():
                clipped[order - 1] += min(count, r_grams.get(gram, 0))
                totals[order - 1] += count

    precisions = [
        (clipped[i] / totals[i]) if totals[i] else 0.0 for i in range(max_n)
    ]
    out = {f"bleu_{i + 1}": round(precisions[i], 4) for i in range(max_n)}
    if min(precisions) <= 0:
        out["bleu"] = 0.0
        out["brevity_penalty"] = 0.0
        return out
    penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    score = penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
    out["bleu"] = round(score, 4)
    out["brevity_penalty"] = round(penalty, 4)
    return out


def _lcs(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length, row-rolled to stay O(min) in memory."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(
                previous[j] + 1 if token == other else max(previous[j + 1], current[j])
            )
        previous = current
    return previous[-1]


def rouge_l(
    predictions: list[str], references: list[str], beta: float = 1.2
) -> dict[str, float]:
    """Mean sentence-level ROUGE-L F-measure.

    Sentence-level and then averaged, which is what the captioning literature
    reports for ROUGE-L -- unlike BLEU, where the corpus form is standard.
    """
    scores = []
    for prediction, reference in zip(predictions, references, strict=True):
        p_tokens, r_tokens = _tokenise(prediction), _tokenise(reference)
        if not p_tokens or not r_tokens:
            scores.append(0.0)
            continue
        common = _lcs(p_tokens, r_tokens)
        if common == 0:
            scores.append(0.0)
            continue
        precision = common / len(p_tokens)
        recall = common / len(r_tokens)
        scores.append(
            ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
        )
    return {"rouge_l": round(sum(scores) / max(1, len(scores)), 4)}


def cider_d(
    predictions: list[str], references: list[str], max_n: int = 4, sigma: float = 6.0
) -> dict[str, float]:
    """CIDEr-D: TF-IDF weighted n-gram similarity with a length penalty.

    IDF is computed over the reference set being scored, which is how CIDEr is
    defined -- it is a corpus-relative metric, so a score is only comparable
    against another score computed over the same references. With one reference
    per image (VRSBench's case) it is noisier than with five, and the absolute
    value should not be read against published numbers computed on five.
    """
    import math

    n_docs = len(references)
    document_freq: list[dict[tuple[str, ...], int]] = [{} for _ in range(max_n)]
    ref_tokens = [_tokenise(r) for r in references]
    for tokens in ref_tokens:
        for order in range(1, max_n + 1):
            for gram in set(_ngrams(tokens, order)):
                d = document_freq[order - 1]
                d[gram] = d.get(gram, 0) + 1

    def vector(tokens: list[str], order: int):
        counts = _ngrams(tokens, order)
        vec, norm = {}, 0.0
        for gram, count in counts.items():
            df = document_freq[order - 1].get(gram, 0)
            idf = math.log(max(n_docs, 1) / max(df, 1))
            value = count * idf
            vec[gram] = value
            norm += value * value
        return vec, math.sqrt(norm)

    scores = []
    for prediction, tokens_r in zip(predictions, ref_tokens, strict=True):
        tokens_p = _tokenise(prediction)
        if not tokens_p or not tokens_r:
            scores.append(0.0)
            continue
        per_order = []
        for order in range(1, max_n + 1):
            vp, np_ = vector(tokens_p, order)
            vr, nr = vector(tokens_r, order)
            if np_ == 0 or nr == 0:
                per_order.append(0.0)
                continue
            # min() before the product is CIDEr-D's clipping: it stops a
            # prediction from scoring by repeating a high-IDF phrase.
            dot = sum(min(vp[g], vr.get(g, 0.0)) * vr.get(g, 0.0) for g in vp)
            per_order.append(dot / (np_ * nr))
        delta = len(tokens_p) - len(tokens_r)
        penalty = math.exp(-(delta**2) / (2 * sigma**2))
        scores.append(penalty * 10.0 * sum(per_order) / max_n)
    return {"cider_d": round(sum(scores) / max(1, len(scores)), 4)}


def caption_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Every caption metric at once, plus length statistics.

    Length is reported because it explains most surprises: a model that writes
    three words scores near zero on everything, and a model that writes an essay
    is punished by the brevity penalty from the other side. Seeing the ratio
    first saves diagnosing a prompt problem as a capability problem.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must align")
    out: dict[str, float] = {}
    out.update(bleu(predictions, references))
    out.update(rouge_l(predictions, references))
    out.update(cider_d(predictions, references))
    p_len = [len(_tokenise(p)) for p in predictions]
    r_len = [len(_tokenise(r)) for r in references]
    out["mean_pred_words"] = round(sum(p_len) / max(1, len(p_len)), 1)
    out["mean_ref_words"] = round(sum(r_len) / max(1, len(r_len)), 1)
    out["empty_predictions"] = sum(1 for n in p_len if n == 0)
    return out
