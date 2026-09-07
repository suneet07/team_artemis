"""The prompt the models were trained on — and only that one.

This module previously assembled its own format: ``<image>{role}</image>``
placeholders followed by ``[sensor: X | GSD: Y m]`` tags. Nothing called it, and
that was lucky, because it is **not** the format any adapter was trained on.
Training builds ``scale_prefix(...) + question`` and hands images to the
processor as separate content items; the placeholder text never existed.

Wiring the old version would not have raised. It would have produced fluent,
confident, worse answers, and every score would have moved without a cause
anyone could point at -- which is exactly what
:func:`satquery.training.dataset.scale_prefix` says in its own docstring:

    **This is the one implementation.** Training, evaluation and the demo
    endpoint all reach the model through here, because a prompt the model was
    never trained on is indistinguishable from a weak model.

So this now delegates. The name stays because it is the one a caller reaches
for, and a correct function under the obvious name is worth more than a removed
one that gets reinvented.
"""

from __future__ import annotations

__all__ = ["assemble_prompt"]


def assemble_prompt(
    images: list[str],
    roles: list[str],
    modalities: list[str],
    effective_gsd_m: list[float],
    question: str,
    *,
    source: str = "",
    loaded_images=None,
) -> str:
    """The text half of a prompt, identical to what training produced.

    ``images``, ``roles`` and ``modalities`` are accepted for call-site
    compatibility and deliberately unused: the vision-language model receives
    images as processor content items, not as placeholder tokens in the string.
    Passing them here is harmless; encoding them into the text is not.

    ``loaded_images`` are PIL images when the caller has them. They are what
    lets the prefix state scene extent, which is computed from the pixels the
    model is actually shown rather than from the manifest, so it cannot drift.

    Returns ``prefix + question``. The prefix is empty when no GSD is known --
    a wrong scale is worse than no scale, because the model learns to trust the
    field.
    """
    from satquery.training.dataset import scale_prefix

    prefix = scale_prefix(effective_gsd_m, question, source, loaded_images)
    return f"{prefix}{question}"
