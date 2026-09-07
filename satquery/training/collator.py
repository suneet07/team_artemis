"""Batch construction and label masking, shared by every training entry point.

This is the single most load-bearing piece of the training path and it was the
first thing the original harness got wrong. ``labels = batch["input_ids"]`` with
padding to a fixed length scores the model on pad tokens and image placeholders.
Evidence it happened: opening loss **11.32** against ``ln(151669) = 11.93`` for
Qwen3-VL's vocabulary -- a pretrained model performing at 95% of uniform random
on its own chat template. The loss then fell to ~4.4 and flatlined, and that
curve was read as training working.

So the masking is done once, here, and every caller inherits it:

1. the prompt is templated twice, the second time with
   ``add_generation_prompt=True``, to find where the assistant answer begins;
2. everything before the answer is set to ``-100``;
3. all padding is set to ``-100`` via the attention mask;
4. image and video placeholder ids are set to ``-100`` explicitly, belt and
   braces, in case a processor emits them inside the answer span.

Padding is **dynamic** -- to the longest sequence in the batch, never to a fixed
cap. A fixed 128-token cap is what made the first run measure ~16 vision tokens
per sample when the real figure is 48 to 512.
"""

from typing import Any

__all__ = ["MaskedCollator", "supervised_span"]


class MaskedCollator:
    """Turn canonical samples into a model batch with correctly masked labels."""

    def __init__(self, processor, *, pad_to_multiple_of: int | None = None):
        self.processor = processor
        self.pad_to_multiple_of = pad_to_multiple_of
        tokenizer = getattr(processor, "tokenizer", processor)
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def messages(self, item: dict[str, Any], *, with_answer: bool) -> list[dict]:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in item["images"]]
        content.append({"type": "text", "text": item["question"]})
        messages = [{"role": "user", "content": content}]
        if with_answer:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}],
                }
            )
        return messages

    def __call__(self, batch: list[dict[str, Any]]):
        full_texts, prompt_texts, images = [], [], []
        for item in batch:
            full_texts.append(
                self.processor.apply_chat_template(
                    self.messages(item, with_answer=True),
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    self.messages(item, with_answer=False),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            images.append(item["images"])

        encoded = self.processor(
            text=full_texts, images=images, return_tensors="pt", padding=True
        )
        prompt_encoded = self.processor(
            text=prompt_texts, images=images, return_tensors="pt", padding=True
        )

        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1)
        for index, length in enumerate(prompt_lengths):
            labels[index, : int(length)] = -100

        for attribute in ("image_token_id", "video_token_id"):
            token_id = getattr(self.processor, attribute, None)
            if token_id is None:
                token_id = getattr(
                    getattr(self.processor, "tokenizer", None), attribute, None
                )
            if isinstance(token_id, int):
                labels[encoded["input_ids"] == token_id] = -100

        encoded["labels"] = labels
        return encoded


def supervised_span(processor, batch) -> str:
    """Decode what the loss is actually computed over, for one batch item.

    The check that catches a broken mask before a run rather than after: this
    must decode to the answer and nothing else. Used by the harness dry-run and
    by ``tests`` wherever a processor is available.
    """
    labels = batch["labels"][0]
    keep = labels[labels != -100]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(keep, skip_special_tokens=True)
