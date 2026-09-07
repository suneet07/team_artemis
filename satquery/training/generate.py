"""Batched generation for evaluation — base model, or base plus an adapter.

Used by the Phase 0 bake-off (item 9), the D2 ablation (item 10) and any
adapter evaluation afterwards. One class so that "zero-shot" and "with adapter"
differ by one argument rather than by two code paths that drift.

Deliberately transformers rather than vLLM. vLLM is the serving path (section
4.9) and its multi-LoRA hot-swap is tested separately by
``satquery.serving.vllm_smoke``; an offline evaluation that has to be
reproducible on whatever hardware is free should not also be a serving test.
"""

import time as _time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["GenerationResult", "VLMRunner"]


@dataclass
class GenerationResult:
    sample_id: str
    prompt: str
    prediction: str


class VLMRunner:
    """Load a Qwen3-VL checkpoint once, answer many samples.

    ``adapter_path`` attaches a trained LoRA. Left ``None`` this is the
    zero-shot configuration the bake-off measures.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | Path | None = None,
        max_pixels: int | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if max_pixels is None:
            from satquery.config import preprocessing_config

            max_pixels = preprocessing_config().tiling.max_pixels

        self.model_id = model_id
        self.max_pixels = max_pixels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)

        # Left, not the tokenizer's default right. A decoder-only model
        # continues from the final position, so under right-padding every
        # sequence shorter than the batch maximum is continued from a PAD token
        # and answers nonsense -- and the prompt-strip below, which slices a
        # single offset off every row, only lines up when the padding is on the
        # left. Batched generation is silently wrong without this, which is
        # worse than an error: it scores as a bad model rather than a bug.
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
        self.adapter_path = str(adapter_path) if adapter_path else None
        #: Adapters attached to THIS base, by the name they were loaded under.
        #: Serving several adapters means several LoRAs over one set of base
        #: weights, not several copies of the model -- see `select_adapter`.
        self._adapters: dict[str, str] = {}
        if self.adapter_path is not None:
            self._adapters[self.adapter_path] = "default"

    def select_adapter(self, adapter_path: str | Path | None) -> None:
        """Attach or switch the active LoRA on the shared base weights.

        Loading one ``VLMRunner`` per adapter loads one **full 4B base** per
        adapter. Three registered tools then need roughly 3x the VRAM and the L4
        dies with `CUDA out of memory` part-way through a query -- which the
        executor records as a tool failure and answers around, so it reads as a
        missing capability rather than an allocation bug.

        The project committed to the opposite (C1): one base, several LoRAs
        swapped over it, which is what makes multi-adapter serving affordable at
        all. PEFT holds every adapter's weights simultaneously -- tens of MB
        each against 8 GB of base -- and ``set_adapter`` chooses which is live.

        ``None`` runs the base model with every adapter disabled, which is how
        `rs_ground_caption` is meant to serve.
        """
        from peft import PeftModel

        if adapter_path is None:
            if isinstance(self.model, PeftModel):
                self.model.disable_adapter_layers()
            self.adapter_path = None
            return

        path = str(adapter_path)
        if path not in self._adapters:
            name = f"a{len(self._adapters)}"
            if isinstance(self.model, PeftModel):
                self.model.load_adapter(path, adapter_name=name)
            else:
                self.model = PeftModel.from_pretrained(
                    self.model, path, adapter_name=name
                ).to(self.device).eval()
            self._adapters[path] = name

        self.model.enable_adapter_layers()
        self.model.set_adapter(self._adapters[path])
        self.adapter_path = path

    def _messages(self, item: dict[str, Any]) -> list[dict]:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in item["images"]]
        content.append({"type": "text", "text": item["question"]})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        items: Sequence[dict[str, Any]],
        *,
        batch_size: int = 8,
        max_new_tokens: int = 64,
        progress: bool = True,
    ) -> list[GenerationResult]:
        """Answer every item. Greedy decoding — evaluation must be reproducible."""
        import torch

        results: list[GenerationResult] = []
        _started = _time.time()
        for start in range(0, len(items), batch_size):
            chunk = list(items[start : start + batch_size])
            texts = [
                self.processor.apply_chat_template(
                    self._messages(item), tokenize=False, add_generation_prompt=True
                )
                for item in chunk
            ]
            inputs = self.processor(
                text=texts,
                images=[item["images"] for item in chunk],
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            # Strip the prompt: generate() returns prompt + completion.
            trimmed = generated[:, inputs["input_ids"].shape[1] :]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)

            for item, text, prediction in zip(chunk, texts, decoded, strict=True):
                results.append(
                    GenerationResult(
                        sample_id=item.get("sample_id", ""),
                        prompt=text,
                        prediction=prediction.strip(),
                    )
                )
            if progress:
                # flush, or nothing is seen until the process exits: stdout is a
                # pipe here, not a terminal, so Python block-buffers it and a
                # 40-minute job looks identical to a hung one.
                #
                # Percentage and remaining time, not a bare count: "generated
                # 512/2012" tells you nothing about whether to wait or go and
                # do something else.
                done = min(start + batch_size, len(items))
                elapsed = _time.time() - _started
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (len(items) - done) / rate if rate > 0 else 0.0
                print(
                    f"  generated {done}/{len(items)}  {100 * done / len(items):5.1f}%"
                    f"  {rate:.1f}/s  eta {remaining / 60:.1f} min",
                    flush=True,
                )
        return results

    def answer_all(self, items: Iterable[dict[str, Any]], **kwargs) -> list[str]:
        return [result.prediction for result in self.generate(list(items), **kwargs)]
