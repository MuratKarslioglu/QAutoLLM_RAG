"""4-bit generator wrapper: greedy decoding only, full token accounting
(Brief D4: no sampling anywhere, not even "just for testing").

Deliberately does not call ``.to("cuda")``/pick a device on its own —
construct through
:meth:`qautollm.core.model_manager.ModelManager.load_large` so the manager
controls when construction (and VRAM allocation) happens (Brief D1). This is
the one large model kept resident for an entire experiment run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qautollm.core.config import GeneratorModelSpec, load_model_catalog

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "generator.yaml"


def _catalog() -> dict[str, GeneratorModelSpec]:
    return load_model_catalog(_CATALOG_PATH, GeneratorModelSpec)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int


class Generator:
    def __init__(self, name: str, device: str = "cuda") -> None:
        spec = _catalog()[name]
        self.name = name
        self.prompt_template_id = spec.prompt_template_id
        self._tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

        quant_config = None
        if spec.quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            quantization_config=quant_config,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self._model.eval()

    def encode(self, text: str) -> list[int]:
        """Token count helper for :func:`qautollm.rag.prompting.pack_context`."""
        return self._tokenizer.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        # Qwen2.5-Instruct is chat-tuned, not a raw completion model — feeding
        # it a bare string (no chat template) means it doesn't reliably know
        # to stop after a short reply and instead free-continues into long,
        # rambling paragraphs that score ~0 F1/EM against HotpotQA's short-span
        # gold answers regardless of whether the underlying answer is correct.
        chat_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False).to(
            self._model.device
        )
        prompt_tokens = int(inputs["input_ids"].shape[1])

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        generated_ids = output_ids[0, prompt_tokens:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=int(generated_ids.shape[0]),
        )
