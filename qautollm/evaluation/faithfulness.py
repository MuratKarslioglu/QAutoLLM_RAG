"""NLI entailment rate: fraction of answer sentences entailed by the context
(Brief §4). Small model — load via
:meth:`qautollm.core.model_manager.ModelManager.load_small`.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from qautollm.core.config import NLIModelSpec, load_model_catalog

_CATALOG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models" / "nli.yaml"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _catalog() -> dict[str, NLIModelSpec]:
    return load_model_catalog(_CATALOG_PATH, NLIModelSpec)


class FaithfulnessScorer:
    def __init__(self, name: str = "default", device: str = "cpu") -> None:
        spec = _catalog()[name]
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(spec.hf_id)
        self._model.eval()
        self._model.to(device)
        self._entail_idx = self._resolve_entailment_index()

    def _resolve_entailment_index(self) -> int:
        id2label = self._model.config.id2label
        for idx, label in id2label.items():
            if str(label).lower().startswith("entail"):
                return int(idx)
        raise ValueError(f"no entailment label found in model config: {id2label}")

    @torch.no_grad()
    def score(self, answer: str, context: str) -> float:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s]
        if not sentences:
            return 0.0

        entailed = 0
        for sentence in sentences:
            inputs = self._tokenizer(
                context, sentence, return_tensors="pt", truncation=True
            ).to(self._device)
            logits = self._model(**inputs).logits
            if int(torch.argmax(logits, dim=-1)) == self._entail_idx:
                entailed += 1
        return entailed / len(sentences)
