"""Pydantic models for every YAML schema in the repo (Brief §3, §7).

Two schema families live here:
  - ``ExperimentConfig`` and friends: the experiment YAML consumed by
    ``scripts/run_experiment.py`` (Brief §7).
  - ``SearchSpaceSpec`` and friends: the search-space YAML consumed by
    :class:`qautollm.space.search_space.SearchSpace` (e.g. ``rag_v2.yaml``).

Nothing outside this module should call ``yaml.safe_load`` directly on a
config file — route every YAML load through one of the ``from_yaml``
classmethods here so schema drift fails loudly in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# --------------------------------------------------------------------------
# Search-space schema (e.g. configs/search_spaces/rag_v2.yaml)
# --------------------------------------------------------------------------


class ConditionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    depends_on: str
    active_when: Any


class VariableSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["categorical", "discrete_numeric", "binary"]
    group: str
    values: list[Any]
    condition: ConditionSpec | None = None

    @field_validator("values")
    @classmethod
    def _non_empty_unique(cls, values: list[Any]) -> list[Any]:
        if len(values) == 0:
            raise ValueError("variable must declare at least one value")
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate values in domain: {values}")
        return values


class SearchSpaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variables: list[VariableSpec]
    groups: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> SearchSpaceSpec:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# --------------------------------------------------------------------------
# Experiment schema (Brief §7)
# --------------------------------------------------------------------------


class ExperimentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int
    budget: int
    checkpoints: list[int]

    @field_validator("budget")
    @classmethod
    def _positive_budget(cls, budget: int) -> int:
        if budget <= 0:
            raise ValueError("budget must be positive")
        return budget

    @model_validator(mode="after")
    def _checkpoints_sorted_and_in_range(self) -> ExperimentMeta:
        if list(self.checkpoints) != sorted(self.checkpoints):
            raise ValueError(f"checkpoints must be sorted ascending: {self.checkpoints}")
        if any(c <= 0 for c in self.checkpoints):
            raise ValueError(f"checkpoints must be positive: {self.checkpoints}")
        if any(c > self.budget for c in self.checkpoints):
            raise ValueError(
                f"checkpoints {self.checkpoints} exceed budget {self.budget}"
            )
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    optimization_split: str
    test_split: str


class QuantumConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shots: int | None = None
    feature_map_depth: int = 1


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["random", "tpe", "nsga2", "structured_bo", "qkernel_bo"]
    init_points: int = 10
    quantum: QuantumConfig = QuantumConfig()


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_new_tokens: int = 64


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_vram_gb: float = 7.0


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentMeta
    dataset: DatasetConfig
    optimizer: OptimizerConfig
    generator: GeneratorConfig
    search_space: str
    hardware: HardwareConfig = HardwareConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


# --------------------------------------------------------------------------
# Model catalogs (configs/models/*.yaml) — short domain name -> HF repo id
# --------------------------------------------------------------------------


class EmbeddingModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str
    dim: int


class RerankerModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str


class NLIModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str


class GeneratorModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hf_id: str
    quantization: Literal["4bit", "none"] = "4bit"
    prompt_template_id: str


def load_model_catalog(path: str | Path, spec_cls: type[BaseModel]) -> dict[str, Any]:
    """Load a ``configs/models/*.yaml`` catalog: ``{short_name: spec_cls(...)}``.

    One loader for every model catalog (embedding/reranker/nli/generator)
    instead of a bespoke ``RootModel`` per file — the files are all the same
    shape (a flat mapping of short name to a spec), only the spec differs.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {name: spec_cls.model_validate(entry) for name, entry in raw.items()}
