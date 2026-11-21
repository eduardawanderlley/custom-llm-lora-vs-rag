"""Typed project configuration, loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    knowledge_base: Path
    knowledge_updates: Path
    out_of_scope_questions: Path
    artifacts_dir: Path = Path("artifacts")

    @property
    def dataset_dir(self) -> Path:
        return self.artifacts_dir / "dataset"

    @property
    def train_file(self) -> Path:
        return self.dataset_dir / "train.jsonl"

    @property
    def test_file(self) -> Path:
        return self.dataset_dir / "test_set.jsonl"

    @property
    def split_file(self) -> Path:
        return self.dataset_dir / "split_manifest.json"

    @property
    def quality_report_file(self) -> Path:
        return self.dataset_dir / "quality_report.json"

    @property
    def review_file(self) -> Path:
        return self.dataset_dir / "review_sample.csv"

    @property
    def adapter_dir(self) -> Path:
        return self.artifacts_dir / "adapter"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"


class GenerationConfig(BaseModel):
    model: str
    questions_per_fact: int = Field(3, ge=1)
    heldout_fact_ratio: float = Field(0.2, gt=0, lt=1)
    dedup_similarity: float = Field(0.9, gt=0, le=1)
    review_sample_size: int = Field(30, ge=1)
    temperature: float = 0.7
    max_tokens: int = 500


class TrainingConfig(BaseModel):
    base_model: str
    max_seq_length: int = 1024
    num_train_epochs: float = 3
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str]
    load_in_4bit: bool = True


class RetrievalConfig(BaseModel):
    embedding_model: str
    top_k: int = Field(3, ge=1)


class EvaluationConfig(BaseModel):
    judge_model: str
    max_new_tokens: int = Field(256, ge=1)


class ProjectConfig(BaseModel):
    seed: int = 42
    paths: PathsConfig
    generation: GenerationConfig
    training: TrainingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path) -> ProjectConfig:
    with open(path, encoding="utf-8") as fh:
        return ProjectConfig.model_validate(yaml.safe_load(fh))
