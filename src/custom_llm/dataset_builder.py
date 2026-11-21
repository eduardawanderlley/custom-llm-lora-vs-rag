"""Builds training rows and evaluation items. Every split happens at the fact level."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from custom_llm.generation.qa_generator import QAPair
from custom_llm.knowledge import FactUpdate, KnowledgeBase
from custom_llm.prompts import SYSTEM_PROMPT
from custom_llm.utils import normalize_text, read_jsonl

logger = logging.getLogger(__name__)

OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class FactSplit:
    train_fact_ids: tuple[str, ...]
    heldout_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalItem:
    id: str
    regime: str  # seen | unseen | update | out_of_scope
    question: str
    gold: str | None
    fact_id: str | None
    key_terms: tuple[str, ...]
    reference: str


def split_facts(fact_ids: list[str], heldout_ratio: float, seed: int) -> FactSplit:
    """Split at the fact level, so held-out facts never appear in the training data."""
    ids = sorted(fact_ids)
    random.Random(seed).shuffle(ids)
    n_heldout = max(1, round(len(ids) * heldout_ratio))
    return FactSplit(
        train_fact_ids=tuple(sorted(ids[n_heldout:])),
        heldout_fact_ids=tuple(sorted(ids[:n_heldout])),
    )


def to_training_row(pair: QAPair) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair.question},
        ],
        "completion": [{"role": "assistant", "content": pair.answer}],
    }


def build_dataset(
    pairs: list[QAPair],
    current_kb: KnowledgeBase,
    updates: list[FactUpdate],
    split: FactSplit,
    seed: int,
) -> tuple[list[dict], list[EvalItem]]:
    """Turn generated pairs into training rows and in-domain evaluation items.

    Held-out facts only produce evaluation items. Facts changed after training are kept out of
    the seen and unseen regimes and are probed through their hand-written probes instead, so two
    versions of the same fact are never compared in one place.
    """
    rng = random.Random(seed)
    facts = current_kb.fact_by_id()
    train_ids = set(split.train_fact_ids)
    heldout_ids = set(split.heldout_fact_ids)
    updated_ids = {update.fact_id for update in updates}

    by_fact: dict[str, list[QAPair]] = defaultdict(list)
    for pair in pairs:
        by_fact[pair.fact_id].append(pair)

    train_rows: list[dict] = []
    items: list[EvalItem] = []

    for fact_id in sorted(by_fact):
        candidates = sorted(by_fact[fact_id], key=lambda p: p.question)
        fact = facts[fact_id]

        if fact_id in updated_ids:
            if fact_id in train_ids:
                train_rows.extend(to_training_row(p) for p in candidates)
            continue

        if fact_id in heldout_ids:
            items.extend(
                EvalItem(
                    id=f"unseen-{fact_id}-{i}",
                    regime="unseen",
                    question=p.question,
                    gold=p.answer,
                    fact_id=fact_id,
                    key_terms=fact.key_terms,
                    reference=fact.passage,
                )
                for i, p in enumerate(candidates)
            )
            continue

        held_out = rng.choice(candidates)
        items.append(
            EvalItem(
                id=f"seen-{fact_id}",
                regime="seen",
                question=held_out.question,
                gold=held_out.answer,
                fact_id=fact_id,
                key_terms=fact.key_terms,
                reference=fact.passage,
            )
        )
        train_rows.extend(to_training_row(p) for p in candidates if p is not held_out)

    for update in updates:
        if update.fact_id in heldout_ids:
            logger.warning(
                "Updated fact %s is held out, so its probe will not measure stale knowledge.",
                update.fact_id,
            )
        items.append(
            EvalItem(
                id=f"update-{update.fact_id}",
                regime="update",
                question=update.probe_question,
                gold=update.probe_answer,
                fact_id=update.fact_id,
                key_terms=update.key_terms,
                reference=facts[update.fact_id].passage,
            )
        )

    return train_rows, items


def load_out_of_scope(path: Path) -> list[EvalItem]:
    return [
        EvalItem(
            id=f"oos-{i:02d}",
            regime=OUT_OF_SCOPE,
            question=row["question"],
            gold=None,
            fact_id=None,
            key_terms=(),
            reference="",
        )
        for i, row in enumerate(read_jsonl(path), start=1)
    ]


def load_eval_items(path: Path) -> list[EvalItem]:
    return [EvalItem(**{**row, "key_terms": tuple(row["key_terms"])}) for row in read_jsonl(path)]


def check_leakage(train_rows: list[dict], items: list[EvalItem]) -> None:
    """Fail loudly if any evaluation question also appears verbatim in the training data."""
    train_questions = {normalize_text(row["prompt"][-1]["content"]) for row in train_rows}
    leaked = [item.id for item in items if normalize_text(item.question) in train_questions]
    if leaked:
        raise ValueError(f"Evaluation questions found in training data: {leaked[:5]}")
