"""Quality filters applied to generated pairs before they reach the training set."""

from __future__ import annotations

import random
from collections import Counter
from difflib import SequenceMatcher

from custom_llm.generation.qa_generator import QAPair
from custom_llm.knowledge import Fact
from custom_llm.utils import normalize_text, term_present

MIN_QUESTION_WORDS = 3
MIN_ANSWER_CHARS = 10
MAX_ANSWER_CHARS = 400


def is_well_formed(pair: QAPair) -> bool:
    question_words = normalize_text(pair.question).split()
    return len(question_words) >= MIN_QUESTION_WORDS and (
        MIN_ANSWER_CHARS <= len(pair.answer) <= MAX_ANSWER_CHARS
    )


def is_consistent(pair: QAPair, fact: Fact) -> bool:
    """The answer must contain at least one key term of the fact it was generated from."""
    return any(term_present(pair.answer, term) for term in fact.key_terms)


def filter_pairs(
    pairs: list[QAPair],
    facts_by_id: dict[str, Fact],
    dedup_threshold: float,
) -> tuple[list[QAPair], dict[str, int]]:
    """Apply the filters in order and report how many pairs each one removed."""
    report: Counter[str] = Counter()
    kept: list[QAPair] = []
    kept_questions: list[str] = []

    for pair in pairs:
        report["input"] += 1
        if not is_well_formed(pair):
            report["malformed"] += 1
            continue
        if not is_consistent(pair, facts_by_id[pair.fact_id]):
            report["inconsistent_with_fact"] += 1
            continue

        normalized = normalize_text(pair.question)
        if any(
            SequenceMatcher(None, normalized, other).ratio() >= dedup_threshold
            for other in kept_questions
        ):
            report["near_duplicate"] += 1
            continue

        kept.append(pair)
        kept_questions.append(normalized)

    report["kept"] = len(kept)
    return kept, dict(report)


def sample_for_review(pairs: list[QAPair], k: int, seed: int) -> list[QAPair]:
    return random.Random(seed).sample(pairs, min(k, len(pairs)))
