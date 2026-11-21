"""Deterministic metrics: exact match, token F1, key-term recall and refusal detection."""

from __future__ import annotations

from collections import Counter

from custom_llm.utils import normalize_text, term_present

REFUSAL_MARKERS = (
    "nao encontrei",
    "nao tenho essa informacao",
    "nao tenho informacoes",
    "nao consta",
    "nao ha informacao",
    "nao possuo",
    "nao sei",
    "fora do escopo",
    "nao consigo responder",
)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def key_term_recall(prediction: str, key_terms: tuple[str, ...]) -> float:
    """Share of the fact's key terms present in the answer, matched on whole tokens."""
    if not key_terms:
        return float("nan")
    hits = sum(term_present(prediction, term) for term in key_terms)
    return hits / len(key_terms)


def is_refusal(prediction: str) -> bool:
    text = normalize_text(prediction)
    return any(marker in text for marker in REFUSAL_MARKERS)
