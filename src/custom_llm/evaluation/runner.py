"""Runs every approach over the evaluation set, scores the answers and aggregates the results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from custom_llm.dataset_builder import OUT_OF_SCOPE, EvalItem
from custom_llm.evaluation.judge import Judge
from custom_llm.evaluation.metrics import exact_match, is_refusal, key_term_recall, token_f1
from custom_llm.inference import Approach

REVIEW_COLUMNS = [
    "approach",
    "item_id",
    "question",
    "gold",
    "reference",
    "prediction",
    "judge_correct",
    "judge_supported",
    "judge_reason",
]


def score_item(item: EvalItem, prediction: str, judge: Judge) -> dict:
    if item.regime == OUT_OF_SCOPE:
        return {"refused": float(is_refusal(prediction))}

    verdict = judge.evaluate(item, prediction)
    return {
        "exact_match": exact_match(prediction, item.gold or ""),
        "token_f1": token_f1(prediction, item.gold or ""),
        "key_term_recall": key_term_recall(prediction, item.key_terms),
        "judge_correct": float(verdict.correct) if verdict else np.nan,
        "judge_supported": float(verdict.supported) if verdict else np.nan,
        "judge_reason": verdict.reason if verdict else "",
    }


def run_evaluation(
    approaches: dict[str, Approach],
    items: list[EvalItem],
    judge: Judge,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name, approach in approaches.items():
        for item in items:
            answer = approach.answer(item.question)
            retrieved = answer.retrieved_fact_ids
            row = {
                "approach": name,
                "item_id": item.id,
                "regime": item.regime,
                "question": item.question,
                "gold": item.gold,
                "reference": item.reference,
                "prediction": answer.text,
                "latency_s": answer.latency_s,
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "retrieval_hit": float(item.fact_id in retrieved)
                if item.fact_id and retrieved
                else np.nan,
            }
            row.update(score_item(item, answer.text, judge))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """Per-approach, per-regime summary for the in-domain items."""
    in_domain = records[records.regime != OUT_OF_SCOPE]
    table = in_domain.groupby(["approach", "regime"]).agg(
        n=("item_id", "count"),
        key_term_recall=("key_term_recall", "mean"),
        exact_match=("exact_match", "mean"),
        token_f1=("token_f1", "mean"),
        judge_correct=("judge_correct", "mean"),
        hallucination_rate=("judge_supported", lambda s: 1 - s.mean()),
        judge_missing=("judge_supported", lambda s: int(s.isna().sum())),
        retrieval_hit=("retrieval_hit", "mean"),
        latency_p50_s=("latency_s", "median"),
        latency_p95_s=("latency_s", lambda s: s.quantile(0.95)),
        completion_tokens=("completion_tokens", "mean"),
    )
    return table.round(3)


def summarize_out_of_scope(records: pd.DataFrame) -> pd.DataFrame:
    ood = records[records.regime == OUT_OF_SCOPE]
    return (
        ood.groupby("approach")
        .agg(n=("item_id", "count"), refusal_rate=("refused", "mean"))
        .round(3)
    )


def judge_review_sample(records: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    in_domain = records[records.regime != OUT_OF_SCOPE]
    sample = in_domain.sample(n=min(n, len(in_domain)), random_state=seed)
    return sample[REVIEW_COLUMNS].assign(human_correct=None, human_supported=None)


def judge_agreement(labelled: pd.DataFrame) -> dict[str, float] | None:
    """Percent agreement between the judge and a human reviewer, on the rows already labelled."""
    done = labelled.dropna(subset=["human_correct", "human_supported"])
    if done.empty:
        return None
    return {
        "labelled": float(len(done)),
        "correct_agreement": float((done["judge_correct"] == done["human_correct"]).mean()),
        "supported_agreement": float((done["judge_supported"] == done["human_supported"]).mean()),
    }
