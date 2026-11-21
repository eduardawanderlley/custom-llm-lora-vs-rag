"""Step 1: build the synthetic dataset.

Generates question/answer pairs from the knowledge base, filters them, splits by fact, and writes
the training file, the evaluation set and a sample for manual review.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict

from custom_llm.config import load_config
from custom_llm.dataset_builder import build_dataset, check_leakage, load_out_of_scope, split_facts
from custom_llm.generation.llm_client import ChatClient
from custom_llm.generation.qa_generator import generate_all
from custom_llm.generation.quality import filter_pairs, sample_for_review
from custom_llm.knowledge import load_knowledge_base, load_updates
from custom_llm.utils import set_seed, write_json, write_jsonl


def write_review_sample(path, pairs, facts_by_id) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fact_id", "question", "answer", "reference", "human_ok", "notes"])
        for pair in pairs:
            writer.writerow(
                [pair.fact_id, pair.question, pair.answer, facts_by_id[pair.fact_id].passage, "", ""]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    base_kb = load_knowledge_base(cfg.paths.knowledge_base)
    update_version, updates = load_updates(cfg.paths.knowledge_updates)
    current_kb = base_kb.with_updates(updates, version=update_version)
    facts = base_kb.facts

    split = split_facts([f.id for f in facts], cfg.generation.heldout_fact_ratio, cfg.seed)
    write_json(cfg.paths.split_file, asdict(split))
    logging.info("Split: %d train facts, %d held-out facts", len(split.train_fact_ids), len(split.heldout_fact_ids))

    client = ChatClient(
        cfg.generation.model,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
    )
    raw_pairs = generate_all(client, facts, cfg.generation.questions_per_fact)
    pairs, filter_report = filter_pairs(
        raw_pairs, base_kb.fact_by_id(), cfg.generation.dedup_similarity
    )
    write_json(cfg.paths.quality_report_file, {"filters": filter_report, "facts": len(facts)})
    logging.info("Filters: %s", filter_report)

    train_rows, items = build_dataset(pairs, current_kb, updates, split, seed=cfg.seed)
    items += load_out_of_scope(cfg.paths.out_of_scope_questions)
    check_leakage(train_rows, items)

    write_jsonl(cfg.paths.train_file, train_rows)
    write_jsonl(cfg.paths.test_file, (asdict(item) for item in items))
    write_review_sample(
        cfg.paths.review_file,
        sample_for_review(pairs, cfg.generation.review_sample_size, cfg.seed),
        base_kb.fact_by_id(),
    )
    logging.info("Wrote %d training rows and %d evaluation items", len(train_rows), len(items))


if __name__ == "__main__":
    main()
